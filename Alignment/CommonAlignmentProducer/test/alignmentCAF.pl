#!/usr/bin/env perl
###############################################################################
# Parallel processing for alignment
# Similar to alignment.pl but 
# - faster running
# - accepts files on diskpools different from default CASTOR
# - saves disk space on output
#
# data.list is the name of a list of root files in the data/ directory
# It must contain filenames in this format:
#     '<file1.root>'
#     '<file2.root>'
#     ...
#     '<fileN.root>'
# NEW: you can choose the number of files processed in each parallel job
# and the STAGE_SVCCLASS location
###############################################################################

# determine location
$host=`echo \$HOST`;
if ( $host =~ /lxplus/ )     { $location="lxplus"; }
elsif ( $host =~ /lxcms/ )   { $location="lxcmsg1"; }

# job steering ----------------------------------------------------------------

# name of job
$jobname="test-caf";

# cfg file
$steering="alignment_noinput.cfg";

# db output files (to be deleted except for last iteration)
$dbfiles="alignments.db";

# db input file
$sqlitefile="";

# list of data
$datalist="data.list";
# number of files per job
$nfilesperjob=2;
# castor data location
# $thepool="wan";
$thepool="cmscaf";

# number of iterations (excluding initial step)
$iterations=10;

# interactive or lxbatch queue
# $farm="I";
# $farm="8nm"; $resource="";
$farm="dedicated -R cmscaf";

$cmsswvers="CMSSW_1_7_5";
$scramarch="slc4_ia32_gcc345";
$scram=scramv1;

# sleep time in seconds between two check cycles
$sleeptime=30;

# site-specific paths etc

if ( $location eq "lxcmsg1" ) {
  $homedir="/afs/cern.ch/user/c/covarell/scratch0/goodalign";
  $basedir="${homedir}/${cmsswvers}";
  $outdir="/data/covarell/joboutput";
}
elsif ( $location eq "lxplus" ) {
  # die "ERROR: Root files for the analysis are on lxcmsg1!\n";
   $homedir="/afs/cern.ch/user/c/covarell/scratch0/goodalign";
   $basedir="${homedir}/${cmsswvers}";
   $outdir="/afs/cern.ch/user/c/covarell/scratch0/joboutput";
}
else {
  die "ERROR: location: $location unknown!\n";
}

# -----------------------------------------------------------------------------

$workdir="${basedir}/src/Alignment/CommonAlignmentProducer/test";
$datalistdir="${basedir}/src/Alignment/CommonAlignmentProducer/data/" . ${datalist};

print "\n";
print "-------------------------------------------------------------------------------\n";
print "H I P   A l i g n m e n t \n";
print "-------------------------------------------------------------------------------\n";
print "Location: $location\n";
print "Workdir: ${workdir}\n";
print "Outdir: ${outdir}/${jobname}/${cmsswvers}\n";

# Calculate njobs (= nfiles / nfilesperjob, last files not taken)
system("more ${datalistdir} | grep root | wc -l >> nofiles.txt;");
open(INFILE,"nofiles.txt") or die "cannot open nofiles.txt";;
@nofiles=<INFILE>;
close(INFILE);
foreach $line1 (@nofiles) {$njobs = int($line1 / ${nfilesperjob});}
system("rm -f nofiles.txt;");

if ($njobs>1) { 
  print "Parallel Jobs: ${njobs} with ${nfilesperjob} file(s) per job\n";
}
else {
  # only a few files 
  print "Only one job (no parallel): please use the default alignment.pl script.\n";
  exit;
}
print "Iterations: ${iterations}\n";

$odir="$outdir/$jobname";

# -----------------------------------------------------------------------------
# create sandbox and set up local environment

system("
rm -rf $outdir/$jobname;
mkdir $outdir/$jobname;
cd $outdir/$jobname;
$scram project CMSSW $cmsswvers > /dev/null;
cd $cmsswvers;
mkdir -p lib/${scramarch};
mkdir -p bin/${scramarch};
mkdir -p external/${scramarch};
mkdir -p module/${scramarch};
mkdir -p tmp;
cp $basedir/lib/${scramarch}/* lib/${scramarch} > /dev/null;
cp $basedir/external/${scramarch}/* external/${scramarch} > /dev/null;
");

# deal with modified cf? files
chdir "$basedir/src";
@cfgfilelist = `ls */*/*/*.cf?`;
foreach $file ( @cfgfilelist ) {
  system("tar rf cfg.tar $file");
}
system("cp cfg.tar $odir/$cmsswvers/src; rm cfg.tar; cd $odir/$cmsswvers/src; tar xf cfg.tar");


# if one cpu create subjob and submit #########################################

if ($njobs == 1) {

  # make subjob
  $dir="$outdir/$jobname/$cmsswvers/main";
  make_subjob($dir,1);

  # create cfg file
  system("cp ${workdir}/$steering $dir/cfgfile");
  $repl="replace maxEvents.input  = -1 ";
  replace("$dir/cfgfile",$repl);

  if ($farm eq "I") {
    print "Run interactively ...\n";
    system("cd $dir; ./subjob");
  }
  else {
    print "Submit to farm ...\n";
    system("bsub -i /tmp/junk -q $farm < $dir/subjob");
  }

}

# PARALLEL ####################################################################

else {

# -----------------------------------------------------------------------------
# create N job directories and set up steering

$ijob=1;
$ifile=0;

open(INFILE2,"$datalistdir") or die "cannot open $datalistdir";;
@listdata=<INFILE2>;
close(INFILE2);

foreach $line (@listdata) { 
    if ($line =~ /root/) {
	${ifile}++;
        if (${nfilesperjob} == 1) {   # 1 FILE PER JOB
            # make job dir and copy aux files
	    $dir="$outdir/$jobname/$cmsswvers/job$ijob";
	    make_subjob($dir,0);
	    
	    # create cfg file
	    system("cp ${workdir}/$steering $dir/cfgfile");
	    if ($dbfiles) {
		$repl= "source = PoolSource { 
		
		   untracked vstring fileNames = {
                   $line
                   }
                   untracked uint32 skipEvents = 0	
	           }
                   replace maxEvents.input  = -1 
                   replace AlignmentProducer.saveToDB = false
                  ";
	    } else {
		$repl= "source = PoolSource { 
		
		   untracked vstring fileNames = {
                   $line
                   }
                   untracked uint32 skipEvents = 0	
	           }
                   replace maxEvents.input  = -1 
                  ";
	    }
	    replace("$dir/cfgfile",$repl);
	    ${ijob}++;
	} else {   # 2 OR MORE FILES PER JOB
	    if ($ifile % ${nfilesperjob} == 0) { 
		
		# make job dir and copy aux files
		$dir="$outdir/$jobname/$cmsswvers/job$ijob";
		make_subjob($dir,0);
		
		# create cfg file
		system("cp ${workdir}/$steering $dir/cfgfile");
		if ($dbfiles) {
		    $repl= $repl . "
                       $line
                      }
                      untracked uint32 skipEvents = 0	
	           }
                   replace maxEvents.input  = -1 
                   replace AlignmentProducer.saveToDB = false
                  ";
		} else {
		    $repl= $repl . "
                       $line
                     }
                      untracked uint32 skipEvents = 0	
	           }
                   replace maxEvents.input  = -1 
                  ";
		}
		replace("$dir/cfgfile",$repl);
		${ijob}++;
	    
	    } elsif ($ifile % ${nfilesperjob} == 1) {
		$repl= "source = PoolSource { 
		
		  untracked vstring fileNames = { 
                  $line,
                   ";	
	    } else {
		$repl= $repl . "$line,";
	    }
	}
    }
};

# -----------------------------------------------------------------------------
# create collector job and steering

$dir="$outdir/$jobname/$cmsswvers/main";
make_subjob($dir,0);

# create cfg file
system("cp ${workdir}/$steering $dir/cfgfile");
$inutile=1;
foreach $line (@listdata) { 
  if ($inutile == 1) {   
      ${inutile}++;
      $repl="
        source = PoolSource { 
		
		untracked vstring fileNames = { 
                $line
                }
        }
        replace maxEvents.input = 1
        replace HIPAlignmentAlgorithm.collectorActive = true
        replace HIPAlignmentAlgorithm.collectorNJobs = $njobs
        replace HIPAlignmentAlgorithm.collectorPath = \"../\"
      ";
  }
}
replace("$dir/cfgfile",$repl);


# -----------------------------------------------------------------------------

$dir="$outdir/$jobname/$cmsswvers";
$iteration=0;

system("rm -f $dir/iteration.txt");
open(FILE,">$dir/iteration.txt");
print FILE "$iteration";
close(FILE);

# enter loop ... ==============================================================

LOOP: {

  # check if all are finished
  if ($iteration gt 0 ) { $alldone=&check_finished(); }

  # need to run collector / resubmit for next iteration
  if ( $alldone eq 1 or $iteration eq 0) {

    # run collector
    print "Run collector for iteration $iteration ...\n";
    $iret=&run_collector();
    if ($iret ne 0) { die "ERROR in collector!\n";}

    # submit jobs for next iteration
    if ($iteration < $iterations) {
      $iteration++;

      system("rm -f $dir/iteration.txt");
      open(FILE,">$dir/iteration.txt");
      print FILE "$iteration";
      close(FILE);

      print "New iteration $iteration --------------------------------------------------------------\n";
      print "Submit all jobs for iteration $iteration ...\n";
      $iret=&submit_jobs;
      if ($iret ne 0) { die "ERROR in submit jobs!\n";}

      redo LOOP;
    }
    # we are done
    else {
      system("
        cd $dir
        rm -rf tmp src logs lib config bin
     ");
      print "Finished!\n";
    }

  }

  # else, just wait ...
  else {
    # print "sleeping for $sleeptime seconds ...\n";
    sleep($sleeptime);
    redo LOOP;
  }

}

# save some disk space
system("cd $dir; rm -rf job*; rm -rf LSFJOB*"); 

}

exit 0;

###############################################################################

sub check_finished  
{

$alldone=1;
$ijob=1;

$timestamp=`date +%H:%M:%S_%d/%m/%y`;

print "Job status: ";

while ( $ijob <= $njobs ) {
  if ( -e "$dir/job$ijob/DONE" ) { print "1 "; }
  else { print "0 "; $alldone=0; }
  $ijob++;
}

print "at $timestamp";
if ($alldone eq 1) { return 1;}
else { return 0; }


}

###############################################################################

sub run_collector 
{
  system(" 
    cd $dir/main
    subjob
  ");
  if ($iteration gt 0) {
    system("cp $dir/job1/HIPAlignmentEvents.root $dir/main");
  }
  $myjob=2;
  while ( $myjob <= $njobs ) {
    system("rm -f $dir/job$myjob/HIPAlignmentEvents.root");
    ${myjob}++;
  }
  return 0;
}

###############################################################################

sub submit_jobs 
{

$ijob=1;
while ( $ijob <= $njobs ) {

  system("
    rm -f $dir/job$ijob/DONE

    cp $dir/main/IOTruePositions.root       $dir/job$ijob
    cp $dir/main/IOMisalignedPositions.root $dir/job$ijob
    cp $dir/main/IOAlignedPositions.root    $dir/job$ijob
    cp $dir/main/IOIteration.root           $dir/job$ijob

  ");

  if ( $farm eq "I" ) {
    print "Run job $ijob interactively ...\n";
    system("$dir/job$ijob/subjob &");
  }
  else {
    print "Submit job $ijob ... ";
    $test=0;
    while($test == 0) {
      $rc=system("cd $dir ; bsub -o /tmp/junk$ijob -q $farm < job$ijob/subjob");
      # $rc=system("cd $dir ; bsub -q $farm < job$ijob/subjob");
      if ($rc == 0) { $test=1; }
      else {
	print "ERROR in submitting job: $rc  .. retrying in 10s ...\n";
	sleep(5);
      }
    }
  }
  sleep(1);
  $ijob++;
}

  return 0;
}

###############################################################################

sub make_subjob {

$dir=@_[0];
$single=@_[1];

system("mkdir $dir");
system("cp ${workdir}/$authfile $dir");
if ($sqlitefile) { system("cp ${workdir}/$sqlitefile $dir"); }
# if ($condbcatalogfile) { system("cp ${workdir}/$condbcatalogfile $dir"); }

if ($single == 1) {

# for single job
system("touch $dir/timing.log");
$subjob="#!/bin/zsh -f
#BSUB -J \"ALIGN\"
#BSUB -C 0
cd $dir
eval \`$scram runtime -sh\`
export STAGE_SVCCLASS=${thepool}
rehash
thisiter=1
until ((thisiter > $iterations )); do
rm -f ${dbfiles}
echo \"Iteration \${thisiter} ...\"
date | read starttime
EdmPluginRefresh
cmsRun cfgfile > /dev/null
date | read endtime
rm -f log.\${thisiter}.log
mv alignment.log alignment.\${thisiter}.log
gzip alignment.\${thisiter}.log
echo \"Iteration \${thisiter}: \${starttime} ... \${endtime}\" >> timing.log
let thisiter++
done
";

}

else {

# for parallel
$subjob="#!/bin/zsh 
#BSUB -J \"ALIGN\" 
#BSUB -C 0
cd $dir
eval \`$scram runtime -sh\`
export STAGE_SVCCLASS=${thepool}
rehash
cat ../iteration.txt | read iter
rm -f ${dbfiles}
EdmPluginRefresh
cmsRun cfgfile > log.\$iter.log
gzip log.\$iter.log
mv alignment.log alignment.\$iter.log
gzip alignment.\$iter.log
touch DONE
";

}

open(FILE,">$dir/subjob");
print FILE "$subjob";
close(FILE);
system("chmod u+x $dir/subjob");

}

###############################################################################

sub replace {

$infile = @_[0];
$repl = @_[1];

open(INFILE,"$infile") or die "cannot open $infile";;
@log=<INFILE>;
close(INFILE);

system("rm -f tmp");
open(OUTFILE,">tmp");

foreach $line (@log) {
  if ($line =~ /REPLACEME/) { print OUTFILE $repl; }
  else { print OUTFILE $line; }
}

close(OUTFILE);
system("mv tmp $infile");

}
