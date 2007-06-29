#!/usr/bin/perl
###############################################################################
# Parallel processing for alignment
###############################################################################

# determine location
$host=`echo \$HOST`;
if    ( $host =~ /fpslife/ ) { $location="fpslife"; }
elsif ( $host =~ /lxplus/ )  { $location="lxplus"; }
elsif ( $host =~ /cnaf/ )    { $location="cnaf"; }
elsif ( $host =~ /lxcms/ )  { $location="lxplus"; }

# job steering ----------------------------------------------------------------

# name of job
$jobname="test";

# cfg file
$steering="alignment.cfg";

# db output files to be deleted except for last iteration
$dbfiles="Alignments.db condbcatalog.xml";

# db authentication file to be copied to exec dir
$authfile="authentication.xml";

# db input file
# $sqlitefile="CSA06Scenario.db"; $condbcatalogfile="condbcatalog.xml";
$sqlitefile=""; $condbcatalogfile="";

# number of events per job
$nevent=10000;

# first event
$firstev=0;

# number of jobs
$njobs=1;

# number of iterations (excluding initial step)
$iterations=3;

# interactive or lxbatch queue
$farm="I";
# $farm="8nm"; $resource="";
# $farm="dedicated -R cmsalca";

$cmsswvers="CMSSW_1_2_0_pre3";
$scramarch="slc3_ia32_gcc323";
$scram=scramv1;

# sleep time in seconds between two check cycles
$sleeptime=30;

# site-specific paths etc

if ( $location eq "lxplus" ) {
  $homedir="/afs/cern.ch/user/f/fpschill";
  $basedir="${homedir}/w1/cmssw/${cmsswvers}";
  $outdir="${homedir}/scratch0/joboutput";
}
elsif ( $location eq "fpslife" ) {
   die "ERROR: location: $location unsupported!\n";
#  $homedir="/afs/cern.ch/user/f/fpschill";
#  $basedir="${homedir}/cms/${cmsswvers}";
#  $outdir="/x01/usr/fpschill/cms/joboutput";
}
elsif ( $location eq "cnaf" ) {
   die "ERROR: location: $location unsupported!\n";
#  $homedir="/home/CMS/fpschill";
#  $basedir="${homedir}/${cmsswvers}";
#  $outdir="/afs/infn.it/cnaf/user/fpschill/joboutput";
}
else {
  die "ERROR: location: $location unknown!\n";
}

# -----------------------------------------------------------------------------

$workdir="${basedir}/src/Alignment/CommonAlignmentProducer/test";

print "\n";
print "-------------------------------------------------------------------------------\n";
print "A l i g n m e n t \n";
print "-------------------------------------------------------------------------------\n";
print "Location: $location\n";
print "Workdir: ${workdir}\n";
print "Outdir: ${outdir}\n";
if ($njobs>1) { 
  print "Parallel Jobs: ${njobs} with $nevent events per job\n";
}
else {
  print "Events: $nevent \n";
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
mkdir -p config/${scramarch};
cp $basedir/lib/${scramarch}/* lib/${scramarch} > /dev/null;
cp $basedir/module/${scramarch}/* module/${scramarch} > /dev/null;
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
  $repl="replace maxEvents  = { untracked int32 input = $nevent }";
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
$ifirst=$firstev;

while ( ${ijob} <= ${njobs} ) {
  print "Job ${ijob}/${njobs} starting at event $ifirst \n";

  # make job dir and copy aux files
  $dir="$outdir/$jobname/$cmsswvers/job$ijob";
  make_subjob($dir,0);

  # create cfg file
  system("cp ${workdir}/$steering $dir/cfgfile");
  $repl="
    replace PoolSource.maxEvents  = $nevent 
    replace PoolSource.skipEvents = $ifirst 
    replace AlignmentProducer.saveToDB = false
  ";
  replace("$dir/cfgfile",$repl);

  ${ijob}++;
  ${ifirst}=${ifirst}+${nevent};
};

# -----------------------------------------------------------------------------
# create collector job and steering

$dir="$outdir/$jobname/$cmsswvers/main";
make_subjob($dir,0);

# create cfg file
system("cp ${workdir}/$steering $dir/cfgfile");
$repl="
  replace PoolSource.maxEvents  = 1
  replace PoolSource.skipEvents = 0
  replace HIPAlignmentAlgorithm.collectorActive = true
  replace HIPAlignmentAlgorithm.collectorNJobs = $njobs
  replace HIPAlignmentAlgorithm.collectorPath = \"../\"
";
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
    ./subjob
  ");
  if ($iteration gt 0) {
    system("cp $dir/job1/HIPAlignmentEvents.root $dir/main");
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
      $rc=system("cd $dir ; bsub -o /tmp/junk -q $farm < job$ijob/subjob");
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
if ($sqlitefile !="") { system("cp ${workdir}/$sqlitefile $dir"); }
if ($condbcatalogfile !="") { system("cp ${workdir}/$condbcatalogfile $dir"); }

if ($single == 1) {

# for single job
system("touch $dir/timing.log");
$subjob="#!/bin/zsh -f
#BSUB -J \"ALIGN\"
#BSUB -C 0
cd $dir
eval \`$scram runtime -sh\`
rehash
thisiter=1
until ((thisiter > $iterations )); do
rm -f ${dbfiles}
echo \"Iteration \${thisiter} ...\"
date | read starttime
cmsRun cfgfile >& log.\${thisiter}.log
date | read endtime
gzip log.\${thisiter}.log
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
rehash
cat ../iteration.txt | read iter
rm -f $dbfiles
cmsRun cfgfile >& log.\$iter.log
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
