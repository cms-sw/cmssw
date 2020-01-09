#!/usr/bin/env perl
#     G. Fluck, Uni Hamburg    13-May-2009
#     $Revision: 1.3 $ by $Author: jbehr $
#     $Date: 2012/09/10 13:10:37 $
#
#  Setup an extra pede
#
#  Usage:
#
#  mps_setupm.pl 

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;

my $chosenMerge = -1;
my $onlyactivejobs = -1;
my $ignoredisabledjobs = -1;

## parse the arguments
my $i = 0;
while (@ARGV) {
  $arg = shift(@ARGV);
  if ($arg =~ /\A-/) {  # check for option 
    if ($arg =~ "h") {
      $helpwanted = 1;
    }
    elsif ($arg =~ "a") {
      $onlyactivejobs = 1;
    }
    elsif ($arg =~ "d") {
      $ignoredisabledjobs = 1;
    }
    else {
	print "\nWARNING: Unknown option: ".$arg."\n\n";
	$helpwanted = 1;
    }
#    elsif ($arg =~ "u") {
#      $updateDb = 1;
  }
  else {  # parameters not related to options
    $i = $i + 1;
    if ($i eq 1) {
	$chosenMerge = $arg;
	if ($chosenMerge < 0) { # not allowed (but probably interpreted as option before...)!
	    $helpwanted = 1;
	}
    }
#    if ($i eq 2) {
#      $cfgTemplate = $arg;
#    }
  }
}


if ( $helpwanted != 0 ) {
  print "Usage:\n  mps_setupm.pl [options] [mergeJobId]";
  print "\nDescription:\n  Sets up an additional merge job (so needs a setup merge job).";
  print "\n  Configuration is copied from";
  print "\n    - last previous merge job if 'mergeJobId' is NOT specified,";
  print "\n    - job directory 'jobm<mergeJobId>' otherwise (so: mergeJobId >= 0).";
  print "\n  Do not forget to edit the configuration file starting the new job.";
  print "\nKnown options:";
  print "\n  -h   This help.\n";
  print "\n  -a   Use only active jobs (jobs with state = 'ok').\n";
  print "\n  -d   Ignore disabled jobs.\n";

  exit 1;
}

read_db();

# check for existing merge job?
if ($driver ne "merge") {
    print "Needs previously setup merge job.\n";
    exit 1;
}

my $iOldMerge = @JOBDIR-1; # merge job where to copy from: here default to last one
if ($chosenMerge >= 0) {
    $iOldMerge = $nJobs + $chosenMerge; # nJobs counts mille jobs
}
if ($iOldMerge >= @JOBDIR) {
    print "Cannot find previous merge job, exit!\n";
    exit 1;
}
my $nMerge = @JOBDIR - $nJobs; # 'index' of this merge job to achieve jobm, jobm1, jobm2, ...

# add a new merge job entry. 
$theJobDir = "jobm".$nMerge;
push @JOBDIR,$theJobDir;
push @JOBID,"";
push @JOBSTATUS,"SETUP";
push @JOBNTRY,0;
push @JOBRUNTIME,0;
push @JOBNEVT,0;
push @JOBHOST,"";
push @JOBINCR,0;
push @JOBREMARK,"";
push @JOBSP1,"";
push @JOBSP2,"";
push @JOBSP3,"";

# create the directory and set up contents
print "Create dir jobData/$theJobDir\n";
system "rm -rf jobData/$theJobDir";
system "mkdir jobData/$theJobDir";
system "ln -s `readlink -e .TrackerTree.root` jobData/$theJobDir/";

# build the absolute job directory path (needed by mps_scriptm)
$thePwd = `pwd`;
chomp $thePwd;
$theJobData = "$thePwd/jobData";

# copy merge job cfg from last merge job
print "cp -p jobData/$JOBDIR[$iOldMerge]/alignment_merge.py $theJobData/$theJobDir\n";
system "cp -p jobData/$JOBDIR[$iOldMerge]/alignment_merge.py $theJobData/$theJobDir";

# copy weights configuration from last merge job
system "cp -p jobData/$JOBDIR[$iOldMerge]/.weights.pkl $theJobData/$theJobDir";

my $tmpc = "";
$tmpc = " -c" if($onlyactivejobs == 1);

# create merge job script
print "mps_scriptm.pl${tmpc} $mergeScript jobData/$theJobDir/theScript.sh $theJobData/$theJobDir alignment_merge.py $nJobs $mssDir $mssDirPool\n";
system "mps_scriptm.pl${tmpc} $mergeScript jobData/$theJobDir/theScript.sh $theJobData/$theJobDir alignment_merge.py $nJobs $mssDir $mssDirPool";

# Write to DB
write_db();

if($onlyactivejobs == 1 || $ignoredisabledjobs == 1) {
  print "try to open <$theJobData/$theJobDir/alignment_merge.py\n";
  open INFILE,"$theJobData/$theJobDir/alignment_merge.py" || die "error: $!\n";
  undef $/;
  my $filebody = <INFILE>;  # read whole file
  close INFILE;
  # build list of binary files
  my $binaryList = "";
  my $iIsOk = 1;
  for (my $i=1; $i<=$nJobs; ++$i) {
    my $sep = ",\n                ";
    if ($iIsOk == 1) { $sep = "\n                " ;}

    next if ( $ignoredisabledjobs == 1 && $JOBSTATUS[$i-1] =~ /DISABLED/gi );
    next if ( $onlyactivejobs == 1 && $JOBSTATUS[$i-1] ne "OK");
    ++$iIsOk;

    my $newName = sprintf "milleBinary%03d.dat",$i;
    if($JOBSP2[$i-1] ne "" && defined $JOBSP2[$i-1])
      {
        my $weight = $JOBSP2[$i-1];
        print "Adding $newName to list of binary files using weight $weight\n";
        $binaryList = "$binaryList$sep\'$newName -- $weight\'";
      }
    else
      {
        print "Adding $newName to list of binary files\n";
        $binaryList = "$binaryList$sep\'$newName\'";
      }

   # print "Adding $newName to list of binary files\n";
   # $binaryList = "$binaryList$sep\'$newName\'";
  }
  my $nn  = ($filebody =~ s/mergeBinaryFiles\s*=\s*\[(.|\n)*?\]/mergeBinaryFiles = \[$binaryList\]/);
  $nn += ($filebody =~ s/mergeBinaryFiles = cms.vstring\(\)/mergeBinaryFiles = \[$binaryList\]/);
  # build list of tree files
  my $treeList = "";
  $iIsOk = 1;
  for (my $i=1; $i<=$nJobs; ++$i) {
    my $sep = ",\n                ";
    if ($iIsOk == 1) { $sep = "\n                " ;}
    next if ( $ignoredisabledjobs == 1 && $JOBSTATUS[$i-1] =~ /DISABLED/gi );
    if ($JOBSTATUS[$i-1] ne "OK" && $onlyactivejobs == 1) {next;}
    ++$iIsOk;
    my $newName = sprintf "treeFile%03d.root",$i;
    $treeList = "$treeList$sep\'$newName\'";
  }
  # replace list of tree files
  $nn = ($filebody =~ s/mergeTreeFiles\s*=\s*\[(.|\n)*?\]/mergeTreeFiles = \[$treeList\]/);
  #print "$filebody\n";
  # store the output file
  open OUTFILE,">$theJobData/$theJobDir/alignment_merge.py";
  print OUTFILE $filebody;
  close OUTFILE;
}
