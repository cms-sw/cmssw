#!/usr/bin/env perl
#
#  This script is part of the Kalman Alignment Production System (KAPS).
#  It is an adapted version of mps_setup.pl, a part of the MillePede
#  Production System (MPS), developed by R. Mankel (DESY).
#
#  Setup local kaps database
#  
#
#  Usage:
#
#  kaps_setup.pl batchScript cfgTemplate infiList nJobs class[:classMerge] jobname [mergeScript] [mssDir]
#
# class can be - any of the normal LSF queues (8nm,1nh,8nh,1nd,2nd,1nw,2nw)
#              - special CAF queues (cmscaf)

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/kapslib");
}
use Kapslib;

$batchScript = "undefined";
$cfgTemplate = "undefined";
$infiList = "undefined";
$nJobs = 0;
$class = "S";
$addFiles = "";
$driver = "";
$mergeScript = "";
$mssDir = "";
$append = 0;

# parse the arguments
while (@ARGV) {
  $arg = shift(ARGV);
  if ($arg =~ /\A-/) {  # check for option 
    if ($arg =~ "h") {
      $helpwanted = 1;
    }
    elsif ($arg =~ "d") {
      $localdir = 1;
    }
    elsif ($arg =~ "u") {
      $updateDb = 1;
    }
    elsif ($arg =~ "m") {
      $driver = "merge";
      print "option sets mode to $driver\n";
    }
    elsif ($arg =~ "a" && -r "kaps.db") {
      $append = 1;
      print "option sets mode to append\n";    
    }

    $optionstring = "$optionstring$arg";
  }
  else {                # parameters not related to options
    $i = $i + 1;
    if ($i eq 1) {
      $batchScript = $arg;
    }
    if ($i eq 2) {
      $cfgTemplate = $arg;
    }
    if ($i eq 3) {
      $infiList = $arg;
    }
    elsif ($i eq 4) {
      $nJobs = $arg;
    }
    elsif ($i eq 5) {
      $class = $arg;
    }
    elsif ($i eq 6) {
      $addFiles = $arg;
    }
    elsif ($i eq 7) {
      $mergeScript = $arg;
    }
    elsif ($i eq 8) {
      $mssDir = $arg;
    }
  }
}

# test input parameters
if ($nJobs eq 0 or $helpwanted != 0 ) {
  print "Usage:\n  kaps_setup.pl [options] batchScript cfgTemplate infiList nJobs class[:classMerge] jobname [mergeScript] [mssDir]";
  print "\nKnown options:";
  print "  \n -m   Setup merging job.";
  print "  \n -a   Append jobs to existing list.\n";
  exit 1;
}

unless (-r $batchScript) {
  print "Bad batchScript script name $batchScript\n";
  exit 1;
}
unless (-r $cfgTemplate) {
  print "Bad cfg template file name $cfgTemplate\n";
  exit 1;
}
unless (-r $infiList) {
  print "Bad input list file $infiList\n";
  exit 1;
}
unless (index("lxplus cmscaf cmsexpress 8nm 1nh 8nh 1nd 2nd 1nw 2nw",get_class())>-1) {
  print "Bad job class '$class'\n";
  exit 1;
}

if ($driver eq "merge") {
  if ($mergeScript eq "") {
    $mergeScript = $batchScript . "merge";
  }
  unless (-r $mergeScript) {
    print "Bad merge script file name $mergeScript\n";
    exit 1;
  }
}
if ($mssDir ne "") {
  $testMssDir = `ls -d $mssDir`;
  chomp $testMssDir;
  if ($testMssDir eq "") {
    print "Bad MSS directory name $mssDir\n";
  }
}

# Create the job directories
my $nJobExist="";
if ($append==1 && -d "jobData") {
# Append mode, and "jobData" exists
  $nJobExist = `ls jobData | grep 'job[0-9][0-9][0-9]' | tail -1`;
  $nJobExist =~ s/job//;
}

if ($nJobExist eq "" || $nJobExist <=0 || $nJobExist>999) {
# Delete all
  system "rm -rf jobData";
  system "mkdir jobData";
  $nJobExist = 0;
}

for ($j = 1; $j <= $nJobs; ++$j) {
  $i = $j+$nJobExist;
  $jobdir = sprintf "job%03d",$i;
  print "jobdir $jobdir\n";
  system "mkdir jobData/$jobdir";
}

# build the absolute job directory path (needed by kaps_script)
$thePwd = `pwd`;
chomp $thePwd;
$theJobData = "$thePwd/jobData";
print "theJobData= $theJobData \n";

if ($append == 1) {
# save current values
  my $tmpNJobs = $nJobs;
  my $tmpInfiList = $infiList;
  my $tmpDriver = $driver;

# Read DB file
  read_db();

# check if last job is a merge job
  if (@JOBDIR[$nJobs] eq "jobm") {
# remove the merge job
    pop @JOBDIR;
    pop @JOBID;
    pop @JOBSTATUS;
    pop @JOBNTRY;
    pop @JOBRUNTIME;
    pop @JOBNEVT;
    pop @JOBHOST;
    pop @JOBINCR;
    pop @JOBREMARK;
    pop @JOBSP1;
    pop @JOBSP2;
    pop @JOBSP3;
  }

# Restore variables
  $nJobs = $tmpNJobs;
  $infiList = $tmpInfiList;
  $driver = $tmpDriver;
}


# Create (update) the local database
for ($j = 1; $j <= $nJobs; ++$j) {
  $i=$j+$nJobExist;
  $theJobDir = sprintf "job%03d",$i;
  push @JOBDIR,$theJobDir;
  push @JOBID,0;
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
  # create the split card files
  print "kaps_split.pl $infiList $j $nJobs >jobData/$theJobDir/theSplit\n";
  system "kaps_split.pl $infiList $j $nJobs >jobData/$theJobDir/theSplit";
  if ($?) {
    print "              split failed\n";
    @JOBSTATUS[$i-1] = "FAIL";
  }
  $theIsn = sprintf "%03d",$i;
  print "kaps_splice.pl $cfgTemplate jobData/$theJobDir/theSplit jobData/$theJobDir/the_cfg.py $theIsn $mssDir\n";
  system "kaps_splice.pl $cfgTemplate jobData/$theJobDir/theSplit jobData/$theJobDir/the_cfg.py $theIsn $mssDir";
  # create the run script
  print "kaps_script.pl $batchScript  jobData/$theJobDir/theScript.sh $theJobData/$theJobDir the_cfg.py jobData/$theJobDir/theSplit $theIsn $mssDir\n";
  system "kaps_script.pl $batchScript  jobData/$theJobDir/theScript.sh $theJobData/$theJobDir the_cfg.py jobData/$theJobDir/theSplit $theIsn $mssDir";
}

# create the merge job entry. This is always done. Whether it is used depends on the "merge" option.
$theJobDir = "jobm";
push @JOBDIR,$theJobDir;
push @JOBID,0;
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

# if merge mode, create the directory and set up contents
if ($driver eq "merge") {

  system "rm -rf jobData/jobm";
  system "mkdir jobData/jobm";
  print "Create dir jobData/jobm\n";

  # We want to merge old and new jobs
  my $nJobsMerge = $nJobs+$nJobExist;

  # create  merge job cfg
  print "kaps_merge.pl $cfgTemplate jobData/jobm/merge_cfg.py $theJobData/jobm $nJobsMerge $mssDir\n";
  system "kaps_merge.pl $cfgTemplate jobData/jobm/merge_cfg.py $theJobData/jobm $nJobsMerge $mssDir";

  # create merge job script
  print "kaps_scriptm.pl $mergeScript jobData/jobm/theScript.sh $theJobData/jobm merge_cfg.py $nJobsMerge $mssDir\n";
  system "kaps_scriptm.pl $mergeScript jobData/jobm/theScript.sh $theJobData/jobm merge_cfg.py $nJobsMerge $mssDir";
}

# Create a backup of batchScript, cfgTemplate, infiList (and mergeScript)
#   in jobData
$i = `ls jobData | grep 'ScriptsAndCfg[0-9][0-9][0-9]' | tail -1`;
$i =~ s/ScriptsAndCfg//;
$i =~ s/.tar//;
$i++;
$ScriptCfg = sprintf "ScriptsAndCfg%03d",$i;
system "mkdir jobData/$ScriptCfg";
system "cp $batchScript $cfgTemplate $infiList jobData/$ScriptCfg/.";
if ($driver eq "merge") {
  system "cp $mergeScript jobData/$ScriptCfg/.";
}
system "tar -cf jobData/$ScriptCfg.tar jobData/$ScriptCfg";
system "rm -rf jobData/$ScriptCfg";


# Write to DB
write_db();
read_db();
print_memdb();
