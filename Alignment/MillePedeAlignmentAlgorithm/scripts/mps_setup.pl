#!/usr/bin/env perl
#     R. Mankel, DESY Hamburg     08-Oct-2007
#     A. Parenti, DESY Hamburg    16-Apr-2008
#     $Revision: 1.26 $
#     $Date: 2011/11/22 14:21:01 $
#
#  Setup local mps database
#  
#
#  Usage:
#
#  mps_setup.pl [options] batchScript cfgTemplate infiList nJobs class[:classMerge] jobname [mergeScript [pool:]mssDir]
#
# class can be - any of the normal LSF queues (8nm,1nh,8nh,1nd,2nd,1nw,2nw)
#              - special CAF queues (cmscaf1nh, cmscaf1nd, cmscaf1nw)
#              - special CAF queues, for pede job (cmscafspec1nh, cmscafspec1nd, cmscafspec1nw). E.g. cmscafspec1nh corresponds to "-q cmscaf1nh -R cmscafspec"
# If class contains a ':', it will be split:
#              - the part before the ':' defines the class for Mille jobs,
#              - the part behind the class for the Pede job.
#              (eg cmscaf:cmscafspec)
# If mssDir contains a ':', it will be split:
#              - the part before ':' defines the pool,
#              - the part behind it the directory.
#              (eg cmscafuser:/castor/cern.ch/cms/store/...)
# Known options:
#  -m          Setup pede merging job.
#  -a          Append jobs to existing list.
#  -M pedeMem  The memory (MB) to be allocated for pede (min: 1024 MB).
#              If not given, it is evinced from the pede executable name.
#              Finally, it is set 2560 MB if neither of the two are available.
#  -N name     The name to be assigned to the jobs. Whitespaces and colons are not allowed.
#  -w          Assign statistical weight.

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;

$batchScript = "undefined";
$cfgTemplate = "undefined";
$infiList = "undefined";
$nJobs = 0;
$class = "S";
$addFiles = "";
$driver = "";
$mergeScript = "";
$mssDirPool = "";
$mssDir = "";
$append = 0;

my $confname = "";
my $weight = 1.0;
my $useweight = 0;

# parse the arguments
while (@ARGV) {
  my $arg = shift(ARGV);
  if ($arg =~ /\A-/) {  # check for option
    if ($arg =~ /-N/g) {
      $confname = $arg;
      $confname =~ s/-N//; # Strips away the "-N"
      if (length($confname) == 0) {
         $confname = shift(ARGV);
       }
      $confname =~ s/\s//g;
      if($confname =~ /\:/)
        {
          $confname =~ s/\://g;
          print "colons were removed in configuration name: $confname\n";
        }
    }
    elsif($arg =~ "w")
      {
        my $tmp = $arg;
        $tmp =~ s/-w//; # Strips away the "-w"
        if (length($tmp) == 0) {
          $tmp = shift(ARGV);
        }
        $tmp =~ s/\s//g;
        $tmp =~ s/\://g;
        $tmp =~ s/\,//g;
        $weight = $tmp;
        $useweight = 1;
      }
    elsif ($arg =~ "h") {
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
    elsif ($arg =~ "a" && -r "mps.db") {
      $append = 1;
      print "option sets mode to append\n";
    }
    elsif ($arg =~ "-M") {
      $pedeMem = $arg;
      $pedeMem =~ s/-M//; # Strips away the "-M"
      if (length($pedeMem) == 0) { # The memory size must be following param
        $pedeMem = shift(ARGV);
      }
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
  print "Usage:\n  mps_setup.pl [options] batchScript cfgTemplate infiList nJobs class[:classMerge] jobname [mergeScript [pool:]mssDir]";
  print "\nKnown options:";
  print "  \n -m          Setup the pede merging job.";
  print "  \n -a          Append jobs to the already existing list.";
  print "  \n -M pedeMem  The memory (MB) to be allocated for pede (min: 1024 MB).";
  print "  \n             If not given, it is evinced from the pede executable name.";
  print "  \n             Finally, it is set 2560 MB if neither of the two are available.";
  print "  \n -N name     Some arbitrary name assigned to the jobs.";
  print "  \n -w value    Statistical weight to be used.";
  print "\n";
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
unless (index(" lxplus cmscaf1nh cmscaf1nd cmscaf1nw cmscafspec1nh cmscafspec1nd cmscafspec1nw 8nm 1nh 8nh 1nd 2nd 1nw 2nw "," ".get_class("mille")." ")>-1) {
  print "Bad job class for mille in class '$class'\n";
  exit 1;
}
unless (index(" lxplus cmscaf1nh cmscaf1nd cmscaf1nw cmscafspec1nh cmscafspec1nd cmscafspec1nw 8nm 1nh 8nh 1nd 2nd 1nw 2nw "," ".get_class("pede")." ")>-1) {
  print "Bad job class for pede in class '$class'\n";
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
  if ($mssDir =~ /:/) { # ':' as delimeter also used in mpedegui.pl
    $mssDirPool = $mssDir;
    $mssDirPool =~ s/:.+?$//; # Remove all that follows ":"

    $mssDir =~ s/^.+?://; # Remove all the precedes ":"
  }

  $testMssDir = `$Mpslib::eos ls -d $mssDir`;
  chomp $testMssDir;
  if ($testMssDir eq "") {
    print "Bad MSS directory name $mssDir\n";
    exit 1;
  }

}

$pedeMemMin = 1024; # Minimum memory allocated for pede: 1024MB=1GB

# Try to guess the memory requirements from the pede executable name.
# 2.5GB is used as default otherwise.
# AP - 23.03.2010
$pedeMemDef = `cat $cfgTemplate | grep "process.AlignmentProducer.algoConfig.pedeSteerer.pedeCommand" | cut -d '#' -f 1 | grep "=" | tail -1 | cut -d '=' -f 2`; # This is the pede executable (full path).
$pedeMemDef = `basename $pedeMemDef`; # This is the pede executable (only the file name, eg "pede_4GB").
$pedeMemDef =~ s/.*_//; # Strip away the first part of the name (eg remains only "4GB").
$pedeMemDef =~ s/GB//; # Strip away "GB" (eg remains only "4").
if ($pedeMemDef =~ /\d/) { # If what remains is a number...
  $pedeMemDef = 1024*$pedeMemDef; # ... convert it into GB.
  if ($pedeMemDef < $pedeMemMin) {$pedeMemDef = $pedeMemMin}; # $pedeMemDef must be >= pedeMemMin.
} else { # Use a default value.
  $pedeMemDef = 1024*2.5;
}

# Allocate memory for the pede job.
# The value specified by the user (-M option) prevails on the one evinced from the executable name.
# AP - 23.03.2010
if ($pedeMem =~ /\D/ || $pedeMem < $pedeMemMin) {
  print "Memory request (".$pedeMem.") non-digit or < ".$pedeMemMin.", use ".$pedeMemDef."\n";
  $pedeMem = $pedeMemDef;
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

# build the absolute job directory path (needed by mps_script)
$thePwd = `pwd`;
chomp $thePwd;
$theJobData = "$thePwd/jobData";
print "theJobData= $theJobData \n";

if ($append == 1) {
# save current values
  my $tmpBatchScript = $batchScript;
  my $tmpCfgTemplate = $cfgTemplate;
  my $tmpInfiList = $infiList;
  my $tmpNJobs = $nJobs;
  my $tmpClass = $class;
  my $tmpMergeScript = $mergeScript;
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
  $batchScript = $tmpBatchScript;
  $cfgTemplate = $tmpCfgTemplate;
  $infiList = $tmpInfiList;
  $nJobs = $tmpNJobs;
  $class = $tmpClass;
  $mergeScript = $tmpMergeScript;
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
  if($useweight)
    {
      push @JOBSP2,"$weight";
    }
  else
    {
      push @JOBSP2,"";
    }
  push @JOBSP3,"$confname";
  # create the split card files
  print "mps_split.pl $infiList $j $nJobs >jobData/$theJobDir/theSplit\n";
  system "mps_split.pl $infiList $j $nJobs >jobData/$theJobDir/theSplit";
  if ($?) {
    print "              split failed\n";
    @JOBSTATUS[$i-1] = "FAIL";
  }
  $theIsn = sprintf "%03d",$i;
  print "mps_splice.py $cfgTemplate jobData/$theJobDir/theSplit jobData/$theJobDir/the.py $theIsn\n";
  system "mps_splice.py $cfgTemplate jobData/$theJobDir/theSplit jobData/$theJobDir/the.py $theIsn";
  # create the run script
  print "mps_script.pl $batchScript  jobData/$theJobDir/theScript.sh $theJobData/$theJobDir the.py jobData/$theJobDir/theSplit $theIsn $mssDir $mssDirPool\n";
  system "mps_script.pl $batchScript  jobData/$theJobDir/theScript.sh $theJobData/$theJobDir the.py jobData/$theJobDir/theSplit $theIsn $mssDir $mssDirPool";
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

write_db();

# if merge mode, create the directory and set up contents
if ($driver eq "merge") {

  system "rm -rf jobData/jobm";
  system "mkdir jobData/jobm";
  print "Create dir jobData/jobm\n";

  # We want to merge old and new jobs
  my $nJobsMerge = $nJobs+$nJobExist;

  # create  merge job cfg
  print "mps_merge.py -w $cfgTemplate jobData/jobm/alignment_merge.py $theJobData/jobm $nJobsMerge\n";
  system "mps_merge.py -w $cfgTemplate jobData/jobm/alignment_merge.py $theJobData/jobm $nJobsMerge";

  # create merge job script
  print "mps_scriptm.pl $mergeScript jobData/jobm/theScript.sh $theJobData/jobm alignment_merge.py $nJobsMerge $mssDir $mssDirPool\n";
  system "mps_scriptm.pl $mergeScript jobData/jobm/theScript.sh $theJobData/jobm alignment_merge.py $nJobsMerge $mssDir $mssDirPool";
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
