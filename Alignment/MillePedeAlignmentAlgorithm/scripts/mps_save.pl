#!/usr/local/bin/perl
#     R. Mankel, DESY Hamburg     11-Oct-2007
#     A. Parenti, DESY Hamburg    16-Apr-2008
#     $Revision: 1.4 $
#     $Date: 2008/04/21 21:16:04 $
#
#  Save output from jobs that have FETCH status
#  
#
#  Usage: mps_save.pl saveDir
#

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;

$saveDir = "undefined";
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
    $optionstring = "$optionstring$arg";
  }
  else {                # parameters not related to options
    $i = $i + 1;
    if ($i eq 1) {
      $saveDir = $arg;
    }
  }
}


if ($saveDir eq "undefined") {
  print "Insufficient information given\n";
  exit 1;
}


# create output directory
if (-d $saveDir) {
  print "Reusing existing directory $saveDir ...\n";
}
else {
  system "mkdir -p $saveDir"; # -p by GF
}

read_db();

# go to merge job 
$i = (@JOBID) - 1;
unless (@JOBDIR[$i] eq "jobm") {
  print "Bad merge job @JOBDIR[$i]\n";
}

if (@JOBSTATUS[$i] eq "FETCH"
    or @JOBSTATUS[$i] eq "OK" or @JOBSTATUS[$i] eq "TIMEL") {

  $dirPrefix = "jobData/@JOBDIR[$i]/";

  @FILENAMES = ("treeFile_merge.root","histograms_merge.root",
		"alignment_merge.cfg","alignment_merge.log",
		"alignment_merge.log.gz","millepede.log","millepede.log.gz",
		"millepede.res","millepede.his","pede.dump",
		"alignments_MP.db");

  while ($theFile = shift @FILENAMES) {
    $copyFile = $dirPrefix.$theFile;
    if (-r $copyFile) {
      print "cp $copyFile $saveDir/\n";
      system "cp $copyFile $saveDir/";
      $retcode = $? >> 8;
      if ($retcode) {
	print "Copy of $copyFile failed, retcode=$retcode\n";
      }
    }
    else {
      print "$copyFile unreadable or not existing\n";
    }
  }

# Now copy the backup of original scripts and cfg's
  $ScriptCfg = `ls jobData/ScriptsAndCfg???.tar`;
  chomp($ScriptCfg);
  $ScriptCfg =~ s/\n/ /g;
  $ScriptCfg =~ s/jobData\///g;

  @FILENAMES = split(' ',$ScriptCfg);

  while ($theFile = shift @FILENAMES) {
    $copyFile = "jobData/".$theFile;
    if (-r $copyFile) {
      print "cp $copyFile $saveDir/\n";
      system "cp $copyFile $saveDir/";
      $retcode = $? >> 8;
      if ($retcode) {
	print "Copy of $copyFile failed, retcode=$retcode\n";
      }
    }
    else {
      print "$copyFile unreadable or not existing\n";
    }
  }
}
