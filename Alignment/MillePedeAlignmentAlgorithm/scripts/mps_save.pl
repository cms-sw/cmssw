#!/usr/local/bin/perl
#     R. Mankel, DESY Hamburg     11-Oct-2007
#     A. Parenti, DESY Hamburg    27-Mar-2008
#     $Revision: 1.1 $
#     $Date: 2008/04/10 16:10:12 $
#
#  Save output from jobs that have FETCH status
#  
#
#  Usage: mps_save.pl saveDir
#

use lib './mpslib';
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
 

  @FILENAMES = ("treeFile_merge.root","histograms_merge.root","alignment_merge.log",
		"alignment_merge.cfg","millepede.res",
		"millepede.log","pede.dump","millepede.his", "alignments_MP.db");

  $dirPrefix = "jobData/@JOBDIR[$i]/";

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
      print "$copyFile unreadable\n";
    }
  }

}
