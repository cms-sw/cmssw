#!/usr/bin/env perl
#     R. Mankel, DESY Hamburg     17-Oct-2007
#     A. Parenti, DESY Hamburg    16-Apr-2008
#     $Revision: 1.3 $
#     $Date: 2008/07/29 15:36:18 $
#
#  Try periodically to fire merge job.
#  Terminate when done
#
#  Usage: mps_auto.pl seconds
#

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;

$seconds = 60;

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
    $optionstring = "$optionstring$arg";
  }
  else {                # parameters not related to options
    $i = $i + 1;
    if ($i eq 1) {
      $seconds = $arg;
    }
  }
}

if ($helpwanted != 0 ) {
  print "Usage:\n  mps_auto.pl [options] [seconds]";
  print "\nTry to submit merge job every 'seconds' (default: 60) seconds.";
  print "\nIn case of a problem in a mille job, will hang in an endless loop...";
  print "\nKnown options:";
  print "\n -h   This help.\n";
  exit;
}


if ($seconds <20) {
  print "Set seconds to 20\n";
  $seconds = 20;
}

$done = 0;

$iter = 0;
while ($done == 0) {
  if ($iter != 0) {
    # In case mps_fetch.pl detects an error, it will never end!
    print  "mps_stat.pl > /dev/null; mps_fetch.pl\n";
    system "mps_stat.pl > /dev/null; mps_fetch.pl";
    print  "In case of error, abort this mps_auto.pl, otherwise endless loop....\n";
    print  "mps_fire.pl -m; sleep $seconds\n";
    system "mps_fire.pl -m; sleep $seconds";
  }
  $iter = $iter + 1;
  # go to merge job 
  read_db();
  $i = (@JOBID) - 1;
  if (@JOBSTATUS[$i] ne "SETUP") {
    $done = 1;
    break;
  }
}
