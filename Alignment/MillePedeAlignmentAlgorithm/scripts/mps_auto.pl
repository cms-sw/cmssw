#!/usr/bin/env perl
#     R. Mankel, DESY Hamburg     17-Oct-2007
#     A. Parenti, DESY Hamburg    16-Apr-2008
#     $Revision: 1.6 $
#     $Date: 2010/08/12 12:55:08 $
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
  print "\nTry to submit merge job every 'seconds' (default: 60, min: 20) seconds.";
  print "\nStops in case of a problem in a mille job.";
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
    print  "mps_stat.pl > /dev/null; mps_fetch.pl\n";
    system "mps_stat.pl > /dev/null; mps_fetch.pl";
    read_db();
    # loop over mille jobs
    for ($i=0; $i<$nJobs; ++$i) {
	 if ($JOBSTATUS[$i] ne "RUN" && $JOBSTATUS[$i] ne "DONE" 
	     && $JOBSTATUS[$i] ne "OK" && $JOBSTATUS[$i] ne  "PEND"
             && !$JOBSTATUS[$i] =~ /DISABLED/) {
	     $done = 1;
	     print "Mille job $i in unknown or bad status $JOBSTATUS[$i], stopping.\n";
	 }
    }
    if ($done == 0) { # only in case of no problem...
	print  "mps_fire.pl -m; date; sleep $seconds\n";
	system "mps_fire.pl -m; date; sleep $seconds";
    }
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
