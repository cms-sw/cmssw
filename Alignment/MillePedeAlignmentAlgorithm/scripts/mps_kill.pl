#!/usr/bin/env perl
#     R. Mankel, DESY Hamburg     16-Jul-2007
#     A. Parenti, DESY Hamburg    16-Apr-2008
#     $Revision: 1.3 $
#     $Date: 2009/01/07 18:26:00 $
#
#  Kill all jobs being processed by ZARAH,
#  i.e. those pending, running or suspended.
#
#  This is useful in case an error has been detected.
#  The killed jobs go to FAIL status.
#  They can be rescheduled with mps_retry.pl once the
#  problem has been fixed.
#  
#
#  Usage:
#      mps_kill.pl
#

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;

@MODSTATES = ();
@MODJOBS = ();
$killAll = 0;
$killMerge = 0;

# parse the arguments
while (@ARGV) {
  $arg = shift(ARGV);
  if ($arg =~ /\A-/) {  # check for option 
    if ($arg =~ "h") {
      $helpwanted = 1;
    }
    elsif ($arg =~ "a") {
      $killAll = 1;
    }
    elsif ($arg =~ "m") {
      $killMerge = 1;
    }
    $optionstring = "$optionstring$arg";
  }
  else {                # parameters not related to options
    $i = $i + 1;
    if (($arg =~ m/\d+/) eq 1) {
      print "Parameter $arg interpreted as job number\n";
      push @MODJOBS,$arg;
    }
    elsif (($arg =~ m/[A-Z]+/) eq 1) {
      print "Parameter $arg interpreted as job state\n";
      push @MODSTATES,$arg;
    }
    else {
      print "Parameter $arg not recognized\n";
    }
  }
}



read_db();

if ($killAll == 1) {
  # loop over pending, running or suspended jobs
  for ($i=0; $i<@JOBID; ++$i) {
    if (@JOBSTATUS[$i] =~ /PEND/
	or @JOBSTATUS[$i] =~ /RUN/
	or @JOBSTATUS[$i] =~ /SUSP/) {
      system "bkill @JOBID[$i]";
      print "bkill @JOBID[$i]\n";
      my $disabled = "";
      $disabled = "DISABLED" if( $JOBSTATUS[$i] =~ /DISABLED/gi);
      @JOBSTATUS[$i] = $disabled."FAIL";
      @JOBHOST[$i] = "user kill";
    }
  }
}
else {
  # only kill certain job numbers or states
  for ($i=0; $i<@JOBID; ++$i) {
    if (@JOBSTATUS[$i] =~ /PEND/
	or @JOBSTATUS[$i] =~ /RUN/
	or @JOBSTATUS[$i] =~ /SUSP/) {
      my $disabled = "";
      $disabled = "DISABLED" if( $JOBSTATUS[$i] =~ /DISABLED/gi);
      $JOBSTATUS[$i] =~ s/DISABLED//gi;
      $stateText = "^@JOBSTATUS[$i]\$";
      $theNum = $i + 1;
      $jobText = "^$theNum\$";
      if ( ( (grep /$stateText/,@MODSTATES) > 0) || (grep /$jobText/,@MODJOBS) > 0) {
	print "bkill @JOBID[$i]\n";
	system "bkill @JOBID[$i]";
	@JOBSTATUS[$i] = $disabled."FAIL";
	@JOBHOST[$i] = "user kill";
      }
    }
  }
}
if ($killMerge eq 1) {
    $i = @JOBID - 1;
	if (@JOBSTATUS[$i] eq "PEND"
	    or @JOBSTATUS[$i] eq "RUN"
	    or @JOBSTATUS[$i] eq "SUSP") {
	    $stateText = "^@JOBSTATUS[$i]\$";
	    $theNum = $i + 1;
	    $jobText = "^$theNum\$";
	    if ( ( (grep /$stateText/,@MODSTATES) > 0) || (grep /$jobText/,@MODJOBS) > 0) {

		print "bkill @JOBID[$i]\n";
		system "bkill @JOBID[$i]";
		@JOBSTATUS[$i] = "FAIL";
		@JOBHOST[$i] = "user kill";
	    }
	}
}
write_db();
