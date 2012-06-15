#!/usr/bin/env perl
#This script disables jobs with the given job numbers.

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;
read_db();

my @disabledjobs;
my $count = 0;
foreach my $arg  (@ARGV) {
  $count++;
  push @disabledjobs, $arg;
}

if($count==0)
  {
    for (my $i=0; $i<@JOBID; ++$i) {
      my $status = $JOBSTATUS[$i];
      if($status =~ /DISABLED/)
        {
          print "mps_disablejob.pl job $j is already disabled!\n";
        }
      else
        {
          unless($JOBDIR[$i] =~ /jobm/)
            {
              $JOBSTATUS[$i] = "DISABLED".$status;
            }
        }
    }
  }
else
  {
    foreach my $j (@disabledjobs) {
      my $status = $JOBSTATUS[$j-1];
      if($status =~ /DISABLED/)
        {
          print "mps_disablejob.pl job $j is already disabled!\n";
        }
      else
        {
          $JOBSTATUS[$j-1] = "DISABLED".$status;
        }
    }
  }
write_db();
