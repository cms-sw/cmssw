#!/usr/bin/env perl
#This script enables jobs with the given job numbers.

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;
read_db();

my @enabledjobs;
my $count = 0;
foreach my $arg  (@ARGV) {
  $count++;
  push @enabledjobs, $arg;
}

if($count==0)
  {
    for (my $i=0; $i<@JOBID; ++$i) {
      my $status = $JOBSTATUS[$i];
      $status =~ s/DISABLED//;
      $JOBSTATUS[$i] = $status;
    }
  }
else
  {
    foreach my $j (@enabledjobs) {
      my $status = $JOBSTATUS[$j-1];
      $status =~ s/DISABLED//;
      $JOBSTATUS[$j-1] = $status;
    }
  }
write_db();
