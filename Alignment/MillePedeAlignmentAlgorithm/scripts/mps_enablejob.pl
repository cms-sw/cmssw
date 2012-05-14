#!/usr/bin/env perl
# Author: Joerg Behr
#This script enables jobs with the given job numbers.

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;
use warnings;
read_db();

my @enabledjobs;
my $count = 0;
my $confname = "";

while (@ARGV) {
   my $arg = shift(@ARGV);
    if ($arg =~ /-N/g) {
      $confname = $arg;
      $confname =~ s/-N//; # Strips away the "-N"
      if (length($confname) == 0) {
        $confname = shift(@ARGV);
       }
      $confname =~ s/\s//g;
      if($confname =~ /\:/)
        {
          $confname =~ s/\://g;
          print "colons were removed in configuration name because they are not allowed: $confname\n";
        }
    }
   else
     {
       $count++;
       push @enabledjobs, $arg;
     }
 }

if($confname ne "")
  {
    print "Enable jobs: ${confname}.\n";
    for (my $i=0; $i<@JOBID; ++$i) {
      my $status = $JOBSTATUS[$i];
      unless($JOBDIR[$i] =~ /jobm/)
        {
          my $name = $JOBSP3[$i];
          if($name eq $confname)
            {
               my $status = $JOBSTATUS[$i];
               $status =~ s/DISABLED//;
               $JOBSTATUS[$i] = $status;
             }
        }
    }
  }
elsif($count==0)
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
