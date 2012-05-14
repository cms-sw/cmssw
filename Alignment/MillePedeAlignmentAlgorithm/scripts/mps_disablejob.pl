#!/usr/bin/env perl
# Author: Joerg Behr
#This script disables jobs with the given job numbers.

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;
use warnings;
read_db();

my @disabledjobs;
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
       push @disabledjobs, $arg;
     }
 }

if($confname ne "")
  {
    print "Disable jobs: ${confname}.\n";
    for (my $i=0; $i<@JOBID; ++$i) {
      my $status = $JOBSTATUS[$i];
      unless($JOBDIR[$i] =~ /jobm/)
        {
          my $name = $JOBSP3[$i];
          if($name eq $confname)
            {
               my $status = $JOBSTATUS[$i];
               $JOBSTATUS[$i] = "DISABLED".$status;
             }
        }
    }
  }
elsif($count==0)
  {
    for (my $i=0; $i<@JOBID; ++$i) {
      my $status = $JOBSTATUS[$i];
      if($status =~ /DISABLED/)
        {
          print "mps_disablejob.pl job $i is already disabled!\n";
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
