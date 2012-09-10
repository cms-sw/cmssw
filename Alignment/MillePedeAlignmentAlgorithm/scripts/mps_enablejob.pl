#!/usr/bin/env perl
# Author: Joerg Behr
#This script enables jobs with the given job numbers.

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;
use warnings;
use POSIX;
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
   elsif ($arg eq "-h")
     {
       print 'mps_enablejob.pl [-h] [-N name] [jobids]
Parameters/Options:
-h         This help ...
-N name    Enable Mille jobs with name "name".
jobids 	   A list of Mille job ids which should be enabled. Does not work together with option -N.

The command mps_enablejob.pl can be used to turn on Mille jobs which have been turn off previously. If no option is provided all jobs are enabled.
';
      exit; 
     }
   else
     {
       if(isdigit $arg)
         {
           $count++;
           push @enabledjobs, $arg;
         }
       else
         {
           print "only integer numbers are allowed for the job ids: $arg\n";
           exit(-1);
         }
     }
 }

if($confname ne "")
  {
    print "Enable jobs: ${confname}.\n";
    for (my $i=0; $i<@JOBID; ++$i) {
      my $status = $JOBSTATUS[$i];
      if(defined $status)
        {
          unless($JOBDIR[$i] =~ /jobm/i)
            {
              my $name = $JOBSP3[$i];
              if($name eq $confname)
                {
                  my $status = $JOBSTATUS[$i];
                  $status =~ s/DISABLED//ig;
                  $JOBSTATUS[$i] = $status;
                }
            }
        }
    }
  }
elsif($count==0)
  {
    for (my $i=0; $i<@JOBID; ++$i) {
      my $status = $JOBSTATUS[$i];
      if(defined $status)
        {
          unless($JOBDIR[$i] =~ /jobm/i)
            {
              $status =~ s/DISABLED//gi;
              $JOBSTATUS[$i] = $status;
            }
        }
    }
  }
else
  {
    foreach my $j (@enabledjobs) {
      my $status = $JOBSTATUS[$j-1];
      if(defined $status)
        {
          unless($JOBDIR[$j-1] =~ /jobm/i)
            {
              $status =~ s/DISABLED//ig;
              $JOBSTATUS[$j-1] = $status;
            }
        }
    }
  }
write_db();
