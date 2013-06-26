#!/usr/bin/env perl
# Author: Joerg Behr
#This script disables jobs with the given job numbers.

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;
use warnings;
use POSIX;
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
   elsif ($arg eq "-h")
     {
       print 'mps_disablejob.pl [-h] [-N name] [jobids]
Parameters/Options:
-h 	     The help.
-N name      Disable Mille jobs with name "name".
jobids 	     A list of Mille job ids which should be disabled. Does not work together with option -N.

The script mps_disablejob.pl can be used to disable several Mille jobs by using either their associated name or by their ids.

#Examples:

#first example:
#create a new Pede job:
% mps_setupm.pl
# disable some Mille jobs:
% mps_disablejob.pl -N ztomumu
# submit the Pede job (works only if the "force" option is used):
% mps_setup.pl -mf
# enable everything
% mps_enablejob.pl

#second example:
#create a new Pede job:
% mps_setupm.pl
# disable some Mille jobs
% mps_disablejob.pl 3 5 6 77 4
# submit the Pede job (works only if the "force" option is used):
% mps_fire.pl -mf

#third example:
# disable a sequence of jobs
% mps_disablejob.pl `seq 2 300`
#create and submit new Pede job. Note if you want to omit the "force" option when the Pede job is submitted, you need to use the -a option for mps_setupm.pl.
% mps_setupm.pl -a
% mps_fire.pl -m
% mps_enablejob.pl
';
      exit;
     }
   else
     {
       if(isdigit $arg)
         {
           push @disabledjobs, $arg;
           $count++;
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
    print "Disable jobs: ${confname}.\n";
    for (my $i=0; $i<@JOBID; ++$i) {
      my $status = $JOBSTATUS[$i];
      if(defined $status)
        {
          unless($JOBDIR[$i] =~ /jobm/)
            {
              my $name = $JOBSP3[$i];
              if($name eq $confname )
                {
                  $JOBSTATUS[$i] = "DISABLED".$status unless ($status =~ /DISABLED/i);
                }
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
          if(defined $status)
            {
              unless($JOBDIR[$i] =~ /jobm/i)
                {
                  $JOBSTATUS[$i] = "DISABLED".$status;
                }
            }
        }
    }
  }
else
  {
    foreach my $j (@disabledjobs) {
      my $status = $JOBSTATUS[$j-1];
      if(defined $status)
        {
          unless($JOBDIR[$j-1] =~ /jobm/i)
            {
              if($status =~ /DISABLED/i)
                {
                  print "mps_disablejob.pl job $j is already disabled!\n";
                }
              else
                {
                  $JOBSTATUS[$j-1] = "DISABLED".$status;
                }
            }
        }
      else
        {
          print "job number ". ($j-1). " was not found.\n";
        }
    }
  }
write_db();
