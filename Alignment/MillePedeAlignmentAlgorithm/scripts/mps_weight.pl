#!/usr/bin/env perl

# Author: Joerg Behr

# Usage:
# Options are:
# -h some help
# -N name weight
# -c remove all weights
# -l list all weights
# Whenever -c is used, all weights are removed from mps.db
# If neither the options -N nor -c are specified, then the first argument is interpreted as weight.
# Consequently, the following list will be treated as a list of Mille jobs to which the weight is assigned.
#
# Examples:
#
# % mps_weight.pl -N ztomumu 5.7
# Assign weight 5.7 to Mille jobs which are called "ztomumu" ("mps_setup.pl -N ztomumu ..." has to be used during job creation).
#
# % mps_weight.pl 6.7 3 4 102
# Assign weight 6.7 to Mille binaries with numbers 3, 4, and 102, respectively.
#
# % mps_weight.pl -c
# Remove all assigned weights.

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;
use warnings;
read_db();

my $confname = "";
my @weightedjobs;
my $firstnumber = 1;

my $cleanall = 0;
my $weight = 1.0;
my $listwtx = 0;

# parse the arguments
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
   elsif($arg =~ /-c/g)
     {
       $weight = 1;
       $cleanall = 1;
     }
   elsif($arg =~ /-h/g)
     {
print 'Usage:
mps_weight.pl [-c] [-h] [-N name] [weight] [jobids]
Parameters:
-h           This help
-l 	     List all weights
-c 	     Remove all weights from the data base
-N name      Assign a weight to the dataset "name". See the option "mps_setup.pl -N".
weight 	     The weight to be used
jobids       A list of job ids to which the weight is assigned. This option is only used if neither -N nor the -c option is specified.

The command mps_weight.pl can be used to associated weights to individual Mille binary files. Some examples are given below.

# Examples:
#
# % mps_weight.pl -N ztomumu 5.7
# Assign weight 5.7 to Mille jobs which are called "ztomumu" ("mps_setup.pl -N ztomumu ..." has to be used during job creation).
#
# % mps_weight.pl 6.7 3 4 102
# Assign weight 6.7 to Mille binaries with numbers 3, 4, and 102, respectively.
#
# % mps_weight.pl -c
# Remove all assigned weights.
';
      exit;
     }
   elsif($arg =~ /-l/g)
     {
       $listwtx = 1;
     }
   else
     {
       if($firstnumber)
         {
           $firstnumber = 0;
           $weight = $arg;
         }
       else
         {
           push @weightedjobs, $arg;
         }
     }
   }

if($listwtx)
  {
    print "list of weights:\n\n";
    print "job directory | name | weight\n";
    for (my $i=0; $i<@JOBID; ++$i) {
      unless($JOBDIR[$i] =~ /jobm/)
        {
           my $name = "undefined";
           my $wtx = 1.0;
           $name = $JOBSP3[$i] if(defined $JOBSP3[$i] && $JOBSP3[$i] ne "");
           $wtx = $JOBSP2[$i] if(defined $JOBSP2[$i] && $JOBSP2[$i] ne "");
           #print "$JOBDIR[$i]\t|\t$name\t|\t$wtx\n";
           printf "%6s | %20s | %10.3f\n",$JOBDIR[$i],$name,$wtx;

        }
    }
  }

if($cleanall)
  {
    print "clean-up mps.db by removing all weights.\n";
     for (my $i=0; $i<@JOBID; ++$i) {
      my $status = $JOBSTATUS[$i];
      unless($JOBDIR[$i] =~ /jobm/)
            {
              $JOBSP2[$i] = "";
            }
    }
   }
elsif($confname ne "")
  {
    print "Assign weight $weight to ${confname}.\n";
    for (my $i=0; $i<@JOBID; ++$i) {
      my $status = $JOBSTATUS[$i];
      unless($JOBDIR[$i] =~ /jobm/)
            {
              my $name = $JOBSP3[$i];
              if($name eq $confname)
                {
                  $weight =~ s/\://g;
                  $weight =~ s/\,//g;
                  $JOBSP2[$i] = $weight;
                }
            }
    }
  }
else
  {
    foreach my $j (@weightedjobs) {
      {
        $weight =~ s/\://g;
        $weight =~ s/\,//g;
        $JOBSP2[$j-1] = $weight;
      }
    }
  }
write_db();
