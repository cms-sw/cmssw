#!/usr/bin/env perl
#     R. Mankel, DESY Hamburg     06-Jul-2007
#     A. Parenti, DESY Hamburg    16-Apr-2008
#     $Revision: 1.7 $ by $Author: jbehr $
#     $Date: 2012/09/10 15:11:05 $
#
#  Update local mps database with batch job status
#  
#
#  Usage:
#

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;

read_db();

# initialize FLAG array. Mark jobs we do not have to worry about
my @FLAG = -1;
my $submittedjobs = 0;
for ($i=0; $i<@JOBID; ++$i) {
  if (         $JOBSTATUS[$i] =~ /SETUP/i
	       or $JOBSTATUS[$i] =~ /DONE/i
	       or $JOBSTATUS[$i] =~ /FETCH/i
	       or $JOBSTATUS[$i] =~ /OK/i
	       or $JOBSTATUS[$i] =~ /ABEND/i
	       or $JOBSTATUS[$i] =~ /FAIL/i
     #          or $JOBSTATUS[$i] =~ /DISABLED/i
     )
    {
    $FLAG[$i] = 1; # no need to care
  }
  else {
    $FLAG[$i] = -1; # care!
    $submittedjobs++;
  }
}
print "submitted jobs $submittedjobs\n";
if ( $submittedjobs > 0) {
  my $in = `bjobs -l`;
  #my $in = `cat output`;
  unless($in =~ /No unfinished job found/i)
    {
      $in =~ s/\n//g;
      $in =~ s/\s\s//g;
      $in =~ s/\s+//g;
      my @result = split /\-{50}/, $in;
      foreach my $i (@result) {
        chomp $i;
        #my ($status) = $i =~ /Status<(.+?)>/;
        #my ($jobid) = $i =~ /Job<(.+?)>,/;
        if (my ($status) = $i =~ /Status<([A-Z]+?)>/) {
          if (my ($jobid) = $i =~ /Job<(\d+?)>,/) {
            my $cputime = 0;
            $cputime = $1 if ($i =~ /TheCPUtimeusedis(\d+?)seconds/);
            print "out $status $jobid $cputime\n";
            my $theIndex = -1;
            my $disabled = "";
            for (my $k=0; $k<@JOBID; ++$k) {
              if ($JOBID[$k] == $jobid) {
                $theIndex = $k;
                $disabled = "DISABLED" if ($JOBSTATUS[$k] =~ /DISABLED/i);
              }
              #print "For index $k check jobid $JOBID[$k] result $theIndex\n";
            }
            if ($theIndex == -1) {
              print "mps_update.pl - the job $jobid was not found in the JOBID array\n";
              #exit(-1);
            }
            next if($theIndex == -1);
            next if($FLAG[$theIndex] == 1);
            $JOBSTATUS[$theIndex] = $disabled.$status;
            if ($status eq "RUN" || $status eq "DONE") {
              if ($cputime>0) {
                my $diff = $cputime - $JOBRUNTIME[$theIndex];
                $JOBRUNTIME[$theIndex] = $1;
                $JOBHOST[$theIndex] = "+$diff";
                $JOBINCR[$theIndex] = $diff;
              } else {
                $JOBRUNTIME[$theIndex] = 0;
                $JOBINCR[$theIndex] = 0;
              }
            }
            $FLAG[$theIndex] = 1;
            print "set flag of job $theIndex with id $JOBID[$theIndex] to 1\n";
          }
        }
      }
    }
}

# loop over remaining jobs to see whether they are done
my $theIndex = -1;
for ($i=0; $i<@JOBID; ++$i) {
  $theIndex = $i;
  my $disabled = "";
  $disabled = "DISABLED" if ($JOBSTATUS[$i] =~ /DISABLED/i);
  print " DB job $JOBID[$i] flag $FLAG[$theIndex]\n";
  if ($FLAG[$theIndex] == 1) {
    next;
  }
  # check whether job may be done
  $theBatchDirectory = sprintf "LSFJOB\_%d",$JOBID[$i]; #GF: $theIndex??
  print "theBatchDirectory $theDirectory\n";
  ## if (-d "LSFJOB\_$JOBID[$i]") {
  ##  print "LSFJOB\_$JOBID[$i] exists\n";
  if (-d $theBatchDirectory) {
    print "Directory $theBatchDirectory exists\n";
    $JOBSTATUS[$theIndex] = $disabled."DONE";
  } else {
    if ($JOBSTATUS[$theIndex] =~ /RUN/i) {
      print "WARNING: Job $theIndex in state RUN, neither found by bjobs nor find LSFJOB directory!\n";
      # FIXME: check if job not anymore in batch system
      # might set to FAIL - but probably $theBatchDirectory is just somewhere else...
    }
  }
}


# check for orphaned jobs
for ($i=0; $i<@JOBID; ++$i) {
  unless ($FLAG[$i] eq 1) {
    unless ($JOBSTATUS[$i] =~ /SETUP/i
            or $JOBSTATUS[$i] =~ /DONE/i
            or $JOBSTATUS[$i] =~ /FETCH/i
            or $JOBSTATUS[$i] =~ /TIMEL/i
            or $JOBSTATUS[$i] =~ /SUBTD/i) {
      print "Funny entry index $i job $JOBID[$i] status $JOBSTATUS[$i]\n";
    }
  }
}
write_db();

