#!/usr/bin/env perl

use strict;

my $db = $ARGV[0];
my $path = $ARGV[1];


#we dont like / at the end of the string.
$path =~ s/\/$//;

if(!defined $ARGV[0] || !defined $ARGV[1])
  {
    print "The location of the mps.db file and the path to the jobData directory has to be provided.\n";
    exit 0;
  }

unless(-e "$db")
  {
    print "Could not find mps.db file: $db\n";
    exit 0;
  }
unless(-d "$path")
  {
    print "Could not find jobData/ directory: $path\n";
    exit 0;
  }

open IN, "< $db" || die "error when opening mps.db file: $!";
my @adb = <IN>;
close IN;

my @ids;
my @names;
my $lastid = 0;

#read the file and apply a chain of filters.
foreach (@adb)
  {
    next unless(/job/);

    if(my $n = () = /\:/g)
      {
        next if ($n != 12);
      }
    my @row = split /:/;
    my $nrow = @row;
    if($nrow != 13)
      {
        print "There seems to be a problem reading the file mps.db. The number of fields per row is $nrow but it should be equal to 13.\n";
        exit 0;
      }
    next if(!defined $row[0] || !defined $row[1]);
    next if($row[0] =~ /\D/g);
    my $jobid = $row[1];
    next if($jobid =~ /jobm/);
    $jobid =~ s/job//;
    next if($jobid =~ /\D/g);
    next if($jobid != $row[0]);
    my $name = pop @row;
    chomp $name;
    next if(!defined $name);
    next if($name eq "");
    push @names, $name;
    push @ids, $jobid;
    $lastid = $jobid if($jobid > $lastid);
  }

my %h;

for(my $i = 1; $i<= $lastid; $i++)
  {
    my $found = 0;
    foreach my $j (@ids)
      {
        $found = 1 if($i == $j);
      }
    if(!$found)
      {
        print "Problem parsing the mps.db file. Could not find job $i.\n";
        exit 0;
      }
    else
      {
        #everything seems to be ok so far ...
        #try two different naming conventions for the millepede monitor file.
        my $foundfile1 = 0;
        my $foundfile2 = 0;

        my $filename1 =  sprintf "$path/jobData/job%03d/millePedeMonitor%03d.root",$i, $i;
        $foundfile1 = 1 if(-e "$filename1");

        my $filename2 =  sprintf "$path/jobData/job%03d/millePedeMonitor.root",$i;
        $foundfile2 = 1 if(-e "$filename2");

        if($foundfile1 && $foundfile2)
          {
            print "Both files were found. This should never happen: $filename1 and $filename2\n";
            exit 0;
          }
        if(!$foundfile1 && !$foundfile2)
          {
            print "Neither $filename1 nor $filename2 was found => ignore job folder.\n";
          }
        else
          {
            my $f = $foundfile1 ? $filename1 : $filename2;
            for(my $j =0; $j <= $#ids; $j++)
              {
                if($i == $ids[$j])
                  {
                    push @{$h{$names[$j]}}, $f;
                    last;
                  }
              }
          }
      }
  }
foreach my $key (keys %h)
  {
    my $haddstring = "hadd -f millepedemonitor_${key}.root ";
    for(my $i =0; $i< @{$h{$key}}; $i++)
      {
         my $a = $h{$key}[$i];
         $haddstring .= "$a ";
       }
    print "$haddstring\n\n";
    system "$haddstring";
  }
