#!/usr/bin/env perl

# Author: Joerg Behr
#
# This script reads the histogram file produced by Pede and it extracts the plot showing the average chi2/ndf per Mille binary number.
# After reading the MPS library, for which the file name has to be provided as first argument, an output file called chi2pedehis.txt is produced
# where the first column corresponds to the associated name, the second column corresponds to the Mille binary number, and the last column
# is equal to <chi2/ndf>. As second argument this scripts expects the file name of the Pede histogram file -- usually millepede.his. The last argument
# represents the location of the Python config which was used by CMSSW.
#
# Standalone usage:
# mps_parse_pedechi2hist.pl mps.db millepede.his $RUNDIR/alignment_merge.py
#
# Use createChi2ndfplot.C to plot the output of this script.

use strict;

my $db = $ARGV[0];
my $pedehis = $ARGV[1];
my $pyconfig = $ARGV[2];


if(!defined $ARGV[0] || !defined $ARGV[1] || !defined $ARGV[2])
  {
    print "The location of the mps.db and Pede histogram file, and the Python config for CMSSW have to be specified.\n";
    exit 0;
  }

unless(-e "$db")
  {
    print "Could not find mps.db file: $db\n";
    exit 0;
  }

unless(-e "$pedehis")
  {
    print "Could not find pedehis file: $pedehis\n";
    exit 0;
  }

unless(-e "$pyconfig")
  {
    print "Could not find Python config file: $pyconfig\n";
    exit 0;
  }


open IN, "< $db" || die "error while opening mps.db file: $!";
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

if(@names == 0)
  {
    print "Names were not associated to Mille binaries (Use mps_setup.pl -N blabla ... for that purpose). Chi2ndf plot is not produced. exit.\n";
    exit(-1);
  }
else
  {
    #now try to guess the used binaries. extract the file number of those
    open INPY, "< $pyconfig" || die "error while opening alignment_merge.py: $!\n";
    my @pycontent = <INPY>;
    close INPY;
    chomp @pycontent;

    my @usedbinaries;
    foreach (@pycontent)
      {
        next unless(/milleBinary/i);
        if(my ($i) = /milleBinary(.+?)\.dat/i)
          {
            push @usedbinaries, int scalar $i;
          }
      }
    #read pede hists
    open INPEDEHIS, "< $pedehis" || die "error while opening pedehis file: $!\n";
    my @ph = <INPEDEHIS>;
    close INPEDEHIS;
    chomp @ph;
    my $foundchi2start = 0;
    my @hisdata;
    for(my $i =0; $i < @ph; $i++)
      {
        my $line = $ph[$i];
        $foundchi2start = 1 if ($line =~ /final \<Chi\^2\/Ndf\> from accepted local fits vs file number/i);
        if($foundchi2start)
          {
            last if($line =~ /end of xy\-data/i);
          }
        next unless($line =~ /\d/);
        next if($line =~ /[a-z]/i);
        if($foundchi2start)
          {
            my @tmp = split " ", $line;
            $hisdata[int scalar $tmp[0] - 1] = $tmp[1];
          }
      }
    if(scalar @hisdata != scalar @usedbinaries)
      {
        print "The number of used binaries is " . scalar @usedbinaries . " whereas in contrast, however, the <chi2/ndf> histogram in Pede has " . scalar @hisdata . " bins (Pede version >= rev92 might help if #bins < #binaries). exit.\n";
        exit 0;
      }

    my $outname = "chi2pedehis.txt";
    unlink "$outname" if(-e "$outname");
    open OUTPUT, "> $outname" || die "error while opening file for writing: $!\n";

    for(my $i = 1; $i <= @usedbinaries; $i++)
     {
       my $found = 0;
       my $name;
       my $index = -1;
       foreach my $j (@ids)
         {
           if($usedbinaries[$i-1] == $j)
             {
               $found = 1;
               $name  = $names[$j-1];
               $index = $j;
               last;
             }
         }
       if(!$found)
         {
           print "Problem parsing the mps.db file. Could not find job $i. Maybe only a fraction of binaries was named?\n";
           exit 0;
         }
       else
         {
           my $tmp2 =  $ids[$index-1];
           if(defined $hisdata[$i-1] && defined $name && defined $tmp2)
             {
               print OUTPUT "$name $tmp2 " . $hisdata[$i-1] . "\n";
             }
         }
     }
    close OUTPUT;
  }
