#!/usr/bin/env perl

# make a nice list of files 

$workdir = ".";
$thisdata = $ARGV[0];
$textfilename = $ARGV[1];
$filesperline = $ARGV[2];

# retrieve cfi file
open(INFILE,"${thisdata}") or die "cannot open ${thisdata}";;
@log=<INFILE>;
close(INFILE);

# lines of cfi file
system("more ${thisdata} | wc -l >> tmpfile2");
open(INFILE2,"tmpfile2") or die "cannot open tmpfile2";;
@log2=<INFILE2>;
close(INFILE2);

my $nlines = 0;
foreach $line2 (@log2) {
    $nlines = $line2;
}

open(MYFILE,">${textfilename}");;

my $ilines = 0;
my $totfilename = '';

foreach $line (@log) {
   $ilines = $ilines + 1;
   chomp($line);
   if ($ilines % $filesperline == 0) {
       $totfilename = $totfilename . "\'" . $line . "\'"; 
       print MYFILE "$totfilename \n";    
       $totfilename = '';
   } else {
       $totfilename = $totfilename . "\'" . $line . "\',"; 
   }
}

system("rm -f tmpfile tmpfile2");
print "Result in ${textfilename}\n";
 
