#!/usr/bin/env perl

# make a nice list of files 

$workdir = ".";
$thisdata = $ARGV[0];
$textfilename = $ARGV[1];
$json = $ARGV[2];

# retrieve cfi file
open(INFILE,"${thisdata}") or die "cannot open ${thisdata}";;
@log=<INFILE>;
close(INFILE);

# lines of cfi file
open(INFILE2,"${json}") or die "cannot open ${json}";;
@log2=<INFILE2>;
close(INFILE2);

open(MYFILE,">${textfilename}");;
my @goodrunsHalf1 = ();
my @goodrunsHalf2 = ();

foreach $line (@log2) {
   if ($line =~ /max/) {
       $half1 = substr($line,1,3);
       $half2 = substr($line,4,3);
       $half1 = $half1 . "/";
       $half2 = $half2 . "/";
       @goodrunsHalf1 = (@goodrunsHalf1,${half1});
       @goodrunsHalf2 = (@goodrunsHalf2,${half2});
   }
}

$goodruns1Size = @goodrunsHalf1;

my $i = 0;   # counter

foreach $line (@log) {
    while ($i < $goodruns1Size-1) {
        $i++;
	my $temp1 = $goodrunsHalf1[$i];
	my $temp2 = $goodrunsHalf2[$i];
        print "$i ${temp1} ${temp2} \n"; 
	if ($line =~ /${temp1}/ && $line =~ /${temp2}/) {
	    print MYFILE "$line";   
	}
    }
    $i = 0;
}


system("rm -f tmpfile tmpfile2");
print "Result in ${textfilename}\n";
 
