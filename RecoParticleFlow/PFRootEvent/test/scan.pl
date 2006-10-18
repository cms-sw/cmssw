#!/usr/bin/perl -w
# Author: C. Bernet 2005

#use strict;
use diagnostics;
use Getopt::Long;

my $tag1=0;
my $tag2=0;
my $sfiles=0;
my $svalues=0;
my $masterfile=0;

GetOptions ('tag1=s' => \$tag1,
	    'tag2=s' => \$tag2,
	    'files=s' => \$sfiles,
	    'values=s' => \$svalues,
	    'master=s' => \$masterfile);



print "========== $tag1 $tag2 /  $svalues / $sfiles ================= \n";

my @rootfiles = `ls $sfiles`;
print "@rootfiles \n";
my @values = split(" ",$svalues);
if($#values == -1) {
    print "no values given\n"
}
else {
    print "@values \n";
}
my $masterprocess = "Macros/process.C";


foreach my $rootfile (@rootfiles) {
    chomp $rootfile;
    print "processing : $rootfile\n";
    
    if($#values == -1) {
	my $outfile = "auto_$masterfile.opt";

	open(IN, "<$masterfile");
	open(OUT, ">$outfile");
	while ( <IN> ) {
	    my $line = $_;
	    
	    if($line =~ /root\s+file\s+/ && $line !~ /\s*\/\//) {
		print OUT "root file $rootfile\n";
	    }
	    else {
		print OUT "$line";
	    }
	}
	close(IN);
	close(OUT);
	
	open(IN, "<$masterprocess");
	`rm -f tmpprocess.C`;
	open(OUT, ">tmpprocess.C");
	while ( <IN> ) {
	    my $line = $_;
	    
	    if($line =~ /^PFRootEventManagerColin\s+em/ && $line !~ /\s*\/\//) {
		print OUT "PFRootEventManagerColin em(\"$outfile\");\n";
	    }
	    else {
		print OUT "$line";
	    }
	}
	`root -b tmpprocess.C`;
	my $outrootfile = "out_$rootfile";
	`mv out.root $outrootfile`;
    }
    
    else {
	foreach my $val (@values) {
	    
	    print "$masterfile: $tag1 $tag2 $val\n";
	    my $outfile = "auto_$masterfile\_$tag1\_$tag2\_$val.opt";
	    
	    open(IN, "<$masterfile");
	    open(OUT, ">$outfile");
	    while ( <IN> ) {
		my $line = $_;

		if($line =~ /$tag1\s+$tag2\s+/ && $line !~ /\s*\/\//) {
		    print OUT "$tag1 $tag2 $val\n";
		}
		elsif($line =~ /root\s+file\s+/ && $line !~ /\s*\/\//) {
		    print OUT "root file $rootfile\n";
		}
		else {
		    print OUT "$line";
		}
	    }
	    close(IN);
	    close(OUT);
	    
	    open(IN, "<$masterprocess");
	    `rm -f tmpprocess.C`;
	    open(OUT, ">tmpprocess.C");
	    while ( <IN> ) {
		my $line = $_;
		
		if($line =~ /^PFRootEventManagerColin\s+em/ && $line !~ /\s*\/\//) {
		    print OUT "PFRootEventManagerColin em(\"$outfile\");\n";
		}
		elsif($line =~ /(.*)<<val<<(.*)/ && $line !~ /\s*\/\//) {
		    print OUT "$1<<$val<<$2\n"; 
		}
		else {
		    print OUT "$line";
		}
	    }
	    `root -b tmpprocess.C`;
	    my $outrootname = "$tag1\_$tag2\_$val.root";
	    `mv out.root out_$outrootname`;
#    print "done $val\n";
	}
	
	my $outdirname = "Out\_$tag1\_$tag2\_$rootfile";
	$outdirname =~ s/\.root//;
	`mkdir $outdirname`;
	my $outpattern = "out_$tag1\_$tag2\_*.root";
	`mv $outpattern $outdirname`;
    }

}

