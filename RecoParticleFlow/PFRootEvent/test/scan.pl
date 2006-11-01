#!/usr/bin/perl -w
# Author: C. Bernet 2005

#use strict;
use diagnostics;
use Getopt::Long;

my $tag1=0;
my $tag2=0;
my $sfiles=0;
my $svalues=0;
my $masterfile="pfRootEvent.opt";
my $doNotSubmit=0;


GetOptions ('tag1=s' => \$tag1,
	    'tag2=s' => \$tag2,
	    'files=s' => \$sfiles,
	    'values=s' => \$svalues,
	    'master=s' => \$masterfile, 
	    'n' => \$doNotSubmit );


if($doNotSubmit) {
    print "will do nothing... \n";
}

print "master : $masterfile\n";
print "========== tags: $tag1 $tag2 ==  values: $svalues == $sfiles ================= \n";

my @rootfiles;
my $basedir = 0;

if( $sfiles =~ /^\/castor/) {
#    print "source files are on castor\n";
    $basedir = "rfio://$sfiles";
    my @tmpfiles = `nsls $sfiles`;
    foreach my $file (@tmpfiles) {
	if( $file =~ /\.root$/ ) {
	    push(@rootfiles, $file); 
	}
    }
}
else {
    @rootfiles = `ls $sfiles`;
}


#print "$basedir\n";
#print "@rootfiles \n";

my @values = split(" ",$svalues);
#if($svalues == 0) {
#    print "no values given\n"
#}
#else {
#    print "values : @values \n";
#}
my $masterprocess = "Macros/process.C";


foreach my $rootfile (@rootfiles) {
    chomp $rootfile;
    my $fullfilename = "$rootfile";
    if($basedir) {
	$fullfilename = "$basedir/$rootfile";
    }
    print "processing :  $fullfilename\n";
    
    if($svalues == 0) {
	my $outfile = "auto_$masterfile.opt";

	open(IN, "<$masterfile");
	open(OUT, ">$outfile");
	while ( <IN> ) {
	    my $line = $_;	    
	    if($line =~ m!root\s+file\s+! ) {
		if($line !~ /^\s*\/\//) {
		    print OUT "root file $fullfilename\n";
		}
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
	if(! $doNotSubmit) {
	    `nice -n 5 root -b tmpprocess.C`;
	    my $outrootfile = "out_$rootfile";
	    `mv out.root $outrootfile`;
	}
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
		    print OUT "root file $fullfilename\n";
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
	    if(! $doNotSubmit) {
		`nice -n 5 root -b tmpprocess.C`;
		my $outrootname = "$tag1\_$tag2\_$val.root";
		`mv out.root out_$outrootname`;
	    }
#    print "done $val\n";
	}
	
	my $outdirname = "Out\_$tag1\_$tag2\_$rootfile";
	$outdirname =~ s/\.root//;
	`mkdir $outdirname`;
	my $outpattern = "out_$tag1\_$tag2\_*.root";
	`mv $outpattern $outdirname`;

    }

}

