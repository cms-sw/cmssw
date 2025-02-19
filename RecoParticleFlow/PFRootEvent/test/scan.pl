#!/usr/bin/env perl -w
# Author: C. Bernet 2005

#use strict;
use diagnostics;
use Getopt::Long;
use File::Basename;

my $tag1=0;
my $tag2=0;

my $sfiles=0;
my $filePattern=0;

my $svalues=0;

my $masterfile="pfRootEvent.opt";
my $masterprocess = "Macros/process.C";

my $doNotSubmit=0;
my $help=0;

my $bsub=0;

my $pwd = `pwd`;
chomp $pwd;
my @args =  @ARGV;

GetOptions ('tag1=s' => \$tag1,
	    'tag2=s' => \$tag2,
	    'files=s' => \$sfiles,
	    'pattern=s' => \$filePattern, 
	    'values=s' => \$svalues,
	    'master=s' => \$masterfile, 
	    'process=s' => \$masterprocess,
	    'h' => \$help,
	    'n' => \$doNotSubmit, 
	    'b=s' => \$bsub );

if($help) {
    print "usage : scan.pl -tag1 clustering -tag2 thresh_Ecal_Endcap -files=\"*.root\" -values \"0.1 0.3 0.5\" -master pfRootEvent.opt -process Macros/tauBenchmarkProcess.C [-n] [-b \"bsub -q 8nm\"]\n";
    print " -n : do not proceed\n";
    print " -b <>: run on the batch system (LSF)\n";
    exit(1);
}


if($doNotSubmit) {
    print "will do nothing... \n";
}

print "master : $masterfile\n";
print "process : $masterprocess\n";
print "======== tags: $tag1 $tag2 ==  values: $svalues == $sfiles ======= \n";


my $date = `date +%d%b%Y_%H%M%S`;
my $scandir = "ScanOut\_$tag1\_$tag2\_$date";
chomp $scandir;
`mkdir $scandir`;
`echo "scan.pl @args" > scan.pl.log`;
`mv scan.pl.log $scandir`;

my $ls = "ls";
my $basedir = "$sfiles"; # contains directory path

if( $sfiles =~ /^\/castor/) {
#    print "source files are on castor\n";
    $ls = "nsls";
    $basedir = "rfio://$sfiles";
}  


my @rootfiles; # will contain base filenames (no path)

my @tmpfiles = `$ls $sfiles`;
foreach my $file (@tmpfiles) {
    if( $file =~ /\.root$/ ) {
	if( ( $filePattern && 
	      $file =~ /$filePattern/ ) ||
	    !$filePattern ) {
	    push(@rootfiles, basename($file) );
		
	}
    }
}


my @values = split(" ",$svalues);




foreach my $value (@values) {
    my $valdir = "$tag1\_$tag2\_$value";
    my $outdir = "$pwd/$scandir/$valdir";
    
    `mkdir $outdir`;

    foreach my $rootfile (@rootfiles) {
	chomp $rootfile;
	my $fullfilename = "$basedir/$rootfile";
	print "processing :  $fullfilename : $tag1 $tag2 $value\n";
	
#    if($svalues == 0) {
	my $optfile = "$outdir/auto_$rootfile.opt";
	my $outrootfile = "$outdir/out_$rootfile";
	
	open(IN, "<$masterfile");
	open(OUT, ">$optfile");

	my $tagsFound = 0;
	while ( <IN> ) {
	    my $line = $_;	    
	    if($line =~ m!root\s+file\s+! ) {
		if($line !~ /^\s*\/\//) {
		    print OUT "root file $fullfilename\n";
		}
	    }
	    elsif($line =~ m!root\s+outfile\s+! ) {
		print OUT "root outfile $outrootfile\n";
	    }
	    
	    elsif($line =~ /$tag1\s+$tag2\s+/ && $line !~ /\s*\/\//) {
		print OUT "$tag1 $tag2 $value\n";
		 $tagsFound = 1;
	    }
	    else {
		print OUT "$line";
	    }
	}
	close(IN);
	close(OUT);
	
	my $macro = "$outdir/auto_$rootfile.C";
	open(IN, "<$masterprocess");
#	`rm -f tmpprocess.C`;
	open(OUT, ">$macro");
	while ( <IN> ) {
	    my $line = $_;
	    
	    if($line =~ /(PFRootEventManager\S*)\s+em/ && $line !~ /\s*\/\//) {
		print OUT "$1 em(\"$optfile\");\n";
	    }
	    else {
		print OUT "$line";
	    }
	}
	
	if(! $tagsFound && $tag1 && $tag2 ) {
	    $doNotSubmit = 1;
	    print "cannot find tags $tag1 $tag2, job not submitted\n";
	}
	if(! $doNotSubmit) {
#	    my $outrootfile = "$scandir/out_$rootfile";
	    if( !$bsub ) { # standard execution
		`nice -n 5 root -b $macro`;
#		`mv out.root $outrootfile`;
	    }
	    else { # batch execution 
		# print "batch \n";
		print "$bsub root -b $macro\n";
		`$bsub root -b $macro`;
		# `mv out.root $outrootfile`;		
	    }
	    #`cp $macro $scandir`;
	    #`cp $optfile $scandir`;	    
	}
	
    }
}
