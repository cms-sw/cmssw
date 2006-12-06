#!/usr/bin/perl -w
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

my $doNotSubmit=0;
my $help=0;

my $bsub=0;

my $pwd = `pwd`;
chomp $pwd;

my $date = `date +%d%b%Y_%H%M%S`;
my $scandir = "ScanOut_$date";
chomp $scandir;
`mkdir $scandir`;
`echo "scan.pl @ARGV" > scan.pl.log`;
`mv scan.pl.log $scandir`;

GetOptions ('tag1=s' => \$tag1,
	    'tag2=s' => \$tag2,
	    'files=s' => \$sfiles,
	    'pattern=s' => \$filePattern, 
	    'values=s' => \$svalues,
	    'master=s' => \$masterfile, 
	    'h' => \$help,
	    'n' => \$doNotSubmit, 
	    'b=s' => \$bsub );

if($help) {
    print "usage : scan.pl -tag1 clustering -tag2 thresh_Ecal_Endcap -files=\"*.root\" -values \"0.1 0.3 0.5\" -master pfRootEvent.opt [-n] [-b \"bsub -q 8nm\"]\n";
    print " -n : do not proceed\n";
    print " -b <>: run on the batch system (LSF)\n";
    exit(1);
}

if($doNotSubmit) {
    print "will do nothing... \n";
}

print "master : $masterfile\n";
print "======== tags: $tag1 $tag2 ==  values: $svalues == $sfiles ======= \n";



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

my $masterprocess = "Macros/process.C";


foreach my $rootfile (@rootfiles) {
    chomp $rootfile;
    my $fullfilename = "$basedir/$rootfile";
    print "processing :  $fullfilename\n";
    
    if($svalues == 0) {
	my $optfile = "$pwd/$scandir/auto_$rootfile.opt";
	my $outrootfile = "$pwd/$scandir/out_$rootfile";

	open(IN, "<$masterfile");
	open(OUT, ">$optfile");
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
	    else {
		print OUT "$line";
	    }
	}
	close(IN);
	close(OUT);
	
	my $macro = "$pwd/$scandir/auto_$rootfile.C";
	open(IN, "<$masterprocess");
#	`rm -f tmpprocess.C`;
	open(OUT, ">$macro");
	while ( <IN> ) {
	    my $line = $_;
	    
	    if($line =~ /^PFRootEventManagerColin\s+em/ && $line !~ /\s*\/\//) {
		print OUT "PFRootEventManagerColin em(\"$optfile\");\n";
	    }
	    else {
		print OUT "$line";
	    }
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

