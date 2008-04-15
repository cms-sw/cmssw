#!/usr/bin/env perl 

use strict;
use Term::ReadKey;


my $destdir = '../MyPFRootEvent/';
my $macro = 'tauBenchmarkDisplay_famos.C';
my $opt = 'tauBenchmark_famos.opt';


protectedCopy( 'template/', "$destdir");

protectedCopy( 'interface/MyPFRootEventManager.h', 
	       "$destdir/interface/MyPFRootEventManager.h");
protectedCopy( 'src/MyPFRootEventManager.cc', 
	       "$destdir/src/MyPFRootEventManager.cc");
protectedCopy( "test/$opt", 
	       "$destdir/workdir/$opt");
protectedCopy( "test/Macros/", 
	       "$destdir/workdir/Macros");
protectedCopy( "test/init.C", 
	       "$destdir/workdir/init.C");

print "\n";
print "success. Now execute: \n";
print "cd $destdir\n";
print "scramv1 b -j 4\n";
print "\n";
print "cd workdir\n";
print "root\n";
print ".x Macros/$macro\n";


sub protectedCopy {

    my $source = shift;
    my $dest = shift;
    if(-e "$dest") {
	print "$dest exists. do you really want to overwrite it (y|n)?\n";
	my $answer = ReadLine(0);

	if($answer =~ /(y|Y)/) {
	    print "cp -r $source $dest\n";
	    `rm -r $dest`;
	    `cp -r $source $dest`;
	}
	elsif($answer =~ /(n|N)/) {
	    print "no\n";
	}
	else {
	    print "what ?\n";
	    protectedCopy( $source, $dest);
	}
    }
    else {
	print "cp -r $source $dest\n";
	`cp -r $source $dest`;
    }
}
