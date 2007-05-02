#! /usr/local/bin/perl -w

use strict;
use Term::ReadKey;


my $destdir = '../MyPFRootEvent/';


protectedCopy( 'template/', "$destdir");

protectedCopy( 'interface/MyPFRootEventManager.h', 
	       "$destdir/interface/MyPFRootEventManager.h");
protectedCopy( 'src/MyPFRootEventManager.cc', 
	       "$destdir/src/MyPFRootEventManager.cc");
protectedCopy( 'test/pfRootEvent.opt', 
	       "$destdir/workdir/pfRootEvent.opt");

print "\n";
print "success. Now execute: \n";
print "cd $destdir\n";
print "scramv1 b\n";
print "\n";
print "cd workdir\n";
print "root\n";
print ".x display.C\n";


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
