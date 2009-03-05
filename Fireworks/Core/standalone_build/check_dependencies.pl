#!/usr/bin/env perl
use strict;
use warnings;

die "Usage:\n\t$0 <source file> <dependencies file> [includes]\n\n"
  if (@ARGV < 2 );

my $output = "";
open (IN,  $ARGV[0]) || die "Cannot open file: $ARGV[0]\n$!\n\n";
while ( my $line = <IN> ){
    if ( my ($include) = ( $line =~ /\#include\s+\"([^\/]+\/[^\/]+\/[^\/]+\/[^\/]+)\"/ ) ){
	if ( -e "src/$include" ){
	    $output .= " src/$include";
	    next;
	}
	if ( -e "cms/$include" ){
	    $output .= " cms/$include";
	    next;
	}
	print "WARNING: couldn't locate include: $include. Ignored\n";
    }
}
my $object_file = $ARGV[1];
$object_file =~ s/\.d$/\.o/;
close IN;
if ( my ($dir) = ($ARGV[1] =~ /^(.*?)\/[^\/]+$/) ){
    system("mkdir -p $dir") if ( ! -e $dir );
}
open (OUT, ">$ARGV[1]") || die "Cannot write to file: $ARGV[1]\n$!\n\n";
print OUT "$object_file: $output\n";
close OUT;
exit
