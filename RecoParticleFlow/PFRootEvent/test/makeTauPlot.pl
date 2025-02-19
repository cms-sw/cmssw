#!/usr/bin/env perl -w

use strict;

my $help = 0;
 
if ($#ARGV != 0) {$help=1;}


if( $help ) {
    print "usage: makeTauPlot.pl <Directory>\n";
    exit(1);
}

my $directory = shift;
if( !-d $directory ) {
    print "directory \"$directory\" does not exist.\n";
    exit(1);
}

my @rootFiles = `ls $directory/*.root`;
if( $#rootFiles == -1) {
    print "check your input root files in directory \"$directory\"\n";
    exit(1);
}

my $outFile = "tauBenchmark.root";
if( -f $outFile ) {
    `rm $outFile`;
}

`hadd $outFile $directory/*.root`;

if( -f $outFile ) {
    `root -b Macros/makeTauPlot.C`;
}
