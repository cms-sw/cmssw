#!/usr/bin/env perl

#
# Generates ER diagrams from configuration files
#
# Use dumpRelationships from an sqlplus session to generate the input files
# then run this script to generate the diagram as .dot and .png file
#
# Author: Giovanni.Organtini@roma1.infn.it 2011
#

use GraphViz;

my $g = GraphViz->new(rankdir => 1);

# read tables

open IN, "TABLES.DAT";
@buffer = <IN>;
close IN;

$lastTable = 'NONE';
@columns;
foreach $line (@buffer) {
    chomp $line;
    ($table, $column) = split / +/,$line;
    $column =~ s/\t*//g;
    $table =~ s/\t*//g;
    if ($table !~ m/$lastTable/) {
	$lastTable = $table;
    }
    push @columns, $table . "%" . $column;
}

$lastTable = 'NONE';
$p = 1;
%port;
foreach $col (@columns) {
    ($table, $column) = split /%/, $col;
    $name = $table . "." . $column;
    if ($table !~ m/$lastTable/) {
	print "Creating node $table\n";
	if ($lastTable !~ m/NONE/) {
	    $g->add_node($lastTable, label => \@label, shape => 'record', fontsize => 10, @default_attrs);	
	}
	@label = ();
	$p = 1;
	push @label, $table;
	$lastTable = $table;
    }
    $port{$name} = $p++;
    push @label, $column;
}

# read relationsips

open IN, "RELATIONSHIPS.DAT";
@buffer = <IN>;
close IN;

foreach $line (@buffer) {
    chomp $line;
    ($part, $parc, $chit, $chic) = split / +/, $line;
    $part =~ s/\t+//g;
    $parc =~ s/\t+//g;
    $chit =~ s/\t+//g;
    $chic =~ s/\t+//g;
    $parent = $part . "." . $parc;
    $child = $chit . "." . $chic;
    print "$parent -> $child\n";
    $g->add_edge($part => $chit, from_port => $port{$parent}, to_port => $port{$child},
	minlen => 10);
}

$g->as_png('test.png');
$g->as_text('test.dot');
