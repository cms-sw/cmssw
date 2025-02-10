#!/usr/bin/perl

open (F, $ARGV[0]) or die "can not open $ARGV[0]";

while (my $l = <F>)
{
    if ($l =~ m/^#include "(.*\.ah)"/)
    {
        my $if = $1;
        print "// BEGIN include $if\n\n";
        open I, $if or die "can not open $if";
        local $/; 
        my $i = <I>;
        close I;
        printf $i;
        printf "// END include $if\n\n";
    }
    else
    {
       print $l;
    }
}

# ./inplace-ahs.pl PropagationMPlex.icc-orig > PropagationMPlex.icc
# ./inplace-ahs.pl PropagationMPlex.cc-orig > PropagationMPlex.cc
