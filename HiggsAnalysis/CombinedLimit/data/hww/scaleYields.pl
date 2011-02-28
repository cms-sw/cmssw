#!/usr/bin/env perl


my %xs = (); 
my $xsfile = "cards/XS.SM.txt";
open XS, $xsfile or die "Cannot read cross sections from $xsfile";
foreach (<XS>) { m/^(\d+)\s+(\S+)/ and $xs{$1} = $2; }

while (<>) {
    s{^(\d+) Yield\s+(\d+)\s+(\S+)}{sprintf('%3d Yield %d %.3f', $1, $2, $3/$xs\{$1\})}e;
    print;
}

