#!/usr/bin/perl

use warnings;
use strict;
$|++;

my $time = shift @ARGV;

die "Provide time."  unless $time;

my $orcon_user = "CMS_COND_CSC";
my $orcon_pass = "__CHANGE_ME__";
my $orcoff_user = "CMS_COND_CSC";
my $orcoff_pass = "__CHANGE_ME__";

my $conn_orcon = "sqlplus -SL ${orcon_user}/${orcon_pass}\@orcon";
my $conn_orcoff = "sqlplus -SL ${orcoff_user}/${orcoff_pass}\@cms_orcoff";

print "Opening pipes to sqlplus...";
open ORCON, "| $conn_orcon >> orcon_poll.txt" or die $!;
print ORCON "set serveroutput on;\nset echo off;\n";

open ORCOFF, "| $conn_orcoff >> orcoff_poll.txt" or die $!;
print ORCOFF "set serveroutput on;\nset echo off;\n";
print "Done.\n";

print "Beginning polling every $time s, use Ctl-c to stop\n\n";

while (1) {
    print ORCON "call poll_db();\n";
    print ORCOFF "call poll_db();\n";
    sleep($time);
    print `echo -n`; # I don't know why, but this program doesn't work with out this line
}

END {
    print "Closing pipes.\n";
    close ORCON;
    close ORCOFF;
}
