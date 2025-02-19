#!/usr/bin/env perl

use warnings;
use strict;
$|++;
use File::Basename;
use Getopt::Long;

my $usage = basename($0)." --source_connect  --dest_connect time \n".
    "Options:\n".
    "--source_connect  source connect string: user/pass\@db(required)\n".
    "--dest_connect    destination database: user/pass\@db(required) \n".
    "--help, -h        Print this message and exit\n";

my $source_connect='' ;
my $dest_connect='' ;
my $help = 0;
GetOptions('source_connect=s' => \$source_connect,
	   'dest_connect=s' => \$dest_connect,
	   'help|h' => \$help );
if ($help) {
    print "$usage";
    exit;
}

my $time = shift @ARGV;

die "Must provide polling time interval in second."  unless $time;

my $conn_source = "sqlplus -SL ${source_connect}";
my $conn_dest = "sqlplus -SL ${dest_connect}";

print "Opening pipes to sqlplus...";
open SOURCE, "| $conn_source >> source_poll.txt" or die $!;
print SOURCE "set serveroutput on;\nset echo off;\n";

open DEST, "| $conn_dest >> dest_poll.txt" or die $!;
print DEST "set serveroutput on;\nset echo off;\n";
print "Done.\n";

print "Beginning polling every $time s, use Ctl-c to stop\n\n";

while (1) {
    print SOURCE "call poll_db();\n";
    print DEST "call poll_db();\n";
    sleep($time);
    print `echo -n`; # I don't know why, but this program doesn't work with out this line
}

END {
    print "Closing pipes.\n";
    close SOURCE;
    close DEST;
}
