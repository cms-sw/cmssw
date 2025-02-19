#!/usr/bin/env perl

use lib "./lib";

use warnings;
use strict;
$|++;

use TB04::RunDB;
use ConnectionFile;

print "Connecting to DB...";
my $condDB = ConnectionFile::connect();
print "Done.\n";

print "Loading runs info...";
my $runDB = new TB04::RunDB;
#$rundb->connect();
#$rundb->load_from_db();
$runDB->load_from_file("rundb");  # XXX hardcoded file name
print "Done.\n";

print "Defining runs table...";
$runDB->fill_runs_table($condDB);
print "Done.\n";

print "Updating bad runs...";
open FILE, '<', "bad_runs" or die $!;
my $status = 0;
while (<FILE>) {
  chomp;
  my ($run, $comment) = split /,/;
  $condDB->update_run(-run_number => $run, 
		      -status => $status, 
		      -comments => $comment);
}
close FILE;
print "Done.\n";

print "All Done.\n";
