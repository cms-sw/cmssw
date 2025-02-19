#!/usr/bin/env perl

use lib "./lib";

use warnings;
use strict;
$|++;

use ConnectionFile;

print "Connecting to DB...";
my $condDB = ConnectionFile::connect();
print "Done.\n";

print "Destroying old DB...";
$condDB->destroydb(-name=>$ConnectionFile::db);
print "Done.\n";

print "Creating new DB...";
$condDB->newdb(-name=>$ConnectionFile::db);
print "Done.\n";

print "Defining stored procedures...";
$condDB->define_procedure(-file=>"../sql/update_online_cndc_iov.sql");
print "Done.\n"
