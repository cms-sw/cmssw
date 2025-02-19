#!/usr/bin/env perl

use lib "./lib";

use warnings;
use strict;
$|++;

use CondDB::channelView;
use ConnectionFile;
use Getopt::Long;

my $all = 0;
GetOptions( 'all' => \$all );

unless ($all || @ARGV) {
  die "Nothing to define.\n";
}

print "Connecting to DB...";
my $condDB = ConnectionFile::connect();
print "Done.\n";

my $cv = CondDB::channelView->new($condDB);

if ($all) {
  $cv->define_all();
} else {
  foreach (@ARGV) {
    $cv->define($_);
  }
}
