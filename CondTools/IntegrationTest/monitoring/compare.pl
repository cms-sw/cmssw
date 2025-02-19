#!/usr/bin/env perl

use warnings;
use strict;
$|++;

my $schema1 = shift @ARGV;
my $schema2 = shift @ARGV;

my @outputs;

foreach ($schema1, $schema2) {
  my ($user, $sid) = ($_ =~ /^(.*)\/.*@(.*)$/);
  unless ($user && $sid) {
    die "$_ is not an sqlplus connection string";
  }
  my $output = $user."-".$sid.".txt";
  print "Writing $output...";
  `sqlplus -SL $_ < compare.sql > $output`;
  print "Done\n";
  push @outputs, $output;
}


my $output = "diff-".$outputs[0]."-".$outputs[1];
print "Writing $output...\n";
my $cmd = "diff $outputs[0] $outputs[1] > $output";
print $cmd."\n";
`$cmd`;
print "Done\n";

print `cat $output`;

exit;
