#!/usr/bin/env perl

use warnings;
use strict;

use DBI;
use File::Basename;

die "Usage: runtype.pl run\n" unless ($#ARGV == 0);

my ($run) = @ARGV;

my $dbName     = '';
my $dbHostName = '';
my $dbUserName = '';
my $dbPassword = '';

my $cfg = dirname($0) . "/.omds_r.conf";
if ( -e "$cfg" ) {
  eval `cat $cfg`;
}

my $dsn = "DBI:Oracle:$dbName";

my $dbh = DBI->connect($dsn, $dbUserName, $dbPassword) || die "DB problems";

my $sql = qq[ SELECT rdef.run_type type
              FROM run_iov riov
              JOIN run_tag rtag ON rtag.tag_id = riov.tag_id
              JOIN run_type_def rdef ON rdef.def_id = rtag.run_type_id
              WHERE riov.run_num = ?];

my $sth = $dbh->prepare_cached($sql);

$sth->execute($run);

my @row = $sth->fetchrow();
print @row;

$sth->finish();

$dbh->disconnect();

exit;

