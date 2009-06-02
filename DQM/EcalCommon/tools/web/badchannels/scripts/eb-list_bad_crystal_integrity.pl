#!/usr/bin/env perl

use warnings;
use strict;

use DBI;
use File::Basename;

die "Usage:  list_bad_crystal_integrity.pl site run status\n" unless ($#ARGV >= 2);

my ($site,$run,$status) = @ARGV;

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

my $sql = qq[ SELECT riov.run_num run,
                     cv.id1 sm,
                     cv.id2 cry,
                     FLOOR((cv.id2-1)/20) eta,
                     MOD((cv.id2-1),20) phi,
                     CAST(con.processed_events AS NUMBER) evts,
                     CAST(con.problematic_events AS NUMBER) prob,
                     CAST(con.problems_id AS NUMBER) id,
                     CAST(con.problems_gain_zero AS NUMBER) gain,
                     CAST(con.problems_gain_switch AS NUMBER) switch,
                     con.task_status status
                FROM run_iov riov
                JOIN run_tag rtag ON rtag.tag_id = riov.tag_id
                JOIN location_def ldef ON ldef.def_id = rtag.location_id
                JOIN mon_run_iov miov ON miov.run_iov_id = riov.iov_id
                JOIN mon_crystal_consistency_dat con ON con.iov_id = miov.iov_id
                LEFT OUTER JOIN channelview cv ON cv.logic_id = con.logic_id AND cv.name = cv.maps_to
               WHERE ldef.location = ?
                 AND riov.run_num = ?
                 AND con.task_status = ?
            ORDER BY run, sm, cry];

my $sth = $dbh->prepare_cached($sql);

$sth->execute($site,$run,$status);

print join("\t", @{$sth->{NAME}}), "\n";

while (my @row = $sth->fetchrow()) {
    @row = map { defined $_ ? $_ : "" } @row;
    print join("\t", @row), "\n";
}

$sth->finish();

$dbh->disconnect();

exit;

