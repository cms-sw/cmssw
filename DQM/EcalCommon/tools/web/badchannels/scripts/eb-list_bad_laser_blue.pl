#!/usr/bin/env perl

use warnings;
use strict;

use DBI;
use File::Basename;

die "Usage:  list_bad_laser_blue.pl site run status\n" unless ($#ARGV >= 2);

my ($site,$run,$status) = @ARGV;

my $dbName     = '';
my $dbHostName = '';
my $dbUserName = '';
my $dbPassword = '';

my $cfg = dirname($0) . "/.cms_tstore_r.pl";
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
                     CAST(lab.apd_mean AS NUMBER(6,1)) mean,
                     CAST(lab.apd_rms AS NUMBER(6,1)) rms,
                     CAST(lab.apd_over_pn_mean AS NUMBER(6,1)) mean1,
                     CAST(lab.apd_over_pn_rms AS NUMBER(6,1)) rms1,
                     lab.task_status status
                FROM run_iov riov
                JOIN run_tag rtag ON rtag.tag_id = riov.tag_id
                JOIN location_def ldef ON ldef.def_id = rtag.location_id
                JOIN mon_run_iov miov ON miov.run_iov_id = riov.iov_id
                JOIN mon_laser_blue_dat lab ON lab.iov_id = miov.iov_id
                LEFT OUTER JOIN channelview cv ON cv.logic_id = lab.logic_id AND cv.name = cv.maps_to
               WHERE ldef.location = ?
                 AND riov.run_num = ?
                 AND lab.task_status = ?
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

