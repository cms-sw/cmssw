#!/usr/bin/env perl

use warnings;
use strict;

use DBI;
use File::Basename;

die "Usage:  list_bad_laser_red.pl site run status\n" unless ($#ARGV >= 2);

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
                     cv.id1 side,
                     cv.id2 ix,
                     cv.id3 iy,
                     cv2.id1 dcc,
                     cv2.id2 ccu,
                     floor((cv2.id3-1)/5)+1 strip,
                     mod(cv2.id3-1,5)+1 chan,
                     CAST(lar.apd_mean AS NUMBER(6,1)) mean,
                     CAST(lar.apd_rms AS NUMBER(6,1)) rms,
                     CAST(lar.apd_over_pn_mean AS NUMBER(6,1)) mean1,
                     CAST(lar.apd_over_pn_rms AS NUMBER(6,1)) rms1,
                     lar.task_status status
                FROM run_iov riov
                JOIN run_tag rtag ON rtag.tag_id = riov.tag_id
                JOIN location_def ldef ON ldef.def_id = rtag.location_id
                JOIN mon_run_iov miov ON miov.run_iov_id = riov.iov_id
                JOIN mon_laser_red_dat lar ON lar.iov_id = miov.iov_id
                LEFT OUTER JOIN channelview cv ON cv.logic_id = lar.logic_id AND cv.name = cv.maps_to
                LEFT OUTER JOIN channelview cv2 ON cv2.logic_id = lar.logic_id AND cv2.name = 'EE_crystal_readout_strip'
               WHERE ldef.location = ?
                 AND riov.run_num = ?
                 AND lar.task_status = ?
            ORDER BY run, dcc, ix, iy];

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

