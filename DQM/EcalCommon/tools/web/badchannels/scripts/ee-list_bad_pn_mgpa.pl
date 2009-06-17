#!/usr/bin/env perl

use warnings;
use strict;

use DBI;
use File::Basename;

die "Usage:  list_bad_pn_mgpa.pl site run status\n" unless ($#ARGV >= 2);

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
                     cv.id2 pn,
                     CAST(pnm.adc_mean_g1 AS NUMBER(6,1)) mean01,
                     CAST(pnm.adc_rms_g1 AS NUMBER(6,1)) rms01,
                     CAST(pnm.adc_mean_g16 AS NUMBER(6,1)) mean16,
                     CAST(pnm.adc_rms_g16 AS NUMBER(6,1)) rms16,
                     CAST(pnm.ped_mean_g1 AS NUMBER(6,1)) mean01,
                     CAST(pnm.ped_rms_g1 AS NUMBER(6,1)) rms01,
                     CAST(pnm.ped_mean_g16 AS NUMBER(6,1)) mean16,
                     CAST(pnm.ped_rms_g16 AS NUMBER(6,1)) rms16,
                     pnm.task_status status
                FROM run_iov riov
                JOIN run_tag rtag ON rtag.tag_id = riov.tag_id
                JOIN location_def ldef ON ldef.def_id = rtag.location_id
                JOIN mon_run_iov miov ON miov.run_iov_id = riov.iov_id
                JOIN mon_pn_mgpa_dat pnm ON pnm.iov_id = miov.iov_id
                LEFT OUTER JOIN channelview cv ON cv.logic_id = pnm.logic_id AND cv.name = cv.maps_to
               WHERE ldef.location = ?
                 AND riov.run_num = ?
                 AND pnm.task_status = ?
            ORDER BY run, sm, pn];

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

