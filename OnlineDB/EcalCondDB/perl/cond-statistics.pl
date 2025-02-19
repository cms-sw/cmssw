#!/usr/bin/env perl

use lib "./lib";

use warnings;
use strict;
$|++;

use CondDB::MySQL;
use ConnectionFile;

my $condDB = ConnectionFile::connect();

my $dbh = $condDB->{dbh};

my @tables = @{$dbh->selectcol_arrayref("SHOW TABLES")};

@tables = grep /^COND_/, @tables;

my $temp1 = qq[ SELECT v.name FROM ? c LEFT JOIN channelView v
		ON c.logic_id = v.logic_id LIMIT 1];

my $temp2 = qq[ SELECT count(distinct logic_id) FROM ? ];

my $temp3 = qq[ SELECT min(since), max(since) FROM ? ];

my $temp4 = qq[ SELECT count(*) FROM ? ];


print join(',',("cond name", "view name", "# ch", "start", "end", "# rows")),
  "\n";

foreach (@tables) {
  my @data = ($_);
  push @data, doit($temp1, $_);
  push @data, doit($temp2, $_);
  push @data, doit($temp3, $_);
  push @data, doit($temp4, $_);

  print join(',', @data), "\n";
}

sub doit {
  my $sql = shift;
  my $table = shift;
  $sql =~ s/\?/$table/g;
  return $dbh->selectrow_array($sql);
}
