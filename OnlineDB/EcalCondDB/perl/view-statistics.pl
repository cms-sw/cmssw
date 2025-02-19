#!/usr/bin/env perl

use lib "./lib";

use warnings;
use strict;
$|++;

use ConnectionFile;

my $condDB = ConnectionFile::connect();

my $dbh = $condDB->{dbh};

my $sql = qq[  select cv.name, cv.maps_to, count(cv.logic_id), 
               min(cv.logic_id) m, max(cv.logic_id),
	       cd.id1name, min(cv.id1), max(cv.id1), 
               cd.id2name, min(cv.id2), max(cv.id2), 
               cd.id3name, min(cv.id3), max(cv.id3), 
               cd.description 
	       from channelView cv, viewDescription cd 
	       where cv.name = cd.name AND cv.name = cv.maps_to
	       group by cv.name, cv.maps_to, cd.id1name, 
	       cd.id2name, cd.id3name, cd.description 
               order by m
            ];

my @data = @{$dbh->selectall_arrayref($sql)};

my @headers = qw(name maps_to count min_logic_id max_logic_id 
                 id1name min_id1 max_id1 
                 id2name min_id2 max_id2
                 id3name min_id3 max_id3
		 description);

my $head = join(",", map("\"$_\"", @headers));

print "DIRECT CHANNEL VIEWS\n";
print $head, "\n";
foreach (@data) {
  no warnings;
  my @row = @{$_};
  $row[$#row] = "\"$row[$#row]\"";
  print join(",", @row), "\n";
}

print "\n";
print "CHANNEL VIEW MAPPINGS\n";
print $head, "\n";
$sql = qq[ select cv.name, cv.maps_to, count(cv.logic_id), 
           min(cv.logic_id) m, max(cv.logic_id),
	   cd.id1name, min(cv.id1), max(cv.id1), 
           cd.id2name, min(cv.id2), max(cv.id2), 
           cd.id3name, min(cv.id3), max(cv.id3), 
           cd.description 
	   from channelView cv, viewDescription cd 
	   where cv.name = cd.name AND cv.name != cv.maps_to
	   group by cv.name, cv.maps_to, cd.id1name, 
	   cd.id2name, cd.id3name, cd.description 
           order by m 
         ];

@data = @{$dbh->selectall_arrayref($sql)};
foreach (@data) {
  no warnings;
  my @row = @{$_};
  $row[$#row] = "\"$row[$#row]\"";
  print join(",", @row), "\n";
}
