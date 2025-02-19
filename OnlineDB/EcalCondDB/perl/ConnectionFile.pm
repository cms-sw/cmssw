#!/usr/bin/env perl

use warnings;
use strict;

use CondDB::Oracle;

package ConnectionFile;

# connection information
our $host = "__CHANGE_ME__";
our $db = "__CHANGE_ME__";
our $user = "__CHANGE_ME__";
our $pass = "__CHANGE_ME__";
our $port = 1521;
our $db_opts = {RaiseError=>1};

# return the conditions database interface
sub connect {
  my $condDB = new CondDB::Oracle;
  
  $condDB->connect(-host=>$host,
		   -db=>$db,
		   -user=>$user,
		   -pass=>$pass,
		   -port=>$port,
		   -db_opts=>$db_opts);
  $condDB->{dbh}->do(qq[ ALTER SESSION SET NLS_DATE_FORMAT='YYYY-MM-DD HH24:MI:SS']);
  return $condDB;
}

1;
