#!/usr/bin/env perl

use warnings;
use strict;

use CondDB::MySQL;

package ConnectionFile;

# connection information
our $host = "localhost";
our $db = "test_condDb2";
our $user = "";
our $pass = "";
our $db_opts = {RaiseError=>1};

# return the conditions database interface
sub connect {
  my $condDB = new CondDB::MySQL;
  
  $condDB->connect(-host=>$host,
		   -db=>$db,
		   -user=>$user,
		   -pass=>$pass,
		   -db_opts=>$db_opts);

  return $condDB;
}

1;
