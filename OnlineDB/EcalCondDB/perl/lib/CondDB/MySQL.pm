#!/usr/bin/env perl

use warnings;
use strict;

use DBI;

package CondDB::MySQL;

our $MAXNAME = 32;

# creates a new database interface
sub new {
  my $proto = shift;
  my $class = ref($proto) || $proto;
  my $this = {};

  $this->{counter} = 0;
  $this->{max_cnt} = 100;

  my %args = @_;
  $this->{check_overlap} = 1;

  bless($this, $class);
  return $this;
}

sub DESTROY {
  my $this = shift;
  if ($this->{transaction}) {
    $this->commit();
  }

  foreach (keys %{$this->{prepared}}) {
    $this->{prepared}->{$_}->finish();
  }

  foreach (keys %{$this->{prepared_overlap}}) {
    $this->{prepared_overlap}->{$_}->finish();
  }

  $this->{dbh}->disconnect() if defined $this->{dbh};
}

sub set_option {
  my $this = shift;
  my %args = @_;

  if (exists $args{-check_overlap}) {
    $this->{check_overlap} = !!$args{-check_overlap};
  }
}

# connect to a database server, optionally selecting the database
sub connect {
  my $this = shift;
  my %args = @_;
  my $host = $args{-host};
  my $user = $args{-user};
  my $pass = $args{-pass};
  my $db_opts = $args{-db_opts};

  unless ($host) {
    die "ERROR:  Must give at least host to connect():, $!"
  }

  $db_opts->{AutoCommit} = 1;

  my $db = $args{-db};

  my $dsn;
  if ($db) {
    $dsn = "DBI:mysql:database=$db;host=$host";
  } else {
    $dsn = "DBI:mysql:host=$host";
  }

  my $dbh = DBI->connect($dsn, $user, $pass, $db_opts)
    or die "Database connection failed, $DBI::errstr";

  $this->{host} = $host;
  $this->{user} = $user;
  $this->{db} = $db;
  $this->{db_opts} = $db_opts;
  $this->{dbh} = $dbh;

  return 1;
}

sub begin_work {
  my $this = shift;
  $this->{dbh}->begin_work();
  $this->{transaction} = 1;
}

sub commit {
  my $this = shift;
  $this->{dbh}->commit();
  $this->{transaction} = 0;
}

sub rollback {
  my $this = shift;
  $this->{dbh}->rollback();
  $this->{transaction} = 0;
}

# creates and uses a new database
sub newdb {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $name = $args{-name} or die "ERROR:  Must give name to newdb(), $!";
  $dbh->do(qq[ CREATE DATABASE $name ]) or die "ERROR:  DB creation failed, ".
    $dbh->errstr;
  $dbh->do(qq[ USE $name ]);

  my $sql =<<END_SQL;
CREATE TABLE conditionDescription (
  name varchar($MAXNAME) NOT NULL,
  description text,
  units varchar(255) default NULL,
  datatype char(1) NOT NULL,
  datasize int NOT NULL default '1',
  hasError tinyint(1) NOT NULL default '0',
  PRIMARY KEY  (name)
) TYPE=InnoDB;
END_SQL

  $dbh->do($sql);

  $sql =<<END_SQL;
CREATE TABLE viewDescription (
  name varchar($MAXNAME) NOT NULL,
  description text,
  id1name varchar($MAXNAME) default NULL,
  id2name varchar($MAXNAME) default NULL,
  id3name varchar($MAXNAME) default NULL,
  PRIMARY KEY  (name)
) TYPE=InnoDB;
END_SQL

  $dbh->do($sql);

  $sql =<<END_SQL;
CREATE TABLE channelView (
  name varchar($MAXNAME) NOT NULL,
  id1 int(11) default NULL,
  id2 int(11) default NULL,
  id3 int(11) default NULL,
  maps_to varchar($MAXNAME) NOT NULL,
  logic_id int(11) NOT NULL,
  UNIQUE (name, id1, id2, id3, logic_id),
  INDEX maps_to (maps_to),
  INDEX logic_id (logic_id)
) TYPE=InnoDB;
END_SQL
  $dbh->do($sql);

$sql = <<END_SQL;
CREATE TABLE runs (
  run_number INT PRIMARY KEY,
  since DATETIME,
  till DATETIME,
  status TINYINT DEFAULT 1,
  comments TEXT,
  INDEX (since, till)
) Type=InnoDB
END_SQL

  $dbh->do($sql);

  $this->{db} = $name;
  return 1;
}

# drop a database
sub destroydb {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $name = $args{-name} or die "Must give name to newdb(), $!";

  $dbh->do(qq[ DROP DATABASE IF EXISTS $name ]) 
    or die "destroydb failed, ".$dbh->errstr;

  return 1;
}

# choose the database to use
sub selectdb {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $name = $args{-name} or die "Must give name to newdb(), $!";

  $dbh->do(qq[ USE $name ]);
}


# delete existing view if it exists and create view in DB
sub new_channelView_type {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $name = $args{-name};
  my $description = $args{-description};
  my ($id1name, $id2name, $id3name) = @{$args{-idnames}};

  check_names($name, $id1name, $id2name, $id3name);

  my $sql;
  { no warnings;
    $sql = qq[ DELETE FROM channelView WHERE name="$name" ];
    $sql =~ s/\"\"/NULL/g;
  }
  $dbh->do($sql);

  { no warnings;
    $sql = qq[ DELETE FROM viewDescription WHERE name="$name" ];
    $sql =~ s/\"\"/NULL/g;
  }
  $dbh->do($sql);

  { no warnings;
    $sql = qq[ INSERT INTO viewDescription
	       SET name="$name",
	           description="$description",
	           id1name="$id1name",
	           id2name="$id2name",
	           id3name="$id3name" ];
    $sql =~ s/\"\"/NULL/g;
  }
  $dbh->do($sql);

  return 1;
}

# delete old condition information if it exists, then create a new condition
sub new_cond_type {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $name = $args{-name} or die "ERROR:  -name required, $!";
  my $description = $args{-description};
  my $datasize = $args{-datasize};
  my $units = $args{-units};
  my $datatype = $args{-datatype} or die "ERROR:  -datatype required, $!";
  my $hasError = $args{-hasError};
  my $ddl = $args{-ddl};

  $name = "COND_".$name unless $name =~ /^COND_/;
  check_names($name);

  $datasize = 1 unless defined $datasize;
  $hasError = 0 unless defined $hasError;

  if ($datasize < 1 || $datasize >= 100) {
    die "ERROR:  datasize is out of range [1, 99], $!\n";
  }

  my $typecode;
  my $mysqltype;
  if ($datatype =~ /float/) {
    $typecode = "f";
    $mysqltype = "float";
  } elsif ($datatype =~ /int/) {
    $typecode = "i";
    $mysqltype = "int";
  } elsif ($datatype =~ /string/) {
    $typecode = "s";
    $mysqltype = "varchar(255)";
  } else {
    die "ERROR:  unknown datatype, $!";
  }

  my $sql;
  # delete existing condition tables and information if it exists
  { no warnings;
    $sql = qq[ DROP TABLE IF EXISTS $name ];
  }
  $dbh->do($sql);

  { no warnings;
    $sql = qq[ DELETE FROM conditionDescription WHERE name="$name" ];
  }
  $dbh->do($sql);

  # create new condition tables and information
  { no warnings;
    $sql = qq[ CREATE TABLE $name ( logic_id INT NOT NULL,
                                    since DATETIME NOT NULL,
                                    till DATETIME NOT NULL, ];
    if (!$hasError) {
      for (0..$datasize-1) {
	$sql .= sprintf "value%02d $mysqltype,", $_;
      }
    } else {
      for (0..$datasize-1) {
	$sql .= sprintf "value%02d $mysqltype, error%02d $mysqltype,", $_, $_;
      }
    }
    $sql .= qq[ PRIMARY KEY (logic_id, since, till),
                INDEX IoV (since, till)
	      ) TYPE=INNODB ];
  }
  $dbh->do($sql);
  $ddl = $sql if $ddl;
  { no warnings;
    $sql = qq[ INSERT INTO conditionDescription
	         SET name="$name",
		     description="$description",
		     units="$units",
		     datatype="$typecode",
                     datasize="$datasize",
		     hasError="$hasError"];
    $sql =~ s/\"\"/NULL/g;
  }
  $dbh->do($sql);

  if ($ddl) { return $ddl; }
  else { return 1; }
}

# insert a channel
sub insert_channel {
    my $this = shift;
    my $dbh = $this->ensure_connect();

    my %args = @_;

    my $name = $args{-name};
    my ($id1, $id2, $id3) = @{ $args{-channel_ids} };
    my $maps_to = $args{-maps_to};
    my $logic_id = $args{-logic_id};

    # it is a direct view  by default
    $maps_to = $name unless $maps_to;

    # XXX check exists, types are ok, etc

    my $sql;
    $sql = qq[ INSERT INTO channelView
		 SET name=?,
		 id1=?,
		 id2=?,
		 id3=?,
                 maps_to=?,
		 logic_id=? ];

    my $sth = $dbh->prepare_cached($sql);
    $sth->execute($name, $id1, $id2, $id3, $maps_to, $logic_id);

    return 1;
}

# validates and inserts a condition to the database
sub insert_condition {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;

  foreach (qw/-name -logic_id -IoV/) {
    unless (defined $args{$_}) {
      die "ERROR:  $_ required, $!";
    }
  }

  my $name = $args{-name};
  my $logic_id = $args{-logic_id};
  my $IoV = $args{-IoV};
  my $value = $args{-value}; # single value
  my $error = $args{-error}; # single error
  my $values = $args{-values}; # arrayref of values
  my $errors = $args{-errors}; # arrayref of errors
  my $hasError = (defined $error || defined $errors);

  $name = "COND_".$name unless $name =~ /^COND_/;

  if ((defined $value && defined $values) ||
      (defined $error && defined $errors)) {
    die "ERROR:  defined input of both scalar and array\n";
  }

  # check that the IoV is valid
  unless (check_iov($IoV)) {
    die "ERROR:  IoV ($IoV->{since}, $IoV->{till}) fails validation\n";
  }

  # check that the IoV does not overlap something in the DB
  if ($this->{check_overlap}) {
    my $overlap = $this->is_overlap(-name=>$name,
				    -logic_id=>$logic_id,
				    -IoV=>$IoV);
    if ($overlap) {
      die "ERROR:  overlapping condition:\n", $overlap;
    }
  }

  # check to see if a statement has been prepared
  unless (exists $this->{prepared}->{$name}) {
    $this->prepare_cached(-name=>$name);
  }

  if ($this->{counter} == 0) { $this->begin_work(); }

  my @vars = ($logic_id, $IoV->{since}, $IoV->{till});
  if (defined $value) {
    push @vars, $value;
    push @vars, $error if $hasError;
  } elsif (defined $values && !defined $errors) {
    push @vars, @{$values};
  } elsif (defined $values && defined $errors) {
    my $num_vals = scalar @{$values};
    my $num_errs = scalar @{$errors};
    unless ($num_vals == $num_errs) {
      die "ERROR:  Number of values different than number of errors, $!";
    }
    for (0..$num_vals-1) {
      push @vars, shift @{$values}, shift @{$errors};
    }
  } else {
    die "ERROR:  undefined data input\n";
  }

  # XXX should check that the number of params matches the expected datasize

  my $sth = $this->{prepared}->{$name};
  $sth->execute(@vars);

  # counter management.  For performance evaluation
  $this->{counter}++;
  if ($this->{counter} >= $this->{max_cnt}) {
    $this->commit();
    $this->{counter} = 0;
  }
  return 1;
}

# insert a run
sub insert_run {
  my $this = shift;
  my $dbh = $this->{dbh};

  my %args = @_;
  my $run_number = $args{-run_number};
  my $IoV = $args{-IoV};

  unless (defined $run_number) {
    die "ERROR:  insert_run needs -run_number\n";
  }

  # check that the IoV is valid
  unless (check_iov($IoV)) {
    die "ERROR:  IoV ($IoV->{since}, $IoV->{till}) fails validation\n";
  }

  my $sql = qq[ INSERT INTO runs VALUES (?, ?, ?, NULL, NULL) ];
  my $insert = $dbh->prepare_cached($sql);
  
  $insert->execute($run_number, $IoV->{since}, $IoV->{till});

  return 1;
}

# update a run
sub update_run {
  my $this = shift;

  my %args = @_;

  my $run = $args{-run_number};
  my $status = $args{-status};
  my $comments = $args{-comments};

  unless (defined $run && defined $status) {
    die "Need to at least send a status to update_run.  skipping.\n";
  }
  
  unless (defined $comments) {
    $comments = "NULL";
  }

  my $sql = qq[ UPDATE runs SET status=$status, comments="$comments"
		WHERE run_number=$run ];

  $this->{dbh}->do($sql);
  return 1;
}

sub prepare_cached {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;

  unless (defined $args{'-name'}) {
    die "ERROR:  $_ required, $!";
  }

  my $name = $args{-name};

  $name = "COND_".$name unless $name =~ /^COND_/;
  my $desc = $this->get_conditionDescription(-name=>$name);
  unless (defined $desc) {
    die "ERROR:  Condition $name is not defined in the DB\n";
  }
  my $hasError = $desc->{hasError};
  my $datasize = $desc->{datasize};

  my $sql;
  { no warnings;
    $sql = qq[ INSERT INTO $name
	       SET logic_id=?,
	       since=?,
	       till=?, ];
    my @fields;
    for (0..$datasize-1) {
      push @fields, sprintf("value%02d=?", $_);
      if ($hasError) {
	push @fields, sprintf("error%02d=?", $_);
      }
    }
    $sql .= join ',', @fields;
  }
  my $sth = $dbh->prepare_cached($sql);
  $this->{prepared}->{$name} = $sth;

  return 1;
}

# returns overlap information if the given $IoV has any overlaps with IoVs 
# in the DB.  Else returns 0
sub is_overlap {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $name = $args{-name};
  my $logic_id = $args{-logic_id};
  my $IoV = $args{-IoV};

  my $t1 = $IoV->{since};
  my $t2 = $IoV->{till};

  my $sql;

  unless (exists $this->{prepared_overlap}->{$name}) {
    $this->prepare_overlap_check(-name=>$name);
  }

  my $sth = $this->{prepared_overlap}->{$name};
  $sth->execute($logic_id, $t1, $t2, $t1, $t2, $t1, $t2);
#  $sth->execute(($logic_id, $t1, $t2)x3);
  my ($db_id, $db_t1, $db_t2) = $sth->fetchrow_array();

  if ($db_id) {
    my $in_str = "input:  ". join ' ', $logic_id, $t1, $t2;
    my $db_str = "   db:  ". join ' ', $db_id, $db_t1, $db_t2;
    return "$in_str\n$db_str";
  } else {
    return 0;
  }
}

sub prepare_overlap_check {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;

  unless (defined $args{'-name'}) {
    die "ERROR:  $_ required, $!";
  }

  my $name = $args{-name};

  $name = "COND_".$name unless $name =~ /^COND_/;

  my $sql;
  { no warnings;
    # argument order is logic_id, t1, t2, t1, t2, t1, t2
    $sql = qq[ SELECT logic_id, since, till FROM $name
	       WHERE logic_id = ?
	         AND ((since >= ? AND since < ?)
	              OR (till  >  ? AND till  < ?)
	              OR (? >= since AND ? < till))
	       LIMIT 1
	     ];
  }


#    { no warnings;
#      $sql = qq[ SELECT logic_id, since, till FROM $name
#  	         WHERE logic_id = ?
#  	         AND (since >= ? AND since < ?)
#  	       UNION
#  	       SELECT logic_id, since, till FROM $name
#  	         WHERE logic_id = ?
#  	         AND (till  >  ? AND till  < ?)
#  	       UNION
#  	       SELECT logic_id, since, till FROM $name
#  	         WHERE logic_id = ?
#  	         AND (? >= since AND ? < till) ];
#    }

  my $sth = $dbh->prepare_cached($sql);
  $this->{prepared_overlap}->{$name} = $sth;

  return 1;
}


# get a condition, returns a scalar only if the condition table is only
# defined as a sigle value.  else it returns an array in the form
# (value00, value01, value03, ...) or
# (value00, error00, value01, error01, ...)
# depending on if there are errors defined
sub get_condition {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $name = $args{-name};
  my $logic_id = $args{-logic_id};
  my $time = $args{-time};

  $name = "COND_".$name unless $name =~ /^COND_/;

  my @fields;
  $name = "COND_".$name unless $name =~ /^COND_/;
  my $desc = $this->get_conditionDescription(-name=>$name);
  unless (defined $desc) {
    die "ERROR:  Condition $name is not defined in the DB\n";
  }
  my $hasError = $desc->{hasError};
  my $datasize = $desc->{datasize};

  for (0..$datasize-1) {
    push @fields, sprintf("value%02d", $_);
    if ($hasError) {
      push @fields, sprintf("error%02d", $_);
    }
  }
  my $fields = join ',', @fields;

  my $sql;
  { no warnings;
    $sql = qq[ SELECT $fields FROM $name
	       WHERE logic_id="$logic_id"
	         AND since <= "$time"
	         AND till  >  "$time" ];
  }

  my @results = @{$dbh->selectall_arrayref($sql)};
  if (scalar @results > 1) {
    warn "ERROR:  Overlapping IoV found in table $name, logic_id, $logic_id, ".
      "time $time";
  }

  if (scalar @{$results[0]} == 1) {
    return $results[0][0];
  } else {
    return @{$results[0]};
  }
}

# return an entire channel map of view_ids pointing to a logic_id
sub get_channelView {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $name = $args{-name};
  my $maps_to = $args{-maps_to};

  # channel is a canonical channel by default
  $maps_to = $name unless $maps_to;

  my $sql;
  { no warnings;
    $sql = qq[ SELECT id1, id2, id3, logic_id
	       FROM channelView WHERE name="$name" AND maps_to="$maps_to"];
  }

  # this recursive subroutine turns an array of values into a hash tree with
  # the last item as the leaf
  # e.g. ($ref, 1, 2, 3, undef, undef, undef, "VALUE")
  # makes $ref->{1}->{2}->{3} = "VALUE"
  # sub tack_on {
#     my ($ref, @values) = @_;
#     my $key = shift @values;
#     if (defined $key && defined $values[0]) {
#       $ref->{$key} = {} unless exists $ref->{$key};
#       tack_on($ref->{$key}, @values);
#     } else {
#       $ref->{$key} = pop @values;
#     }
#   }

#   my $view = {};
#   my @results = @{$dbh->selectall_arrayref($sql)};
#   foreach (@results) {
#     my @row = @{$_};
#     tack_on($view, @row);
#   }

  my $view = {};
  my @results = @{$dbh->selectall_arrayref($sql)};
  
  if (scalar @results == 0) {
    return undef;
  }

  foreach (@results) {
    my ($id1, $id2, $id3, $logic_id) = map {defined $_ ? $_ : ''} @{$_};
    $view->{$id1}->{$id2}->{$id3} = $logic_id;
  }

  return $view;
}

# returns an array of logic_ids for used in a channelView
sub get_channelView_logic_ids {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $name = $args{-name};
  my $maps_to = $args{-maps_to};

  # channel is a canonical channel by default
  $maps_to = $name unless $maps_to;

  my $sql;
  $sql = qq[ SELECT logic_id
	     FROM channelView WHERE name="$name" AND maps_to="$maps_to"];
  return @{$dbh->selectcol_arrayref($sql)};
}

# returns true if a condition type has an error defined
sub hasError {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $name = $args{-name};

  my $desc = $this->get_conditionDescription(-name=>$name);
  return $desc->{hasError};
}

# return a list of condition types defined in the DB
sub get_conditionDescription {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $name = $args{-name};

  my $sql;
  { no warnings;
    $sql = qq[ SELECT * FROM conditionDescription WHERE name="$name"];
  }

  return $dbh->selectrow_hashref($sql);
}

# return a list of channel view types defined in the DB
sub get_view_description {
}

# die if we are not connected
sub ensure_connect {
  my $this = shift;

  unless (exists $this->{dbh} && defined $this->{dbh}) {
    die "ERROR:  Not connected to database.\n";
  }

  # XXX really check the connection

  return $this->{dbh};
}

###
###   PRIVATE FUNCTIONS
###

sub check_names {
  no warnings;
  foreach (@_) {
    my $count = length $_;
    if ($count > $MAXNAME) {
      die "ERROR:  Name \"$_\" is too long.  Names for conditions and ids ".
	"can only be $MAXNAME characters long\n";
    }
  }
}

sub check_date {
  my $date = shift;

  return 0 unless defined $date;

  my @date = ($date =~ /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})$/);

  foreach (@date) {
    return 0 unless defined $_;
  }

  if ($date[0] < 0 || $date[0] > 9999 || # year
      $date[1] < 1 || $date[1] > 12 ||   # month
      $date[2] < 1 || $date[2] > 31 ||   # day
      $date[3] < 0 || $date[3] > 23 ||   # hour
      $date[4] < 0 || $date[4] > 59 ||   # minute
      $date[5] < 0 || $date[5] > 59) {   # second
    return 0;
  }

  return 1;
}

sub check_iov {
  my $IoV = shift;
  return (check_date($IoV->{since}) &&               # since valid
	  check_date($IoV->{till})  &&               # till valid
	  ($IoV->{since} lt $IoV->{till})            # since < till
	 );
}

1;
