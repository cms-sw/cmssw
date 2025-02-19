#!/usr/bin/env perl

use warnings;
use strict;

use DBI;
use DBD::Oracle qw(:ora_types);

package CondDB::Oracle;

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

sub connect {
  my $this = shift;
  my %args = @_;
  my $host = $args{-host};
  my $user = $args{-user};
  my $pass = $args{-pass};
  my $port = $args{-port} || 1521;
  my $db_opts = $args{-db_opts};

  unless ($host) {
    die "ERROR:  Must give at least host to connect():, $!"
  }

  my $db = $args{-db};

  # here we use the TNS_ADMIN file and $db is the SID
  my $dsn;
  if ($db) {
    $dsn = "DBI:Oracle:$db";
  } else {
    die "Oracle needs to have database defined on connection!\n";
  }

  my $dbh = DBI->connect($dsn, $user, $pass, $db_opts)
    or die "Database connection failed, $DBI::errstr";

  $dbh->do(qq[ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD HH24:MI:SS']);

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
  my $ix_tablespace = $this->{ix_tablespace};

  my $sql;

  $sql =<<END_SQL;
CREATE TABLE conditionDescription (
  name VARCHAR2($MAXNAME) NOT NULL,
  description VARCHAR2(4000),
  units VARCHAR2(255) DEFAULT NULL,
  datatype CHAR(1) NOT NULL,
  datasize NUMBER DEFAULT '1' NOT NULL,
  hasError NUMBER DEFAULT '0' NOT NULL
)
END_SQL
  $dbh->do($sql);

  $sql = qq[ALTER TABLE conditionDescription ADD CONSTRAINT c_desc_pk
            PRIMARY KEY (name)];
  $sql .= " USING INDEX TABLESPACE $ix_tablespace" if $ix_tablespace;
  $dbh->do($sql);

$sql =<<END_SQL;
CREATE TABLE conditionColumns (
  name VARCHAR2($MAXNAME) NOT NULL,
  colindex NUMBER NOT NULL,
  colname VARCHAR2(64)
)
END_SQL
  $dbh->do($sql);

  $sql = qq[ALTER TABLE conditionColumns ADD CONSTRAINT c_cols_pk
            PRIMARY KEY (name, colindex)];
  $sql .= " USING INDEX TABLESPACE $ix_tablespace" if $ix_tablespace;
  $dbh->do($sql);

  $sql =<<END_SQL;
CREATE TABLE viewDescription (
  name VARCHAR2($MAXNAME) NOT NULL,
  description VARCHAR2(4000),
  id1name VARCHAR2($MAXNAME) DEFAULT NULL,
  id2name VARCHAR2($MAXNAME) DEFAULT NULL,
  id3name VARCHAR2($MAXNAME) DEFAULT NULL
)
END_SQL

  $dbh->do($sql);

  $sql = qq[ALTER TABLE viewDescription ADD CONSTRAINT cvd_pk
            PRIMARY KEY (name)];
  $sql .= " USING INDEX TABLESPACE $ix_tablespace" if $ix_tablespace;  
  $dbh->do($sql);

  $sql =<<END_SQL;
CREATE TABLE channelView (
  name VARCHAR2($MAXNAME) NOT NULL,
  id1 NUMBER DEFAULT NULL,
  id2 NUMBER DEFAULT NULL,
  id3 NUMBER DEFAULT NULL,
  maps_to VARCHAR2($MAXNAME) NOT NULL,
  logic_id NUMBER NOT NULL
)
END_SQL
  $dbh->do($sql);

  $sql = qq[ALTER TABLE channelView ADD CONSTRAINT cv_ix1
            UNIQUE (name, id1, id2, id3, logic_id)];
  $sql .= " USING INDEX TABLESPACE $ix_tablespace" if $ix_tablespace;
  $dbh->do($sql);

  $sql = qq[CREATE INDEX cv_ix2 ON channelView (maps_to)];
  $sql .= " TABLESPACE $ix_tablespace" if $ix_tablespace;
  $dbh->do($sql);

  $sql = qq[CREATE INDEX cv_ix3 ON channelView (logic_id)];
  $sql .= " TABLESPACE $ix_tablespace" if $ix_tablespace;
  $dbh->do($sql);

$sql = <<END_SQL;
CREATE TABLE runs (
  run_number NUMBER,
  since DATE,
  till DATE,
  status NUMBER(1) DEFAULT 1,
  comments VARCHAR2(4000)
)
END_SQL

  $dbh->do($sql);

  $sql = qq[ALTER TABLE runs ADD CONSTRAINT runs_pk
            PRIMARY KEY (run_number)];
  $sql .= " USING INDEX TABLESPACE $ix_tablespace" if $ix_tablespace;  
  $dbh->do($sql);


  $sql = qq[CREATE INDEX run_ix ON runs (since, till)];
  $sql .= " TABLESPACE $ix_tablespace" if $ix_tablespace;
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

  my @tables = qw(conditionDescription conditionColumns viewDescription channelView runs);
  @tables = map uc, @tables;
  my $table_list = join ',', map "\'$_\'", @tables;
  my $sql = qq[ SELECT table_name FROM user_tables WHERE table_name
                IN ($table_list)
                OR table_name LIKE 'COND_%'
                OR table_name LIKE 'CNDC_%' ];
  @tables = @{$dbh->selectcol_arrayref($sql)};

  foreach (@tables) {
    $dbh->do( qq[ DROP TABLE $_ ]);
  }
  return 1;
}

# define a stored procedure from the code in an SQL file
sub define_procedure {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $file = $args{-file} or die "Must give file to define_procedure, $!";
  
  open FILE, '<', $file or die $!;
  my $code = join("", grep( $_ !~ /^\/$/, <FILE>));

  $dbh->do($code);
  
  close FILE;

  return 1;
}

# TODO:  is it possible to switch tablespaces?
# choose the database to use
sub selectdb {
#   my $this = shift;
#   my $dbh = $this->ensure_connect();

#   my %args = @_;
#   my $name = $args{-name} or die "Must give name to newdb(), $!";

#   $dbh->do(qq[ USE $name ]);
  die "selectdb not supported by Oracle.\n";
}


# delete existing view if it exists and create view in DB
sub new_channelView_type {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $name = $args{-name};
  my $description = $args{-description};
  my ($id1name, $id2name, $id3name) = @{$args{-idnames}};
  my $maps_to = $args{-maps_to};

  $maps_to = $name unless $maps_to;

  check_names($name, $id1name, $id2name, $id3name, $maps_to);

  my $sql;
  { no warnings;
    $sql = qq[ DELETE FROM channelView 
                 WHERE name='$name' AND maps_to='$maps_to' ];
    $sql =~ s/\"\"/NULL/g;
  }
  $dbh->do($sql);

  if ($name eq $maps_to) {
    { no warnings;
      $sql = qq[ DELETE FROM viewDescription WHERE name='$name' ];
      $sql =~ s/\"\"/NULL/g;
    }
    $dbh->do($sql);
  }

  { no warnings;
    my $fieldlist = join ',', qw(name description id1name id2name id3name);
    my $valuelist = join ',', map "\'$_\'", ($name, $description, $id1name,
					     $id2name, $id3name);
    $sql = qq[ INSERT INTO viewDescription ($fieldlist)
                 VALUES ($valuelist) ];
    $sql =~ s/\'\'/NULL/g;
  }
  $dbh->do($sql);

  return 1;
}

# standard data table
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
  my $datalist = $args{-datalist};
  my $ddl = $args{-ddl};

  my $ix_tablespace = $this->{ix_tablespace};

  $name = "COND_".$name unless $name =~ /^COND_/;
  check_names($name);

  $datasize = 1 unless defined $datasize;
  $hasError = 0 unless defined $hasError;

  if ($datasize < 1 || $datasize >= 100) {
    die "ERROR:  datasize is out of range [1, 99], $!\n";
  }

  my $oracletype;
  if ($datatype =~ /float/) {
    $datatype = "f";
    $oracletype = "NUMBER";
  } elsif ($datatype =~ /int/) {
    $datatype = "i";
    $oracletype = "NUMBER";
  } elsif ($datatype =~ /string/) {
    $datatype = "s";
    $oracletype = "VARCHAR2(255)";
  } else {
    die "ERROR:  unknown datatype, $!";
  }

  my $sql;
  
  # drop the condition type
  $this->drop_condition_type(-name=>$name);

  # create new condition tables and information
  { no warnings;
#     $sql = qq[ CREATE TABLE $name ( logic_id NUMBER NOT NULL,
#                                     since DATE NOT NULL,
#                                     till DATE DEFAULT to_date('9999-12-31 23:59:59', 'YYYY-MM-DD HH24:MI:SS') NOT NULL,];
    $sql = qq[ CREATE TABLE $name ( logic_id NUMBER NOT NULL,
                                    since NUMBER NOT NULL,
                                    till NUMBER DEFAULT 9999999999999999 NOT NULL,];
    if (!$hasError) {
      for (0..$datasize-1) {
	$sql .= sprintf "value%02d $oracletype,", $_;
      }
    } else {
      for (0..$datasize-1) {
	$sql .= sprintf "value%02d $oracletype, error%02d $oracletype,", $_, $_;
      }
    }
    chop $sql;
    $sql .= ")";
  }
  $dbh->do($sql);
  $ddl = $sql if $ddl;

  # make indexes
  my $keyname = $name;
  $keyname =~ s/COND_/C_/i;

  $sql = qq[ALTER TABLE ${name} ADD CONSTRAINT ${keyname}_pk
            PRIMARY KEY (logic_id, since, till)];
  $sql .= " USING INDEX TABLESPACE $ix_tablespace" if $ix_tablespace;
  $dbh->do($sql);

  $sql = qq[CREATE INDEX ${keyname}_ix ON ${name} (since, till)];
  $sql .= " TABLESPACE $ix_tablespace" if $ix_tablespace;
  $dbh->do($sql);

  # write metadata
  $this->insert_conditionDescription($name, $description, $units, $datatype,
				     $datasize, $hasError, $datalist);

  if ($ddl) { return $ddl; }
  else { return 1; }
}


# CLOB type data tables
sub new_cndc_type {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $name = $args{-name} or die "ERROR:  -name required, $!";
  my $description = $args{-description};
  my $datasize = $args{-datasize};
  my $units = $args{-units};
  my $datatype = $args{-datatype} or die "ERROR:  -datatype required, $!";
  my $hasError = $args{-hasError};
  my $datalist = $args{-datalist};
  my $ddl = $args{-ddl};

  my $ix_tablespace = $this->{ix_tablespace};

  $name = "CNDC_".$name unless $name =~ /^CNDC_/;
  check_names($name);

  $datasize = 1 unless defined $datasize;
  defined $hasError ? $hasError = 1 : $hasError = 0;

  if ($datatype =~ /float/) {
    $datatype = "f";
  } elsif ($datatype =~ /int/) {
    $datatype = "i";
  } elsif ($datatype =~ /string/) {
    $datatype = "s";
  } else {
    die "ERROR:  unknown datatype, $!";
  }



  my $sql;
  # delete existing condition tables and information if it exists
  $this->drop_condition_type(-name=>$name);

  # create new condition tables and information
  # TODO:  require since and till NOT NULL
  { no warnings;
    $sql = qq[ CREATE TABLE $name ( since DATE NOT NULL,
                                    till DATE NOT NULL,
                                    data CLOB ) ];
  }
  $dbh->do($sql);
  $ddl = $sql if $ddl;

  # make indexes
  my $keyname = $name;
  $keyname =~ s/COND_/C_/i;

  $sql = qq[ALTER TABLE ${name} ADD CONSTRAINT ${keyname}_pk
            PRIMARY KEY (since, till)];
  $sql .= " USING INDEX TABLESPACE $ix_tablespace" if $ix_tablespace;
  $dbh->do($sql);

  # make IoV update trigger
  my $trigname = $keyname.'_tg';
  $sql = qq[CREATE OR REPLACE TRIGGER $trigname
	    BEFORE INSERT ON $name
	    REFERENCING NEW AS newiov
	    FOR EACH ROW
	    CALL update_online_cndc_iov('$name', :newiov.since, :newiov.till)];
  $dbh->do($sql);

  # make metadata
  $this->insert_conditionDescription($name, $description, $units, $datatype,
				     $datasize, $hasError, $datalist);

  if ($ddl) { return $ddl; }
  else { return 1; }
}

# delete existing condition tables and information if it exists
sub drop_condition_type {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $name = $args{-name} or die "drop_condition_type requires -name";

  my $sql;
  { no warnings;
    $sql = qq[ DROP TABLE $name ];
    $dbh->do($sql) if $this->table_exists($name);
  }

  { no warnings;
    $sql = qq[ DELETE FROM conditionDescription WHERE name='$name' ];
  }
  $dbh->do($sql);

  { no warnings;
    $sql = qq[ DELETE FROM conditionColumns WHERE name='$name' ];
  }
  $dbh->do($sql);

  return 1;
}


# metadata for data tables
sub insert_conditionDescription {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my ($name, $description, $units, $datatype, $datasize, $hasError, $datalist) = @_;

  my $listsize = $hasError ? $datasize * 2 : $datasize;
  if ($datalist) {
    if ($listsize != scalar @{$datalist}) {
      die "datalist size does not match number of data elements.";
    }
  } else {
    if ($listsize == 1) {
      $datalist = [$name];
    } elsif ($listsize == 2 && $hasError) {
      $datalist = [$name, $name.'_err'];
    } elsif ($listsize > 2 && $hasError) {
      $datalist = [];
      for (0..($listsize/2 - 1) ) {
	my $num = sprintf '%03d', $_;
	push @{$datalist}, $name.$num, $name.'_err'.$num;
      }
    } else {
      for (0..($listsize - 1) ) {
	my $num = sprintf '%03d', $_;
	push @{$datalist}, $name.$num;
      }
    }
  }
  
  my $sql;
  { no warnings;
    my $fieldlist = join ',', qw(name description units
				 datatype datasize hasError);
    my $valuelist = join ',', map "\'$_\'", ($name, $description, $units,
					     $datatype, $datasize, $hasError);
    $sql = qq[ INSERT INTO conditionDescription ($fieldlist)
                 VALUES ($valuelist) ];
    $sql =~ s/\'\'/NULL/g;
  }
  $dbh->do($sql);

  my $colindex = 0;
  $sql = qq[ INSERT into conditionColumns VALUES (?, ?, ?) ];
  my $insert = $dbh->prepare($sql);
  foreach (@{$datalist}) {
    $insert->execute($name, $colindex, $_);
    $colindex++;
  }

  return 1;
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
    my $fieldlist = join ',', qw(name id1 id2 id3 maps_to logic_id);
    my $valuelist = join ',', ('?')x6;
    $sql = qq[ INSERT INTO channelView ($fieldlist)
		 VALUES ($valuelist) ];

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

# insert a bunch of data into a CLOB
sub insert_condition_clob {
  my $this = shift;

  my $dbh = $this->ensure_connect();

  my %args = @_;

  foreach (qw/-name -data -logic_ids -IoV/) {
    unless (defined $args{$_}) {
      die "ERROR:  $_ required, $!";
    }
  }

  my $name = $args{-name};
  my $logic_ids = $args{-logic_ids};
  my $data = $args{-data};
  my $IoV = $args{-IoV};

  $name = "CNDC_".$name unless $name =~ /^CNDC_/;
  
  unless (scalar @{$logic_ids} == scalar @{$data}) {
    die "Missmatched arrays of logic_ids and data\n";
  }
  
  my $clob = "";
  for my $i (0..$#{$logic_ids}) {
    my $datalist = join ',', @{$data->[$i]};
    $clob .= $logic_ids->[$i] . '=' . $datalist . ';';
  }
  chop $clob;

  my $insert = $dbh->prepare(qq[INSERT INTO ${name} VALUES (?, ?, ?)]);

  { no warnings;
    $insert->bind_param(1, $IoV->{since});
    $insert->bind_param(2, $IoV->{till});
    $insert->bind_param(3, $clob);
    $insert->execute();
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

  my $sql = qq[ UPDATE runs SET status=$status, comments='$comments'
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
  my $hasError = $desc->{HASERROR};
  my $datasize = $desc->{DATASIZE};

  my $sql;
  { no warnings;
    my @fields = qw(logic_id since till);

    for (0..$datasize-1) {
      push @fields, sprintf("value%02d", $_);
      if ($hasError) {
	push @fields, sprintf("error%02d", $_);
      }
    }
    my $fieldlist = join ',', @fields;
    my $valuelist = join ',', ('?') x scalar @fields;

    $sql = qq[ INSERT INTO $name ($fieldlist) VALUES ($valuelist) ];
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
  my $desc = $this->get_conditionDescription(-name=>$name);
  unless (defined $desc) {
    die "ERROR:  Condition $name is not defined in the DB\n";
  }
  my $hasError = $desc->{HASERROR};
  my $datasize = $desc->{DATASIZE};

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
	       WHERE logic_id='$logic_id'
	         AND since <= '$time'
	         AND till  >  '$time' ];
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
	       FROM channelView WHERE name='$name' AND maps_to='$maps_to'];
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
	     FROM channelView WHERE name='$name' AND maps_to='$maps_to'
             ORDER BY id1, id2, id3
           ];
  return @{$dbh->selectcol_arrayref($sql)};
}

# returns true if a condition type has an error defined
sub hasError {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $name = $args{-name};

  my $desc = $this->get_conditionDescription(-name=>$name);
  return $desc->{HASERROR};
}

# return a list of condition types defined in the DB
sub get_conditionDescription {
  my $this = shift;
  my $dbh = $this->ensure_connect();

  my %args = @_;
  my $name = $args{-name};

  my $sql;
  { no warnings;
    $sql = qq[ SELECT * FROM conditionDescription WHERE name='$name'];
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

sub table_exists {
  my $this = shift;
  my $dbh = $this->{dbh};
  my $table = shift;
  $table = uc $table;
  my $sql = qq[ SELECT 1 FROM user_tables WHERE table_name='$table'];
  return $dbh->selectrow_array($sql);
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
