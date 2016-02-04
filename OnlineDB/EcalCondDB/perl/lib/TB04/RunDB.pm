#/usr/bin/perl

use warnings;
use strict;
$|++;

use DBI;

package TB04::RunDB;

use POSIX;
my $dummy = time;

sub new {
  my $proto = shift;
  my $class = ref($proto) || $proto;
  my $this = {};

  bless($this, $class);
  return $this;
}

# connect to the database
sub connect {
  my $this = shift;
  # initialize DB connection and other important stuff here
  my $dbh = DBI->connect("DBI:mysql:host=suncms100.cern.ch;db=rclog",
			 "WWW", "myWeb01", {RaiseError => 1})
    or die DBI->errstr();

  $this->{dbh} = $dbh;
}

# load into memory from the database connection
sub load_from_db {
  my $this = shift;
  my $dbh = $this->{dbh} or die "Not connected to DB, $!\n";

  my $sql = qq[ SELECT run_number, start_time, stop_time FROM runs ];
  my $sth = $dbh->prepare($sql);
  $sth->execute();
  while (my ($run, $start, $stop) = $sth->fetchrow_array()) {
    $this->{runs}->{$run}->{since} = $start;
    $this->{runs}->{$run}->{till} = $stop;
  }
}

# load into memory from a simple file
sub load_from_file {
  my $this = shift;
  my $file = shift;

  open FILE, "<", $file or die $!;
  while (<FILE>) {
    chomp;
    my ($run, $start, $stop) = split /,/;
    $this->{runs}->{$run}->{since} = $start;
    $this->{runs}->{$run}->{till} = $stop;
  }
  close FILE;
}

# make make a run table in a given database
sub fill_runs_table {
  my $this = shift;
  my $dest_db = shift;
 
  foreach my $run (sort keys %{$this->{runs}}) {
    my $since = $this->{runs}->{$run}->{since};
    my $till = $this->{runs}->{$run}->{till};
    my $IoV = { since => $since, till => $till };
    $dest_db->insert_run(-run_number => $run,
			 -IoV => $IoV );
  }
}

# given a run, get the iov from what is loaded into memory
sub get_iov {
  my $this = shift;
  my $run = shift;

  return $this->{runs}->{$run};
}

sub get_dummy_iov {
  my $since = POSIX::strftime("%Y-%m-%d %H:%M:%S", localtime($dummy));
  $dummy++;
  my $till  = POSIX::strftime("%Y-%m-%d %H:%M:%S", localtime($dummy));
  return {since=>$since, till=>$till};
}

# prints all run durations in the format run,start_time,stop_time
sub dump_iov {
  my $this = shift;
  my $run = shift;

  my $dbh = $this->{dbh} or die "Not connected to DB, $!\n";;
  my $sql = qq[ SELECT run_number, start_time, stop_time FROM runs ];
  my $sth = $dbh->prepare($sql);
  $sth->execute();
  while (my @row = $sth->fetchrow_array()) {
    print join(",", @row), "\n";
  }
}

1;
