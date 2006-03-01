#/usr/bin/perl

use warnings;
use strict;
$|++;

package TB04::Importer;
use ConnectionFile;
use Data::Dumper;


# the maximum time for MySQL 4.1
our $MAX_DATETIME = "9999-12-31 23:59:59";

my $cnt_fmt = "%10d ";
my $cnt_del = "\b"x11;

sub new {
  my $proto = shift;
  my $class = ref($proto) || $proto;
  my $this = {};

  # initialize DB connection and other important stuff here
  $this->{buffer} = {};
  $this->{condDB} = ConnectionFile::connect();

  bless($this, $class);
  return $this;
}

# flush the stream if we're done
sub DESTROY {
  my $this = shift;
  $this->flush_stream();
  if (defined $this->{count}) {
    print "\n";
  }
}

sub start_counter {
  my $this = shift;
  $this->{count} = 0;
  print "\nInsertions:  ";
  printf $cnt_fmt, $this->{count};
}

sub count {
  my $this = shift;
  $this->{count}++;
  print $cnt_del;
  printf $cnt_fmt, $this->{count};
}

sub load_view {
  my $this = shift;
  my $view_name = shift;
  my $view = $this->{condDB}->get_channelView(-name=>"$view_name");
  if ($view) {
    $this->{views}->{$view_name} = $view;
  } else {
    die "ERROR:  Failed to load view $view_name\n";
  }
}

sub dump_view {
  my $this = shift;
  my $view_name = shift;
  print "DUMP VIEW $view_name:  ", Dumper($this->{views}->{$view_name}), "\n";
}

# stream in data from some source (parser).  Data is sent in with time stamps
# and IoV must be constructed from within this class before it is inserted
# into the database.  Data should be sent in chronological order.
sub stream {
  my $this = shift;
  my ( $cond_name,   # the name of the condition (eg. "HV_vMon")
       $view_name,   # the name of the channel view to use (eg. "HV_channl")
       $view_ids,    # an array ref to the channel keys
       $datetime,    # time in YYYY-MM-DD HH:MM:SS format
       $value,       # a value
       $error        # the (optional) error to the value
     )
    = @_;

  my $buffer = $this->{buffer};
  my $view_key = join(',', $view_name, @{$view_ids});

  # insert the condition if there is a previos value in the boffer
  if (exists $buffer->{$cond_name}->{$view_key}) {
    my ($since, $l_value, $l_error)
      = @{$buffer->{$cond_name}->{$view_key}};
    my $IoV = {since=>$since, till=>$datetime};
    if ($since ne $datetime) {
      $this->insert($cond_name, $view_name, $view_ids, $IoV, $l_value, $l_error);
    } else {
      warn "WARN:  Duplicate time in stream:\n\tFILE $ARGV\n".
	"\tLINE $.\n\tVAR $cond_name $view_key\n".
	  "\tTIME $since\n\tskipping...\n";
    }
  }

  # buffer the condition
  $buffer->{$cond_name}->{$view_key} = [$datetime, $value, $error];
  
  return 1;
}

# this flushes out the buffer, a zero length IoV is written...
# XXX is this what we want?
sub flush_stream {
  my $this = shift;
  my $buffer = $this->{buffer};
  foreach my $cond_name (keys %{$buffer}) {
    foreach my $view_key (keys %{$buffer->{$cond_name}}) {
      my @view_info = split /,/, $view_key;
      my $view_name = shift @view_info;
      my $view_ids = [@view_info];
      my ($since, $value, $error)
	= @{$buffer->{$cond_name}->{$view_key}};
      my $IoV = {since=>$since, till=>$MAX_DATETIME};
      $this->insert($cond_name, $view_name, $view_ids,
		    $IoV, $value, $error);
    }
  }

  $this->{buffer} = {};  # clear buffer

  return 1;
}

# inserts into the database, with the IoV already constructed
sub insert {
  my $this = shift;
  my ( $cond_name,   # the name of the condition (eg. "HV_vMon")
       $view_name,   # the name of the channel view to use (eg. "HV_channl")
       $view_ids,    # an array ref to the channel keys
       $IoV,         # hashref to IoV, { since=>$s, till=>$t }
       $value,       # a value
       $error        # the (optional) error to the value
     )
    = @_;

  # print the data we recieved
  # $this->dummy($cond_name, $view_name, $view_ids, $IoV, $value, $error);

  # make sure the view is loaded
  my $view = $this->{views}->{$view_name};
  unless ($view) {
    die "ERROR:  view \"$view_name\" was not loaded, $!";
  }

  # this is just to avoid a warning about undef hash keys...
  my ($id1, $id2, $id3) = map {defined $_ ? $_ : ''} (@{$view_ids}, (undef)x3);

  # get the logic_id based on the view_ids
  my $logic_id = $view->{$id1}->{$id2}->{$id3};

  eval {
    # insert to the DB
    unless (ref $value) {
      $this->{condDB}->insert_condition(-name=>$cond_name,
					-logic_id=>$logic_id,
					-IoV=>$IoV,
					-value=>$value,
					-error=>$error);
    } else {
      $this->{condDB}->insert_condition(-name=>$cond_name,
					-logic_id=>$logic_id,
					-IoV=>$IoV,
					-values=>$value,
					-errors=>$error);
    }
  };
  if ($@) {
    warn ("ERROR:  insertion failed.  $@\n".
	  "Tried to insert:\n".
	  "\t-name=>$cond_name,\n".
	  "\t-logic_id=>$logic_id,\n".
	  "\t-IoV=>$IoV->{since}, $IoV->{till},\n".
	  "\t-value=>$value,\n".
	  "\t-error=>$error");
  }

  if (defined $this->{count}) {
    $this->count();
  }

  return 1;
}

sub dummy {
  my $this = shift;
  my ($cond_name, $view_name, $view_ids, $IoV, $value, $error) = @_;

  # this is just a dummy until the schema is ready
  $value = $value."+-".$error if defined $error;
  print join('|',
	     $cond_name,
	     $view_name,
	     join(',', @{$view_ids}),
	     join(',', "since=$IoV->{since}", "till=$IoV->{till}"),
	     $value), "\n";

}

1;
