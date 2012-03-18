#
# subroutines
#

use strict 'vars';
use vars qw( %opts );
use vars qw( %config );

sub ec_mon_find_conf {
  if ( $opts{'conf'} ) {
    if ( -d "$opts{'conf'}" ) {
      print "Using config file: $opts{'conf'}\n" if ( $opts{'debug'} );

      if ( -e "$opts{'conf'}/ec_mon.conf" ) {
        return "$opts{'conf'}/ec_mon.conf";
      } else {
        die "Sorry you have specified a directory for the config rc file, but\n";
        "no ec_mon.conf was found in $opts{'conf'}.\n";
      }
    } elsif ( -e "$opts{'conf'}" ) {
      print "Using config: $opts{'conf'}\n" if ( $opts{'debug'} );
      return "$opts{'conf'}";
    } elsif ( -e "$config{'ec_mon_path'}/$opts{'conf'}" ) {
      print "Using config: $config{'ec_mon_path'}/$opts{'conf'}\n" if ( $opts{'debug'} );
      return "$config{'ec_mon_path'}/$opts{'conf'}";
    } else {
      die "Sorry, $opts{'conf'} does not seem to exist.\n".
      "Please check your use of the --conf option.\n";
    }
  } else {
    if ( -e "ec_mon.conf" ) {
      print "Using config: ec_mon.conf\n" if ( $opts{'debug'} );
      return "ec_mon.conf";
    } elsif ( -e "$config{'ec_mon_path'}/ec_mon.conf" ) {
      print "Using config: $config{'ec_mon_path'}/ec_mon.conf\n" if ( $opts{'debug'} );
      return "$config{'ec_mon_path'}/ec_mon.conf";
    }
  }
}

sub ec_mon_conf_file {
  $config{'ec_mon_path'} = dirname($0);
  if ( $config{'ec_mon_path'} eq "." ) {
    $config{'ec_mon_path'} = $ENV{'PWD'};
  }

  $config{'$ec_mon_conf'} = ec_mon_find_conf();

  die "Sorry, but no ec_mon.conf was found.\n" if ( ! defined($config{'$ec_mon_conf'}) );

  $config{'ec_mon_master_runcontrol'} = 0;
  $config{'ec_mon_master_runrange'} = '';
  $config{'ec_mon_master_name'} = '';
  $config{'ec_mon_master_rundir'} = '';
  $config{'ec_mon_master_max_jobs_running'} = 1;
  $config{'ec_mon_master_submit_wait'} = 120;
  $config{'ec_mon_master_archive_wait'} = 600;
  $config{'ec_mon_archive_dir'} = '';

  open FILE, "<$config{'$ec_mon_conf'}";
  print "Found control file, opening...\n" if ( $opts{'debug'} );
  while ( my $line = <FILE> ) {
    #print $line if ( $opts{'debug'} ); 
    if ( $line =~ /^ec_mon_master_runcontrol=(.*)$/ ) {
      $config{'ec_mon_master_runcontrol'} = 0 if ( $1 eq 'stop' );
      $config{'ec_mon_master_runcontrol'} = 1 if ( $1 eq 'suspend' );
      $config{'ec_mon_master_runcontrol'} = 2 if ( $1 eq 'run' );
    }
    if ( $line =~ /^ec_mon_master_runrange=(.*)$/ ) {
      $config{'ec_mon_master_runrange'} = $1;
      print "Found ec_mon_master_runrange = $config{'ec_mon_master_runrange'}\n" if ( $opts{'debug'} );
    }
    if ( $line =~ /^ec_mon_master_rundir=(.*)$/ ) {
      $config{'ec_mon_master_rundir'} = $1;
      print "Found ec_mon_master_rundir = $config{'ec_mon_master_rundir'}\n" if ( $opts{'debug'} );
    }
    if ( $line =~ /^ec_mon_master_name=(.*)$/ ) {
      $config{'ec_mon_master_name'} = $1;
      print "Found ec_mon_master_name = $config{'ec_mon_master_name'}\n" if ( $opts{'debug'} );
    }
    if ( $line =~ /^ec_mon_master_max_jobs_running=(.*)$/ ) {
      $config{'ec_mon_master_max_jobs_running'} = $1;
      print "Found ec_mon_master_max_jobs_running =  $config{'ec_mon_master_max_jobs_running'}\n" if ( $opts{'debug'} );
    }
    if ( $line =~ /^ec_mon_master_submit_wait=(.*)$/ ) {
      $config{'ec_mon_master_submit_wait'} = $1;
      print "Found ec_mon_master_submit_wait = $config{'ec_mon_master_submit_wait'}\n" if ( $opts{'debug'} );
    }
    if ( $line =~ /^ec_mon_master_archive_wait=(.*)$/ ) {
      $config{'ec_mon_master_archive_wait'} = $1;
      print "Found ec_mon_master_archive_wait = $config{'ec_mon_master_archive_wait'}\n" if ( $opts{'debug'} );
    }
    if ( $line =~ /^ec_mon_archive_dir=(.*)$/ ) {
      $config{'ec_mon_archive_dir'} = $1;
      print "Found ec_mon_archive_dir = $config{'ec_mon_archive_dir'}\n" if ( $opts{'debug'} );
    }

  }
  close FILE;

  $config{'ec_mon_master_log'} = 'ec_mon_master'.$config{'ec_mon_master_name'}.'.log';

}

sub ec_mon_filelist {
    my $searchdir = $_[0];
    my $num = $_[1];
    my $fun = $_[2];

    my @namelist = (
		    "ecal_local.$num.*.A.$fun.*.*.dat",
		    "Global*.$num.*.A.$fun.*.*.dat",
		    "MW*.$num.*.A.$fun.*.*.dat",
		    "PrivCal*.$num.*.A.$fun.*.*.dat",
		    "TransferTestWithSafety.$num.*.A.$fun.*.*.dat",
		    "CRUZET*.$num.*.A.$fun.*.*.dat",
		    "CRAFT*.$num.*.A.$fun.*.*.dat",
		    "Commissioning*.$num.*.A.$fun.*.*.dat",
		    "Run*.$num.*.A.$fun.*.*.dat",
		    "Data.$num.*.A.$fun.*.*.dat",
#		    "Minidaq.$num.*.A.$fun.*.*.dat",
		    "Minidaq.$num.*.Calibration.$fun.*.*.dat",
		    "PrivMinidaq.$num.*.A.$fun.*.*.dat"
		    );

    my $result = "find $searchdir -maxdepth 1 \\( -name '$namelist[0]'";
    for (my $i = 1; $i < @namelist; $i++) {
	$result = "$result -or -name '$namelist[$i]'";
    }
    $result = "$result \\) -printf '%f\n' 2>&1 | sort";

    return $result;
}

############################################################################

1;
