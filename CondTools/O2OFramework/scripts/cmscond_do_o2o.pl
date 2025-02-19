#!/usr/bin/env perl

use warnings;
use strict;
use File::Basename;
use Getopt::Long;
use Data::Dumper;

my $cmssw_base = $ENV{'CMSSW_BASE'};
unless ($cmssw_base) {
    die "CMSSW_BASE is not set.  Be sure to eval `scramv1 runtime` first!\n";
}

# Manually including library
push @INC, "${cmssw_base}/src/CondTools/OracleDBA/perllib";
require CMSDBA;

# Directories used
my $o2o_dbconfigdir = $cmssw_base.'/src/CondTools/O2OFramework/dbconfig';
my $dba_dbconfigdir = $cmssw_base.'/src/CondTools/OracleDBA/dbconfig';
my $dba_xmldir = $cmssw_base.'/src/CondTools/OracleDBA/xml';

my $usage = basename($0)." [options] [[--all] | [object1 object2 ...]]\n".
    "Options:\n".
    "--dbaconfig       Offline DB configuration file (hardcoded default in project area)\n".
    "--o2oconfig       O2O configuration file (hardcoded default in project area)\n".
    "--auth            DB connection file (hardcoded default in project area)\n".
    "--general_connect Connect string to the offline DB general schema (default in dbaconfig)\n".
    "--offline_connect Connect string to the offline DB detector schema (default in dbaconfig)\n".
    "--all             Setup all objects in O2O configuration file\n".
    "--fake            Don't actually do anything, only print commands\n".
    "--debug           Print additional debug information\n".
    "--log             Log file\n".
    "--help, -h        Print this message and exit\n";


my $cmd_general_connect = '';
my $cmd_offline_connect = '';
my $o2o_configfile = $o2o_dbconfigdir.'/o2oconfiguration.xml';
my $dba_configfile = $dba_dbconfigdir.'/dbconfiguration.xml';
my $authfile = $dba_dbconfigdir.'/authentication.xml';
my $doall = 0;
my $fake = 0;
my $debug = 0;
my $log = '';
my $help = 0;

GetOptions('o2oconfig=s' => \$o2o_configfile,
	   'dbaconfig=s' => \$dba_configfile,
	   'auth=s' => \$authfile,
	   'all' => \$doall,
	   'general_connect=s' => \$cmd_general_connect,
	   'offline_connect=s' => \$cmd_offline_connect,
	   'fake' => \$fake,
	   'help|h' => \$help,
	   'debug' => \$debug,
	   'log=s' => \$log);

if ($help) {
    print "$usage";
    exit;
}


# Parse config files
foreach($o2o_configfile, $dba_configfile, $authfile) {
    unless (-e $_) { die "Configuration file $_ does not exist!\n"; }
    else { print "Using config file $_\n"; }
}

my $o2o_config = CMSDBA::parse_o2oconfiguration($o2o_configfile);
print "Result of parsing $o2o_configfile:\n".Dumper($o2o_config) if $debug;

my $dba_config = CMSDBA::parse_dbconfiguration($dba_configfile);
print "Result of parsing $dba_configfile:\n".Dumper($dba_config) if $debug;

my $auth = CMSDBA::parse_authentication($authfile);
print "Result of parsing $authfile:\n".Dumper($auth) if $debug;

# Determine General Connect
my $general_connect;
if (!$cmd_general_connect && exists $dba_config->{general}->{general_connect}) {
    $general_connect = $dba_config->{general}->{general_connect};
} elsif ($cmd_general_connect) {
    $general_connect = $cmd_general_connect;
} else {
    die "general_connect not defined at command line or at $dba_configfile";
}

# Get connection info
my ($general_user, $general_pass, $general_db, $general_schema) = CMSDBA::connection_test($auth, $general_connect);

my $catalog = "relationalcatalog_oracle://".$general_db.'/'.$general_schema; # XXX Add to config?


my @commands;
my $cmd;

# Determine what objects to set up
my $objects = {}; # Build hash $objects->{$detector}->[ @objects ]

die "Must provide object name(s) or --all" unless(@ARGV || $doall);
my @userobjects = @ARGV;

if ($doall) {
    foreach my $detector (keys %{$o2o_config->{detector}}) {
	$objects->{$detector} = [keys %{$o2o_config->{detector}->{$detector}->{object}}];
    }
} else {
    foreach my $object (@userobjects) {
	my $foundit = 0;
	foreach my $detector (keys %{$o2o_config->{detector}}) {
	    my @det_objects = keys %{$o2o_config->{detector}->{$detector}->{object}};

	    if (grep {$_ eq $object} @det_objects) {
		$foundit = 1;
		unless (exists $objects->{$detector}) {
		    $objects->{$detector} = [];
		}
		push @{$objects->{$detector}}, $object;
	    }
	}
	if (!$foundit) {
	    die "Object $object is not defined in $o2o_configfile";
	}
    }
}

print "Object array:  ", Dumper($objects), "\n" if $debug;

# Begin O2O of objects
foreach my $detector (keys %{$objects}) {

    my $offline_connect;
    if (!$cmd_offline_connect && exists $dba_config->{detector}->{$detector}->{offline_connect}) {
	$offline_connect = $dba_config->{detector}->{$detector}->{offline_connect};
    } elsif ($cmd_offline_connect) {
	$offline_connect = $cmd_offline_connect;
    } else {
	die "offline_connect not defined at command line or at $dba_configfile";
    }

    my ($offline_user, $offline_pass, $offline_db, $offline_schema) = CMSDBA::connection_test($auth, $offline_connect);

    foreach my $object (@{$objects->{$detector}}) {
	# Transfer the payload data
	my $sql = qq[call master_payload_o2o('$object')];
	$cmd = CMSDBA::get_sqlplus_cmd('user' => $general_user, 'pass' => $general_pass, 'db' => $general_db, 
				       'sql' => $sql);
	push(@commands, { 'info' => "Executing master_payload_o2o('$object')",
			  'cmd'  => $cmd });
    }

    # Register the new objects to POOL
    my $library = $dba_config->{detector}->{$detector}->{poolsetup}->{library};
    
    my $dbsetup = $dba_xmldir.'/'.
	$dba_config->{detector}->{$detector}->{poolsetup}->{dbsetup};


    CMSDBA::check_files($dbsetup);
    
    $cmd = "pool_setup_database -f $dbsetup -d $library -c $offline_connect -u $offline_user -p $offline_pass";
    push(@commands, { 'info' => "Registering new $detector objects to POOL",
		      'cmd'  => $cmd });

    foreach my $object (@{$objects->{$detector}}) {
	my $objref = $o2o_config->{detector}->{$detector}->{object}->{$object};

	# Build the IOV
	my $table = $objref->{table};
	my $tagsuffix = $objref->{tagsuffix};

	unless ($table && $tagsuffix) {
	    die "Attributes table and tagsuffix are required for object name='$object' in $o2o_configfile";
	}

	my $tag = $object.'_'.$tagsuffix;
	my $timetype = '';
	my $infiniteiov = '';
	my $append = '-a';
	my $query = '';

	if (exists $objref->{timetype} &&
	    $objref->{timetype} eq 'timestamp') {
	    $timetype = '-t';
	} 
	
	if (exists $objref->{infiniteiov} &&
	    $objref->{infiniteiov} eq 'true') {
	    $infiniteiov = '-i';
	    $append = '';
	}

	if (exists $objref->{query}) {
	    $query = '-q '.$objref->{query};
	}
	 
	$cmd = "cmscond_build_iov -c $offline_connect -f $catalog -u $offline_user -p $offline_pass -d $library -t $table -o $object $timetype $query $append $infiniteiov $tag";
	push(@commands, { 'info' => "Building IOV for $object using tag '$tag'",
			  'cmd'  => $cmd });
	
    }
}

# Execution of commands
CMSDBA::execute_commands('cmd_array' => \@commands, 'fake' => $fake, 'debug' => $debug, 'log' => $log);
