#!/usr/bin/perl

use warnings;
use strict;
$|++;

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
my $o2o_sqldir = $cmssw_base.'/src/CondTools/O2OFramework/sql';
my $dba_dbconfigdir = $cmssw_base.'/src/CondTools/OracleDBA/dbconfig';


my $usage = basename($0)." [options] [detector1 detector2 ...]\n".
    "Options:\n".
    "--general    Setup the general offline schema for O2O\n".
    "--dbaconfig  Offline DB configuration file (hardcoded default in project area)\n".
    "--o2oconfig  O2O configuration file (hardcoded default in project area)\n".
    "--auth       DB connection file (hardcoded default in project area)\n".
    "--online_db  Name of online DB (default in o2oconfig)\n".
    "--offline_db Name of offline DB (default in dbaconfig)\n".
    "--all        Setup all detectors in O2O configuration file\n".
    "--fake       Don't actually do anything, only print commands\n".
    "--debug      Print additional debug information\n".
    "--log        Log file\n".
    "--help, -h   Print this message and exit\n";


my $online_db = '';
my $offline_db = '';
my $o2o_configfile = $o2o_dbconfigdir.'/o2oconfiguration.xml';
my $dba_configfile = $dba_dbconfigdir.'/dbconfiguration.xml';
my $authfile = $dba_dbconfigdir.'/authentication.xml';
my $doall = 0;
my $dogeneral = 0;
my $fake = 0;
my $debug = 0;
my $log = '';
my $help = 0;

GetOptions('general' => \$dogeneral,
	   'o2oconfig=s' => \$o2o_configfile,
	   'dbaconfig=s' => \$dba_configfile,
	   'auth=s' => \$authfile,
	   'all' => \$doall,
	   'online_db=s' => \$online_db,
	   'offline_db=s' => \$offline_db,
	   'fake' => \$fake,
	   'help|h' => \$help,
	   'log=s' => \$log,
	   'debug' => \$debug);

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

# Determing DB to set up
if (!$offline_db && exists $dba_config->{general}->{offline_db}) {
    $offline_db = $dba_config->{general}->{offline_db};
}

if (!$offline_db) {
    die "offline_db not defined at command line or at $dba_configfile";
}

if (!$online_db && exists $o2o_config->{general}->{online_db}) {
    $online_db = $o2o_config->{general}->{online_db};
}

if (!$online_db) {
    die "online_db not defined at command line or at $o2o_configfile";
}

# Build commands
my @commands;
my $cmd;


if ($dogeneral) {
    # Get connection info
    my $general_schema = $dba_config->{general}->{general_schema};
    my $general_connect = "oracle://".$offline_db.'/'.$general_schema;
    
    my ($general_user, $general_pass) = CMSDBA::connection_test($auth, $general_connect);

    # Create all database objects
    foreach my $dbobject ( qw(o2o_setup o2o_log master_payload_o2o) ) { # XXX hardcoded because order matters
	my $sqlfile = $o2o_sqldir.'/'.
	    $o2o_config->{general}->{dbobject}->{$dbobject}->{sqlfile};

	CMSDBA::check_files($sqlfile);
	
	$cmd = CMSDBA::get_sqlplus_cmd('user' => $general_user, 'pass' => $general_pass, 'db' => $offline_db, 
				       'file' => $sqlfile);
	push(@commands, { 'info' => "Creating $dbobject",
			  'cmd'  => $cmd });
    }
}


# Determine what detectors to set up
my @detectors;
if ($doall) {
    @detectors = keys %{$o2o_config->{detector}}
} else {
    die "Must provide detector name(s) or --all, unless --general" unless(@ARGV || $dogeneral);
    @detectors = @ARGV;
    foreach (@detectors) {
	unless (exists $o2o_config->{detector}->{$_}) {
	    die "$_ not configured in $o2o_configfile";
	}
    }
}

foreach my $detector (@detectors) {
    if (!exists $dba_config->{detector}->{$detector}) {
	die "Detector $detector is defined in o2oconfig but not in dbconfig";
    }

    # Get connection info
    my $general_schema = $dba_config->{general}->{general_schema};
    my $general_connect = "oracle://".$offline_db.'/'.$general_schema;
    my ($general_user, $general_pass) = CMSDBA::connection_test($auth, $general_connect);
    
    my $online_schema = $o2o_config->{detector}->{$detector}->{online_schema};
    my $online_connect = "oracle://".$online_db.'/'.$online_schema;
    my ($online_user, $online_pass) = CMSDBA::connection_test($auth, $online_connect);

    my $offline_schema = $dba_config->{detector}->{$detector}->{offline_schema};
    my $offline_connect = "oracle://".$offline_db.'/'.$offline_schema;
    my ($offline_user, $offline_pass) = CMSDBA::connection_test($auth, $offline_connect);

    # Create a database link from offline schema to online schema
    $sql = qq[CREATE DATABASE LINK $online_db CONNECT TO $online_schema IDENTIFIED BY $online_pass USING '$online_db'];
    $cmd = CMSDBA::get_sqlplus_cmd('user' => $offline_user, 'pass' => $offline_pass, 'db' => $offline_db, 
				   'sql' => $sql);
    push(@commands, { 'info' => "Creating database link from $offline_db to $online_db",
		      'cmd'  => $cmd });


    foreach my $object (keys %{$o2o_config->{detector}->{$detector}->{object}}) {
	my $object_name = $object;
	$object = $o2o_config->{detector}->{$detector}->{object}->{$object_name};
	
        # Add time column to top-level-table in offline schema
	my $table = $object->{table};
	my $sql = qq[ALTER TABLE $table ADD time NUMBER(38)];
	$cmd = CMSDBA::get_sqlplus_cmd('user' => $offline_user, 'pass' => $offline_pass, 'db' => $offline_db, 
				       'sql' => $sql);
	push(@commands, { 'info' => "Adding TIME column to $table",
			  'cmd'  => $cmd });


	# Write payload procedure to offline schema
	my $procedure = $object_name.'_payload_o2o';  # XXX Should this be hardcoded?
	my $procedure_sqlfile = $o2o_sqldir.'/'.$procedure.'.sql';
	
	CMSDBA::check_files($procedure_sqlfile);
	$cmd = CMSDBA::get_sqlplus_cmd('user' => $offline_user, 'pass' => $offline_pass, 'db' => $offline_db, 
				       'file' => $procedure_sqlfile);
	push(@commands, { 'info' => "Adding $procedure to offline schema",
			  'cmd'  => $cmd });

	# Grant access to top-level-table and procedure to general schema
	$sql = qq[GRANT SELECT ON $table TO $general_schema];
	$cmd = CMSDBA::get_sqlplus_cmd('user' => $offline_user, 'pass' => $offline_pass, 'db' => $offline_db, 
				       'sql' => $sql);
	push(@commands, { 'info' => "Granting $general_schema access to $table",
			  'cmd'  => $cmd });
	
	$sql = qq[GRANT EXECUTE ON $procedure TO $general_schema];
	$cmd = CMSDBA::get_sqlplus_cmd('user' => $offline_user, 'pass' => $offline_pass, 'db' => $offline_db, 
				       'sql' => $sql);
	push(@commands, { 'info' => "Granting $general_schema access to $procedure",
			  'cmd'  => $cmd });
	
	# Register the object to O2O_SETUP
	$sql = qq[INSERT INTO o2o_setup VALUES ('$object_name', '$offline_schema', '$table')];
	$cmd = CMSDBA::get_sqlplus_cmd('user' => $general_user, 'pass' => $general_pass, 'db' => $offline_db, 
				       'sql' => $sql);
	push(@commands, { 'info' => "Registering $object_name to O2O_SETUP",
			  'cmd'  => $cmd });

    }
    

}

# Execution of commands
CMSDBA::execute_commands('cmd_array' => \@commands, 'fake' => $fake, 'debug' => $debug, 'log' => $log);
