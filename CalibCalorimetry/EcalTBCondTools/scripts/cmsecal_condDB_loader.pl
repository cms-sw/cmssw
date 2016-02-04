#!/usr/bin/perl

# A script to generate cfg files for the loading of the offline condDB

use warnings;
use strict;
$|++;

use File::Spec;

# Options
my $smSlot = 1;

# Connection and resource locations
my $connect = "oracle://cmsr/CMS_ECAL_H4_COND";
my $catalog = "relationalcatalog_oracle://cmsr/CMS_ECAL_H4_COND";
my $logfile = "/afs/cern.ch/cms/ECAL/testbeam/pedestal/2006/LOGFILE/cms_ecal_h4_cond.log";
my $cfgdir = "/afs/cern.ch/cms/ECAL/testbeam/pedestal/2006/config_files/write/cms_ecal_h4_cond";
$ENV{TNS_ADMIN} = "/afs/cern.ch/project/oracle/admin";

unless ($#ARGV == 3) {
  die "ERROR:  Use Args:  object inputfile tagsuffix since\n";
}

my ($object, $inputfile, $tagsuffix, $since) = @ARGV;

# "Fake mode" if since < 0
my $fake = $since < 0 ? 1 : 0;

# Rework the parameters
$cfgdir = File::Spec->rel2abs( $cfgdir );
$inputfile = File::Spec->rel2abs( $inputfile );
$logfile = File::Spec->rel2abs( $logfile );
my $record = $object.'Rcd';
my $tag = $object.'_'.$tagsuffix;
my $appendIOV = "false";
my $mode = "create";
if ($since != 0) { 
  $appendIOV = "true"; 
  $mode = "append";
}

# Get the number of the config file
opendir DIR, $cfgdir or die $!;
my @cfgfiles = sort(grep(/^\d{3}_.+\.cfg$/, readdir( DIR )));
my $cfgnum = @cfgfiles ? $#cfgfiles + 1 : 0;
my $cfgfile = sprintf("%03s_${object}_${mode}_${tagsuffix}.cfg", $cfgnum);
$cfgfile = File::Spec->catfile($cfgdir, $cfgfile) or die $!;
closedir DIR;

# Print our settings
print <<ENDPRINT;
=== Config Arguments ===
connect    $connect
catalog    $catalog
logfile    $logfile
object     $object
inputfile  $inputfile
tag        $tag
since      $since
appenIOV   $appendIOV
cfgfile    $cfgfile
ENDPRINT

# Make the config file
my $essource = '';
$essource =<<ENDCONFIG if $since != 0;
    es_source = PoolDBESSource
    {
        string connect = "$connect"
        untracked string catalog = "$catalog"
        untracked uint32 authenticationMethod = 1
        bool loadAll = true
        string timetype = "runnumber"
        VPSet toGet =
        {
            {
                string record = "$record"
                string tag = "$tag"
            }
        }
     }
ENDCONFIG

my $config =<<ENDCONFIG;
process TEST =
{
    source = EmptyIOVSource
    {
        string timetype = "runnumber"
	untracked uint32 firstRun = 1
        untracked uint32 lastRun  = 1
        uint32 interval = 1
    }

$essource

    service = PoolDBOutputService
    {
	string connect = "$connect"
	untracked string catalog = "$catalog"
	untracked uint32 authenticationMethod = 1
	string timetype = "runnumber"
	VPSet toPut =
	{
	    {
		untracked string containerName = "$object"
		untracked bool appendIOV = $appendIOV
		string tag ="$tag"
	    }
	}
    }
    
    module ecalModule = StoreEcalCondition
    {
        string logfile = "$logfile"
        untracked uint32 smSlot = $smSlot
	VPSet toPut =
	{
	    {
		untracked string conditionType = "$object"
		untracked string inputFile = "$inputfile"
		untracked uint32 since = $since
	    }
	}
    }

    path p =
    {
	ecalModule
    }

}
ENDCONFIG

# Write the config file
if ($fake) {
  print $config, "\n";
} else {
  open FILE, '>', $cfgfile or die $!;
  print FILE $config;
  close FILE;
}

# Execute cmsRun on the config file
print "=== Executing cmsRun ===\n";
system("cmsRun $cfgfile") unless $fake;

exit $?;
