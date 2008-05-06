#!/usr/bin/perl -w
# $Id: InjectWorker.pl,v 1.2 2008/05/06 08:27:25 loizides Exp $

use strict;
use DBI;
use Getopt::Long;
use Sys::Hostname;

#date routine for log file finding / writing
sub getdatestr()
{
    my @ltime = localtime(time);
    my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = @ltime;
    $year += 1900;
    $mon++;

    my $datestr=$year;
    if ($mon < 10) {
        $datestr=$datestr . "0";
    }

    $datestr=$datestr . $mon;
    if ($mday < 10) {
        $datestr=$datestr . "0";
    }
    $datestr=$datestr . $mday;
    return $datestr;
}


# injection subroutine 
# if called from insertFile, 2nd arg is 0 - no notify
# if called from closeFile msg, 2nd arg is 1 - notifies file to be transferred
sub inject($$)
{
    my $dbh = $_[0]; #shift;
    my $doNotify = $_[1]; #shift;

    my $filename    = $ENV{'SM_FILENAME'};
    my $count       = $ENV{'SM_FILECOUNTER'};
    my $nevents     = $ENV{'SM_NEVENTS'};;
    my $filesize    = $ENV{'SM_FILESIZE'};
    my $starttime   = $ENV{'SM_STARTTIME'};
    my $stoptime    = $ENV{'SM_STOPTIME'};
    my $status      = $ENV{'SM_STATUS'};
    my $runnumber   = $ENV{'SM_RUNNUMBER'};
    my $lumisection = $ENV{'SM_LUMISECTION'};
    my $pathname    = $ENV{'SM_PATHNAME'};
    my $hostname    = $ENV{'SM_HOSTNAME'};
    my $dataset     = $ENV{'SM_DATASET'};
    my $stream      = $ENV{'SM_STREAM'};
    my $instance    = $ENV{'SM_INSTANCE'};
    my $safety      = $ENV{'SM_SAFETY'};
    my $appversion  = $ENV{'SM_APPVERSION'};
    my $appname     = $ENV{'SM_APPNAME'};
    my $type        = $ENV{'SM_TYPE'};
    my $checksum    = $ENV{'SM_CHECKSUM'};
    my $producer    = 'StorageManager';

    my $SQL = "INSERT INTO CMS_STOMGR.TIER0_INJECTION (" .
        "RUNNUMBER,LUMISECTION,INSTANCE,COUNT,START_TIME,STOP_TIME,FILENAME,PATHNAME," .
        "HOSTNAME,DATASET,PRODUCER,STREAM,STATUS,TYPE,SAFETY,NEVENTS,FILESIZE,CHECKSUM) " .
        "VALUES ($runnumber,$lumisection,$instance,$count,'$starttime','$stoptime'," .
        "'$filename','$pathname','$hostname','$dataset','$producer','$stream','$status'," .
        "'$type',$safety,$nevents,$filesize,$checksum)";

    if (!defined $dbh) { 
        print "$SQL\n";
        return 0;
    }

    my $rows = $dbh->do($SQL) or 
        die $dbh->errstr;

    if ($rows==1 && $doNotify) {
        my $notscript = $ENV{'SM_NOTIFYSCRIPT'};
        if (!defined $notscript) {
            $notscript = "/nfshome0/cmsprod/TransferTest/injection/sendNotification.sh";
        }

        my $indfile     = $filename;
        $indfile =~ s/\.dat$/\.ind/;

        my $TIERZERO = "$notscript --APP_NAME=$appname --APP_VERSION=$appversion --RUNNUMBER $runnumber --LUMISECTION $lumisection --INSTANCE $instance --COUNT $count --START_TIME $starttime --STOP_TIME $stoptime --FILENAME $filename --PATHNAME $pathname --HOSTNAME $hostname --DATASET $dataset --STREAM $stream --STATUS $status --TYPE $type --SAFETY $safety --NEVENTS $nevents --FILESIZE $filesize --CHECKSUM $checksum --INDEX $indfile";
        system($TIERZERO);
    }
    return $rows-1;
}



# main starts here
if (!defined $ARGV[0]) {
    die "Syntax: ./injectIntoDB.pl inpath logpath errpath SMinstance";
}
my $inpath="$ARGV[0]";

if (!defined $ARGV[1]) {
    die "Syntax: ./injectIntoDB.pl inpath logpath errpath SMinstance";
}
my $outpath=">$ARGV[1]";


if (!defined $ARGV[2]) {
    die "Syntax: ./injectIntoDB.pl inpath logpath errpath SMinstance";
}
my $errpath=">$ARGV[2]";

if (!defined $ARGV[3]) {
    die "Syntax: ./injectIntoDB.pl inpath logpath errpath SMinstance";
}
my $sminstance=$ARGV[3];

#overwrite TNS to be sure it points to new DB
$ENV{'TNS_ADMIN'} = '/etc/tnsnames.ora';

# connect to DB
my $dbh; #my DB handle
if (!defined $ENV{'SM_DONTACCESSDB'}) { 
    my $dbi    = "DBI:Oracle:cms_rcms";
    my $reader = "CMS_STOMGR_W";
    $dbh = DBI->connect($dbi,$reader,"qwerty") or 
        die "Error: Connection to Oracle failed: $DBI::errstr\n";
}


my $thedate = getdatestr();
my $host = hostname();

#path names have the >  i almost forgot i did that
my $infile = "$inpath/$thedate-$host-$sminstance.log";
my $outfile = "$outpath/$thedate-$host-$sminstance.log"; #should this have a different name?
my $errfile = "$errpath/$thedate-$host-$sminstance.err";

my $line;
my $lnum = 1;

open(STDERR, $errfile) or
    die("Error:cannot open error file '$errfile'\n");

open(INDATA, $infile) or 
    die("Error: cannot open file '$infile'\n");

open(OUTDATA, $outfile) or 
    die("Error: cannot open file '$outfile'\n");


#loop over input files - sleep and try to reread file once end is reached

$endflag=0;
$SIG{'INT'}= 'SETFLAG';
$SIG{'KILL'}='SETFLAG';
sub SETFLAG {
    $endflag=1;
}

while( !$endflag ) {
    while( $line = <INDATA> ){
	chomp($line);
	#if ($line =~ m/export/i) {
	#    my @exports = split(';', $line);
	#    my $lexports = scalar(@exports);
	#    for (my $count = 0; $count < $lexports; $count++) {
	#	my $field = $exports[$count];
	#	if ($field =~ m/export (.*)=(.*)/i) {
	#	    $ENV{$1}=$2;
	#	}
	#    }
	if ($line =~ m/insertFile/i) {
	    my @exports = split(' ', $line);
	    my $lexports = scalar(@exports);
	    for (my $count = 0; $count < $lexports; $count++) {
		#my $field = "SM_$exports[2*$count]=$exports[2*$count+1]";
		#my $field = $exports[$count];
		my $field = "$exports[$count]=$exports[$count+1]";
		if ($field =~ m/^\-\-(.*)=(.*)/i) {    
		    my $fname = "SM_$1";
		    if    ($1 eq "COUNT")      { $fname = "SM_FILECOUNTER";}
		    elsif ($1 eq "START_TIME") { $fname = "SM_STARTTIME";}
		    elsif ($1 eq "STOP_TIME")  { $fname = "SM_STOPTIME";}
		    elsif ($1 eq "APP_VERSION") { $fname = "SM_APPVERSION";}
		    elsif ($1 eq "APP_NAME") { $fname = "SM_APPNAME";}
		    $ENV{$fname}=$2;
		    $count++;
		}

	    } 
	
	    my $ret=inject($dbh,0);
	    if ($ret == 0) {
		print OUTDATA "$line\n";
		my $cmd=$ENV{'SM_HOOKSCRIPT'};
		if (defined $cmd) {
		    system($cmd);
		}
	    }

	} elsif ($line =~ m/closeFile/i) {


	    my @exports = split(' ', $line);
	    my $lexports = scalar(@exports);
	    for (my $count = 0; $count < $lexports; $count++) {
		#my $field = "SM_$exports[2*$count]=$exports[2*$count+1]";
		#my $field = $exports[$count];
		my $field = "$exports[$count]=$exports[$count+1]";
		if ($field =~ m/^\-\-(.*)=(.*)/i) {    
		    my $fname = "SM_$1";
		    if    ($1 eq "COUNT")      { $fname = "SM_FILECOUNTER";}
		    elsif ($1 eq "START_TIME") { $fname = "SM_STARTTIME";}
		    elsif ($1 eq "STOP_TIME")  { $fname = "SM_STOPTIME";}
		    elsif ($1 eq "APP_VERSION") { $fname = "SM_APPVERSION";}
		    elsif ($1 eq "APP_NAME") { $fname = "SM_APPNAME";}
		    $ENV{$fname}=$2;
		    $count++;
		}

	    } 
	
	    my $ret=inject($dbh,1);
	    if ($ret == 0) {
		print OUTDATA "$line\n";
		my $cmd=$ENV{'SM_HOOKSCRIPT'};
		if (defined $cmd) {
		    system($cmd);
		}
	    }
 
	}

	$lnum++;
    }


    #sleep a little bit
    sleep(5);
    #Seek nowhere in file to reset EOF flag
    seek(INDATA,0,1);


    if($thedate!=getdatestr()) {
	#When the date changes (next day), we want to spawn a new copy of this process that goes to work on the new log
	#But need to check also that we got everything from the old file!
	
#Wait some number of hours, then end the process (can be done by setting the flag)
	my @ltime = localtime(time);
	if($ltime[2] > 6) { $endflag=1;}
    }

}


close INDATA;
close OUTDATA;

# Disconnect from DB
if (defined $dbh) { 
    $dbh->disconnect or 
        warn "Warning: Disconnection from Oracle failed: $DBI::errstr\n";
}

$SIG{'INT'}='DEFAULT';
$SIG{'KILL'}='DEFAULT';
