#!/usr/bin/perl -w
# $Id: injectIntoDB.pl,v 1.11 2008/05/02 10:47:47 loizides Exp $

use strict;
#use DBI;
use Getopt::Long;
use File::Basename;
use Sys::Hostname;


my $debug=1;

my $endflag=0;
$SIG{'INT'}= 'SETFLAG';
$SIG{'KILL'}='SETFLAG';
sub SETFLAG {
    $endflag=1;
}

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
# if called from insertFile, 2nd arg is 0 - insert into DB, no notify
# if called from closeFile msg, 2nd arg is 1 - updates DB, notifies file to be transferred
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

    my $SQL;
    if($doNotify==0) {
        $SQL = "INSERT INTO CMS_STOMGR.TIER0_INJECTION (" .
            "RUNNUMBER,LUMISECTION,INSTANCE,COUNT,START_TIME,STOP_TIME,FILENAME,PATHNAME," .
            "HOSTNAME,DATASET,PRODUCER,STREAM,STATUS,TYPE,SAFETY,NEVENTS,FILESIZE,CHECKSUM) " .
            "VALUES ($runnumber,$lumisection,$instance,$count,'$starttime','$stoptime'," .
            "'$filename','$pathname','$hostname','$dataset','$producer','$stream','$status'," .
            "'$type',$safety,$nevents,$filesize,$checksum)";
    } else {
        $SQL = "UPDATE CMS_STOMGR.TIER0_INJECTION SET FILESIZE=$filesize, STATUS='$status',  STOP_TIME='$stoptime', NEVENTS=$nevents, CHECKSUM=$checksum, PATHNAME='$pathname', SAFETY=$safety WHERE FILENAME = '$filename'";
    }
    
    if (!defined $dbh) { 
        if($debug) { print "DB not defined, just printing and returning 0\n";}
        print "$SQL\n";
        if($doNotify) {
        
            my $notscript = $ENV{'SM_NOTIFYSCRIPT'};
            if (!defined $notscript) {
                $notscript = "/nfshome0/cmsprod/TransferTest/injection/sendNotification.sh";
            }

            my $indfile     = $filename;
            $indfile =~ s/\.dat$/\.ind/;
            print "$notscript --APP_NAME=$appname --APP_VERSION=$appversion --RUNNUMBER $runnumber --LUMISECTION $lumisection --INSTANCE $instance --COUNT $count --START_TIME $starttime --STOP_TIME $stoptime --FILENAME $filename --PATHNAME $pathname --HOSTNAME $hostname --DATASET $dataset --STREAM $stream --STATUS $status --TYPE $type --SAFETY $safety --NEVENTS $nevents --FILESIZE $filesize --CHECKSUM $checksum --INDEX $indfile \n";
        }
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
my $oldflag=0;
my $thedate = getdatestr();
my $host = hostname();
my $inpath;
my $infile;
my $outfile;
my $errfile;
my $line;
my $lnum = 1;
my $lastline;

if (!defined $ARGV[0]) {
    die "Syntax: ./injectWorker.pl inpath(or file) donepath logpath SMinstance(required if in not a file)";
}
if ( -d $ARGV[0]) {
    $inpath="$ARGV[0]";
} elsif ( -e $ARGV[0] ) {
    $infile="$ARGV[0]";
    $oldflag=1;
} else {
    die("Specified infile does not exist");
}

if (!defined $ARGV[1]) {
    die "Syntax: ./injectWorker.pl inpath(or file) donepath logpath SMinstance(required if in not a file)";
}
my $outpath="$ARGV[1]";


if (!defined $ARGV[2]) {
    die "Syntax: ./injectWorker.pl inpath(or file) donepath logpath SMinstance(required if in not a file)";
}
my $errpath="$ARGV[2]";

my $sminstance;

if (!defined $ARGV[3] && !defined $infile) {
    die "Syntax: ./injectWorker.pl inpath(or file) donepath logpath SMinstance(required if in not a file)";
} elsif(!defined $infile) {
    my $sminstance=$ARGV[3];
    $infile = "$inpath/$thedate-$host-$sminstance.log";
    $outfile = "$outpath/$thedate-$host-$sminstance.log"; #should this have a different name?
    #if the output file exists (has been worked on before) then find what the last thing done was
    if( -e $outfile ) {
        open QUICKSEARCH, "<$outfile";
        while($line=<QUICKSEARCH>) { $lastline=$line;}
        close QUICKSEARCH;
    }
    $errfile = "$errpath/$thedate-$host-$sminstance.log";

} else {
    my $inbase = basename($infile);
    $outfile = "$outpath/$inbase";
    #if the output file exists (has been worked on before) then find what the last thing done was
    if( -e $outfile ) {
        open QUICKSEARCH, "<$outfile" or die("Error: cannot open output file '$outfile'\n");
        if($debug){ print "Found old file...searching for last thing done.\n";}
        while($line=<QUICKSEARCH>) { $lastline=$line;}
        close QUICKSEARCH;
        if ($debug) { print "Last line done was:\n $lastline\n";}
    }
    $errfile = "$errpath/$inbase";
}



open(STDOUT, ">$errfile") or
    die("Error:cannot open log file '$errfile'\n");
open(STDERR, ">&STDOUT");

if($debug) {print "Infile = $infile\nOutfile = $outfile\nLogfile = $errfile\n";}

if( !(-e "$infile")) { print "In file for today does not already exist, creating it"; system("touch $infile");}

open(INDATA, $infile) or 
    die("Error: cannot open input file '$infile'\n");

#Find the last thing done in a previously opened outfile - then read till that point
if( defined($lastline) ) {
    if ($debug) {print "Last line done was:\n $lastline\n"; print "Skipping previously done work\n";}
    while($line = <INDATA>) {
        if($line eq $lastline) { if($debug) {print "Found last line previously done\n";} last;}
    }
}

open(OUTDATA, ">>$outfile") or 
    die("Error: cannot open output file '$outfile'\n");
    
#Make the output file hot so that buffer is flushed on every printed line
my $ofh = select OUTDATA;
$| = 1;
select $ofh;

#overwrite TNS to be sure it points to new DB
$ENV{'TNS_ADMIN'} = '/etc/tnsnames.ora';

# connect to DB
my $dbh; #my DB handle
if (!defined $ENV{'SM_DONTACCESSDB'}) { 
    
    if($debug) {print "env var is $ENV{'SM_DONTACCESSDB'}"; print "Setting up DB connection\n";}
    my $dbi    = "DBI:Oracle:cms_rcms";
    my $reader = "CMS_STOMGR_W";
    $dbh = DBI->connect($dbi,$reader,"qwerty") or 
        die "Error: Connection to Oracle failed: $DBI::errstr\n";
} else {print "Don't access DB flag set \n"."Following commands would have been processed: \n";}



#loop over input files - sleep and try to reread file once end is reached

while( !$endflag ) {
    while($line=<INDATA>){
        if($endflag) {last;}
        if($debug) {print $line;}
	chomp($line);
	if ($line =~ m/insertFile/i) {
	    if($debug) {print "Found file insertion\n";}
	    my @exports = split(' ', $line);
	    my $lexports = scalar(@exports);
	    for (my $count = 0; $count < $lexports; $count++) {

		my $field = "$exports[$count]=$exports[$count+1]";
		if ($field =~ m/^\-\-(.*)=(.*)/i) {    
		    my $fname = "SM_$1";
		    if    ($1 eq "COUNT")      { $fname = "SM_FILECOUNTER";}
		    elsif ($1 eq "START_TIME") { $fname = "SM_STARTTIME";}
		    elsif ($1 eq "STOP_TIME")  { $fname = "SM_STOPTIME";}
		    elsif ($1 eq "APP_VERSION") { $fname = "SM_APPVERSION";}
		    elsif ($1 eq "APP_NAME") { $fname = "SM_APPNAME";}
		    $ENV{$fname}=$2;
		    if($debug) {print "$fname = $ENV{$fname}\n";}
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
	    if($debug) {print "Found file close\n";}

	    my @exports = split(' ', $line);
	    my $lexports = scalar(@exports);
	    for (my $count = 0; $count < $lexports; $count++) {

		my $field = "$exports[$count]=$exports[$count+1]";
		if ($field =~ m/^\-\-(.*)=(.*)/i) {    
		    my $fname = "SM_$1";
		    if    ($1 eq "COUNT")      { $fname = "SM_FILECOUNTER";}
		    elsif ($1 eq "START_TIME") { $fname = "SM_STARTTIME";}
		    elsif ($1 eq "STOP_TIME")  { $fname = "SM_STOPTIME";}
		    elsif ($1 eq "APP_VERSION") { $fname = "SM_APPVERSION";}
		    elsif ($1 eq "APP_NAME") { $fname = "SM_APPNAME";}
		    $ENV{$fname}=$2;
		    if($debug) {print "$fname = $ENV{$fname}\n";}
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


    if($thedate!=getdatestr() && $oldflag==0) {
	#When the date changes (next day), we want to spawn a new copy of this process that goes to work on the new log
	#But need to check also that we got everything from the old file!
	exec('./InjectWorker.pl $inpath $outpath $errpath $sminstance') or warn("Can't launch new process after date change.");
        #Wait some number of hours, then end the process (can be done by setting the flag)
	my @ltime = localtime(time);
	if($ltime[2] > 6) { $endflag=1;}
    } elsif ($oldflag) {
        $endflag=1;
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
