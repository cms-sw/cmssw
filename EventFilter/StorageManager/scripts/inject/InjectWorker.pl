#!/usr/bin/perl -w
# $Id: InjectWorker.pl,v 1.8 2008/05/14 12:53:14 loizides Exp $

use strict;
use DBI;
use Getopt::Long;
use File::Basename;
use Sys::Hostname;
use Cwd;
use Cwd 'abs_path';

############################################################################################################
my $debug=0;
############################################################################################################
my $endflag=0; 

# printout syntax and die
sub printsyntax()
{
    die "Syntax: ./InjectWorker.pl inputpath/file outputpath logpath sm_instance(not required if file input)";
}

# date routine for log file finding / writing
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

# time routine for printouts
sub gettimestr()
{
    my @ltime = localtime(time);
    my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = @ltime;
    $year += 1900;
    $mon++;

    my $timestr="";
    if ($hour < 10) {
	$timestr=$timestr . "0";
    }
    $timestr=$timestr . $hour;
    $timestr=$timestr . ":";
    if ($min < 10) {
	$timestr=$timestr . "0";
    }
    $timestr=$timestr . $min;
    $timestr=$timestr . ":";
    if ($sec < 10) {
	$timestr=$timestr . "0";
    }
    $timestr=$timestr . $sec;
    return $timestr;
}

# execute on terminate
sub TERMINATE {
    my $timestr = gettimestr();
    if ($endflag!=1) {
        print "$timestr: Terminating on request\n"; 
        $endflag=1;
    }
}

# injection subroutine 
# if called from insertFile, 2nd arg is 0: insert into DB, no notify
# if called from closeFile msg, 2nd arg is 1: updates DB, notifies file to be transferred
sub inject($$)
{
    my $dbh = $_[0];      #shift;
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
    if ($doNotify==0) {
        $SQL = "INSERT INTO CMS_STOMGR.TIER0_INJECTION (" .
            "RUNNUMBER,LUMISECTION,INSTANCE,COUNT,START_TIME,STOP_TIME,FILENAME,PATHNAME," .
            "HOSTNAME,DATASET,PRODUCER,STREAM,STATUS,TYPE,SAFETY,NEVENTS,FILESIZE,CHECKSUM) " .
            "VALUES ($runnumber,$lumisection,$instance,$count,'$starttime','$stoptime'," .
            "'$filename','$pathname','$hostname','$dataset','$producer','$stream','$status'," .
            "'$type',$safety,$nevents,$filesize,$checksum)";
    } else {
        $SQL = "UPDATE CMS_STOMGR.TIER0_INJECTION SET FILESIZE=$filesize, STATUS='$status'," .
               "STOP_TIME='$stoptime',NEVENTS=$nevents,CHECKSUM=$checksum,PATHNAME='$pathname'," . 
               "SAFETY=$safety WHERE FILENAME = '$filename'";
    }

    my $notscript = $ENV{'SM_NOTIFYSCRIPT'};
    if (!defined $notscript) {
        $notscript = "/nfshome0/cmsprod/TransferTest/injection/sendNotification.sh";
    }

    my $indfile     = $filename;
    $indfile =~ s/\.dat$/\.ind/;

    my $TIERZERO = "$notscript --APP_NAME=$appname --APP_VERSION=$appversion --RUNNUMBER $runnumber " . 
        "--LUMISECTION $lumisection --INSTANCE $instance --COUNT $count --START_TIME $starttime " . 
        "--STOP_TIME $stoptime --FILENAME $filename --PATHNAME $pathname --HOSTNAME $hostname " .
        "--DATASET $dataset --STREAM $stream --STATUS $status --TYPE $type --SAFETY $safety " .
        "--NEVENTS $nevents --FILESIZE $filesize --CHECKSUM $checksum --INDEX $indfile";
    
    if (!defined $dbh) { 
        if ($debug) { print "DB not defined, just printing and returning 0\n";}
        print "$SQL\n";
        if ($doNotify) {
            print "$TIERZERO\n";
        }
        return 0;
    }

    my $rows = $dbh->do($SQL) or die $dbh->errstr;
    if ($doNotify) {
        if ($rows==1) {
            system($TIERZERO);
        } else {
            print "Error in DB access: Did not execute $TIERZERO\n";
        }
    }
    return $rows-1;
}


############################################################################################################
# Main starts here                                                                                         #
############################################################################################################

# redirect signals
$SIG{ABRT} = \&TERMINATE;
$SIG{INT}  = \&TERMINATE;
$SIG{KILL} = \&TERMINATE;
$SIG{QUIT} = \&TERMINATE;
$SIG{TERM} = \&TERMINATE;

# figure out how I am called
my $mycall = abs_path($0);

# check arguments
if (!defined $ARGV[2]) {
    printsyntax();
}

my $infile;
my $inpath;
my $fileflag=0;
my $sminstance;
if (-d $ARGV[0]) {
    $inpath="$ARGV[0]";
    if (!defined $ARGV[3]) {
        printsyntax();
    }
    $sminstance=$ARGV[3];
} elsif (-e $ARGV[0] ) {
    $infile="$ARGV[0]";
    $fileflag=1;
} else {
    die("Error: Specified input \"$ARGV[0]\" does not exist");
}

my $outpath="$ARGV[1]";
if (!-d $outpath) {
    die("Error: Specified output path \"$outpath\" does not exist");
}
my $errpath="$ARGV[2]";
if (!-d $errpath) {
    die("Error: Specified output path \"$errpath\" does not exist");
}

my $errfile;
my $outfile;
my $thedate = getdatestr();
my @harray  = split(/\./,hostname());
my $host    = $harray[0];

my $waiting = -1;
if ($fileflag==0) {
    $infile  = "$inpath/$thedate-$host-$sminstance.log";
    $outfile = "$outpath/$thedate-$host-$sminstance.log";
    $errfile = "$errpath/$thedate-$host-$sminstance.log";
} else {
    my $inbase = basename($infile);
    $outfile = "$outpath/$inbase";
    $errfile = "$errpath/$inbase";
}

# lockfile
my $lockfile = "/tmp/." . basename($outfile) . ".lock";
if (-e $lockfile) {
    die("Error: Lock \"$lockfile\" exists.");
} else {
    system("touch $lockfile");
}

# redirecting output
open(STDOUT, ">>$errfile") or
    die("Error: Cannot open log file \"$errfile\"\n");
open(STDERR, ">>&STDOUT");
if ($debug) {print "Infile = $infile\nOutfile = $outfile\nLogfile = $errfile\n";}

# if input file does not exist - we'll wait for it
while (!(-e "$infile") && !$endflag) {
    if ($debug) {print "Input file \"$infile\" does not already exist, sleeping\n";} 
    sleep(10);
    # if day changes we can immediately spawn a new process for a new file for the new day 
    # (since old file never showed up dont wait)
    if (!(-e "$infile") && ($thedate ne getdatestr())) {
        if ($debug) {print "Spawning new process: $mycall $inpath $outpath $errpath $sminstance\n";}
	system("$mycall $inpath $outpath $errpath $sminstance &"); 
    	$endflag=1;
    }
}

# if told to exit while waiting for input, we exit here
if($endflag) {
    system("rm -f $lockfile");
    exit 0;
}

# if the output file exists (has been worked on before) then find what the last thing done was
my $line;
my $lastline;
if (-e $outfile) {
    open QUICKSEARCH, "<$outfile" or die("Error: Cannot open output file \"$outfile\"\n");
    if ($debug) {print "Found old output file \"$outfile\": Searching for last line.\n";}
    while($line=<QUICKSEARCH>) {$lastline=$line;}
    close QUICKSEARCH;
    if ($debug) {
        if (defined($lastline)) {print "Last line done was:\n $lastline\n";}
    }
}

open(INDATA, $infile) or 
    die("Error: Cannot open input file \"$infile\"\n");

# find the last thing done in a previously opened outfile - then read till that point
if (defined($lastline)) {
    if ($debug) {print "Last line done was:\n $lastline\n"; print "Skipping previously done work\n";}
    while($line = <INDATA>) {
        if ($line eq $lastline) { 
            if ($debug) {print "Found last line previously done\n";} 
            last;
        }
    }
}

# open output file
open(OUTDATA, ">>$outfile") or 
    die("Error: Cannot open output file \"$outfile\"\n");
    
# make the output file hot so that buffer is flushed on every printed line
my $ofh = select OUTDATA;
$| = 1;
select $ofh;

# overwrite TNS to be sure it points to new DB
$ENV{'TNS_ADMIN'} = '/etc/tnsnames.ora';

# connect to DB
my $dbh; #my DB handle
if (!defined $ENV{'SM_DONTACCESSDB'}) { 
    my $dbi    = "DBI:Oracle:cms_rcms";
    my $reader = "CMS_STOMGR_W";
    if ($debug) {print "Setting up DB connection for $dbi and $reader\n";}
    $dbh = DBI->connect($dbi,$reader,"qwerty") or 
        die "Error: Connection to Oracle failed: $DBI::errstr\n";

    my $timestr = gettimestr();
    print "$timestr: Setup DB connection\n";
} else { 
    print "Don't access DB flag set \n".
          "Following commands would have been processed: \n";
}

#loop over input files: sleep and try to reread file once end is reached
my $lnum=0;
my $livecounter=0;
while( !$endflag ) {

    while($line=<INDATA>) {

        if ($endflag) {last;}

        if ($debug) {print $line;}
	chomp($line);
        my $type=-1;
	if ($line =~ m/insertFile/i) {
	    if ($debug) {print "Found file insert\n";}
            $type=0;
	} elsif ($line =~ m/closeFile/i) {
	    if ($debug) {print "Found file close\n";}
            $type=1;
        } else {
	    if ($debug) {print "Unknown line: $line\n";}
            next;
        }

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
                if ($debug) {print "$fname = $ENV{$fname}\n";}
                $count++;
            }

        } 
	
        my $ret=inject($dbh,$type);
	    
        if ($ret == 0) {
            print OUTDATA "$line\n";
            my $cmd=$ENV{'SM_HOOKSCRIPT'};
            if (defined $cmd) {
                system($cmd);
            }
        }

	$lnum++;
        $livecounter=0;
    }

    if ($fileflag==1) {
        $endflag=1;
        last;
    }

    # when the date changes (next day), we want to spawn a new copy of this process 
    # that goes to work on the new log. But need to check also that we got everything 
    # from the old file!
    if ($waiting<0 && $thedate ne getdatestr()) {
        sleep(5);
        if ($debug) {print "Spawning new process: $mycall $inpath $outpath $errpath $sminstance\n";}
	system("$mycall $inpath $outpath $errpath $sminstance &"); 
        $waiting=0; #start the waiting counter
    } elsif ($waiting>=0) {
        $waiting++;
        if ($waiting>10) {
            $endflag=1;
            last;
        }
    }

    # sleep a little bit
    sleep(10);

    # seek nowhere in file to reset EOF flag
    seek(INDATA,0,1);

    # check live counter
    $livecounter++;
    if ($livecounter>360) {
        my $timestr = gettimestr();
        print "$timestr: Still alive in main loop\n";
        $livecounter = 0;
    }
}

# disconnect from DB
if (defined $dbh) { 
    $dbh->disconnect or 
        warn "Warning: Disconnection from Oracle failed: $DBI::errstr\n";
}

# close files
close INDATA;
close OUTDATA;

# remove lock file
system("rm -f $lockfile");

# reset signal
$SIG{ABRT} = 'DEFAULT';
$SIG{INT}  = 'DEFAULT';
$SIG{KILL} = 'DEFAULT';
$SIG{QUIT} = 'DEFAULT';
$SIG{TERM} = 'DEFAULT';
