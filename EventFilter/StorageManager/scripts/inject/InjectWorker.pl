#!/usr/bin/env perl
# $Id: InjectWorker.pl,v 1.36 2009/05/14 13:33:33 loizides Exp $

use warnings;
use strict;
use DBI;
use Getopt::Long;
use File::Basename;
use Cwd;
use Cwd 'abs_path';

############################################################################################################
my $debug       = 0;  # toggled by SM_DEBUG
my $nodbint     = 0;  # toggled by SM_DONTACCESSDB
my $justnoti    = 0;  # toggled by SM_JUSTNOTI (only if SM_DONTACCESSDB)
my $nofilecheck = 0;  # toggled by SM_NOFILECHECK
############################################################################################################

# global vars
my $endflag = 0; 
my $host    = ""; 

# printout syntax and die
sub printsyntax()
{
    die "Syntax: ./InjectWorker.pl inputpath/file outputpath logpath configfile instance(not needed for file)";
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

#time routine for SQL commands timestamp
sub gettimestamp($)
{
    my $stime = shift;
    my @ltime = localtime($stime);
    my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = @ltime;

    $year += 1900;
    $mon++;

    my $timestr = $year."-";
    if ($mon < 10) {
	$timestr=$timestr . "0";
    }

    $timestr=$timestr . $mon . "-";

    if ($mday < 10) {
	$timestr=$timestr . "0";
    }

    $timestr=$timestr . $mday . " " . $hour . ":" . $min . ":" . $sec;
    return $timestr;
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

# execute instead of die
sub mydie($$) 
{
    my $msg = $_[0];
    my $lf = $_[1];

    if (length($lf)>0) {
        system("rm -f $lf");
    }
    my $timestr = gettimestr();
    print "$timestr: $msg"; 
    die "Aborted\n";
}

# execute on terminate
sub TERMINATE 
{
    my $timestr = gettimestr();
    if ($endflag!=1) {
        print "$timestr: Terminating on request\n"; 
        $endflag=1;
    }
}

# reset environment variables for next input
sub initenv
{
    $ENV{'SM_FILENAME'}    = "unspecified";
    $ENV{'SM_FILECOUNTER'} = "unspecified";
    $ENV{'SM_NEVENTS'}     = "unspecified";
    $ENV{'SM_FILESIZE'}    = "unspecified";
    $ENV{'SM_STARTTIME'}   = "unspecified";
    $ENV{'SM_STOPTIME'}    = "unspecified";
    $ENV{'SM_STATUS'}      = "unspecified";
    $ENV{'SM_RUNNUMBER'}   = "unspecified";
    $ENV{'SM_LUMISECTION'} = "unspecified";
    $ENV{'SM_PATHNAME'}    = "unspecified";
    $ENV{'SM_HOSTNAME'}    = "unspecified";
    $ENV{'SM_SETUPLABEL'}  = "unspecified";
    $ENV{'SM_STREAM'}      = "unspecified";
    $ENV{'SM_INSTANCE'}    = "unspecified";
    $ENV{'SM_SAFETY'}      = "unspecified";
    $ENV{'SM_APPVERSION'}  = "unspecified";
    $ENV{'SM_APPNAME'}     = "unspecified";
    $ENV{'SM_TYPE'}        = "unspecified";
    $ENV{'SM_CHECKSUM'}    = "unspecified";
    $ENV{'SM_HLTKEY'}      = "unspecified";
}

# injection subroutine 
# if called from insertFile, 2nd arg is 0: insert into DB, no notify
# if called from closeFile msg, 2nd arg is 1: updates DB, notifies file to be transferred
sub inject($$)
{
    my $sth = $_[0];      #shift;
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
    my $setuplabel  = $ENV{'SM_SETUPLABEL'};
    my $stream      = $ENV{'SM_STREAM'};
    my $instance    = $ENV{'SM_INSTANCE'};
    my $safety      = $ENV{'SM_SAFETY'};
    my $appversion  = $ENV{'SM_APPVERSION'};
    my $appname     = $ENV{'SM_APPNAME'};
    my $type        = $ENV{'SM_TYPE'};
    my $checksum    = $ENV{'SM_CHECKSUM'};
    my $hltkey      = $ENV{'SM_HLTKEY'};
    my $producer    = 'StorageManager';
    my $destination = 'Global';
    my $commentstr  = 'HLTKEY=' . $hltkey;

    if ( ($filename   eq "unspecified") || ($count     eq "unspecified") || ($nevents     eq "unspecified") ||
         ($filesize   eq "unspecified") || ($starttime eq "unspecified") || ($stoptime    eq "unspecified") ||
         ($status     eq "unspecified") || ($runnumber eq "unspecified") || ($lumisection eq "unspecified") ||
         ($pathname   eq "unspecified") || ($hostname  eq "unspecified") || ($setuplabel  eq "unspecified") ||
         ($stream     eq "unspecified") || ($instance  eq "unspecified") || ($safety      eq "unspecified") ||
         ($appversion eq "unspecified") || ($appname   eq "unspecified") || ($type        eq "unspecified") ||
         ($checksum   eq "unspecified") ) {

        print "Error in obtained parameters\n";
        return -1;
    }

    # index file name and size
    my $indfile     = $filename;
    $indfile =~ s/\.dat$/\.ind/;
    my $indfilesize = -1;
    if ($host eq $hostname) {
        if (-e "$pathname/$indfile") {
            $indfilesize = -s "$pathname/$indfile";
        }

        if ($nofilecheck==0) {
            if (not -e "$pathname/$indfile") {
                $indfile = '';
            }
        }
    }
    
    # fix a left over bug from CMSSW_2_0_4
    $appversion=$1 if $appversion =~ /\"(.*)'/; #'for emacs syntax highlighting
    $appversion=$1 if $appversion =~ /\"(.*)\"/;

    # redirect setuplabel/streams to different destinations according to 
    # https://twiki.cern.ch/twiki/bin/view/CMS/SMT0StreamTransferOptions
    return 0 if ($stream eq 'EcalCalibration' || $stream =~ '_EcalNFS$'); #skip EcalCalibration
    return 0 if ($stream =~ '_NoTransfer$'); #skip if NoTransfer option is set

    if ($setuplabel =~ 'TransferTest' || $stream =~ '_TransferTest$') {
	$destination = 'TransferTest'; # transfer but delete after
    } elsif ($stream =~ '_NoRepack$'|| $stream eq 'Error') {
	$destination = 'GlobalNoRepacking'; # do not repack 
        $indfile     = '';
        $indfilesize = -1;
    }

    if ($doNotify==0) {
	my $stime = gettimestamp($starttime);

        if (defined $sth) {
	    $sth->bind_param(1,$filename);
	    $sth->bind_param(2,$pathname);
	    $sth->bind_param(3,$hostname);
	    $sth->bind_param(4,$setuplabel);
	    $sth->bind_param(5,$stream);
	    $sth->bind_param(6,$type);
	    $sth->bind_param(7,$producer);
	    $sth->bind_param(8,$appname);
	    $sth->bind_param(9,$appversion);
	    $sth->bind_param(10,$runnumber);
	    $sth->bind_param(11,$lumisection);
	    $sth->bind_param(12,$count);
	    $sth->bind_param(13,$instance);
	    $sth->bind_param(14,$stime);
        }
    } else {
        my $stime = gettimestamp($stoptime);

        if (defined $sth) {
    	    $sth->bind_param(1,$filename);
	    $sth->bind_param(2,$pathname);
	    $sth->bind_param(3,$destination);
	    $sth->bind_param(4,$nevents);
	    $sth->bind_param(5,$filesize);
	    $sth->bind_param(6,$checksum);
	    $sth->bind_param(7,$stime);
	    $sth->bind_param(8,$indfile);
	    $sth->bind_param(9,$indfilesize);
	    $sth->bind_param(10,$commentstr);
        }
    }

    my $notscript = $ENV{'SM_NOTIFYSCRIPT'};
    if (!defined $notscript) {
        $notscript = "/nfshome0/cmsprod/TransferTest/injection/sendNotification.sh";
    }

    my $TIERZERO = "$notscript --APP_NAME=$appname --APP_VERSION=$appversion --RUNNUMBER $runnumber " . 
        "--LUMISECTION $lumisection --START_TIME $starttime --STOP_TIME $stoptime --FILENAME $filename " .
        "--PATHNAME $pathname --HOSTNAME $hostname --DESTINATION $destination --SETUPLABEL $setuplabel " .
        "--STREAM $stream --TYPE $type --NEVENTS $nevents --FILESIZE $filesize --CHECKSUM $checksum " . 
        "--HLTKEY $hltkey";

    if ($indfile ne '') {
      $TIERZERO .= " --INDEX $indfile"; # --INDEXFILESIZE $indfilesize"
    }

    if (!defined $sth) { 
        if ($debug) { 
            print "DB not defined, just returning 0\n";
            if ($doNotify) {
                print "$TIERZERO\n";
                if ($justnoti) {
                    system($TIERZERO);
                }
            }
        } else {
            if ($justnoti) {
                system($TIERZERO);
            }
        }
        return 0;
    }

    my $errflag=0;
    my $rows = $sth->execute() or $errflag=1;

    if ($errflag>0) {
        print "Error in DB access when executing, DB returned $sth->errstr\n";
        return -1;
    }
    
    if ($rows!=1) {
        print "Strange error related to DB access when executing , DB returned rows=$rows\n";
        if ($doNotify) {
            print "Error related to DB: Since rows!=1, did not execute $TIERZERO\n";
        }
        return -1; 
    }

    if ($doNotify) {
	if ($debug) {print "Executing notification: $TIERZERO\n";}
        system($TIERZERO);
    }
    return 0;
}

############################################################################################################
# Main starts here                                                                                         #
############################################################################################################

# get options from environment
if (defined $ENV{'SM_DEBUG'}) { 
    $debug=1;
}

if (defined $ENV{'SM_NOFILECHECK'}) { 
    $nofilecheck=1;
}

if (defined $ENV{'SM_DONTACCESSDB'}) { 
    $nodbint=1;
    
    if (defined $ENV{'SM_JUSTNOTI'}) { 
        $justnoti=1;
    }
}

# redirect signals
$SIG{ABRT} = \&TERMINATE;
$SIG{INT}  = \&TERMINATE;
$SIG{KILL} = \&TERMINATE;
$SIG{QUIT} = \&TERMINATE;
$SIG{TERM} = \&TERMINATE;

# figure out how I am called
my $mycall = abs_path($0);

# check arguments
if (!defined $ARGV[3]) {
    printsyntax();
}

my $infile;
my $inpath;
my $fileflag=0;
my $sminstance;
if (-d $ARGV[0]) {
    $inpath="$ARGV[0]";
    if (!defined $ARGV[4]) {
        printsyntax();
    }
    $sminstance=$ARGV[4];
} elsif (-e $ARGV[0] ) {
    $infile="$ARGV[0]";
    $fileflag=1;
} else {
    mydie("Error: Specified input \"$ARGV[0]\" does not exist","");
}

my $outpath="$ARGV[1]";
if (!-d $outpath) {
    mydie("Error: Specified output path \"$outpath\" does not exist","");
}
my $errpath="$ARGV[2]";
if (!-d $errpath) {
    mydie("Error: Specified output path \"$errpath\" does not exist","");
}
my $config="$ARGV[3]";
if (!-e $config) {
    mydie("Error: Specified config file \"$config\" does not exist","");
}

my $reader = "xxx";
my $phrase = "xxx";
if (-r $config) {
    eval `cat $config`;
} else {
    mydie("Error: Can not read config file \"$config\"","");
    usageShort();
}

my $errfile;
my $outfile;
my $thedate  = getdatestr();
my $hostname = `hostname -f`;
my @harray   = split(/\./,$hostname);
$host        = $harray[0];

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
    mydie("Error: Lock \"$lockfile\" exists.","");
} else {
    system("touch $lockfile");
}

# redirecting output
open(STDOUT, ">>$errfile") or
    mydie("Error: Cannot open log file \"$errfile\"\n",$lockfile);
open(STDERR, ">>&STDOUT");
if ($debug) {print "Infile = $infile\nOutfile = $outfile\nLogfile = $errfile\n";}

# if input file does not exist - we will wait for it
while (!(-e "$infile") && !$endflag) {
    if ($debug) {print "Input file \"$infile\" does not already exist, sleeping\n";} 
    sleep(30);
    # if day changes we can immediately spawn a new process for a new file for the new day 
    # (since old file never showed up dont wait)
    if (!(-e "$infile") && ($thedate ne getdatestr())) {
        if ($debug) {print "Spawning new process: $mycall $inpath $outpath $errpath $config $sminstance\n";}
	system("$mycall $inpath $outpath $errpath $config $sminstance &"); 
    	$endflag=1;
    }
}

# if told to exit while waiting for input, we exit here
if ($endflag) {
    system("rm -f $lockfile");
    open(STDOUT,">>/dev/null");
    open(STDERR,">>/dev/null");
    system("rm -f $errfile");
    exit 0;
}

# if the output file exists (has been worked on before) then find what the last thing done was
my $line;
my $lastline;
if (-e $outfile) {
    open QUICKSEARCH, "<$outfile" or mydie("Error: Cannot open output file \"$outfile\"\n",$lockfile);
    if ($debug) {print "Found old output file \"$outfile\": Searching for last line.\n";}
    while($line=<QUICKSEARCH>) {$lastline=$line;}
    close QUICKSEARCH;
    if ($debug) {
        if (defined($lastline)) {print "Last line done was:\n $lastline\n";}
    }
}

open(INDATA, $infile) or 
    mydie("Error: Cannot open input file \"$infile\"\n",$lockfile);

# find the last thing done in a previously opened outfile - then read till that point
if (defined($lastline)) {
    if ($debug) {print "Last line done was:\n $lastline\n"; print "Skipping previously done work\n";}
    while($line = <INDATA>) {
        if ($lastline =~ /$line/) {
            if ($debug) {print "Found last line previously done\n";} 
            last;
        }
    }
}

# open output file
open(OUTDATA, ">>$outfile") or 
    mydie("Error: Cannot open output file \"$outfile\"\n",$lockfile);
    
# make the output file hot so that buffer is flushed on every printed line
my $ofh = select OUTDATA;
$| = 1;
select $ofh;

# overwrite TNS to be sure it points to new DB
$ENV{'TNS_ADMIN'} = '/etc/tnsnames.ora';

# connect to DB
my $dbh;          #my DB handle
my $newHandle;    #for new files
my $injectHandle; #for injections
my $SQLn;
my $SQLi;
my $dbi    = "DBI:Oracle:cms_rcms";
my $dbhlt;        #my DB handle for HLT key
my $dbihlt = "DBI:Oracle:cms_omds_lb";
my $hltHandle;    #for HLT key queries
my $SQLh;
my %hltkeys;      #cache hlt keys

if ($nodbint==0) { 

    if ($debug) {print "Setting up DB connection for $dbi and $reader\n";}
    my $retry = 0;
    while (!$retry) {
        $retry=1;
        $dbh = DBI->connect($dbi,$reader,$phrase) or $retry=0;
        if ($retry == 0) {
            print("Error: Connection to Oracle failed: $DBI::errstr\n",$lockfile);
            sleep(10);
        }
    }
    
    my $timestr = gettimestr();
    print "$timestr: Setup main DB connection\n";

    $SQLn = "INSERT INTO CMS_STOMGR.FILES_CREATED (" .
        "FILENAME,CPATH,HOSTNAME,SETUPLABEL,STREAM,TYPE,PRODUCER,APP_NAME,APP_VERSION," .
        "RUNNUMBER,LUMISECTION,COUNT,INSTANCE,CTIME) " .
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?," .
        "TO_DATE(?,'YYYY-MM-DD HH24:MI:SS'))";
    $newHandle = $dbh->prepare($SQLn) or mydie("Error: Prepare failed for $SQLn: $dbh->errstr \n",$lockfile);
    
    $SQLi = "INSERT INTO CMS_STOMGR.FILES_INJECTED (" .
        "FILENAME,PATHNAME,DESTINATION,NEVENTS,FILESIZE,CHECKSUM,ITIME,INDFILENAME,INDFILESIZE,COMMENT_STR) " .
        "VALUES (?,?,?,?,?,?," . 
        "TO_DATE(?,'YYYY-MM-DD HH24:MI:SS'),?,?,?)";
    $injectHandle = $dbh->prepare($SQLi) or mydie("Error: Prepare failed for $SQLi: $dbh->errstr \n",$lockfile);
        
    # this is for HLT key queries
    if ($debug) {print "Setting up DB connection for $dbihlt and $reader\n";}
    $retry = 0;
    while (!$retry) {
        $retry=1;
        $dbhlt = DBI->connect($dbihlt,$reader,$phrase) or $retry=0;
        if ($retry == 0) {
            print("Error: Connection to Oracle failed: $DBI::errstr\n",$lockfile);
            sleep(10);
        }
    }

    $timestr = gettimestr();
    print "$timestr: Setup DB connection for HLT key retrieval\n";
    
    $SQLh = "SELECT STRING_VALUE FROM CMS_RUNINFO.RUNSESSION_PARAMETER " . 
        "WHERE RUNNUMBER=? and NAME='CMS.LVL0:HLT_KEY_DESCRIPTION'";
    $hltHandle = $dbhlt->prepare($SQLh) or mydie("Error: Prepare failed for $SQLh: $dbh->errstr \n",$lockfile);

} else { # no DB interaction
    if ($debug) {
        print "Don't access DB flag set \n".
            "Following commands would have been processed: \n";
    }
}

#loop over input files: sleep and try to reread file once end is reached
my $lnum=0;
my $livecounter=0;

if (1) {
    my $timestr = gettimestr();
    print "$timestr: Entering main while loop now\n";
}

while(!$endflag) {

    while($line=<INDATA>) {
        
        if ($endflag) {last;}

        if ($debug) {print $line;}
	chomp($line);
        my $type=-1;
	my $useHandle;
	if ($line =~ m/insertFile/i) {
	    if ($debug) {print "Found file insert\n";}
            $type=0;
	    $useHandle=$newHandle;
	} elsif ($line =~ m/closeFile/i) {
	    if ($debug) {print "Found file close\n";}
            $type=1;
	    $useHandle=$injectHandle;
        } else {
	    if ($debug) {print "Unknown line: $line\n";}
            next;
        }

        initenv;

        my @exports = split(' ', $line);
        my $lexports = scalar(@exports);
        for (my $count = 0; $count < $lexports; $count++) {

            my $field = "$exports[$count]=$exports[$count+1]";
            if ($field =~ m/^\-\-(.*)=(.*)/i) {    
                my $fname = "SM_$1";
                if    ($1 eq "COUNT")       { $fname = "SM_FILECOUNTER";}
                elsif ($1 eq "DATASET")     { $fname = "SM_SETUPLABEL";}
                elsif ($1 eq "START_TIME")  { $fname = "SM_STARTTIME";}
                elsif ($1 eq "STOP_TIME")   { $fname = "SM_STOPTIME";}
                elsif ($1 eq "APP_VERSION") { $fname = "SM_APPVERSION";}
                elsif ($1 eq "APP_NAME")    { $fname = "SM_APPNAME";}
                $ENV{$fname}=$2;
                if ($debug) {print "$fname = $ENV{$fname}\n";}
                $count++;
            }
        } 

        # query hlt db if hlt was not already obtained
        $ENV{'SM_HLTKEY'} = "UNKNOWN";
        if (defined $dbhlt) {
            my $runnumq = $ENV{'SM_RUNNUMBER'};
            my $hltkey = $hltkeys{$runnumq};
            if (defined $hltkey) {
                $ENV{'SM_HLTKEY'}=$hltkey;
            } else {
                my $errflag = 0;
                if ($debug) {print "Quering DB for runnumber $runnumq\n";}
                $hltHandle->execute($runnumq) or $errflag=1;
                if ($errflag>0) {
                    print "Error in DB for HLT KEY when executing, DB returned $hltHandle->errstr\n";
                } else {
                    my @row = $hltHandle->fetchrow_array or $errflag=1;
                    if ($errflag>0) {
                        print "Error in DB for HLT KEY when fetching, DB returned $hltHandle->errstr\n";
                    } else {
                        if (defined $row[0]) {
                            $hltkey = $row[0];
                            $ENV{'SM_HLTKEY'}=$hltkey;
                            $hltkeys{$runnumq} = $hltkey;
                            if ($debug) {print "Obtained $hltkey for run $runnumq\n";}
                        }
                    }
                }
                $hltHandle->finish;
            }
	}

        # inject and possibly notify
        my $ret=inject($useHandle,$type);
	    
        if ($ret == 0) {
            print OUTDATA "$line\n";
            if ($type == 1) {
              my $cmd=$ENV{'SM_HOOKSCRIPT'};
              if (defined $cmd) {
                  system($cmd);
              }
            }
        } else {
            my $timestr = gettimestr();
            print "$timestr: Inject returned error ($ret) for $line\n"; 
            print OUTDATA "Error for $line\n";
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
        if ($debug) {print "Spawning new process: $mycall $inpath $outpath $errpath $config $sminstance\n";}
	system("$mycall $inpath $outpath $errpath $config $sminstance &"); 
        $waiting=0; #start the waiting counter
    } elsif ($waiting>=0) {
        $waiting++;
        if ($waiting>50) {
            $endflag=1;
            last;
        }
    }

    # sleep a little bit
    sleep(20);

    # seek nowhere in file to reset EOF flag
    seek(INDATA,0,1);

    #If can't ping the db will reconnect and re-prepare statements
    unless(defined($dbh) && $dbh->ping()) {
        my $retry = 0;
        while (!$retry) {
            $retry=1;
            $dbh = DBI->connect($dbi,$reader,$phrase) or $retry=0;
            if ($retry == 0) {
                print("Error: Re-Connection to Oracle failed: $DBI::errstr\n",$lockfile);
                sleep(10);
            }
        }
	$newHandle = $dbh->prepare($SQLn) or mydie("Error: Prepare failed for $SQLn: $dbh->errstr \n",$lockfile);
	$injectHandle = $dbh->prepare($SQLi) or mydie("Error: Prepare failed for $SQLi: $dbh->errstr \n",$lockfile);
    }

    unless(defined($dbhlt) && $dbhlt->ping()) {
        my $retry = 0;
        while (!$retry) {
            $retry=1;
            $dbhlt = DBI->connect($dbihlt,$reader,$phrase) or $retry=0;
            if ($retry == 0) {
                print("Error: Re-Connection to Oracle failed: $DBI::errstr\n",$lockfile);
                sleep(10);
            }
        }
        $hltHandle = $dbhlt->prepare($SQLh) or mydie("Error: Prepare failed for $SQLh: $dbh->errstr \n",$lockfile);
    }

    # check live counter
    $livecounter++;
    if ($livecounter>60) {
        my $timestr = gettimestr();
        print "$timestr: Still alive in main loop\n";
        $livecounter = 0;
    }
}

# disconnect from DB
if (defined $dbh) { 
    my $timestr = gettimestr();
    print "$timestr: Disconnect from main DB connection\n";
    $dbh->disconnect or 
        warn "Warning: Disconnection from Oracle failed: $DBI::errstr\n";
}
if (defined $dbhlt) { 
    my $timestr = gettimestr();
    print "$timestr: Disconnect from DB connection for HLT key retrieval\n";
    $dbhlt->disconnect or 
        warn "Warning: Disconnect from Oracle for HLT failed: $DBI::errstr\n";
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
