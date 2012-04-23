# remove this line before using!!!!
#!/usr/bin/perl

# od_mon_write.pl
# Written by Giovanni.Organtini@roma1.infn.it
#
# Read OD monitoring data as text files and build SQL commands to
# fill the DB table OD_MON_DAT.
# The file containing the SQL commands is /tmp/od_mon_dat.sql 
#
# Can also run in a completely automatic mode (read last entry from
# DB, process existing new data and update DB. 
#
# Usage: run this script according to instructions, then 
#        log on DB sqlplus cms_ecal_cond@cms_omds_lb and 
#        issue the command > @/tmp/od_mon_dat
# 
# 2012 vers. 1.0
# 2012 vers. 2.0: process automatically new files

use Getopt::Long;
use strict;
use DBI;

my $ME = $0;
$ME =~ s/^\..//g;
$ME = '[' . $ME . ']';

# 
# get options
#
my $help;
my $debug = 0;
my $sm;
my $after;
my $auto;
my $nmax = -1;
my $pw;
my $result = GetOptions("verbose=n" => \$debug, "after=s" => \$after, 
			"debug=n" => \$debug, "sm=s" => \$sm, "help" => \$help,
			"auto" => \$auto, "nmax=i" => \$nmax, "pw=s" => \$pw);

#
# Oracle connection
#
my $ORACLE_HOME = `pwd`;
my $ORACLE_SID  = 'INT2R_LB';
my $ORACLE_PW   = $pw;

$ENV{ORACLE_HOME} = $ORACLE_HOME;
$ENV{ORACLE_SID}  = $ORACLE_SID;
my $CONNECT_STRING = 'dbi:Oracle:' . $ORACLE_SID;
my $dbh = DBI->connect($CONNECT_STRING,
		       'CMS_ECAL_COND', 
		       $ORACLE_PW) || 
    die "Can't connect to $ORACLE_SID";

$dbh->{AutoCommit}    = 0;
$dbh->{RaiseError}    = 1;
$dbh->{ora_check_sql} = 0;
$dbh->{RowCacheSize}  = 16;
my $sql = '';

if ($help) {
    help();
    exit 0;
}

if ($auto) {
    # 
    # get the last timestamp in the DB 
    #
    my $sql = "SELECT TO_CHAR(MAX(RUN_START), 'YYYY-MM-DD HH24:MI:SS') " .
	"FROM OD_MON_DAT";
    my $sth = $dbh->prepare($sql);
    $sth->execute();
    my $last_timestamp = '1970-01-01 00:00:00';
    while (my @row = $sth->fetchrow_array()) {
	$last_timestamp = $row[0];    
    }
    $after = $last_timestamp;
    print "$ME Analyzing data taken after $last_timestamp\n";
} 

#
# check format
#
`/bin/touch -d '1970-01-01 00:00:00' /tmp/od_mon_write.now`;
if ($after) {
    if ($after !~ m/[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}/) {
	print "$ME ERROR: date format not recognized (use YYY-MM-DD HH24:MI:SS)\n";
	exit -1;
    }
    my $cmd = "/bin/touch -d '" . $after . "' /tmp/od_mon_write.now";
    `$cmd`;
} 

if (($debug < 0) || ($debug > 2)) {
    print "$ME ERROR: verbose level can be either 0, 1 or 2.\n";
}

#
# prepare: read valid channels from the map and build a hash
#
open VL, "/nfshome0/vlassov/delays/endcap/fibmaptr.txt";
my @buffer = <VL>;
close VL;
my @validChannels;
foreach my $line (@buffer) {
    chomp $line;
    if ($line =~ m/^6/) {
	my @l = split / +/, $line;
	my $n = @l;
	my $fed = $l[0];
	for (my $i = 11; $i <$n; $i++) {
	    my $ccu = $l[$i];
	    my $ch = $fed * 1000 + $ccu;
	    if ($debug >= 2) {
		print "$ME Found valid channel: $ch\n"; 
	    }
	    push @validChannels, $ch;
	}
    }
}

#
# list directories
#
open OUT, ">/tmp/od_mon_dat.sql";
my $lscmd = "ls -d1 /data/ecalod-disk01/dcu-data/ccs-data/*"; #list command
my $slscmd = $lscmd; #list command for specific SM (set below)
if ($sm) {
    $slscmd .= "/" . $sm;
}

#
# scan directories
#
my $nStatements = 0;
my @dirs = `$slscmd`;
foreach my $dir (@dirs) {
    if ($debug >= 1) {
	print "$ME Listing files in $dir\n";
    }
    chomp $dir;
    my $fed =$dir;
    $fed =~ s/^.*E(B|E)//; # remove the first characters and get the FED number
    my $prefix = "EB";
    if ($dir =~ m/EB/) {
	if ($fed < 0) {
	    $fed = 610 - $fed - 1;
	} else {
	    $fed = 628 + $fed - 1;
	}
    } else {
	$prefix = "EE";
	if ($fed < 0) {
	    $fed = 601 + (- $fed + 2) % 9;
	} else {
	    $fed = 646 + (+ $fed + 2) % 9;
	}
    }
    #
    # find files newer than the last enrty in the DB (or according to the
    # --after option)
    #
    my $lsfcmd = "/usr/bin/find $dir -type f -cnewer /tmp/od_mon_write.now";
    #
    # list files to read
    #
    my @files = `$lsfcmd`;
    foreach my $file (@files) {
	# 
	# read files
	#
	chomp $file;
	if ($debug >= 1) {
	    print "$ME   Analyzing $file\n";
	}
	open IN, $file;
	my @buffer = <IN>;
	close IN;
	#
	# clear hashes
	#
	my %word1;
	my %word2;
	for (keys %word1) {
	    delete $word1{$_};
	}
	for (keys %word2) {
	    delete $word2{$_};
	}
	my $timestamp;
	foreach my $line (@buffer) {
	    #
	    # analyze eache line of the file
	    #
	    if ($debug >= 2) {
		print "$ME      " . $line;
	    }
	    chomp $line;
	    if ($line =~ m/TimeStamp=/) {
		#
		# get the timestamp and reformat it
		#
		(my $dummy, $timestamp) = split / +/, $line;
		$timestamp =~ s/h/:/;
		$timestamp =~ s/m/:/;
		$timestamp =~ s/s/:/;
		$timestamp =~ s/_/-/g;
		$timestamp =~ s/.$/\'/;
		$timestamp =~ s/-([0-9]{2}):/ \1:/;
		print OUT "/* $fed $file $timestamp */\n";
	    } elsif ($line =~ m/^(.[1-9])/) {
		#
		# get CCU and 2 words if the line starts with a number 
		#
		$line =~ s/^ +//;
		(my $ccu, my $w1, my $w2) = split / +/, $line;
		$word1{$ccu} = hex($w1);
		$word2{$ccu} = hex($w2);
	    } 
	}
	#
	# get the current date and time in the plain format
	#
	my $daytime = $timestamp;
	$daytime =~ s/( |-|:)//g;
	my $mafter = $after;
	$mafter =~ s/( |-|:)//g;
	if ((($after) && ($daytime > $mafter)) || 
	    (!$after)) {
	    foreach my $key (keys %word1) {
		#
		# loop on all CCU's 
		#
		if ($debug >= 2) {
		    print "$ME     key: $key\n"; 
		}
		if (($key != 0) && ($key != 71)) {
		    #
		    # special CCU numbers
		    #
		    my $ccu = $key;
		    my $w20 = $word2{0}; # this is the DAQ state when data got
		    my $w171 = $word2{71}; # ccs board status 1 
		    my $w271 = $word2{71}; # ccs board status 2
		    my $w2k = $word2{$key};
		    #
		    # not all data have a third column, i.e. a 2nd word
		    #
		    if (length($w20) == 0) {
			$w20 = 0; 
		    } 
		    if (length($w271) == 0) {
			$w271 = 0;
		    } 
		    if (length($w2k) == 0) {
			$w2k = 0;
		    } 
		    if (length($w171) == 0) {
			$w171 = 0;
		    }
		    #
		    # build SQL string
		    #
		    $sql = "INSERT INTO OD_MON_DAT VALUES (" .
			"TO_DATE('" . $timestamp . 
			", 'YYYY-MM-DD HH24:MI:SS'), " .
			"(SELECT LOGIC_ID FROM CHANNELVIEW WHERE ID1 = " .
			$fed . " AND ID2 = " . $ccu . 
			" AND NAME = '" . $prefix . "_readout_tower' AND " .
			"NAME = MAPS_TO), " . 
			$w20 . ", " . $word1{$key} . ", " .
			$w2k . ", " . $w171 . ", " .
			$w271 . ", DEFAULT)";
		    #
		    # check if this is a valid channel
		    #
		    my $ch = $fed * 1000 + $ccu;
		    my @found = grep(/$ch/, @validChannels);
		    my $nFound = @found;
		    if ($nFound > 0) {
			if ($auto) {
			    my $sth = $dbh->prepare($sql);
			    $sth->execute();
			} else {
			    print OUT "$sql;\n";
			}
			$nStatements++;
			if (($nmax >= 0) && ($nStatements >= $nmax)) {
			    $dbh->disconnect();
			    exit 0; # not very well structured, in fact...
			}
		    }
		    if ($debug >= 2) {
			print "$ME $sql\n";
			#
			# print just one SQL command
			#
			$debug = 1;
		    }
		    if (($debug >= 0) && ($nFound <= 0)) {
			print "$ME *** IGNORED: ";
			print "$ME $file $fed $ccu --> $ch " . $found[0] . "\n";
		    }
		}
	    }
	} else {
	    if ($debug >= 1) {
		print "$ME     *** SKIP (already in DB)\n";
	    }
	}
    }
}

print "$ME Inserted $nStatements rows in DB\n";

close OUT;
$dbh->disconnect();
exit 0;

sub help() {
    print "Create SQL commands to be used to fill the OD_MON_DAT table\n";
    print "for OMDS. The commands are written into file /tmp/od_mon_dar.sql\n\n";
    print "Usage: od_mon_write.pl [options]\n";
    print "\n";
    print "Options:\n";
    print "help          : show this help\n";
    print "verbose=n     : be verbose. n can be 1 or 2\n";
    print "after=s       : analyze only files created after date s\n";
    print "sm=n          : analyze only files of SM n\n";
    print "\n";
    print "dates must be in the format YYYY-MM-DD HH24:MI:SS\n";
    print "SM n is a string that may contains wildcards (e.g. E*+1*)\n";
}
