# remove this line before using
#!/usr/bin/perl

# od_mon_write.pl
# Written by Giovanni.Organtini@roma1.infn.it
#
# Read OD monitoring data as text files and build SQL commands to
# fill the DB table OD_MON_DAT.
# The file containing the SQL commands is /tmp/od_mon_dat.sql 
#
# Usage: run this script according to instructions, then 
#        log on DB sqlplus cms_ecal_cond@cms_omds_lb and 
#        issue the command > @/tmp/od_mon_dat
# 
# 2012 vers. 1.0

use Getopt::Long;

# 
# get options
#
my $help;
my $debug = 0;
my $sm;
my $after;
$result = GetOptions("verbose=n" => \$debug, "after=s" => \$after, 
		     "debug=n" => \$debug, "sm=s" => \$sm, "help" => \$help);

if ($help) {
    help();
    exit 0;
}
#
# check format
#
`/bin/touch -d '1970-01-01 00:00:00' /tmp/od_mon_write.now`;
if ($after) {
    if ($after !~ m/[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}/) {
	print "ERROR: date format not recognized (use YYY-MM-DD HH24:MI:SS)\n";
	exit -1;
    }
    $cmd = "/bin/touch -d '" . $after . "' /tmp/od_mon_write.now";
    `$cmd`;
}

if (($debug < 0) || ($debug > 2)) {
    print "ERROR: verbose level can be either 0, 1 or 2.\n";
}
#
# prepare
#
open VL, "/nfshome0/vlassov/delays/endcap/fibmaptr.txt";
@buffer = <VL>;
close VL;
@validChannels;
foreach $line (@buffer) {
    chomp $line;
    if ($line =~ m/^6/) {
	@l = split / +/, $line;
	$n = @l;
	$fed = $l[0];
	for ($i = 11; $i <$n; $i++) {
	    $ccu = $l[$i];
	    $ch = $fed * 1000 + $ccu;
	    if ($debug >= 2) {
		print "Found valid channel: $ch\n"; 
	    }
	    push @validChannels, $ch;
	}
    }
}

#
# list directories
#
open OUT, ">/tmp/od_mon_dat.sql";
$lscmd = "ls -d1 /data/ecalod-disk01/dcu-data/ccs-data/*";
$slscmd = $lscmd;
if ($sm) {
    $slscmd .= "/" . $sm;
}
@dirs = `$slscmd`;
foreach $dir (@dirs) {
    #
    # list files
    #
    if ($debug >= 1) {
	print "$dir\n";
    }
    chomp $dir;
    $fed =$dir;
    $fed =~ s/^.*E(B|E)//; # remove the first characters and get the FED number
    $prefix = "EB";
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
    $lsfcmd = "/usr/bin/find $dir -type f -cnewer /tmp/od_mon_write.now";
    @files = `$lsfcmd`;
    foreach $file (@files) {
	# 
	# read files
	#
	chomp $file;
	if ($debug >= 1) {
	    print "  $file\n";
	}
	open IN, $file;
	@buffer = <IN>;
	close IN;
	#
	# clear hashes
	#
	%word1;
	%word2;
	for (keys %word1) {
	    delete $word1{$_};
	}
	for (keys %word2) {
	    delete $word2{$_};
	}
	$timestamp;
	foreach $line (@buffer) {
	    #
	    # analyze eache line of the file
	    #
	    if ($debug >= 2) {
		print "     " . $line;
	    }
	    chomp $line;
	    if ($line =~ m/TimeStamp=/) {
		#
		# get the timestamp and reformat it
		#
		($dummy, $timestamp) = split / +/, $line;
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
		($ccu, $w1, $w2) = split / +/, $line;
		$word1{$ccu} = hex($w1);
		$word2{$ccu} = hex($w2);
	    } 
	}
	#
	# get the current date and time in the plain format
	$daytime = $timestamp;
	$daytime =~ s/( |-)//g;
	$mafter = $after;
	$mafter =~ s/( |-)//g;
	if ((($after) && ($daytime > $mafter)) || 
	    (!$after)) {
	    foreach $key (keys %word1) {
		#
		# loop on all CCU's 
		#
		if ($debug >= 2) {
		    print "    key: $key\n"; 
		}
		if (($key != 0) && ($key != 71)) {
		    #
		    # special CCU numbers
		    #
		    $ccu = $key;
		    $w20 = $word2{0}; # this is the DAQ state when data got
		    $w171 = $word2{71}; # ccs board status 1 
		    $w271 = $word2{71}; # ccs board status 2
		    $w2k = $word2{$key};
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
			$w271 . ", DEFAULT);";
		    #
		    # check if this is a valid channel
		    #
		    $ch = $fed * 1000 + $ccu;
		    @found = grep(/$ch/, @validChannels);
		    $nFound = @found;
		    if ($nFound > 0) {
			print OUT "$sql\n";
		    }
		    if ($debug >= 2) {
			print "$sql\n";
			#
			# print just one SQL command
			#
			$debug = 1;
		    }
		    if (($debug >= 0) && ($nFound <= 0)) {
			print "*** IGNORED: ";
			print "$file $fed $ccu --> $ch " . $found[0] . "\n";
		    }
		}
	    }
	} else {
	    if ($debug >= 1) {
		print "    *** SKIP\n";
	    }
	}
    }
}

close OUT;
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
