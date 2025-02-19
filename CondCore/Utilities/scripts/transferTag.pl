#!/usr/bin/env perl

##########################################################################
#
# transferTag.pl
# written by Giovanni.Organtini@roma1.infn.it
#
# generates commands to transfer tags from a db to another
#
##########################################################################

use Getopt::Long;

# initialization
$help;
$account = 'oracle://cms_orcoff_prep/CMS_COND_ECAL';
$tag     = 'EcalLaserAPDPNRatios_fit_20110830_160400_175079';
$newtag  = 'EcalLaserAPDPNRatios_fit_20110830_offline';
$listOfIovs = 'EcalLaserAPDPNRatios_fit_20110830_160400_175079.iov_list'; 

GetOptions("help" => \$help, "account=s" => \$account, "source_tag=s" => \$tag,
	   "destination_tag=s" => \$newtag, "list_of_iovs=s" => \$listOfIovs);

if ($help) {
    help();
    exit(0);
}

# check if list of iov's exists
if (!(-e $listOfIovs)) {
    print "File $listOfIovs does not exist!\n";
    print "Create it using the following command, first: \n";
    $cmd = 'cmscond_list_iov -c $account -P /nfshome0/popcondev/conddb -t $tag';
    print "$cmd\n";
    exit(0);
}

# ask green light to the user
print "Generating commands to transfer tag $tag to tag $newtag\n";
print "using account $account and IOV's in file $listOfIovs\n";
print "Proceed [y/n]?";

$yesorno = <STDIN>;
if ($yesorno !=~ /^(y|Y|yes|YES|Yes)/) {
    print "\nAbandoned...\n";
    exit(0);
}    

# get the list of iovs as in the format given by cmscond_list_iov
open IN, $listOfIovs;
@buffer = <IN>;
close IN;

$cmd = 'cmscond_export_iov -s ' . $account . ' -P/nfshome0/popcondev/conddb -i ' . 
    $tag . ' -t ' . $newtag . 
    ' -d oracle://cms_orcon_prod/CMS_COND_42X_ECAL_LAS -b BEGIN -e END  -l sqlite_file:DBLog.db';
$count = 0;
$first_iov = 0;
$last_iov = 0;
foreach $line (@buffer) {
    chomp $line;
    if ($line =~ m/^[0-9]+/) {
	# ignoring lines not containing IOV's
	@row = split / +/, $line;
	if (($count % 100) == 0) {
	    # every 100 rows define the first_iov
	    $first_iov = $row[0];
	    $cmd =~ s/BEGIN/$first_iov/;
	}
	if (($count % 100) == 99) {
	    # after 100 rows define the last_iov
	    $last_iov = $row[0];
	    $cmd =~ s/END/$last_iov/;
	    # print and redefine the command for transfer
	    print "$cmd\n";
	    $cmd = 'cmscond_export_iov -s ' . $account . ' -P/nfshome0/popcondev/conddb -i ' . 
		$tag . ' -t ' . $newtag . 
		' -d oracle://cms_orcon_prod/CMS_COND_42X_ECAL_LAS -b BEGIN -e END  -l sqlite_file:DBLog.db';
	}
	$count++;
    }
}
# finish lines (they may be not a multiple of 100)
$last_iov = $row[0];
$cmd =~ s/END/$last_iov/;
print "$cmd\n";
$cmd = 'cmscond_export_iov -s ' . $account . ' -P/nfshome0/popcondev/conddb -i ' . 
    $tag . ' -t ' . $newtag . 
    ' -d oracle://cms_orcon_prod/CMS_COND_42X_ECAL_LAS -b BEGIN -e END  -l sqlite_file:DBLog.db';

exit(0);

sub help() {
    print "Usage: transferTag.pl [options]\n\n";
    print "help             :shows this help\n";
    print "account          :account string (ex $account)\n";
    print "source_tag       : source tag name (ex $tag)\n";
    print "destination_tag  :destination tag name (ex $newtag)\n";
    print "list_of_iovs     :name of the file containing the list of IOV to transfer (ex $listOfIovs)\n\n";
    print "Options can be shortened (e.g. use -s instead of -source_tag)\n";
    print "Note that the script does not execute commands. It just prompt them on stdout.\n";
}
