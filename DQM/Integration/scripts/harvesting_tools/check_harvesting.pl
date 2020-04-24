#!/usr/bin/env perl 

###############################################################################
#                                                                             #
#   Script to check the status of DQM Harvesting jobs                         #
#                                                                             #
#   It basically checks whether the root tree is present for all jobs         #
#   If not, it resubmits the failed jobs on request                           #
#                                                                             #
#  Usage: check_harvesting.pl [crab_status_output]                            #
#                                                                             #
#  NOTE: it needs to be run from the directory from which you submitted       #
#        multicrab jobs                                                       #
#                                                                             #
#  Optional argument:                                                         #
#   crab_status_output: file containing output of "multicrab -status" command #
#                       if not given, the script will get it itself           #
#                                                                             #
#  Author:        Vuko Brigljevic, Rudjer Boskovic Institute                  #
#  First Version: January 22, 2010                                            #
#                                                                             #
###############################################################################




use strict;




# Gets the crab status from input or produce it

my $nargs = @ARGV;


my $crabstatus = "";

if ($nargs>0 && -e $ARGV[0] ) {
    $crabstatus= $ARGV[0];
}

if ($crabstatus eq "") {
    $crabstatus = "multicrabstatus.out";
    print "Getting multicrab jobs status ... \n";
    system "multicrab -status > $crabstatus ";
    print "Done \n";
}




# Open the multicrab cfg file to extract the list of job ids
#

my $crabcfg = "multicrab.cfg";

if (! -r $crabcfg ) {
    print "No multicrab.cfg in this directory, quitting... \n";
    exit;
}


open (CRABCFG,$crabcfg);

#my @lines=<STDIN>;
my @lines=<CRABCFG>;

my $line="";

my $tempfile=".check_harvesting_output";

my $castorbase="/castor/cern.ch";

my $job;

my $nsuccess=0;
my $njobs=0;
my $nfailed=0;

my @failedjobs=();

foreach $line (@lines) {

    if ( $line =~ /\[(.*?)\]/ )
    {
	if ( $1 ne "MULTICRAB") {
	    $job = $1;
	    $njobs++;
	}
    }

    if ($line =~ /USER.user_remote_dir/) {

	my @words = split (/ = /,$line);
	chop($words[1]);
	my $castordir=$castorbase.$words[1];  # ADD JOB NAME IN CASTOR DIRECTORY!
	# MODIFICATION: to account for the fact that multicrab adds job name in directory tree for output!!!
	$castordir=$castordir."/".$job;
#	print "CASTOR DIR: $castordir \n";
	# this lines needs to be commented out to account for the above modification
#	chop($castordir); #remove newline at the end

	if ( -e $tempfile ) {
	    system "rm $tempfile";
	}

	my $rfcmd="rfdir $castordir > $tempfile";

	# Check if root file present in output directory
	system $rfcmd ;
	open(LIST,$tempfile);

	my $tmpline;
	my $nrootfiles=0;
	while ($tmpline = <LIST> ) {
	    if ( $tmpline =~ /root/ ) {
		$nrootfiles++;
	    }
	}
	if ($nrootfiles > 0) {
	    $nsuccess++;
	} else {
	    $failedjobs[$nfailed]=$job;
	    $nfailed++;
	}
    }
}



print "Number of jobs                               : $njobs \n";
print "Number of successful jobs (root file present : $nsuccess \n";
print "Number of failed or still running jobs       : $nfailed \n";

my %jobstatus;

##################################################
# Get job status for jobs without root output

foreach $job (@failedjobs) {

    my $jobstatus  = get_multicrab_job_status($job,$crabstatus);
    $jobstatus{$job} = $jobstatus;

    print " $job : $jobstatus \n";
}

if ($nfailed == 0) {
    exit;
}

# Resubmit failed jobs if desired

print "Would you like to resubmit the failed jobs? [y/n] \n";

my $reply=<STDIN>;
chop ($reply);

print "reply: $reply \n";

if ( $reply eq "y" || $reply eq "Y" ) {

    print "Resubmitting jobs... \n";

    foreach $job (@failedjobs) {


	my $status = $jobstatus{$job};

	my $retrieve = 0;
	my $resubmit = 0;

	if ( $status eq "Scheduled" 
	     || $status eq "Pending"
	     || $status eq "Running" ) {
	    print "$job running: leave it in peace... \n";
	} elsif ($status eq "Aborted") {
	    $resubmit = 1;
	} elsif ($status eq "Done") {
	    $retrieve = 1;
	    $resubmit = 1;
	} elsif ($status eq "Retrieved") {
	    $resubmit = 1;
	}

	if ($retrieve) {
	    print "retrieving... \n";
	    system "crab -c $job -getoutput all";
	}
	if ($resubmit) {
	    print "resubmitting... \n";
	    system "crab -c $job -resubmit all";
	}

    }

}





##################################################################################

sub get_multicrab_job_status {
    
    #-------------------------------------------------------------------#
    # Aim: Get the CRAB job status of a multicrab job                   #
    #      by parsing the "multicrab -status" output                    #
    #                                                                   #
    #  two input arguments                                              #
    #      $job         : CRAB job name                                 #
    #      $crabstatus  : file with output of "multicrab -status"       #
    #-------------------------------------------------------------------#


    if ( scalar(@_) != 2 ) {
	print "get_multicrab_job_status() called with wrong number of arguments: @_ \n";
	return "";
    }
    
    my ($job, $crabstatus) = @_ ;

    open(CRABSTATUS,$crabstatus);
   
    my $jobfound=0;
    my $linecounter=0;
    my $jobstatus="undefined";
    while ( <CRABSTATUS>) {
	if ($linecounter > 0) {
	    if ($linecounter == 2) { 
		# This is the line with the job status
		
		my @words = split (" ");
		$jobstatus=$words[1];
		last;
	    }
	    $linecounter++;
	}
	if ($jobfound) {
	    if (/ID/ && /STATUS/ ) {
		$linecounter++
		}
	    next;
	}
	
	if (/$job/) {
	    $jobfound = 1;
	}
    }

    close(CRABSTATUS);

    return $jobstatus;
    
}
