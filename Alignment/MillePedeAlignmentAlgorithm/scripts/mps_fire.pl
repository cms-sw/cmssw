#!/usr/local/bin/perl
#     R. Mankel, DESY Hamburg      3-Jul-2007
#     A. Parenti, DESY Hamburg    21-Apr-2008
#     $Revision: 1.5 $
#     $Date: 2008/05/02 13:13:08 $
#
#  Submit jobs that are setup in local mps database
#  
#
#  Usage:
#
#  mps_fire.pl [-m[f]] [maxjobs]

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;

$maxJobs = 1;  # by default, fire one job only
$fireMerge = 0;
# parse the arguments
while (@ARGV) {
  $arg = shift(ARGV);
  if ($arg =~ /\A-/) {  # check for option 
    if ($arg =~ "h") {
      $helpwanted = 1;
    }
    if ($arg =~ "m") {
      $fireMerge = 1;
      if ($arg =~ "f") {
# Run merge job even if some mille job are not "OK"
        $forceMerge = 1;
      }
    }
    elsif ($arg =~ "u") {
      $updateDb = 1;
    }
    $optionstring = "$optionstring$arg";
  }
  else {                # parameters not related to options
    $i = $i + 1;
    if ($i eq 1) {
      $maxJobs = $arg;
    }
  }
}

read_db();

# build the absolute job directory path (needed by mps_script)
$thePwd = `pwd`;
chomp $thePwd;
$theJobData = "$thePwd/jobData";


if ($fireMerge == 0) {
    # fire the "normal" parallel jobs
    # set the resource string coming from mps.db
    $resources = get_class("mille");
    if ($resources eq "cmscafspec") {  # special cmscaf resource
	print "\nWARNING:\n  Running mille jobs on cmscafspec, intended for pede only!\n\n";
	$resources = "-q cmscaf -R ".$resources;
    } else {
	$resources = "-q ".$resources;
    }

    # set the job name
    $theJobName = "mpalign";
    if ($addFiles ne "") { $theJobName = $addFiles; }

    $nSub = 0;
    for ($i = 0; $i < $nJobs; ++$i) {
	if (@JOBSTATUS[$i] eq "SETUP") {
	    if ($nSub < $maxJobs) {
		# for some reasons LSF wants script with full path
		print "bsub -J $theJobName $resources $theJobData/@JOBDIR[$i]/theScript.sh\n";
		$result = `bsub -J $theJobName $resources $theJobData/@JOBDIR[$i]/theScript.sh`;
		print "      $result";
		chomp $result;
		$nn = ($result =~ m/Job \<(\d+)\> is submitted/);
		if ($nn eq 1) {
		    # need standard format for job number
		    @JOBSTATUS[$i] = "SUBTD";
		    ## @JOBID[$i] = $1;
		    @JOBID[$i] = sprintf "%07d",$1;
		    ## print "jobid is @JOBID[$i]\n";
		}
		else {
		    $jid = $i + 1;
		    print "Submission of $jid seems to have failed: $result\n";
		}
		++$nSub;
	    }
	}
    }
}
else {
    # fire the merge job
    print "fire merge\n";
    # set the resource string coming from mps.db
    $resources = get_class("pede");
    if ($resources eq "cmscafspec") {  # special cmscaf resource
	$resources = "-q cmscaf -R ".$resources;
    } else {
	$resources = "-q ".$resources;
    }

    # check whether all other jobs OK
    $mergeOK = 1;
    for ($i = 0; $i < $nJobs; ++$i) {
	if (@JOBSTATUS[$i] ne "OK") {
	    $mergeOK = 0;
	    break;
	}
    }

    $i = $nJobs;
    if (@JOBSTATUS[$i] ne "SETUP") {
      print "Merge job $i status @JOBSTATUS[$i] not submitted.\n";
    } elsif ($mergeOK!=1 && $forceMerge!=1) {
      print "Merge job not submitted since Mille jobs error/unfinished (Use -mf to force).\n";
    } else {
        if ($mergeOk!=1) { # some mille jobs are not OK
	  # Make first a backup copy of the script
	  if (!(-e "$theJobData/@JOBDIR[$i]/theScript.sh.bak")) {
	    system "cp -p $theJobData/@JOBDIR[$i]/theScript.sh $theJobData/@JOBDIR[$i]/theScript.sh.bak";
	  }

	  # Get then the name of merge cfg file (-> $mergeCfg)
          $mergeCfg = `cat $theJobData/@JOBDIR[$i]/theScript.sh.bak | grep cmsRun | grep "\.cfg" | head -1 | awk '{gsub("^.*cmsRun ","");print \$1}'`;
	  $mergeCfg = `basename $mergeCfg`;
          $mergeCfg =~ s/\n//;

	  # And make a backup copy of the cfg
	  if (!(-e "$theJobData/@JOBDIR[$i]/$mergeCfg.bak")) {
	    system "cp -p $theJobData/@JOBDIR[$i]/$mergeCfg $theJobData/@JOBDIR[$i]/$mergeCfg.bak";
	  }

          # Rewrite the mergeCfg, using only "OK" jobs
          system "mps_merge.pl -c $theJobData/@JOBDIR[$i]/$mergeCfg.bak $theJobData/@JOBDIR[$i]/$mergeCfg $theJobData/@JOBDIR[$i] $nJobs";

          # Rewrite theScript.sh, using only "OK" jobs
	  system "mps_scriptm.pl -c $mergeScript $theJobData/@JOBDIR[$i]/theScript.sh $theJobData/@JOBDIR[$i] $mergeCfg $nJobs $mssDir";
	  exit;
        }

	print "bsub -J almerge $resources $theJobData/@JOBDIR[$i]/theScript.sh\n";
	$result = `bsub -J almerge $resources $theJobData/@JOBDIR[$i]/theScript.sh`;
	print "     $result";
	chomp $result;
	$nn = ($result =~ m/Job \<(\d+)\> is submitted/);
	if ($nn eq 1) {
	    # need standard format for job number
	    @JOBSTATUS[$i] = "SUBTD";
	    ## @JOBID[$i] = $1;
	    @JOBID[$i] = sprintf "%07d",$1;
	    print "jobid is @JOBID[$i]\n";
	}
	else {
	    print "Submission of merge job seems to have failed: $result\n";
	}
    }

}
write_db();
