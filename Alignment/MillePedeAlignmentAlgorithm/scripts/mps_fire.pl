#!/usr/bin/env perl
#     R. Mankel, DESY Hamburg      3-Jul-2007
#     A. Parenti, DESY Hamburg    21-Apr-2008
#     $Revision: 1.15 $ by $Author: flucke $
#     $Date: 2009/06/24 10:29:47 $
#
#  Submit jobs that are setup in local mps database
#  
#
#  Usage:
#
#  mps_fire.pl [-m[f]] [maxjobs]
#  mps_fire.pl -h

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;

$maxJobs = 1;  # by default, fire one job only
$fireMerge = 0;

$pedeMemDef = 2560; # Default memory allocated for pede: 2560MB=2.5GB
$pedeMemMin = 1024; # Minimum memory allocated for pede: 1024MB=1GB

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
  } else {                # parameters not related to options
    $i = $i + 1;
    if ($i eq 1) {
      $maxJobs = $arg;
    }
  }
}

if ( $helpwanted != 0 ) {
  print "Usage:\n  mps_fire.pl [-m[f]] [maxjobs]";
  print "\nmaxjobs:       Number of Mille jobs to be submitted (default is one)";
  print "\nKnown options:";
  print "\n  -m   Submit the Pede job.";
  print "\n  -mf  Force the submission of the Pede job in case";
  print "\n          some Mille jobs are not in the OK state.\n";
  print "\n  -h   This help.";
  exit 1;
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
    if ($resources =~ "cmscafspec") {  # "cmscafspec" found in $resources: special cmscaf resource
	print "\nWARNING:\n  Running mille jobs on cmscafspec, intended for pede only!\n\n";
        $queue = $resources;
        $queue =~ s/cmscafspec/cmscaf/;
 	$resources = "-q ".$queue." -R cmscafspec";
    } elsif ($resources =~ "cmscaf") { # "cmscaf" found in $resources
	# g_cmscaf for ordinary caf queues, keeping 'cmscafspec' free for pede jobs: 
 	$resources = "-q ".$resources." -m g_cmscaf";
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
		print "bsub -J $theJobName -R\"type==SLC5_64 || type==SLC4_64\" $resources $theJobData/@JOBDIR[$i]/theScript.sh\n";
		$result = `bsub -J $theJobName -R\"type==SLC5_64 || type==SLC4_64\" $resources $theJobData/@JOBDIR[$i]/theScript.sh`;
		print "      $result";
		chomp $result;
		$nn = ($result =~ m/Job \<(\d+)\> is submitted/);
		if ($nn eq 1) {
		    # need standard format for job number
		    @JOBSTATUS[$i] = "SUBTD";
		    ## @JOBID[$i] = $1;
		    @JOBID[$i] = sprintf "%07d",$1;
		    ## print "jobid is @JOBID[$i]\n";
		} else {
		    $jid = $i + 1;
		    print "Submission of $jid seems to have failed: $result\n";
		}
		++$nSub;
	    }
	}
    }
} else {
    # fire the merge job
    print "fire merge\n";
    # set the resource string coming from mps.db
    $resources = get_class("pede");
    if ($resources =~ "cmscafspec") {  # "cmscafspec" found in $resources: special cmscaf resource
        $queue = $resources;
        $queue =~ s/cmscafspec/cmscaf/;
 	$resources = "-q ".$queue." -R cmscafspec";
    } else {
	$resources = "-q ".$resources;
    }

# Allocate memory for pede job
    if ($pedeMem =~ /\D/ || $pedeMem < $pedeMemMin) {
        print "Memory request (".$pedeMem.") non-digit or < ".$pedeMemMin.", use ".$pedeMemDef."\n";
	$pedeMem = $pedeMemDef;
    }
    $resources = $resources." -R \"rusage[mem=".$pedeMem."]\"";

    # check whether all other jobs OK
    $mergeOK = 1;
    for ($i = 0; $i < $nJobs; ++$i) {
	if (@JOBSTATUS[$i] ne "OK") {
	    $mergeOK = 0;
	    break;
	}
    }

    $i = $nJobs;
    while ($i < @JOBDIR) { # loop on all possible merge jobs (usually just 1...)
      if (@JOBSTATUS[$i] ne "SETUP") {
        print "Merge job $i status @JOBSTATUS[$i] not submitted.\n";
      } elsif ($mergeOK!=1 && $forceMerge!=1) {
        print "Merge job not submitted since Mille jobs error/unfinished (Use -mf to force).\n";
      } else {
        if ($forceMerge==1) { # force option invoked
	  # Make first a backup copy of the script
          if (!(-e "$theJobData/@JOBDIR[$i]/theScript.sh.bak")) {
	    system "cp -p $theJobData/@JOBDIR[$i]/theScript.sh $theJobData/@JOBDIR[$i]/theScript.sh.bak";
       	  }
          # Get then the name of merge cfg file (-> $mergeCfg)
          $mergeCfg = `cat $theJobData/@JOBDIR[$i]/theScript.sh.bak | grep cmsRun | grep "\.py" | head -1 | awk '{gsub("^.*cmsRun ","");print \$1}'`;
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
        } else {
          # Restore the backup copy of the script
          if (-e "$theJobData/@JOBDIR[$i]/theScript.sh.bak") {
            system "cp -pf $theJobData/@JOBDIR[$i]/theScript.sh.bak $theJobData/@JOBDIR[$i]/theScript.sh";
          }
          # Then get the name of merge cfg file (-> $mergeCfg)
          $mergeCfg = `cat $theJobData/@JOBDIR[$i]/theScript.sh | grep cmsRun | grep "\.py" | head -1 | awk '{gsub("^.*cmsRun ","");print \$1}'`;
          $mergeCfg = `basename $mergeCfg`;
          $mergeCfg =~ s/\n//;

          # And finally restore the backup copy of the cfg
          if (-e "$theJobData/@JOBDIR[$i]/$mergeCfg.bak") {
            system "cp -pf $theJobData/@JOBDIR[$i]/$mergeCfg.bak $theJobData/@JOBDIR[$i]/$mergeCfg";
          }
        } # end of 'else' from if($forceMerge)

        print "bsub -J almerge -R\"type==SLC5_64 || type==SLC4_64\" $resources $theJobData/@JOBDIR[$i]/theScript.sh\n";
        $result = `bsub -J almerge -R\"type==SLC5_64 || type==SLC4_64\" $resources $theJobData/@JOBDIR[$i]/theScript.sh`;
        print "     $result";
        chomp $result;
        $nn = ($result =~ m/Job \<(\d+)\> is submitted/);
        if ($nn eq 1) {
          # need standard format for job number
          @JOBSTATUS[$i] = "SUBTD";
          ## @JOBID[$i] = $1;
          @JOBID[$i] = sprintf "%07d",$1;
          print "jobid is @JOBID[$i]\n";
        } else {
          print "Submission of merge job seems to have failed: $result\n";
        }
      }
      ++$i;  
    } # end while on merge jobs

}
write_db();
