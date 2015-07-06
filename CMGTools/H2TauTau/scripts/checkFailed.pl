#!/usr/bin/perl
# -w

$taskdir=shift;
$gzipfiles=shift;
$resubmit=shift;
print "checkFailed.pl will check for failed jobs in $taskdir\n";

opendir(DIR,$taskdir) or die "can't opendir ${taskdir} $!";
closedir(DIR);
@executefiles = `ls ${taskdir} | grep Job_ `;
$njobs=@executefiles;
if($njobs<1){ die "no jobs found in ${taskdir} $!";}

$ngoodfiles=0;
$totalfiles=0;
$ngoodevents=0;
$ntotpassedevents=0;
$nresubmitted=0;
foreach $job (@executefiles){
	chomp($job);
	$logfile=`find $taskdir/$job/ | grep STDOUT `;	
	chomp($logfile);


	$success=0; 
	$exitcode="";
	$execemption=0;
	$nevents=0;
	$npassedevents=0;	
	$logexists=`[ -f ${logfile} ] && echo 1 || echo 0`;
	if($logexists==1){

	    $gzip=0;
	    @logfiletype=split("STDOUT",$logfile);
	    if($logfiletype[1] eq ".gz"){
		system("gunzip $logfile");
		$logfile=`find $taskdir/$job/ | grep STDOUT `;	
		chomp($logfile);
		$gzip=1;
	    }

	    $success=1; 
	    open FILE, "< $logfile";	
	    while($buffer=<FILE>){
		chomp $buffer;
		#print "$buffer \n";
		@exitword=split(" ",$buffer);
		

		if( $exitword[0] eq "cms::Exception" and $execemption==0){
		    $success=0;
		    $exitcode="${exitcode} , cms::Exception";
		    $execemption=1;
		}	
		if( $exitworkd[0] eq "Problem" and $exitworkd[1] eq "with" and $exitworkd[0] eq "configuration" and $exitworkd[0] eq "file"){
		    $success=0;
		    $exitcode="${exitcode} , configuration";
		}
		if( $exitword[0] eq "----" and 	$exitword[1] eq "ProductNotFound"){
		    $success=0;
		    $exitcode="${exitcode} , ProductNotFound";
		}
		if( $exitword[0] eq "std::bad_alloc" and $exitword[1] eq "exception"){
		    $success=0;
		    $exitcode="${exitcode} , std::bad_alloc";
		}	


		############look for bad collections
		if($exitword[0] eq "RootInputFileSequence::initFile():" and $exitword[1] eq "Input" and $exitword[2] eq "file"
		   and $exitword[4] eq "was" and $exitword[5] eq "not"  and $exitword[6] eq "found,"){
		    print "bad collection : $exitword[3] \n";
		    $badcolletionlist[$badcollectionindex]=$exitword[3];
		    $badcollectionindex++;
		}

		##########check number of processed events
		if($exitword[0] eq "TrigReport" and $exitword[1] eq "Events" and $exitword[2] eq "total" and $exitword[5] eq "passed" and $exitword[8] eq "failed" ){
		    $nevents = $exitword[4];
		    $npassedevents=$exitword[7];
		}				
	    }
	    close FILE;
	    
	    if($nevents < 2){ $success=0; $exitcode="${exitcode} EventsProcessed"; $noevents++;}
	    
	    ##gzip the log file only if it was previously gzipped otherwise job may still be running
	    if($gzip==1 || $gzipfiles==1){
		system("gzip $logfile");
	    }
	    
	}

	

	############
	if($success==1){
	    $ngoodfiles++;
	}	
	if($success==0){
	    print "${taskdir}/${job} : Failed : EventsPassed $npassedevents/$nevents : $exitcode \n";
	    print "$logfile\n";

	    if($resubmit==1){
		$pwd=`pwd`;
		chomp($pwd);
		chdir("./${taskdir}/${job}/") or die "Cant chdir to ./${taskdir}/${job}/ $!";		
		system("rm -rf LSFJOB_*");
		system("bsub -q 2nd -J RESUB < ./batchScript.sh");
		chdir("$pwd");
	    }

	}		
	$totalfiles++;
	$ngoodevents+=$nevents; 
	$ntotpassedevents+=$npassedevents;

}    

print "checkFailed.pl Summary : files: $ngoodfiles/$totalfiles, events = $ntotpassedevents/$ngoodevents \n";
