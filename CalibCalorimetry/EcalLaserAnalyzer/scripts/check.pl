#!/usr/bin/env perl

# SELMA - Saclay Ecal Laser Monitoring and Analysis
# contacts: Julie Malcles, Gautier Hamel de Monchenault
# last modified: Sat Jul 28 09:58:19 CEST 2007

use Term::ANSIColor;
#use Date::Manip; 
use Cwd;

$firstRun     = @ARGV[0];
$lastRun      = @ARGV[1];
$user         = @ARGV[2];
$nmaxjobs     = @ARGV[3];
$cfgfile      = @ARGV[4];
$ecalPart     = @ARGV[5];
$debug        = @ARGV[6];
$usealsoab    = @ARGV[7];

do "/nfshome0/ecallaser/config/readconfig.pl";
readconfig(${cfgfile});

${MON_CMSSW_REL_DIR}=~ s/\s+//;
${LMF_LASER_PERIOD}=~ s/\s+//;
${MON_OUTPUT_DIR}=~ s/\s+//;
${SORT_WORKING_DIR}=~ s/\s+//;
${MON_CMSSW_CODE_DIR}=~ s/\s+//;
${MON_CALIB_PATH}=~ s/\s+//;

$proddir= "${MON_OUTPUT_DIR}/${LMF_LASER_PERIOD}";
$donefile         = "$proddir/lmfDoneList.txt";
$doneabfile         = "$proddir/lmfDoneABList.txt";
$globalstatusfile = "$proddir/lmfStatusList.txt";
$laserstatusfile  = "$proddir/lmfLaserStatusList.txt";
$tpstatusfile     = "$proddir/lmfTestPulseStatusList.txt";
$ledstatusfile    = "$proddir/lmfLedStatusList.txt";
$scriptdir        = "${MON_CMSSW_CODE_DIR}/scripts";



#  Load some usefull functions
#==============================

do "${scriptdir}/monitoringFunctions.pl";

print color("red"),"\n\n ***** Saclay Ecal Laser Monitoring & Analysis *****\n", color("reset");
print color("red"),     " *****           Checking : ${ecalPart}              *****\n\n", color("reset");

while( 1 ) 
{
    system date;
    doCheck();
    sleep 1;
}

exit;


sub doCheck

{ 
    while(! -e ${donefile}){
	print " No file ${donefile} yet ... sleeping \n";
	sleep 5;
    }

    open( DONEFILE, "$donefile"               ) || die "cannot open $donefile log file \n";
    #open( DONEABFILE, "$doneabfile"               ) || die "cannot open $doneabfile log file \n";
    open( GLOBALSTATUS, ">>$globalstatusfile" ) || die "cannot open file $globalstatusfile\n";
    open( LASERSTATUS, ">>$laserstatusfile"   ) || die "cannot open file $laserstatusfile\n";
    open( LEDSTATUS, ">>$ledstatusfile"       ) || die "cannot open file $ledstatusfile\n";
    open( TPSTATUS, ">>$tpstatusfile"         ) || die "cannot open file $tpstatusfile\n";

    while( <DONEFILE> )
    { 
	chomp ($_);
	my $theLine = $_;
	
	if( $debug == 1 ) {
	    print "$theLine n";
	}
	next unless ( $theLine =~ /(.*)\/Runs\/(.*) START=(.*)_(.*) STOP=(.*)_(.*) NPROC=(.*)/ );
	my $ecalmod=$1;
	my $dirname=$2;
	my $datestart=$3;
	my $timestart=$4;
	my $datedone=$5;
	my $timedone=$6;
	my $nprocdone=$7;
	my $jobdir="${proddir}/${ecalmod}/Runs/${dirname}";
	my $shortjobdir="${ecalmod}/Runs/${dirname}";
	my $procfile="${proddir}/${ecalmod}/Runs/${dirname}/nproc";
	
	my $nproc=0;
	if( -e $procfile) {
	    $nproc=`tail -1 $procfile`;
	    $nproc=$nproc+0;
	}
	my $failed=0;
	
	next unless( $ecalmod =~ /EE(\d*)/ || $ecalmod =~ /EB(\d*)/ );
	next unless( $dirname =~ /Run(\d*)_LB(\d*)/ );

	my $run    = $1;
	my $lb     = $2;
	
	my $fed=getFedNumber($ecalmod);
 
	next unless( doProcessFed($fed, $ecalPart) == 1 );


	next unless ( $run >= $firstRun && $run <= $lastRun ) ;
	
	if( $debug == 1 ) {
	    print "$ecalmod $fed $dirname $run $lb \n";
	}
	
	my $lmfdir       = "${sortdir}/${ecalmod}";
	my $smdir        = "${proddir}/${ecalmod}";
	my $runsdir      = "${proddir}/${ecalmod}/Runs";

	my $laserdir     = "${proddir}/${ecalmod}/Laser";
	my $testpulsedir = "${proddir}/${ecalmod}/TestPulse";
	my $leddir       = "${proddir}/${ecalmod}/LED";
	my $jobdir       = "${runsdir}/${dirname}";

	my $detecteddir     = "${runsdir}/Detected";
	my $analyzeddir     = "${runsdir}/Analyzed";
	my $faileddir       = "${runsdir}/Analyzed/Failed";
	my $statusfile      = "${jobdir}/status.txt";
    
	my $mydate = `date +%s`;

	print " Run ${run} and LB ${lb} found for fed ${fed} (${ecalmod}) at: ${mydate} ${shortjobdir}\n";

	
        # skip if already analyzed
	
	next if(-e "${analyzeddir}/${dirname}");
	next unless(-e "${jobdir}");

	# proceed differently if already processed and failed

	if( -e "${faileddir}/${dirname}" ){
	    
	    my $lastfailedline = `grep  '${shortjobdir}' ${globalstatusfile} | tail -1`;
	    next unless ( $lastfailedline=~ /NPROC=(\d{1})/ );
	    my $nprocfromstat=$1;
	    	    
	    $nproc=$nproc+0;
	    $nprocfromstat=$nprocfromstat+0;
	    
	    # skip if already checked

	    next if( ${nprocfromstat} == ${nproc} );
	    next if( ${nprocdone} == ${nprocfromstat} );
	    next unless (${nprocdone} == ${nproc} );

	    # otherwise, remove failed link
	    
	    system "rm -f ${faileddir}/${dirname}";   

	    if( -e "${laserdir}/Analyzed/Failed/${dirname}" ){
		system "rm -f ${laserdir}/Analyzed/Failed/${dirname}";   
	    }
	    if( -e "${testpulsedir}/Analyzed/Failed/${dirname}" ){
		system "rm -f ${testpulsedir}/Analyzed/Failed/${dirname}";   
	    }
	    if( -e "${leddir}/Analyzed/Failed/${dirname}" ){
		system "rm -f ${leddir}/Analyzed/Failed/${dirname}";   
	    }	    
	}
	
	
	my @reshead=readHeader(${jobdir},${fed});

	my $isHead=@reshead[0];
	my $isLas=@reshead[1];
	my $isLed=@reshead[2];
	my $isTP=@reshead[3];
	my $nLas=@reshead[4];
	my $nLed=@reshead[5];
	my $nTP=@reshead[6];

	my $DumpFile="${shortjobdir} START=${datestart}_${timestart} STOP=${datedone}_${timedone}";
	my $Dump   =$DumpFile;
	my $DumpLas=$DumpFile;
	my $DumpLed=$DumpFile;
	my $DumpTP=$DumpFile;
	
	
	open( STATUSFILE, ">>$statusfile" ) || die "cannot open file $statusfile\n";
	
	print STATUSFILE " ${shortjobdir}\n";
	print STATUSFILE " START=${datestart}_${timestart}\n";
	print STATUSFILE " STOP=${datedone}_${timedone}\n";
	print STATUSFILE " \n";

	if( $isHead != 1){
	    print " NO HEADER \n";
	    
	    print STATUSFILE  " HEADER_ANALYSIS=FAILED \n";
	    print STATUSFILE  " GLOBAL_STATUS=FAILED\n";
	    system "rm -f ${detecteddir}/${dirname}; ln -sf ../../${dirname} ${faileddir}/${dirname}";   
	    close STATUSFILE;
	    
	    
	    $Dump  ="$Dump HEADER_ANALYSIS=FAILED";
	    
	    my $areThereErrors=0;
	    if( -e "${jobdir}/all.log" ){
		$areThereErrors=checkErrors(${jobdir},"all");
	    }
	    my $error="ERRORS=NO";
	    if( $areThereErrors == 1){
		$error="ERRORS=YES";   
	    }
	    $Dump  ="$Dump ${error} NPROC=${nproc} GLOBAL_STATUS=FAILED";
	    print GLOBALSTATUS "$Dump \n "; 
	    
	    next;
	}else{	    
	    $Dump  ="$Dump HEADER_ANALYSIS=OK";
	    print STATUSFILE " HEADER_ANALYSIS=OK\n";
	    print STATUSFILE " \n";
	}
	
	my $areThereErrors=0;
	if( -e "${jobdir}/all.log" ){
	    $areThereErrors=checkErrors(${jobdir},"all");
	}
	my $error="ERRORS=NO";
	if( $areThereErrors == 1){
	    $error="ERRORS=YES";   
	}
	
	my $globalstatus="OK";
	my $gs=1;
	
	# LASER EVENTS
	#=============

	if( $isLas == 1 ){ 

	    my $analstatus="OK";
	    my $gainstatus="OK";
	    my $timestatus="OK";
	    my $matdata="YES";
	    my $matanalstatus="OK";
	    my $sigstatus="OK";
	    my $status="OK";
	    my $nMat=0;

	    # check matacq
            #--------------
	    
	    my @resmat = checkJob(${jobdir},"all","MATACQ");
	    
	    my $matDone=@resmat[0];
	    $nMat=@resmat[3];

	    if ( $matDone == 0 ){
		$matanalstatus="FAILED";
		$status="FAILED";
	    } 
	    
	    my $checknomat = checkNoType(${jobdir},"all","MATACQ");
	    if ( $checknomat == 1 ){
		$matdata="NO";
	    }
	    
	    # check laser
            #--------------

	    my @reslas = checkJob(${jobdir},"all","LASER");
	    my $lasDone      = @reslas[0];
	    my $lasGainOK    = @reslas[1];
	    my $lasTimeOK    = @reslas[2];
	    my $lasSigOK     = @reslas[3];
	    my $lasRawSig    = @reslas[4];
	
	    if( $lasDone == 1 ){
		
		if ( $lasGainOK == 0 ){
		    $gainstatus="FAILED";
		    $status="FAILED";
		    $globalstatus="FAILED";
		    $gs=0;
		}
		if( $lasSigOK == 0 ){
		    $sigstatus="FAILED";
		    $status="FAILED";
		    $globalstatus="FAILED";
		    $gs=0;
		}
		if ( $lasTimeOK == 0 ){
		    $timestatus="FAILED";
		}
		
		# create analyzed link or failed link
		#-------------------------------------

		if( $status=="OK" ){ 
		    system "ln -sf ../../Runs/${dirname} ${laserdir}/Analyzed/${dirname}";
		}else{
		    system "ln -sf ../../../Runs/${dirname} ${laserdir}/Analyzed/Failed/${dirname}";
		}
	    }else{
		system "ln -sf ../../../Runs/${dirname} ${laserdir}/Analyzed/Failed/${dirname}";
		$analstatus="FAILED";
		$status="FAILED";
		$globalstatus="FAILED";
		$gainstatus="UNKNOWN";
		$timestatus="UNKNOWN";
		$sigstatus="UNKNOWN";
		$lasRawSig="UNKNOWN";

		$gs=0;
	    }
	    
	    
	    # Dump to status file
	    #----------------------
	    
	    
	    print STATUSFILE " LASER_DATA=YES\n";
	    print STATUSFILE " LASER_EVTS=${nLas}\n";
	    print STATUSFILE " LASER_ANALYSIS=${analstatus}\n";
	    print STATUSFILE " LASER_STATUS=${status}\n";
	    print STATUSFILE " LASER_GAIN=${gainstatus}\n";
	    print STATUSFILE " LASER_TIMING=${timestatus}\n";
	    print STATUSFILE " LASER_SIGNAL=${sigstatus}\n";
	    print STATUSFILE " LASER_RAW_MEAN=${lasRawSig}\n";
	    print STATUSFILE " \n";
	    print STATUSFILE " MATACQ_DATA=${matdata}\n";
	    print STATUSFILE " MATACQ_EVTS=${nMat}\n";
	    print STATUSFILE " MATACQ_ANALYSIS=${matanalstatus}\n";
	    print STATUSFILE " \n";
 	    	    
	    $Dump   ="$Dump LASER_DATA=YES";
	    $Dump   ="$Dump LASER_ANALYSIS=${analstatus}";
	    $Dump   ="$Dump MATACQ_DATA=${matdata}";
	    $Dump   ="$Dump MATACQ_ANALYSIS=${matanalstatus}";
	    $Dump   ="$Dump LASER_STATUS=${status}";
	    
	    $DumpLas="$DumpLas LASER_DATA=YES";
	    $DumpLas="$DumpLas LASER_EVTS=${nLas}";
	    $DumpLas="$DumpLas LASER_ANALYSIS=${analstatus}";
	    $DumpLas="$DumpLas LASER_STATUS=${status}";
	    $DumpLas="$DumpLas LASER_GAIN=${gainstatus}";
	    $DumpLas="$DumpLas LASER_TIMING=${timestatus}";
	    $DumpLas="$DumpLas LASER_SIGNAL=${sigstatus}";
	    $DumpLas="$DumpLas LASER_RAW_MEAN=${lasRawSig}";
	    $DumpLas="$DumpLas MATACQ_DATA=${matdata}";
	    $DumpLas="$DumpLas MATACQ_EVTS=${nMat}";
	    $DumpLas="$DumpLas MATACQ_ANALYSIS=${matanalstatus}";
	     
	    print LASERSTATUS "$DumpLas \n "; 
	    
	}else{
	    
	    my $laserdat="UNKNOWN";	    
	    my $checknolas = checkNoType(${jobdir},"all","LASER");
	    
	    if ( $checknolas == 1 ){
		$laserdat="NO";
	    }
	    print STATUSFILE " LASER_DATA=$laserdat\n";
	    $Dump   ="$Dump LASER_DATA=$laserdat";	    
	}
    	
	# LED EVENTS
	#===========

	if( $isLed == 1 ){ 

	    my $analstatus="OK";
	    my $gainstatus="OK";
	    my $timestatus="OK";
	    my $sigstatus="OK";
	    my $status="OK";

	    # check led
	    #-----------

	    my @resled = checkJob(${jobdir},"all","LED");
	    my $ledDone      = @resled[0];
	    my $ledGainOK    = @resled[1];
	    my $ledTimeOK    = @resled[2];
	    my $ledSigOK     = @resled[3];
	    my $ledRawSig    = @resled[4];

	    # create analyzed link or failed link
	    #-------------------------------------

	    if( $ledDone == 1 ){
		
		if ( $ledGainOK == 0 ){
		    $gainstatus="FAILED";
		    $status="FAILED";
		}
		if ( $ledSigOK == 0 ){
		    $sigstatus="FAILED";
		    $status="FAILED";
		}
		if ( $ledTimeOK == 0 ){
		    $timestatus="FAILED";
		}
		
		if( $status=="OK" ){ 
		    system "ln -sf ../../Runs/${dirname} ${leddir}/Analyzed/${dirname}";
		}else{
		    system "ln -sf ../../../Runs/${dirname} ${leddir}/Analyzed/Failed/${dirname}";
		}
	    }else{
		system "ln -sf ../../../Runs/${dirname} ${leddir}/Analyzed/Failed/${dirname}";
		
		$analstatus="FAILED";
		$globalstatus="FAILED";
		$sigstatus="UNKNOWN";
		$gainstatus="UNKNOWN";
		$timestatus="UNKNOWN";
		$status="UNKNOWN";
		$gs=0;
	    }
	    
	    # Dump to status file
	    #----------------------
	    
	    print STATUSFILE " LED_DATA=YES\n";
	    print STATUSFILE " LED_EVTS=${nLed}\n";
 	    print STATUSFILE " LED_ANALYSIS=${analstatus}\n";
	    print STATUSFILE " LED_STATUS=${status}\n";
	    print STATUSFILE " LED_GAIN=${gainstatus}\n";
	    print STATUSFILE " LED_TIMING=${timestatus}\n";
	    print STATUSFILE " LED_SIGNAL=${sigstatus}\n";
	    print STATUSFILE " LED_RAW_MEAN=${ledRawSig}\n";
	    print STATUSFILE " \n";
	    
	    $Dump   ="$Dump LED_DATA=YES";	    
	    $Dump   ="$Dump LED_ANALYSIS=${analstatus}";
	    $Dump   ="$Dump LED_STATUS=${status}";
	    
	    $DumpLed="$DumpLed LED_DATA=YES";
	    $DumpLed="$DumpLed LED_EVTS=${nLed}";	    
	    $DumpLed="$DumpLed LED_ANALYSIS=${analstatus}";
	    $DumpLed="$DumpLed LED_STATUS=${status}";
	    $DumpLed="$DumpLed LED_GAIN=${gainstatus}";
	    $DumpLed="$DumpLed LED_TIMING=${timestatus}";
	    $DumpLed="$DumpLed LED_SIGNAL=${sigstatus}";
	    $DumpLed="$DumpLed LED_RAW_MEAN=${ledRawSig}";
	    
	    print LEDSTATUS "$DumpLed \n "; 

	}else{

	    my $leddat="UNKNOWN";
	    
	    my $checknoled = checkNoType(${jobdir},"all","LED");
	    
	    if ( $checknoled == 1 ){
		$leddat="NO";
	    }

	    print STATUSFILE " LED_DATA=${leddat}\n";
	    print STATUSFILE " \n";
	    
	    $Dump   ="$Dump LED_DATA=${leddat}";	
	    
	}
	
	# TESTPULSE EVENTS
	#==================

	if( $isTP == 1 ){ 

	    my $analstatus="OK";

	    # check tp   
	    #----------

	    my @restp = checkJob(${jobdir},"all","TESTPULSE");
	    my $TPDone      = @restp[0];
	    
	    if( $TPDone == 1 ){
		system "ln -sf ../../Runs/${dirname} ${testpulsedir}/Analyzed/${dirname}";
	    }else{	
		system "ln -sf ../../../Runs/${dirname} ${testpulsedir}/Analyzed/Failed/${dirname}";
		$analstatus="FAILED";
		$globalstatus="FAILED";
		$gs=0;
	    }

	    # Dump to status file
	    #----------------------
	    
	    print STATUSFILE " TESTPULSE_DATA=YES\n";
	    print STATUSFILE " TESTPULSE_EVTS=${nTP}\n";
 	    print STATUSFILE " TESTPULSE_ANALYSIS=${analstatus}\n";
	    print STATUSFILE " TESTPULSE_STATUS=${analstatus}\n";
	    print STATUSFILE " \n";

	    $Dump   ="$Dump TESTPULSE_DATA=YES";
	    $Dump   ="$Dump TESTPULSE_ANALYSIS=${analstatus}";
	    $Dump   ="$Dump TESTPULSE_STATUS=${analstatus}";

	    $DumpTP   ="$DumpTP TESTPULSE_DATA=YES";
	    $DumpTP   ="$DumpTP TESTPULSE_EVTS=${nTP}\n";
	    $DumpTP   ="$DumpTP TESTPULSE_ANALYSIS=${analstatus}";
	    $DumpTP   ="$DumpTP TESTPULSE_STATUS=${analstatus}";

	    print TPSTATUS "$DumpTP \n "; 
	    
	}else{

	    my $tpdat="UNKNOWN";
	    
	    my $checknotp = checkNoType(${jobdir},"all","TESTPULSE");
	    
	    if ( $checknotp == 1 ){
		$tpdat="NO";
	    }

	    print STATUSFILE " TESTPULSE_DATA=${tpdat}\n";
	    print STATUSFILE " \n";
	    
	    $Dump   ="$Dump TESTPULSE_DATA=${tpdat}";	
	    
	}
	
	print STATUSFILE " GLOBAL_STATUS=${globalstatus}\n";
	$Dump="$Dump ${error} NPROC=${nproc} GLOBAL_STATUS=${globalstatus}";
	
	
	# create global links
	#=====================

	# remove detected link
	system "rm -f ${detecteddir}/${dirname}";

	# create analyzed or failed links
	if( $gs == 1 ){
	    system "ln -sf ../${dirname} ${analyzeddir}/${dirname}";
	}else{
	    system "ln -sf ../../${dirname} ${faileddir}/${dirname}";
	}
	
	print GLOBALSTATUS "$Dump \n ";
	close STATUSFILE;
    }
    close TPSTATUS;
    close LEDSTATUS;
    close LASERSTATUS;
    close GLOBALSTATUS;
    close DONEFILE;

}


sub readHeader
{
    my $dir    = $_[0];
    my $fed    = $_[1];

    my $isHead=0;
    my $isLas=0;
    my $isLed=0;
    my $isTP=0;
    my $nLas=0;
    my $nLed=0;
    my $nTP=0;

    my $type="";

    my ${headerfile}="${dir}/header.txt";
    
    if( -e ${headerfile} ){
	if (open( HEADERFILE, "${headerfile}")){
	    
	    while (<HEADERFILE>)
	    {
		chomp($_);
		my $theLine = $_;
		if( $theLine =~ /... header done/ ){
		    $isHead=1;
		}
		if( $theLine =~ /RUNTYPE = (\d*)/ )
		{
		    if ( $1 eq 0 ) 
		    { $type = "COSMIC"; } 
		    elsif ( $1 eq 4 )
		    { $type = "LASER_STD"; } 
		    elsif ( $1 eq 5 )
		    { $type = "LASER_POWER_SCAN"; } 
		    elsif ( $1 eq 6 )
		    { $type = "LASER_DELAY_SCAN"; } 
		    elsif ( $1 eq 7 ) 
		    { $type = "TESTPULSE_SCAN_MEM"; }
		    elsif ( $1 eq 8 ) 
		    { $type = "TESTPULSE_MGPA"; }
		    elsif ( $1 eq 9 ) 
		    { $type = "PEDESTAL_STD"; }
		    elsif ( $1 eq 10 ) 
		    { $type = "PEDESTAL_OFFSET_SCAN"; }
		    elsif ( $1 eq 11 ) 
		    { $type = "PEDESTAL_25NS_SCAN"; }
		    elsif ( $1 eq 12 ) 
		    { $type = "LED_STD"; }
		    elsif ( $1 eq 16 ) 
		    { $type = "LASER_GAP"; }
		    elsif ( $1 eq 17 ) 
		    { $type = "TESTPULSE_GAP"; }
		    elsif ( $1 eq 18 ) 
		    { $type = "PEDESTAL_GAP"; }
		    elsif ( $1 eq 19 ) 
		    { $type = "LED_GAP"; }
		    next;
		}
		
		if( $theLine =~ /FEDID = (\d*)/ )
		{
		    my $fedid = $1;
		    if($fed != $fedid){
			print " Problems matching fedid from name and file: $fed $fedid \n";
		    }
		    
		    if ( $type =~ /LASER/ ){
			$isLas=1;
		    }
		    if ( $type =~ /TESTPULSE/ ){
			$isTP=1;
		    }
		    if ( $type =~ /LED/ ){
			$isLed=1;
		    }
		}
		if( $theLine =~ /LASER EVENTS = (\d*)/ ){
		    $nLas=$1;
		}
		if( $theLine =~ /LED EVENTS = (\d*)/ ){
		    $nLed=$1;
		}
		if( $theLine =~ /TESTPULSE EVENTS = (\d*)/ ){
		    $nTP=$1;
		}
	    }
	    close HEADERFILE;
	}
    }else{
	print "cannot open header file ${headerfile} in readHeader \n";
    }
    
    my @output;
    push(@output, $isHead);
    push(@output, $isLas);
    push(@output, $isLed);
    push(@output, $isTP);
    push(@output, $nLas);
    push(@output, $nLed);
    push(@output, $nTP);
    
    return @output;
}

sub checkErrors {

    my $dir=$_[0];
    my $logname=$_[1];

    print "Check errors in data analysis\n";

    my $file = "${dir}/${logname}.log";
    
    my $errors = 0;

    
    open ( FILE, $file) || die "cannot open $file log file \n";
    while (<FILE>)
    {
	chomp ($_);
	$theLine = $_;
	if( $theLine =~ /ERROR/ || $theLine =~ /error/ || $theLine =~ /Error/
	    || $theLine =~ /segmentation/ ||  $theLine =~ /Segmentation/  )
	{
	    $errors = 1;
	}	
    }
    
    return $errors;
}

sub checkJob {

    my $dir=$_[0];
    my $logname=$_[1];
    my $type =$_[2]; # LASER, LED, TESTPULSE, MATACQ

    print "Check result of ${type} data analysis\n";

  
    my $file = "${dir}/${logname}.log";
    
    my $analyze      = 0;
    my $done         = 0;
    my $gainok       = 1;
    my $timeok       = 1;
    my $sigok        = 1;
    my $rawsig;
    
    open ( FILE, $file) || die "cannot open $file log file \n";
    while (<FILE>)
    {
	chomp ($_);
	$theLine = $_;
	if( $theLine =~ /\*\*\*  No ${type} Events  \*\*\*/ )
	{
	    print "$theLine\n";
	}	
	if( $theLine =~ /\+=\+/ )
	{
	    print "$theLine\n";
	    if( $analyze == 1 )
	    {
		$done = 1;
	    }
	    else
	    {
		if( $analyze == 2 && $theLine =~ /Analyzing/ ){
		    $analyze = 0;
		}
		if( $analyze == 2 && $theLine =~ /... done/ ){
		    $analyze = 1;
		}
		if( $analyze == 2 && $theLine =~ /events:(\s{1})(\S*)(\s{2})\+=\+/){
		    $sigok=$2;
		}
		if( $analyze == 2 && $theLine =~ /APD GAIN WAS NOT 1/ ){
		    $gainok=0;
		}
		if( $analyze == 2 && $theLine =~ /TIMING WAS BAD/ ){
		    $timeok=0;
		}
		if( $analyze == 2 && $theLine =~ /LIGHT SIGNAL WAS BAD/ ){
		    $sigok=0;
		}
		if( $analyze == 2 && $theLine =~ /MEAN RAW SIG: (\S*)/ ){
		    print "here test $1 \n" ;
		    $rawsig=$1;   
		}
		
		if( $theLine =~ /Analyzing ${type} data/ ){
		    
		    if( $theLine =~ /alpha/ || $theLine =~ /beta/ ){
			$analyze = 0;
		    }else{
			$analyze = 2;
		    }
		}		
	    }
	}
    }
    
    my @output;
    push(@output, $done);
    push(@output, $gainok);
    push(@output, $timeok);
    push(@output, $sigok);
    push(@output, $rawsig);
    
    return @output;
}

sub checkNoType {

    my $dir=$_[0];
    my $logname=$_[1];
    my $type =$_[2]; # LASER, LED, TESTPULSE, MATACQ

    
    my $file = "${dir}/${logname}.log";
    
    my $notypeok     = 0;


    
    open ( FILE, $file) || die "cannot open $file log file \n";
    while (<FILE>)
    {
	chomp ($_);
	$theLine = $_;
	if( $theLine =~ /\*\*\*  No ${type} Events  \*\*\*/ )
	{
	    print "$theLine\n";
	    $notypeok     = 1;
	}	
    }
    return $notypeok;
}
