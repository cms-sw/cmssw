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
$nprocmax     = 1;

do "/nfshome0/ecallaser/config/readconfig.pl";
readconfig(${cfgfile});

${MON_CMSSW_REL_DIR}=~ s/\s+//;
${SORT_WORKING_DIR}=~ s/\s+//;
${MON_OUTPUT_DIR}=~ s/\s+//;
${MON_CMSSW_CODE_DIR}=~ s/\s+//;
${MON_CALIB_PATH}=~ s/\s+//;
${MON_AB_PATH}=~ s/\s+//;

$proddir      = "${MON_OUTPUT_DIR}/${LMF_LASER_PERIOD}";
$sortdir      = "${SORT_WORKING_DIR}/out";

$sortfile     = "${SORT_WORKING_DIR}/lmfFileList.txt";
$sentfile     = "$proddir/lmfSentList.txt";
$donefile     = "$proddir/lmfDoneList.txt";
$doneabfile     = "$proddir/lmfDoneABList.txt";

$scriptdir    = "${MON_CMSSW_CODE_DIR}/scripts";
$templatesdir = "${MON_CMSSW_CODE_DIR}/data/pytemplates";


#  Load some usefull functions
#==============================

do "${scriptdir}/monitoringFunctions.pl";

print color("red"),"\n\n ***** Saclay Ecal Laser Monitoring & Analysis *****\n", color("reset");
print color("red"),     " *****           Analyzing : ${ecalPart}              *****\n\n", color("reset");

while( 1 ) 
{
    system date;
    doDetect();
    sleep 1;
}

exit;


sub doDetect

{ 
    print color("green"), "\n *** Restarting File Detection ***\n\n", color("reset");

    open( SORTED, "$sortfile" )  || die "cannot open $sortfile file \n";
    while( <SORTED> )
    { 
	chomp ($_);
	my $theLine = $_;
	next unless ( $theLine =~ /out\/(.*)\/(.*).lmf/ );

	my $ecalmod=$1;
	my $dirname=$2;
	my $filename="$2.lmf";
	my $filenamezip="$2.lmf.bz2";


	next unless( $ecalmod =~ /EE(\d*)/ || $ecalmod =~ /EB(\d*)/ );

	next unless( $dirname =~ /Run(\d*)_LB(\d*)/ );
	my $run    = $1;
	my $lb     = $2;	

	
	my $fed=getFedNumber( ${ecalmod} );
 
	
	next unless( doProcessFed($fed, $ecalPart) == 1 );
	next unless ( $run >= $firstRun && $run <= $lastRun ) ;
	
	if( $debug == 1 ) {
	    print "$ecalmod $fed $dirname $run $lb $filename $filenamezip \n";
	}

	my $lmfdir       = "${sortdir}/${ecalmod}";
	my $smdir        = "${proddir}/${ecalmod}";
	my $runsdir      = "${proddir}/${ecalmod}/Runs";
	my $laserdir     = "${proddir}/${ecalmod}/Laser";
	my $testpulsedir = "${proddir}/${ecalmod}/TestPulse";
	my $leddir       = "${proddir}/${ecalmod}/LED";
	my $jobdir       = "${runsdir}/${dirname}";
	my $nprocfile="${jobdir}/nproc";  
	 
	next unless(-e "${lmfdir}/${filename}" || -e "${lmfdir}/${filenamezip}");

	my $shortjobdir       = "${ecalmod}/Runs/${dirname}";

	my $mydate = `date +%s`;

	print " Run ${run} and LB ${lb} found for fed ${fed} (${ecalmod}) at: ${mydate} ${shortjobdir} \n";

	# skip if already detected or analyzed
	
	next if(-e "${runsdir}/Detected/${dirname}");
	next if(-e "${runsdir}/Analyzed/${dirname}");

	my $nproc=`tail -1 ${nprocfile}`;
	$nproc=$nproc + 0;
	
	if( $debug == 1 ){
	    print " nproc= $nproc \n";
	}
	
	next if( $nproc >= $nprocmax );
	
	
	# prepare unzip commands if file is zipped!
	
	my $dormzip=0; 
	my $commandunzip;
	my $commandrmzip;
	if( -e "${lmfdir}/$filenamezip" && not -e "${lmfdir}/$filename" ){
	    $dormzip=1;
	    $commandunzip="bzcat ${lmfdir}/${filenamezip} > ${jobdir}/input.lmf";
	    $commandrmzip="rm ${jobdir}/input.lmf";
	}
	if( $debug == 1 ){
	    print " dormzip=$dormzip \n";
	    print " commandunzip=$commandunzip \n";
	    print " commandrmzip=$commandrmzip \n";
	}

	
	if(-e "${runsdir}/Analyzed/Failed/${dirname}"){
	    my $statusfile="${runsdir}/${dirname}/status.txt";
	    my $resend=0;
	    my $nproc=`tail -1 ${nprocfile}`;
	    $nproc=$nproc + 0;
	    
	    if( $debug == 1 ){
		print " nproc= $nproc \n";
	    }

	    next if( $nproc >= $nprocmax );
	    
	    if( -e "${statusfile}" ){
		open( STATUSFILE, "$statusfile" ) || die "cannot open file $statusfile\n";
		while( <STATUSFILE> )
		{ 
		    chomp ($_);
		    my $theStatLine = $_;
		    if( $theStatLine =~ /LASER_ANALYSIS=FAILED/ 
			|| $theStatLine =~ /HEADER_ANALYSIS=FAILED/
			|| $theStatLine =~ /LED_ANALYSIS=FAILED/
			|| $theStatLine =~ /TESTPULSE_ANALYSIS=FAILED/
			)
		    {
			$resend=1;
		    }
		}
		close STATUSFILE;
	    }
	    
	    if( $resend == 0 ){
		next;
	    }else{
		
                # delete appropriate links here
		system "rm ${runsdir}/Analyzed/Failed/${dirname}";
		
		if( -e "$laserdir/Analyzed/Failed/${dirname}" ){
		    system "rm laserdir/Analyzed/Failed/${dirname}";
		}
		if(-e "$leddir/Analyzed/Failed/${dirname}" ){
		    system "rm leddir/Analyzed/Failed/${dirname}";
		}
		if(-e "$testpulsedir/Analyzed/Failed/${dirname}" ){
		    system "rm testpulsedir/Analyzed/Failed/${dirname}";
		}
	    }
	}
	
	my $command;

	if( -e ${smdir} ) 
	{
	    if( $debug == 1 ) {
		print "${smdir} exists.\n";
	    }
	}else{
	    print "creating directory ${smdir}\n";
	    $command =  "${scriptdir}/createDirs.csh ${proddir} ${ecalmod}" ;
	    system ${command};
	}
	
	if(-e ${jobdir} ){
	    if( $debug == 1 ) {
		print "${jobdir} exists.\n";
	    }
	    # check input exists or dounzip again (for reprocessed jobs)
	    if( not -e "${jobdir}/input.lmf" &&  -e "${jobdir}/input.lmf.bz2"){
		system ${commandunzip};
		$dormzip=1; 
	    }
	    
	}else{
	    
	    print "creating directory ${jobdir}\n";
	    $command =  "mkdir ${jobdir}" ;
	    system ${command};
	    
	    # link to the pool file if it exists
	    if( $dormzip == 1 ){
		system ${commandunzip};
		$command = "ln -sf ${lmfdir}/${filenamezip} ${jobdir}/input.lmf.bz2";
	    }else{
		$command = "ln -sf ${lmfdir}/${filename} ${jobdir}/input.lmf";
	    } 
	    if( $debug == 1 ) {
		print "linking pool command: $command $dormzip \n";
	    }
	    
	    system  ${command}; 
            
	}

	my $mydate = `date +%s`;
	print " ... done at: ${mydate} \n";
	system date;	


        #  Generate cfg file for led, laser shape, testpulse and matacq analysis:
	# ========================================================================
	my $part="EB";
	my $digis="ebDigis";
	if( $ecalmod =~ /EE/ ){
	    $part="EE";
	    $digis="eeDigis";
	}
		
	open( TEMPLATE, "${templatesdir}/all.py" );
	open( CFGFILE, ">${jobdir}/all.py" );
	while( <TEMPLATE> )
	{  
	    $_ =~ s/CCCC/$digis/g;
	    $_ =~ s/PPPP/$part/g;
	    $_ =~ s/FFFF/$fed/g;
	    $_ =~ s/DDDD/$debug/g;
	    $_ =~ s/CALIBPATH/${MON_CALIB_PATH}/g;
	    $_ =~ s/ABPATH/${MON_AB_PATH}/g;
	    print CFGFILE $_;
	    
	}
	close CFGFILE;
	close TEMPLATE;


	if( $usealsoab == 1){
	    
	    #  Generate cfg file for ab laser analysis:
	    # ==========================================
	    my $part="EB";
	    my $digis="ebDigis";
	    if( $ecalmod =~ /EE/ ){
		$part="EE";
		$digis="eeDigis";
	    }
	    
	    open( TEMPLATE, "${templatesdir}/ab.py" );
	    open( CFGFILE, ">${jobdir}/ab.py" );
	    while( <TEMPLATE> )
	    {  
		$_ =~ s/CCCC/$digis/g;
		$_ =~ s/PPPP/$part/g;
		$_ =~ s/FFFF/$fed/g;
		$_ =~ s/DDDD/$debug/g;
		$_ =~ s/CALIBPATH/${MON_CALIB_PATH}/g;
		$_ =~ s/ABPATH/${MON_AB_PATH}/g;
		print CFGFILE $_;
		
	    }
	    close CFGFILE;
	    close TEMPLATE;
	    
	}

        #  Send job: 
	# ===========
	
	my $key="all";
	my $key2="ab";

	my $jobkey="NoKey";
	my $date = `date +'%F_%R:%S'`;
	
	
	my $isItSent=0;
	if( $usealsoab == 1){
	    $isItSent= send2Jobs(${key}, ${key2} , ${nmaxjobs}, ${user}, ${jobdir}, ${scriptdir}, ${donefile}, ${nprocfile}, ${shortjobdir}, ${dormzip} ); 
	}else{
	    $isItSent= sendJob(${key}, ${jobkey} , ${nmaxjobs}, ${user}, ${jobdir}, ${scriptdir}, ${donefile}, ${nprocfile}, ${shortjobdir} , ${dormzip}); 
	}
	
	
	if( $isItSent == 1 ){ print "....... done\n"; }
	
        #  Create links and write to sent file: 
	# =====================================
	
	if(! -e "${runsdir}/Detected/${dirname}"){
	    print " Setting Pointers for ${dirname} \n"; 
	    if( -e "${runsdir}/lastDetected" ) { system "rm -f ${runsdir}/lastDetected"; }
	    system "ln -sf ${dirname} ${runsdir}/lastDetected";
	    system "ln -sf ../${dirname} ${runsdir}/Detected/${dirname}";
	    
	    open( SENTFILE, ">>$sentfile" ) || die "cannot open file $sentfile\n";
	    print SENTFILE  "${shortjobdir} START=$date";
	    close SENTFILE;
	    
	}
    }
    close SORTED;
    
}

