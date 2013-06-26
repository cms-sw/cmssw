#!/usr/bin/env perl


# SELMA - Saclay Ecal Laser Monitoring and Analysis
# contacts: Julie Malcles, Gautier Hamel de Monchenault, Serguei Ganjour, Patrice Verrecchia
# last modified: Sat Jul 28 09:58:19 CEST 2007

use Term::ANSIColor;
#use Date::Manip; 
use Cwd;

$fed         = @ARGV[0];
$ecalmod     = @ARGV[1];
$useMatacq   = @ARGV[2];
$useShape    = @ARGV[3];
$fitAB       = @ARGV[4];
$linkname    = @ARGV[5];
$user        = @ARGV[6];
$nmaxjobstot = @ARGV[7];
$debug       = @ARGV[8];

die unless( $ecalmod =~ /E(E|B)(\+|-)(\d*)/ );

$zone = "Barrel";
$ecalPart="EB";
$digis="ebDigis";
if( $1 eq "E" ) { 
    $zone = "EndCap"; 
    $ecalPart="EE";
    $digis="eeDigis";
}
$side = "positive";
if( $2 eq "-" ) { $side = "negative" };
$sm = $3;

$localdir     = cwd;
if (  $linkname eq "" ) {
    $proddir      = "${localdir}/LaserMonitoring";
}else{
    $proddir      = "${localdir}/${linkname}";
}
$fitab="True";
if($fitAB==0){
    $fitab="False";
}

$sortdir      = "$proddir/sorting";
$scriptdir    = "$proddir/scripts";
$shapedir     = "$proddir/shapes";
$templatesdir = "$proddir/templates";
$runsdir      = "$proddir/${ecalmod}/Runs";
$laserdir     = "$proddir/${ecalmod}/Laser";
$pedestaldir  = "$proddir/${ecalmod}/Pedestal";
$leddir       = "$proddir/${ecalmod}/LED";
$testpulsedir = "$proddir/${ecalmod}/TestPulse";
$mailboxdir   = "$proddir/MailBox";

do "${scriptdir}/monitoringFunctions.pl";

print color("red"), "\n\n***** Saclay Ecal Laser Monitoring & Analysis *****\n\n", color("reset");


while( 1 ) 
{
    system date;
  #  doLaser();
  #  doTestPulse();
    doBoth(); 
    sleep 1;
}

exit;

sub doBoth 
{
    print color("green"), "\n *** Restarting Analyzer for FED: ${fed} ***\n\n", color("reset");
    
    my $detected = "$laserdir/Detected";
    my $analyzed = "$laserdir/Analyzed";
    my $failed   = "$laserdir/Analyzed/Failed";

    my $TPdetected = "$testpulsedir/Detected";
    my $TPanalyzed = "$testpulsedir/Analyzed";
    my $TPfailed   = "$testpulsedir/Analyzed/Failed";

    opendir( DIR, $detected ) || die "cannot open $detected directory\n";
    
    my @dirs = sort  readdir( DIR );
    
    foreach my $dirname (@dirs)
    { 
	# skip if already analyzed

	next if( -e "${laserdir}/Analyzed/${dirname}");
	
	my $isThereTP=0;
	if( -e "$TPdetected/${dirname}") {
	    $isThereTP=1;
	}

	# skip if already analyzed
	
	if( -e "$TPanalyzed/${dirname}"){
	    $isThereTP=0;
	}
	
	if( $dirname =~ /Run(\d*)_LB(\d*)/ )
	{
	    my $run = $1;
	    my $lb = $2;
	    print color("blue"), " Analyzing Laser data for Run: $run, and Lumi Block $lb \n", color("reset");
	    my $mydate = `date +%s`;
	    print " Starting at: ${mydate} \n";
	    system date;	

	    my $jobdir = "${runsdir}/${dirname}";
	    
	    my $statusfile   = "$jobdir/statusLaser.txt";
	    my $TPstatusfile   = "$jobdir/statusTestPulse.txt";
	    my $command;
	    
	    die unless( opendir( CURRENTDIR, ${jobdir} ) );
 
	    open( STATUSFILE, ">>$statusfile" ) || die "cannot open file $statusfile\n";

	    # set pointer to the last detected run
	    if( -e "${laserdir}/lastDetected" ) { system "rm -f ${laserdir}/lastDetected"; }
	    system "ln -sf ../Runs/${dirname} ${laserdir}/lastDetected";

	    if($isThereTP==1){
		
		open( STATUSFILETP, ">>$TPstatusfile" ) || die "cannot open file $TPstatusfile\n";
		
		# set pointer to the last detected run
		if( -e "${testpulsedir}/lastDetected" ) { system "rm -f ${testpulsedir}/lastDetected"; }
		system "ln -sf ../Runs/${dirname} ${testpulsedir}/lastDetected";
	    }
	    
	    # check MATACQ
	    # ==============
	    
	    my $matacqOK=0;
	    my $matacqlog="header";
	    
	    if( $useMatacq == 1 ){
		$matacqOK = checkMatacqJob($dirname, $matacqlog );
	    }
	    my $laserOK2=0;
	    my $laserOK=0;
	    my $TPOK=0;
	    my $ABOK=0;

	    
	    # analyze LASER data
	    # ====================

	    if( $matacqOK && $useMatacq && $useShape ){ 
		
		# with templates method if matacq is OK
		$laserOK2 = analyzeLaser2($dirname);
		
		# get alpha and beta if laser analysis OK
		if($laserOK2) {
		    $ABOK = analyzeAB($dirname);
		}
		
		# analyze data with alpha beta if not
		else {
		    
		    @OKBoth = analyzeBoth($dirname, $isThereTP ); 
		    $laserOK = @OKBoth[0];
		    $TPOK = @OKBoth[1];
		}
		
	    }else{
		
		# with alpha and beta 
		
		@OKBoth = analyzeBoth($dirname, $isThereTP ); 
		$laserOK = @OKBoth[0];
		$TPOK = @OKBoth[1];
		
		if( $laserOK ) {
		    $ABOK = 1;
		}
	    }
	    
	    my $mydate = `date +%s`;
	    print " ....... done at: ${mydate} \n";
	    system date;	
	    
	    print STATUSFILE  "RUN = $run\n";
	    print STATUSFILE  "LB = $lb\n";
	    if( $matacqOK ) { print STATUSFILE  "MATACQ = OK\n" } else  { print STATUSFILE  "MATACQ = FAILED\n" }
	    if( $laserOK2 )  { 
		print STATUSFILE  "LASER = OK\n";
		print STATUSFILE "Templates method\n";
		if( $ABOK )  { print STATUSFILE     "AB = OK\n" } 
	    }elsif( $laserOK )  { 	
		print STATUSFILE  "LASER = OK\n";
		if($fitAB==0){ print STATUSFILE "Fixed AlphaBeta method\n";}
		if($fitAB==1){ print STATUSFILE "Fitted AlphaBeta method\n";}
	    }
	    
#	    if ( ($laserOK && $useMatacq==0) || (($laserOK || $laserOK2) && $matacqOK )) 
	    if ( ($laserOK) || ($laserOK2 && $matacqOK) )  # if redundant prb with mtq 
	    {
		
		print STATUSFILE  "STATUS = ANALYZED\n";
		system "ln -sf ../../Runs/${dirname} ${laserdir}/Analyzed/${dirname}";
		
		# set pointer to the last analyzed run
		if( -e "${laserdir}/lastAnalyzed" ) { system "rm -f ${laserdir}/lastAnalyzed"; }
		system "ln -sf ../Runs/${dirname} ${laserdir}/lastAnalyzed";
		
	    }  else 
	    {
		print STATUSFILE  "STATUS = FAILED\n";
		system "rm -f ${detected}/${dirname}; ln -sf ../../../Runs/${dirname} ${failed}/${dirname}";
	    }
	    
	    close STATUSFILE;
	    
	    if($isThereTP==1){
		
		print TPSTATUSFILE  "RUN = $run\n";
		print TPSTATUSFILE  "LB = $lb\n";
		if ( $TPOK == 1 ) 
		{
		    
		    print TPSTATUSFILE  "STATUS = ANALYZED\n";
		    system "rm -f ${TPdetected}/${dirname}; ln -sf ../../Runs/${dirname} ${TPanalyzed}/${dirname}";
		    
		    # set pointer to the last analyzed run
		    if( -e "${testpulsedir}/lastAnalyzed" ) { system "rm -f ${testpulsedir}/lastAnalyzed"; }
		    system "ln -sf ../Runs/${dirname} ${testpulsedir}/lastAnalyzed";
		}
		else 
		{
		    print TPSTATUSFILE  "STATUS = FAILED\n";
		    system "rm -f ${TPdetected}/${dirname}; ln -sf  ../../../Runs/${dirname} ${TPfailed}/${dirname}";
		}
		close TPSTATUSFILE;
	    }
	    
	}
    }
    closedir( DIR ); 
}

sub doLaser 
{

    print color("green"), "\n *** Restarting Laser Analyzer for FED: ${fed} ***\n\n", color("reset");
    
    my $detected = "$laserdir/Detected";
    my $analyzed = "$laserdir/Analyzed";
    my $failed   = "$laserdir/Analyzed/Failed";

    opendir( DIR, $detected ) || die "cannot open $detected directory\n";

    my @dirs = sort  readdir( DIR );

    foreach my $dirname (@dirs)
    { 
	# skip if already analyzed

	next if( -e "${laserdir}/Analyzed/${dirname}");
	
	if( $dirname =~ /Run(\d*)_LB(\d*)/ )
	{
	    my $run = $1;
	    my $lb = $2;
	    print color("blue"), " Analyzing Laser data for Run: $run, and Lumi Block $lb \n", color("reset");
	    
	    my $jobdir = "${runsdir}/${dirname}";
	
	    my $statusfile   = "$jobdir/statusLaser.txt";
	    my $command;

	    die unless( opendir( CURRENTDIR, ${jobdir} ) );
 
	    open( STATUSFILE, ">>$statusfile" ) || die "cannot open file $statusfile\n";

	    # set pointer to the last detected run
	    if( -e "${laserdir}/lastDetected" ) { system "rm -f ${laserdir}/lastDetected"; }
	    system "ln -sf ../Runs/${dirname} ${laserdir}/lastDetected";

	   
	    # analyze MATACQ
	    # ===============
	    my $matacqOK=0;
	    my $laserOK2=0;
	    my $laserOK=0;
	    my $ABOK=0;

	    my $matacqlog="header";
	    
	    if( $useMatacq == 1 ){
		$matacqOK = checkMatacqJob($dirname, $matacqlog );
	    }
	    
	    # analyze LASER data
	    # ====================

	    if( $matacqOK && $useMatacq && $useShape ){ 
		
		# with templates method if matacq is OK
		$laserOK2 = analyzeLaser2($dirname);
		
		# get alpha and beta if laser analysis OK
		if($laserOK2) {
		    $ABOK = analyzeAB($dirname);
		}
		
		# analyze data with alpha beta if not
		else {
		    $laserOK = analyzeLaser($dirname); 
		}
		
	    }else{
		
		# with alpha and beta if there is no matacq
		$laserOK = analyzeLaser($dirname); 
		if( $laserOK ) {
		    $ABOK = 1;
		}
	    }
	    
	    print STATUSFILE  "RUN = $run\n";
	    print STATUSFILE  "LB = $lb\n";
	    if( $matacqOK ) { print STATUSFILE  "MATACQ = OK\n" } else  { print STATUSFILE  "MATACQ = FAILED\n" }
	    if( $laserOK2 )  { 
		print STATUSFILE  "LASER = OK\n";
		print STATUSFILE "Templates method\n";
		if( $ABOK )  { print STATUSFILE     "AB = OK\n" } 
	    }elsif( $laserOK )  { 	
		print STATUSFILE  "LASER = OK\n";
		if($fitAB==0){ print STATUSFILE "Fixed AlphaBeta method\n";}
		if($fitAB==1){ print STATUSFILE "Fitted AlphaBeta method\n";}
	    }
	    
#	    if ( ($laserOK && $useMatacq==0) || (($laserOK || $laserOK2) && $matacqOK )) 
	    if ( ($laserOK ) || ($laserOK2 && $matacqOK )) # if redundant prb with mtq 
	    {
		
		print STATUSFILE  "STATUS = ANALYZED\n";
		system "ln -sf ../../Runs/${dirname} ${laserdir}/Analyzed/${dirname}";
		
		# set pointer to the last analyzed run
		if( -e "${laserdir}/lastAnalyzed" ) { system "rm -f ${laserdir}/lastAnalyzed"; }
		system "ln -sf ../Runs/${dirname} ${laserdir}/lastAnalyzed";
		
	    }  else 
	    {
		print STATUSFILE  "STATUS = FAILED\n";
		system "rm -f ${detected}/${dirname}; ln -sf ../../../Runs/${dirname} ${failed}/${dirname}";
	    }
	    
	    close STATUSFILE;
	}
    }
    closedir( DIR ); 
}

sub doTestPulse 
{
    
    print color("green"), "\n *** Restarting TestPulse Analyzer for FED: ${fed} ***\n\n", color("reset");

    my $detected = "$testpulsedir/Detected";
    my $analyzed = "$testpulsedir/Analyzed";
    my $failed   = "$testpulsedir/Analyzed/Failed";


    opendir( DIR, $detected ) || die "cannot open $detected directory\n";

    my @dirs = sort  readdir( DIR );
    foreach my $dirname (@dirs)
    {
	# skip if already analyzed
	
	next if( -e "${testpulsedir}/Analyzed/${dirname}");
	
	if( $dirname =~ /Run(\d*)_LB(\d*)/ )
	{
	    my $run = $1;
	    my $lb = $2;
	    print color("blue"), " Analyzing Test-Pulse data for Run: $run, and Lumi Block $lb \n", color("reset");
    
	    my $jobdir = "${runsdir}/${dirname}";
	    my $statusfile   = "$jobdir/statusTestPulse.txt";
	    my $command;

	    die unless( opendir( CURRENTDIR, ${jobdir} ) );
 
	    open( STATUSFILE, ">>$statusfile" ) || die "cannot open file $statusfile\n";

	    # set pointer to the last detected run
	    if( -e "${testpulsedir}/lastDetected" ) { system "rm -f ${testpulsedir}/lastDetected"; }
	    system "ln -sf ../Runs/${dirname} ${testpulsedir}/lastDetected";

	    # analyze TESTPULSE
	    my $testpulseOK =  analyzeTestPulse($dirname);

	    print STATUSFILE  "RUN = $run\n";
	    print STATUSFILE  "LB = $lb\n";
	    if ( $testpulseOK ) 
	    {
			
		print STATUSFILE  "STATUS = ANALYZED\n";
		system "rm -f ${detected}/${dirname}; ln -sf ../../Runs/${dirname} ${analyzed}/${dirname}";

		# set pointer to the last analyzed run
		if( -e "${testpulsedir}/lastAnalyzed" ) { system "rm -f ${testpulsedir}/lastAnalyzed"; }
		system "ln -sf ../Runs/${dirname} ${testpulsedir}/lastAnalyzed";
	    }
	    else 
	    {
		print STATUSFILE  "STATUS = FAILED\n";
		system "rm -f ${detected}/${dirname}; ln -sf  ../../../Runs/${dirname} ${failed}/${dirname}";
	    }
	    close STATUSFILE;
	}
    }
    closedir(DIR)
}

sub analyzeMatacq 
{
    my $arg = 0;
    my $dirname    = shift;
    my $cfgfile = "matacq";
    my $jobdir = "${runsdir}/${dirname}";
    open( TEMPLATE, "${templatesdir}/${cfgfile}.py" );
    open( CFGFILE, ">${jobdir}/${cfgfile}.py" );
    while( <TEMPLATE> )
    {  
	$_ =~ s/FFFF/$fed/g;
	$_ =~ s/DDDD/$debug/g;
	print CFGFILE $_;
    }
    close CFGFILE;
    close TEMPLATE;

    print " - Getting Matacq primitives";

    # submit the MATACQ job

    my $key="matacq";
    my $key2="cmsRun";
    
    my $isItSent = sendCMSJob(${key}, ${key2}, ${nmaxjobstot}, ${user}, ${jobdir}, ${scriptdir});        
    if( $isItSent == 1 ){ 
	print "....... done\n"; 
    }
    

    $arg = checkMatacqJob($dirname,${cfgfile});
    return $arg;
}

sub analyzeLaser 
{
    my $arg = 0;
    my $dirname    = shift;
    my $cfgfile = "laser";
    my $jobdir = "${runsdir}/${dirname}";
    open( TEMPLATE, "${templatesdir}/${cfgfile}.py" );
    open( CFGFILE, ">${jobdir}/${cfgfile}.py" );
    while( <TEMPLATE> )
    {
	$_ =~ s/CCCC/$digis/g;
	$_ =~ s/PPPP/$ecalPart/g;
	$_ =~ s/FFFF/$fed/g;
	$_ =~ s/DDDD/$debug/g;
	$_ =~ s/AAAA/$fitab/g;
	print CFGFILE $_;
    }
    close CFGFILE;
    close TEMPLATE;
    
    print " - Getting APD primitives (AB method)";

    # submit the LASER job

    my $key="laser";
    my $key2="cmsRun";
    
    my $isItSent = sendCMSJob(${key}, ${key2}, ${nmaxjobstot}, ${user}, ${jobdir}, ${scriptdir});        
    if( $isItSent == 1 ){ print "....... done\n"; }

    $arg = checkLaserJob($dirname,$cfgfile);
    return $arg;
}

sub analyzeLaser2 
{
    my $arg = 0;
    my $dirname    = shift;
    my $cfgfile = "laser2";
    my $jobdir = "${runsdir}/${dirname}";
    open( TEMPLATE, "${templatesdir}/${cfgfile}.py" );
    open( CFGFILE, ">${jobdir}/${cfgfile}.py" );
    while( <TEMPLATE> )
    {
	$_ =~ s/CCCC/$digis/g;
	$_ =~ s/PPPP/$ecalPart/g;
	$_ =~ s/FFFF/$fed/g;
	$_ =~ s/DDDD/$debug/g;
	print CFGFILE $_;
    }
    close CFGFILE;
    close TEMPLATE;

    print " - Getting APD primitives (shape method)";

    # submit the LASER job

    my $key="laser2";
    my $key2="cmsRun";
    
    my $isItSent = sendCMSJob(${key}, ${key2}, ${nmaxjobstot}, ${user}, ${jobdir}, ${scriptdir});        
    if( $isItSent == 1 ){ print "....... done\n"; }

    $arg = checkLaserJob2($dirname,${cfgfile});
    return $arg;
}
sub analyzeAB 
{
    my $arg = 0;
    my $dirname    = shift;
    my $cfgfile = "ab";
    my $jobdir = "${runsdir}/${dirname}";
    open( TEMPLATE, "${templatesdir}/${cfgfile}.py" );
    open( CFGFILE, ">${jobdir}/${cfgfile}.py" );
    while( <TEMPLATE> )
    {
	$_ =~ s/CCCC/$digis/g;
	$_ =~ s/FFFF/$fed/g;
	print CFGFILE $_;
    }
    close CFGFILE;
    close TEMPLATE;
    
    print "Getting alpha and beta";

    # submit the AB job
    my $key="ab";
    my $key2="cmsRun";
    
    my $isItSent = sendCMSJob(${key}, ${key2}, ${nmaxjobstot}, ${user}, ${jobdir}, ${scriptdir});        
    if( $isItSent == 1 ){ print "....... done\n"; }

    $arg = checkABJob($dirname, $cfgfile);
    return $arg;
}

sub analyzeTestPulse 
{
    my $arg = 0;
    my $dirname    = shift;
    my $cfgfile = "testpulse";
    my $jobdir = "${runsdir}/${dirname}";
    open( TEMPLATE, "${templatesdir}/${cfgfile}.py" );
    open( CFGFILE, ">${jobdir}/${cfgfile}.py" );
    while( <TEMPLATE> )
    {
	$_ =~ s/CCCC/$digis/g;
	$_ =~ s/FFFF/$fed/g;
	print CFGFILE $_;
    }
    close CFGFILE;
    close TEMPLATE;

    print " - Getting APD primitives";

    # submit the TESTPULSE job 
    my $key="testpulse";
    my $key2="cmsRun";
    
    my $isItSent = sendCMSJob(${key}, ${key2}, ${nmaxjobstot}, ${user}, ${jobdir}, ${scriptdir});        
    if( $isItSent == 1 ){ print "....... done\n"; }

    $arg = checkTestPulseJob($dirname, $cfgfile);
    return $arg;
}

sub checkMatacqJob 
{
    my $arg = 1;
    my $dirname=$_[0];
    my $logname=$_[1];

    print "Check result of Matacq data analysis\n";

    my $jobdir = "${runsdir}/${dirname}";
    my $file = "$jobdir/${logname}.log";
    my $run = 0;
    my $event = 0;
    my $nrun = 0;
    my $ii = 0;
    
    open ( FILE, $file) || die "cannot open $file log file \n";
    while (<FILE>)
    {
	chomp ($_);
	$theLine = $_;
	if( $theLine =~ /BeforeEvents/ )
	{
	    print "$theLine\n";
	}
	if( $theLine =~ /PostModule/ )
	{
	    $ii++;
	    if($ii==1) { print "$theLine\n"; }
	}
	if( $theLine =~ /Run: (.*) Event: (.*)/)
	{
	    if( $1 != $run ) 
	    {
		$nrun++;
		$run= $1;
	    }
	    $event = $2;
	}
	if( $theLine =~ /\+=\+/ )
	{
	    print "$theLine\n";
	    
	}
	if( $theLine =~ /WARNING! NO MATACQ/ ){
	    print "$theLine\n";
	    $arg =0;   
	}
	
    }
    close FILE;
    if( $nrun > 1 )
    {
	print "WARNING! More than one run analyzed!\n";
	$arg = 0;
    } 
    if( $arg ) 
    {
	print "....... OK\n";
    }
    else
    {
	print "....... Failure\n";
    }
    return $arg;
}

sub checkLaserJob 
{
    my $dirname=$_[0];
    my $logname=$_[1];
    my $arg = 1;
    print "Check result of Laser data analysis\n";

    my $jobdir = "${runsdir}/${dirname}";
    my $file = "$jobdir/${logname}.log";
    my $run = 0;
    my $event = 0;
    my $nrun = 0;
    my $ii = 0;


    my $computeAB    = 0;
    if($fitAB==0){ 
	$computeAB=1; 
    }
    my $analyzeAPD   = 0;
    my $allOK        = 0;
    
    open ( FILE, $file) || die "cannot open $file log file \n";
    while (<FILE>)
    {
	chomp ($_);
	$theLine = $_;
	if( $theLine =~ /FwkReport:  main_input:source (.*) CEST BeforeEvents/ )
	{
	    $t1 = $1;
	    print "$theLine\n";
	}
	if( $theLine  =~ /FwkJob:  PostModule (.*) CEST/ )
	{
	    $t2 = $1;
	    $ii++;
	    if($ii==1) { print "$theLine\n"; }
	}
	if( $theLine =~ /Run: (.*) Event: (.*)/)
	{
	    if( $1 != $run ) 
	    {
		$nrun++;
		$run= $1;
	    }
	    $event = $2;
	}

	if( $theLine =~ /No Laser Events/)
	{
	    print "$theLine\n";
	}
	if( $theLine =~ /\+=\+/ )
	{
	    print "$theLine\n";
	    if( $computeAB == 1 )
	    {
		if( $analyzeAPD == 1 )
		{
		    $allOK = 1;
		}
		else
		{
		    if( $analyzeAPD == 2 && $theLine =~ /... done/ )
		    {
			$analyzeAPD = 1;
		    }
		    elsif( $theLine =~ /Analyzing laser data/ )
		    {
			$analyzeAPD = 2;
			
		    }
		    elsif( $theLine =~ /APD GAIN WAS NOT 1/ )
		    {
			$analyzeAPD = 0;
			
		    }
		}
	    }
	    else
	    {
		if( $computeAB == 2 && $theLine =~ /... done/ )
		{
		    $computeAB = 1;
		}
		elsif( $theLine =~ /Analyzing data/ )
		{
		    $computeAB = 2;
		}
	    }
	}
    }

    close FILE;
   
    if( ! $allOK ) 
    {
	$arg = 0;
    }
    
    if( $nrun > 1 )
    {
	print "WARNING! More than one run analyzed!\n";
	$arg = 0;
    } 
    
    if( $arg ) 
    {
	print "....... OK\n";
    }
    else
    {
	print "....... Failure\n";
    }
    return $arg;
}

sub checkLaserJob2 
{
    my $dirname=$_[0];
    my $logname=$_[1];
    my $arg = 1;
    print "Check result of Laser data analysis\n";

    my $jobdir = "${runsdir}/${dirname}";
    my $file = "$jobdir/${logname}.log";
    my $run = 0;
    my $event = 0;
    my $nrun = 0;
    my $ii = 0;

    my $analyzeAPD   = 0;
    my $allOK        = 0;
    my $shape        = 1;
    
    open ( FILE, $file) || die "cannot open $file log file \n";
    while (<FILE>)
    {
	chomp ($_);
	$theLine = $_;
	if( $theLine =~ /FwkReport:  main_input:source (.*) CEST BeforeEvents/ )
	{
	    $t1 = $1;
	    print "$theLine\n";
	}
	if( $theLine  =~ /FwkJob:  PostModule (.*) CEST/ )
	{
	    $t2 = $1;
	    $ii++;
	    if($ii==1) { print "$theLine\n"; }
	}
	if( $theLine =~ /Run: (.*) Event: (.*)/)
	{
	    if( $1 != $run ) 
	    {
		$nrun++;
		$run= $1;
	    }
	    $event = $2;
	}
	if( $theLine =~ /No Laser Events/)
	{
	    print "$theLine\n";
	}
	if( $theLine =~ /\+=\+/ )
	{
	    print "$theLine\n";
	    
	    if( $analyzeAPD == 1 )
	    {
		$allOK = 1;
	    }
	    else
	    {
		if( $analyzeAPD == 2 && $theLine =~ /... done/ )
		{
		    $analyzeAPD = 1;
		}
		elsif( $theLine =~ /Analyzing laser data/ )
		{
		    $analyzeAPD = 2;
		}
		elsif( $theLine =~ /APD GAIN WAS NOT 1/ )
		{
		    $analyzeAPD = 0;
		    
		}
	    }
	}

	if( $theLine =~ /No matacq shape available/ )
	{
	    print "$theLine\n";
	    $shape = 0;    
	    
	}
	if( $theLine =~ /Matacq shape file not found/ )
	{
	    print "$theLine\n";
	    $shape = 0;    
	}
	
    }
    
    close FILE;
 
    if( $allOK && $shape ){  
	
	$arg = 1;
    }else
    {
	$arg = 0;
    } 
    
    if( $nrun > 1 )
    {
	print "WARNING: More than one run analyzed!\n";
	$arg = 0;
    } 
    
    if( $arg ) 
    {
	print "....... OK\n";
    }
    else
    {
	print "....... Failure\n";
    }
    return $arg;
}

sub checkTestPulseJob 
{
    my $arg = 1;
    my $dirname=$_[0];
    my $logname=$_[1];
    print "Check result of Test-Pulse data analysis\n";

    my $jobdir = "${runsdir}/${dirname}";
    my $file = "$jobdir/${logname}.log";
    my $run = 0;
    my $event = 0;
    my $nrun = 0;
    my $ii = 0;
    
    open ( FILE, $file) || die "cannot open $file log file \n";
    while (<FILE>)
    {
	chomp ($_);
	$theLine = $_;
	if( $theLine =~ /FwkReport:  main_input:source (.*) CEST BeforeEvents/ )
	{
	    $t1 = $1;
	    print "$theLine\n";
	}
	if( $theLine  =~ /FwkJob:  PostModule (.*) CEST/ )
	{
	    $t2 = $1;
	    $ii++;
	    if($ii==1) { print "$theLine\n"; }
	}
	if( $theLine =~ /Run: (.*) Event: (.*)/)
	{
	    if( $1 != $run ) 
	    {
		$nrun++;
		$run= $1;
	    }
	    $event = $2;
	}
	if( $theLine =~ /\+=\+/ )
	{
	    $arg = 1;
	    print "$theLine\n";
	}
    }

    close FILE;
    if( $arg ) 
    {
	print "....... OK\n";
    }
    else
    {
	print "....... Failure\n";
    }
    return $arg;
}

sub checkABJob 
{

    my $arg = 1;
    my $dirname=$_[0];
    my $logname=$_[1];
    print "Check result of alpha and beta computation \n";

    my $jobdir = "${runsdir}/${dirname}";
    my $file = "$jobdir/${logname}.log";
    my $run = 0;
    my $event = 0;
    my $nrun = 0;
    my $ii = 0;

    my $computeAB    = 0;
    my $allOK        = 0;
    
    open ( FILE, $file) || die "cannot open $file log file \n";
    while (<FILE>)
    {
	chomp ($_);
	$theLine = $_;
	if( $theLine =~ /FwkReport:  main_input:source (.*) CEST BeforeEvents/ )
	{
	    $t1 = $1;
	    print "$theLine\n";
	}
	if( $theLine  =~ /FwkJob:  PostModule (.*) CEST/ )
	{
	    $t2 = $1;
	    $ii++;
	    if($ii==1) { print "$theLine\n"; }
	}
	if( $theLine =~ /Run: (.*) Event: (.*)/)
	{
	    if( $1 != $run ) 
	    {
		$nrun++;
		$run= $1;
	    }
	    $event = $2;
	}
	if( $theLine =~ /\+=\+/ )
	{
	    print "$theLine\n";
	    if( $computeAB != 1 )
	    {
		if( $computeAB == 2 && $theLine =~ /... done/ )
		{
		    $computeAB = 1;
		}
		elsif( $theLine =~ /Analyzing data/ )
		{
		    $computeAB = 2;
		}
	    }
	}
    }
    
    close FILE;

    if( ! $allOK ) 
    {
	$arg = 0;
    }
    
    if( $nrun > 1 )
    {
	print "WARNING: More than one run analyzed!\n";
	$arg = 0;
    } 
    
    if( $arg ) 
    {
	print "....... OK\n";
    }
    else
    {
	print "....... Failure\n";
    }
    return $arg;
}

sub analyzeBoth 
{
    my $dirname=$_[0];
    my $isThereTP=$_[1];

    my $cfgfile = "";

    if($isThereTP == 1){
	$cfgfile = "both";
    }else{
	$cfgfile = "laser";
    }
    
    my $jobdir = "${runsdir}/${dirname}";
    open( TEMPLATE, "${templatesdir}/${cfgfile}.py" );
    open( CFGFILE, ">${jobdir}/${cfgfile}.py" );
    while( <TEMPLATE> )
    {
	$_ =~ s/CCCC/$digis/g;
	$_ =~ s/PPPP/$ecalPart/g;
	$_ =~ s/FFFF/$fed/g;
	$_ =~ s/DDDD/$debug/g;  
	$_ =~ s/AAAA/$fitab/g;
	print CFGFILE $_;
    }
    close CFGFILE;
    close TEMPLATE;
    
    print " - Getting APD primitives for Laser (AB method) and TP";

    # submit the LASER job

    my $key="NoKey";
    
    my $isItSent = sendCMSJob(${cfgfile}, ${key}, ${nmaxjobstot}, ${user}, ${jobdir}, ${scriptdir});        
    if( $isItSent == 1 ){ print "....... done\n"; }
    
    my @arg=checkBothJob($dirname, $isThereTP, $cfgfile);

    return @arg;
}

sub checkBothJob 
{
    my $dirname=$_[0];
    my $isThereTP=$_[1];
    my $logname=$_[2];
    
    my $arg1 = checkLaserJob($dirname,$logname);
    if( $isThereTP==1 ){
	my $arg2 = checkTestPulseJob($dirname,$logname);
    }else{
	my $arg2=2;
    }
    my @output;
    push(@output, $arg1);
    push(@output, $arg2);
    
    return @output;
}
