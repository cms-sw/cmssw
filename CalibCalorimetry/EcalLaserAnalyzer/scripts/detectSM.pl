#!/usr/bin/env perl

# SELMA - Saclay Ecal Laser Monitoring and Analysis
# contacts: Julie Malcles, Gautier Hamel de Monchenault, Serguei Ganjour, Patrice Verrecchia
# last modified: Sat Jul 28 09:58:19 CEST 2007

use Term::ANSIColor;
#use Date::Manip; 
use Cwd;

$firstRun     = @ARGV[0];
$lastRun      = @ARGV[1];
$fed          = @ARGV[2];
$ecalmod      = @ARGV[3];
$useShape     = @ARGV[4];
$fitAB        = @ARGV[5];
$linkname     = @ARGV[6];
$user         = @ARGV[7];
$nmaxjobshead = @ARGV[8];
$debug        = @ARGV[9];
 
#  Load some usefull functions
#==============================


die unless( $ecalmod =~ /E(E|B)(\+|-)(\d*)/ );

$localdir     = cwd;
if (  $linkname eq "" ) {
    $proddir      = "${localdir}/LaserMonitoring";
}else{
    $proddir      = "${localdir}/${linkname}";
}

$sortdir      = "$proddir/sorting";
$scriptdir    = "$proddir/scripts";
$shapedir     = "$proddir/shapes";
$alphabetadir = "$proddir/alphabeta";
$templatesdir = "$proddir/templates";
$lmfdir       = "$proddir/sorting/${ecalmod}";
$runsdir      = "$proddir/${ecalmod}/Runs";
$laserdir     = "$proddir/${ecalmod}/Laser";
$testpulsedir = "$proddir/${ecalmod}/TestPulse";

do "${scriptdir}/monitoringFunctions.pl";

print color("red"), "\n\n***** Saclay Ecal Laser Monitoring & Analysis *****\n\n", color("reset");

while( 1 ) 
{
    system date;
    doDetect();
    sleep 1;
}

exit;


sub doDetect

{ 
    print color("green"), "\n *** Restarting File Detection for FED: ${fed} ***\n\n", color("reset");
    
    opendir(POOLDIR, $lmfdir) || die "cannot open sorting directory  $lmfdir \n";
    
    $dirname = "";
    $lbname = "";
    $run     = 0;
    
    my @poolfiles = sort  readdir( POOLDIR );

    # Detect news sequences
    #=======================

    foreach my $file (@poolfiles)
    {
	if( $debug == 1 ) {
	    print "LMF File: ${file} detected.\n";
	}
	next unless( $file =~ /(.*)_(.*).lmf/ );

	$dirname = $1; 
	$lbname = $2;

	# check file is fully copied (no .part)
	next unless (  $file eq "${dirname}_${lbname}.lmf" );

	if( $debug == 1 ) {
	    print "  dirname = $dirname \n";
	    print "  lbname = $lbname \n";
	}

	next unless( $dirname =~ /Run(\d*)/ );
	$run    = $1;

	next unless( $lbname =~ /LB(\d*)/ );
	$lb    = $1;

	$dirname=$dirname."_LB".$lb;
	
	if( $debug == 1 ) {
	    print "DirName: $dirname \n ";
	    print " $run $firstRun $lastRun $fed \n";
	}
	
	next unless( $run >= $firstRun && $run <= $lastRun );

	my $ecalmod = "";
	$ecalmod = getSMname($fed);

	my $mydate = `date +%s`;
	print " Run ${run} and LB ${lb} found for fed ${fed} (${ecalmod}) at: ${mydate} \n";	
	system date;	


	my $smdir      = "${proddir}/${ecalmod}";
	my $runsdir    = "${smdir}/Runs";
	my $jobdir     = "${runsdir}/${dirname}";
	my $command;

	# skip if already analyzed
	
	my $doesHeaderExist=checkHeader(${jobdir});
	if( $debug == 1 ){
	    print "doesHeaderExist: $doesHeaderExist\n";
	}	
	next if( $doesHeaderExist == 1 );
	next if(-e "${laserdir}/Detected/${dirname}");
	

	if( $debug == 1 ) {
	    print "\nRun number = $run  ";
	    print "\n";
	}
	
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
	}else{
	    
	    print "creating directory ${jobdir}\n";
	    $command =  "mkdir ${jobdir}" ;
	    system ${command};

	    # link to the pool file
	    $command = "ln -sf ${lmfdir}/${file} ${jobdir}/input.lmf";
	    system  ${command};
	    
	    # link to the shapes (elec shapes)
	    if( $useShape == 1 ) {
		$command = "ln -sf ${shapedir}/ElecMeanShape.root  ${jobdir}/ElecShape.root ";
		system  ${command};
	    }elsif( $fitAB ){
		# link to the alpha beta init file
		$command = "ln -sf ${alphabetadir}/AB${fed}.root  ${jobdir}/ABInit.root ";
		if(-e "${alphabetadir}/AB${fed}.root"){
		    system  ${command};
		}
	    }else{
		# link to the alpha beta file
		$command = "ln -sf ${alphabetadir}/AB${fed}.root  ${jobdir}/ABInit.root ";
		if(-e "${alphabetadir}/AB${fed}.root"){
		    system  ${command};
		}
		$command = "ln -sf ${alphabetadir}/AB${fed}.root  ${jobdir}/AB.root ";
		if(-e "${alphabetadir}/AB${fed}.root"){
		    system  ${command};
		}
	    }

	}

	my $mydate = `date +%s`;
	print " ... done at: ${mydate} \n";
	system date;	

	
	# Analyze
	#=========
	
	my $type = getType( $file );
    }
    
    close(POOLDIR);
}

sub getType
{

    my $type = "UNKNOWN";
    my $runfromname  = 0;
    my $lb  = 0;
    my $dirname = "";
    my $lbname = "";

    my $file = shift;
    
    if($debug==1){
	print "In getType, file: $file \n";
    }
    
    if( $file  =~ /(.*)_(.*).lmf/ ){
	$dirname = $1;
	$lbname = $2;   	
    }

    if( $dirname =~ /Run(\d*)/ ){
	$runfromname   = $1;
    }
    if( $lbname =~ /LB(\d*)/ ){
	$lbfromname   = $1;
    }
    
    $dirname=$dirname."_".$lbname;
    
    my $ecalmodfromname = getSMname($fed);
    
    my $smdir   = "${proddir}/${ecalmodfromname}";
    my $runsdir = "${smdir}/Runs";
    my $jobdir  = "${runsdir}/${dirname}";

    # Run StatusMon to get the sequence header
    #==========================================

    print "Get the sequence header\n";
   
   
   # my $command = "cp ${templatesdir}/header.py ${jobdir}/header.py";
   # system $command;
 
    open( TEMPLATE, "${templatesdir}/header.py" );
    open( CFGFILE, ">${jobdir}/header.py" );
    while( <TEMPLATE> )
    {  
	$_ =~ s/FFFF/$fed/g;
	$_ =~ s/DDDD/$debug/g;
	print CFGFILE $_;
    }
    close CFGFILE;
    close TEMPLATE;
    
    my $key="header";
    my $isItSent = sendCMSJob(${key}, ${key}, ${nmaxjobshead}, ${user}, ${jobdir}, ${scriptdir});        
    if( $isItSent == 1 ){ print "....... done\n"; }
    

    # Read the sequence header to run other jobs for each SM
    #=======================================================

    my $headerfile = "${jobdir}/header.txt";

    if (open( HEADERFILE, "${headerfile}")){
	
	while (<HEADERFILE>)
	{
	    chomp($_);
	    my $theLine = $_;
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
		$fed = $1;
		$ecalmod = getSMname($fed);
		
		if( ( $type =~ /LASER/ ) ||  ( $type =~ /TESTPULSE/ ) ){
		    
		    print "Type is $type\n";
		    print "ECAL module is ${ecalmod} \n";
		    
		    $smdir   = "${proddir}/${ecalmod}";
		    $runsdir = "${smdir}/Runs";
		    $jobdir  = "${runsdir}/${dirname}";
		    
		    
		    if  ( $ecalmodfromname eq $ecalmod ){
			
			# Set Pointers 
			#==============
			
			my $analyzedlaserdir     = "${laserdir}/Analyzed";
			my $analyzedtestpulsedir = "${testpulsedir}/Analyzed";
			
			if( $type =~ /LASER/ )
			{
			    # skip if already analyzed
			    
			    if(-e "${analyzedlaserdir}/${dirname}") {
				print " Laser sequence already analyzed: ${laserdir}, ${dirname} \n"; 
			    }else{
				
				# otherwise, set pointer to the last detected run
				print " Setting Laser Pointers  ${laserdir}, ${dirname} \n"; 
				if( -e "${laserdir}/lastDetected" ) { system "rm -f ${laserdir}/lastDetected"; }
				system "ln -sf    ../Runs/${dirname} ${laserdir}/lastDetected";
				system "ln -sf ../../Runs/${dirname} ${laserdir}/Detected/${dirname}";
			    }
			}
			elsif( $type =~ /TESTPULSE/ )
			{
			    
			    # skip if already analyzed
			    
			    if(-e "${analyzedtestpulsedir}/${dirname}") {
				print " TP sequence already analyzed: ${laserdir}, ${dirname} \n"; 
			    }else{
				
				# otherwise, set pointer to the last detected run
				print " Setting TP Pointers ${testpulsedir}, ${dirname} \n"; 
				# set pointer to the last detected run
				if( -e "${testpulsedir}/lastDetected" ) { system "rm -f ${testpulsedir}/lastDetected"; }
				system "ln -sf    ../Runs/${dirname} ${testpulsedir}/lastDetected";
				system "ln -sf ../../Runs/${dirname} ${testpulsedir}/Detected/${dirname}";
			    }
			}
		    }  
		    else
		    { 
			print " FED ID $ecalmod does not match fed id in file name $ecalmodfromname: problem must have occured while sorting ";
			
		    }
		}  
		next;
	    }
	}
	
	close HEADERFILE;
	
    }else{
	print "cannot open header file in getType \n";
    }
    
    return $type;
}


sub checkHeader
{

    my $dir    = shift;
    my $headerDone = 0;

    if( -e "${dir}/header.txt" ){
	my $nlines=`grep done ${dir}/header.txt | wc -l`;  
	if ( $nlines == 0 ){
	    $headerDone=0;
	}else{ 
	    $headerDone=1;
	}
    }
    return $headerDone;
}
