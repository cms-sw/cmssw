#!/usr/bin/env perl

# SELMA - Saclay Ecal Laser Monitoring and Analysis
# contacts: Julie Malcles, Gautier Hamel de Monchenault
# last modified: Sat Nov 24 08:56:52 CET 2007

use Term::ANSIColor;
use Cwd;
use File::stat;
use Time::localtime;

$firstRun     = @ARGV[0];
$lastRun      = @ARGV[1];
$useMatacq    = @ARGV[2];
$useShape     = @ARGV[3];
$fitAB        = @ARGV[4];
$linkName     = @ARGV[5];
$user         = @ARGV[6];
$ecalPart     = @ARGV[7];
$nmaxjobshead = @ARGV[8];
$nmaxjobstot  = @ARGV[9];

# Defaults arguments values 
#===========================

$firstRunDefault = "0" ;
$lastRunDefault  = "100000" ;
$useMatacqDefault = 1 ;
$useShapeDefault = 1 ;
$linkNameDefault = "LaserMonitoring" ;
$userDefault="ecallaser";
$ecalPartDefault="Ecal";
$debug=0;

@analyzeSMRunning = (0,0,0,0,0,0,0,0,0,0,
		     0,0,0,0,0,0,0,0,0,0,
		     0,0,0,0,0,0,0,0,0,0,
		     0,0,0,0,0,0,0,0,0,0,
		     0,0,0,0,0,0,0,0,0,0,
		     0,0,0,0);
@detectSMRunning  = (0,0,0,0,0,0,0,0,0,0,
		     0,0,0,0,0,0,0,0,0,0,
		     0,0,0,0,0,0,0,0,0,0,
		     0,0,0,0,0,0,0,0,0,0,
		     0,0,0,0,0,0,0,0,0,0,
		     0,0,0,0);

# Setup defaults arguments values 
#=================================

if( $firstRun eq "" )
{
    $firstRun = $firstRunDefault; 
    $lastRun = $lastRunDefault;
}
elsif( $lastRun eq "+" ) 
{ 
    $lastRun = "99999"; 
}
elsif( $lastRun eq "" ) 
{ 
    $lastRun = $firstRun; 
}

if( $useMatacq eq "" )
{
    $useMatacq=$useMatacqDefault;
}
if( $useShape eq "" ){
    $useShape=$useShapeDefault;
}

if( $useMatacq == 0 ) {
    $useShape = 0;
}

if( $user eq "" ){
    $user=$userDefault;
}
if( $ecalPart eq "" ){
    $ecalPart = $ecalPartDefault;
}

# Directories settings
#======================

$localdir     = cwd;
if (  $linkName eq "" ) {
    $proddir      = "${localdir}/${linkNameDefault}";
}else{
    $proddir      = "${localdir}/${linkName}";
}

$scriptdir    = "$proddir/scripts";
$shapedir     = "$proddir/shapes";
$sortdir      = "$proddir/sorting";

# ECAL numbering settings
#=========================

$firsteem     = 601;
$firstebm     = 610;
$firstebp     = 628;
$firsteep     = 646;
$lastecal     = 654;

do "${scriptdir}/monitoringFunctions.pl";

print color("red"),"\n\n ***** Saclay Ecal Laser Monitoring & Analysis *****\n", color("reset");
print color("red"),     " *****           Analyzing : ${ecalPart}              *****\n\n", color("reset");



# Analyze the sorting directory:
#===============================

while( 1 ) 
{
    system date;    
    analyze();
    sleep 60;
}

exit;

sub analyze 
{
    print color("green"), " \n ** Restarting Analyzer ** \n\n", color("reset");
    
    opendir(SORTDIR, $sortdir) || die "cannot open sorting directory $sortdir \n";
    
    $dirname = "";
    $fedname = "";
    $lbname = "";
    $run     = 0;
    $fed     = 0;
    
    my @sortdirs = sort  readdir( SORTDIR );

    # Detect news super-modules
    #===========================

    foreach my $dir (@sortdirs)
    {
	if( $debug == 1 ) { print "Directory: ${dir} detected.\n"; }
	
	next unless( $dir =~ /E(E|B)(\+|-)(\d{1})/ );
	$fed = getFedNumber(${dir});  
	
	if( $debug == 1 ) { print "Cut1Passed fed:${fed},dir:${dir},ecalPart:${ecalPart} \n";}

	next unless( doProcessFed($fed, $ecalPart) == 1 );

	if( $debug == 1 ) { print "Cut2Passed (doProcessFed) \n";}


	# Create eventually new directory
	# ================================

	my $smdir      = "${proddir}/${dir}";

	if( -e ${smdir} ) 
	{
	    if( $debug == 1 ) { print "${smdir} exists.\n"; }

	}else{
	    print "creating directory ${smdir}\n";
	    $command =  "${scriptdir}/createDirs.csh ${proddir} ${dir}" ;
	    system ${command};
	}
	
	# Check if analyzer is running and eventuelly launch it
	# ======================================================

	if( $detectSMRunning[$fed-600] == 0 ){
	    $detectSMRunning[$fed-600]= isDetectSMRunning($fed, $dir, $user, $linkName);
	}
	if( $detectSMRunning[$fed-600] == 0 ){   
	    print "now launch the script to detect new files for module ${dir}\n";
	    
	    my $curtime=time();
	    
	    my $smlog="${smdir}/detectSM_${curtime}.log";
	    
	    $command = "nohup ${scriptdir}/detectSM.pl ${firstRun} ${lastRun} ${fed} ${dir} ${useShape} ${fitAB} ${linkName} ${user} ${nmaxjobshead} ${debug} > ${smlog} &\n"; 
	    system ${command};    
	    $detectSMRunning[$fed-600]=1;
	}
	

	if( $analyzeSMRunning[$fed-600] == 0 ){
	    $analyzeSMRunning[$fed-600]= isAnalyzeSMRunning($fed, $dir, $user, $linkName);
	}
	if( $analyzeSMRunning[$fed-600] == 0 ){   
	    print "now launch the script to analyze module ${dir}\n";
	    
	    my $curtime=time();
	    
	    my $smlog="${smdir}/analyzeSM_${curtime}.log";
	    
	    $command = "nohup ${scriptdir}/analyzeSM.pl ${fed} ${dir} ${useMatacq} ${useShape} ${fitAB} ${linkName} ${user} ${nmaxjobstot} ${debug} > ${smlog} &\n"; 
	    system ${command};    
	    $analyzeSMRunning[$fed-600]=1;
	}
    
	sleep 1;
    }
        
    close(SORTDIR);
}

sub isAnalyzeSMRunning
{    
    my $fedToCheck=$_[0];
    my $smToCheck=$_[1];
    my $username=$_[2];
    my $linkName=$_[3];
    
    my $isItRunning=1;

    my $command = "ps x -u $username";
     
    my $nproc=0;
    
    open(COMMAND, "${command}|" ) or die $!;
    while( my $ligne = <COMMAND> ) {
	next unless( $ligne =~ /(.*)analyzeSM.pl $fedToCheck $smToCheck(.*)$linkName(.*)/ ); 
	$nproc=$nproc+1;
    }
    close COMMAND;
    
    if( $debug ==1 ) {print "isAnalyzeSMRunning: $fed nproc= $nproc \n"; }
    if( $nproc == 0 ) {
	$isItRunning=0;
    }
    return $isItRunning; 
}

sub isDetectSMRunning
{    
    my $fedToCheck=$_[0];
    my $smToCheck=$_[1];
    my $username=$_[2];
    my $linkName=$_[3];
    
    my $isItRunning=1;

    my $command = "ps x -u $username";
     
    my $nproc=0;
    
    open(COMMAND, "${command}|" ) or die $!;
    while( my $ligne = <COMMAND> ) {
	next unless( $ligne =~ /(.*)detectSM.pl $fedToCheck $smToCheck(.*)$linkName(.*)/ ); 
	$nproc=$nproc+1;
    }
    close COMMAND;
    
    if( $debug ==1 ) {print "isDetectSMRunning: $fed nproc= $nproc \n"; }
    if( $nproc == 0 ) {
	$isItRunning=0;
    }
    return $isItRunning; 
}


