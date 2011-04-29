#!/usr/bin/env perl

# SELMA - Saclay Ecal Laser Monitoring and Analysis
# contacts: Julie Malcles, Gautier Hamel de Monchenault
# last modified: Tue Dec  2 10:43:34 CET 2008


use Term::ANSIColor;
use Cwd;
use File::stat;
use Time::localtime;

#===================#
# Parameters to set #
#===================#

#=============================================================================#
$firstRun            = "84000"; 
#$firstRun            = "65000";
$lastRun             = "130000";
#$lastRun             = "84000";
$useMatacq           =  1;
$useShape            =  0;
$fitAB               =  0;
$linkName            = "Cosmics09_310"; 
$user                = "ecallaser";
$ecalPart            = "";
$nmaxjobshead        = 3;
$nmaxjobstot         = 8;
#=============================================================================#

$runMon  = 1;
$runPrim = 0;

$machine=`uname -n`;

$mon_host_ebeven=`echo \${MON_HOSTNAME_EBEVEN}`;
$mon_host_ebodd=`echo \${MON_HOSTNAME_EBODD}`;
$mon_host_ee=`echo \${MON_HOSTNAME_EE}`;
$mon_host_prim=`echo \${MON_HOSTNAME_PRIM}`;


if ( $machine =~ /$mon_host_ebeven/ ){
    if( $ecalPart eq "") {
	$ecalPart="EBEven";
    }
}elsif( $machine =~ /$mon_host_ebodd/ ){
    if( $ecalPart eq "") {
	$ecalPart="EBOdd";
    }
}elsif( $machine =~ /$mon_host_ee/ ){
    if( $ecalPart eq "") {
	$ecalPart="EE";
	$runMon  = 1;
	$runPrim = 0;
    }
}elsif( $machine =~ /$mon_host_prim/ ){
    if( $ecalPart eq "") {
	$ecalPart="All";
    }
    $runMon  = 0;
    $runPrim = 1;

#reprocessing: 
#}elsif ( $machine =~ /srv-C2D17-15/ ){
#    if( $ecalPart eq "") {
#	$ecalPart="EBEven";
#    }

#}elsif( $machine =~ /srv-C2D17-16/ ){
#    if( $ecalPart eq "") {
#	$ecalPart="EBOdd";
#    }
}else {
    print "unknown machine: $machine ... abort \n";
    die;
}

#==============#
# Run the jobs #
#==============#

if( $runMon==1 ){


print color("red"), "\n\n***** You are about to run the monitoring with the following parameters: *****\n\n", color("reset");

print "  machine             = ${machine} ";
print " firstRun            = ${firstRun} \n ";
print " lastRun             = ${lastRun} \n ";
print " useMatacq           = ${useMatacq} \n ";
print " useShape            = ${useShape} \n ";
print " fitAB               = ${fitAB} \n ";
print " linkName            = ${linkName} \n ";
print " user                = ${user} \n ";
print " ecalPart            = ${ecalPart} \n ";
print " nmaxjobshead        = ${nmaxjobshead} \n ";
print " nmaxjobstot         = ${nmaxjobstot} \n ";

}
if($runPrim==1){
    
    print color("red"), "\n\n***** You are about to generate primitives with the following parameters: *****\n\n", color("reset");
    
    print "  machine             = ${machine} ";
    print " linkName            = ${linkName} \n ";
    
}


$localdir     = cwd;  
$proddir      = "${localdir}/${linkName}";
$scriptdir    = "${proddir}/scripts";  
$musecaldir    = "${proddir}/musecal";  
$logdir    = "${proddir}/log";  

my $isAnswerCorrect=0;
my $answer=" ";

while( $isAnswerCorrect == 0 ){
    print color("red"), "\n\n Do you want to proceed? [yes/no] \n \n", color("reset");
    $answer=<STDIN>;

    if ( $answer =~ /yes/ ){
	
	print "... Proceeding \n";
	
	my $curtime=time();
	
	if( $runMon==1 ){
	    my $log="${logdir}/LM_${ecalPart}_${curtime}.log";
	    my $command = "nohup ${scriptdir}/LM.pl ${firstRun} ${lastRun} ${useMatacq} ${useShape} ${fitAB} ${linkName} ${user} ${ecalPart} ${nmaxjobshead} ${nmaxjobstot} > ${log} &";
	    system $command;
	}
	
	if( $runPrim==1 ){
	    
	    my $log="${logdir}/Prim_${curtime}.log";
	    my $command = "nohup ${musecaldir}/generatePrim.pl ${linkName} > ${log} &";
	    system $command;   
	}
	
	$isAnswerCorrect=1;
    }elsif ( $answer =~ /no/ ){ 
	print "... Aborting \n";
	$isAnswerCorrect=1;
    }else{
	print "... Unknown answer \n";
    }
}

exit;
