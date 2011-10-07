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
$debug               =  0; 
$nmaxjobs            =  8;
$cfgfile             = @ARGV[0];
$cfgfileDef="/nfshome0/ecallaser/config/lmf_cfg";
#=============================================================================#

if( ${cfgfile} eq ""){
    $cfgfile=$cfgfileDef;
}

$runMon  = 0;
$runPrim = 0;

$machineu=`uname -n`;
$machineu=~ s/\s+//;

do "/nfshome0/ecallaser/config/readconfig.pl";
readconfig(${cfgfile});

# define run range from cfg:
# ===========================

$firstRunDef         = "120000"; 
$lastRunDef          = "900000";

if( ${MON_FIRST_RUN} eq ""){
    $firstRun=$firstRunDef;
}else{
$firstRun=${MON_FIRST_RUN};
}
if( ${MON_LAST_RUN} eq ""){
    $lastRun=$lastRunDef;
}else{
$lastRun=${MON_LAST_RUN};
}
$firstRun=~ s/\s+//;
$lastRun=~ s/\s+//;
$firstRun=$firstRun+0;
$lastRun=$lastRun+0;

${MON_USEALSOAB}=~ s/\s+//;
${MON_CMSSW_CODE_DIR}=~ s/\s+//;
$scriptdir    = "${MON_CMSSW_CODE_DIR}/scripts";
do "${scriptdir}/monitoringFunctions.pl";

$machine=VirtualMachineName($machineu);

# define user from cfg:
# =====================
if( ${MON_USER} eq ""){
    $user                = "ecallaser";
}else{
    $user                = ${MON_USER};
}
$user =~ s/\s+//;


# define usealsoab from cfg:
# ============================

if( ${MON_USEALSOAB} eq ""){
    $usealsoab           = 1;
}else{
    $usealsoab           = ${MON_USEALSOAB};
}
$usealsoab =~ s/\s+//;
$usealsoab=$usealsoab+0;


$linkName=${LMF_LASER_PERIOD}; 
$linkName=~ s/\s+//;

$localdir     = cwd; 
$localdir=~ s/\s+//;
$reldir=$MON_CMSSW_REL_DIR;
$reldir=~ s/\s+//;

if( $reldir != $localdir){
    print "You are not in the right release for this config ! \n";
    die;
}

$mon_host_ebeven=${MON_HOSTNAME_EBEVEN};
$mon_host_ebodd=${MON_HOSTNAME_EBODD};
$mon_host_ee=${MON_HOSTNAME_EE};
$mon_host_prim=${MON_HOSTNAME_PRIM};

$mon_host_ebeven=~ s/\s+//;
$mon_host_ebodd=~ s/\s+//;
$mon_host_ee=~ s/\s+//;
$mon_host_prim=~ s/\s+//;




$ecalPart            = "";

if ( $machine =~ /$mon_host_ebeven/ && $machine =~ /$mon_host_ebodd/ && $machine =~ /$mon_host_ee/ ){
    if( $ecalPart eq "") {
	$ecalPart="Ecal";
    }
    $runMon  = 1;
}elsif ( $machine =~ /$mon_host_ebeven/ && $machine =~ /$mon_host_ebodd/ ){
    if( $ecalPart eq "") {
	$ecalPart="EB";
    }
    $runMon  = 1;
}
if ( $machine =~ /$mon_host_ebeven/ ){
    if( $ecalPart eq "") {
	$ecalPart="EBEven";
    }
    $runMon  = 1;
}elsif( $machine =~ /$mon_host_ebodd/ ){
    if( $ecalPart eq "") {
	$ecalPart="EBOdd";
    }
    $runMon  = 1;
}elsif( $machine =~ /$mon_host_ee/ ){
    if( $ecalPart eq "") {
	$ecalPart="EE";
    }
    $runMon  = 1;
}
if( $machine =~ /$mon_host_prim/ ){
    if( $ecalPart eq "") {
	$ecalPart="Ecal";
    }
    $runPrim = 1;
}
if( $runPrim==0 && $runMon==0 ){
    print "unknown machine: $machine ... abort \n";
    die;
}

#==============#
# Run the jobs #
#==============#

if( $runMon==1 ){


print color("red"), "\n\n***** You are about to run the monitoring with the following parameters: *****\n\n", color("reset");

print "  machine             = ${machine} \n";
print "  firstRun            = ${firstRun} \n ";
print " lastRun             = ${lastRun} \n ";
print " period              = ${linkName} \n ";
print " user                = ${user} \n ";
print " usealsoab           = ${usealsoab} \n ";
print " ecalPart            = ${ecalPart} \n ";
print " nmaxjobs            = ${nmaxjobs} \n ";
print " cfgfile             = ${cfgfile}  \n "; 
print " debug               = ${debug}  \n "; 
print " release dir         = ${reldir}   \n "; 

}

#if($runPrim==1){
    
#    print color("red"), "\n\n***** You are about to generate primitives with the following parameters: *****\n\n", color("reset");
    
#    print "  machine             = ${machine} \n ";
#    print "  period              = ${linkName} \n ";
    
#}

 
$proddir      = "${localdir}/${linkName}";
$logdir       = "${proddir}/log";  

$scriptdir    = "${MON_CMSSW_CODE_DIR}/scripts";
$scriptdir    =~ s/\s+//;

$musecaldir   = "${MON_MUSECAL_DIR}" ;
$musecaldir   =~ s/\s+//;

#$scriptdir    = "${proddir}/scripts";  
#$musecaldir   = "${proddir}/musecal";  
#$logdir       = "${proddir}/log";  

my $isAnswerCorrect=0;
my $answer=" ";

while( $isAnswerCorrect == 0 ){
    print color("red"), "\n\n Do you want to proceed? [yes/no] \n \n", color("reset");
    $answer=<STDIN>;

    if ( $answer =~ /yes/ ){
	
	print "... Proceeding runMon=${runMon} ${ecalPart} ${firstRun} ${lastRun} ${user} ${nmaxjobs} ${cfgfile} \n";
	
	my $curtime=time();
	
	if( $runMon==1 ){

	    my $log="${logdir}/analyze_${ecalPart}_${curtime}.log";
	    my $command = "nohup ${scriptdir}/analyze.pl ${firstRun} ${lastRun} ${user} ${nmaxjobs} ${cfgfile} ${ecalPart} ${debug} ${usealsoab} >& ${log} &";
	    print "$command \n";

	    system ${command};
     
	    my $log2="${logdir}/check_${ecalPart}_${curtime}.log";
	    my $command3 = "nohup ${scriptdir}/check.pl ${firstRun} ${lastRun}  ${user} ${nmaxjobs} ${cfgfile} ${ecalPart} ${debug} ${usealsoab} >& ${log2} &";
	    print "$command3 \n";

	    system ${command3};

	}
	
#	if( $runPrim==1 ){
#	    
#	    my $log="${logdir}/Prim_${curtime}.log";
#	    my $command = "nohup ${musecaldir}/generatePrim.pl ${cfgfile} > ${log} &";
#	    system $command;   
#	}
	
	$isAnswerCorrect=1;
    }elsif ( $answer =~ /no/ ){ 
	print "... Aborting \n";
	$isAnswerCorrect=1;
    }else{
	print "... Unknown answer \n";
    }

    sleep 3;
}

exit;
