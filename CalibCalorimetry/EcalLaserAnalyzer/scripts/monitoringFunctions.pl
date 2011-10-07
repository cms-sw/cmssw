#!/usr/bin/env perl

# SELMA - Saclay Ecal Laser Monitoring and Analysis
# contacts: Julie Malcles, Gautier Hamel de Monchenault 
# last modified: Wed Jun 18 16:48:40 CEST 2008

use Term::ANSIColor;
use Cwd;
use File::stat;
use Time::localtime;


$firsteem     = 601;
$firstebm     = 610;
$firstebp     = 628;
$firsteep     = 646;
$lastecal     = 654;

$debugMF=0;

sub RealMachineName{

    my $virtual    = shift;
    my $realName="";
    my $realNameBase="srv-C2F38-";
    if( $virtual=~ /srv-ecal-laser-(\d*)/ ){
	my $end=$1;
	$realName=$realNameBase.$end;
	
    }else{
	print "Unknown name : $virtual \n";
    }
    return $realName;
    
}
sub VirtualMachineName{

    my $real    = shift;
    my $virtual="";
    my $virtualBase="srv-ecal-laser-";
    if( $real=~ /srv-C2F38-(\d*)/ ){
	my $end=$1;
	$virtual=$virtualBase.$end;
    }
    return $virtual;
}

sub getSMname
{

    my $fed    = shift;
    
    my $ecal = "";
    my $ecalmod = "";
    my $nsm  = 0;
    
    if( $fed<$firsteem || $fed>$lastecal )
    {
	print "Wrong SM number in getSMname: $fed \n";
    }
    if( $fed<$firstebm )
    {
	$ecal = "EE-";
	$nsm = $fed-$firsteem+1-3;
	if ($nsm <= 0 ){ $nsm+=9; }
    }
    elsif( $fed<$firstebp )
    {
	$ecal = "EB-";
	$nsm = $fed-$firstebm+1;
    }
    elsif( $fed<$firsteep )
    {
	$ecal = "EB+";
	$nsm = $fed-$firstebp+1;
    }
    else
    {
	$ecal = "EE+";
	$nsm = $fed-$firsteep+1-3;
	if($nsm <= 0) { $nsm+=9; }
    }
    
    $ecalmod = $ecal.$nsm;

    return $ecalmod;

}


sub getFedNumber
{

    my $smname    = shift;
    my $fed = 0;

    if( $smname =~ /EE-(\d*)/ ){
	
	$nsm = $1;
	my $num=$nsm+3;
	if($num>9) {$num-=9;}

	$fed =  $num + $firsteem -1;
	

    }elsif ( $smname =~ /EE\+(\d*)/ ){

	$nsm = $1;
	my $num=$nsm+3;
	if($num>9) {$num-=9;}
	$fed = $num + $firsteep -1;
	
    }elsif ( $smname =~ /EB-(\d*)/ ){
	
	$nsm = $1;
	$fed = $nsm + $firstebm - 1;


    }elsif ( $smname =~ /EB\+(\d*)/ ){

	$nsm = $1;
	$fed =  $nsm+ $firstebp-1;
	
    }else {

	print "Wrong SM name in getFedNumber: $fed\n";
	$fed =0;
    }
   
    return $fed;
    
}

sub doProcessFed{
    
    my $fed=$_[0];
    my $ecalpart=$_[1];
    my $doprocess=0;
    
    
    if( $fed<$firsteem || $fed>$lastecal )
    {
	print "Wrong SM number in doProcessFed: $fed \n";
	return $doprocess;
    }
    
    if( $ecalpart eq "Ecal" ){
	$doprocess=1;
    }
    elsif( $ecalpart eq "EB" ){
	if( getSMname($fed) =~ /EB(.*)/ ){
	    $doprocess=1;
	}
    }
    elsif( $ecalpart eq "EE" ){
	if( getSMname($fed) =~ /EE(.*)/ ){
	    $doprocess=1;
	}
    }	
    elsif( $ecalpart eq "EB-" ){
	if( getSMname($fed) =~ /EB-(\d*)/ ){
	    $doprocess=1;
	}
    }	
    elsif( $ecalpart eq "EB+" ){
	if( getSMname($fed) =~ /EB\+(\d*)/ ){
	    $doprocess=1;
	}
    }	
    elsif( $ecalpart eq "EE-" ){
	if( getSMname($fed) =~ /EE-(\d*)/ ){
	    $doprocess=1;
	}
    }	
    elsif( $ecalpart eq "EE+" ){
	if( getSMname($fed) =~ /EE\+(\d*)/ ){
	    $doprocess=1;
	}
    } 
    elsif( $ecalpart eq "EcalEven" ){
	my $odd=$fed%2;
	if( $odd == 0 ) {
	    $doprocess=1;
	}
    }
    elsif( $ecalpart eq "EcalOdd" ){
	my $odd=$fed%2;
	if( $odd%2  == 1 ) {
	    $doprocess=1;
	}
    }
    elsif( $ecalpart eq "EBEven" ){
	
	if( getSMname($fed) =~ /EB(.*)/ ){
	    if( $fed%2 == 0 ){	
		$doprocess=1;
	    }
	}
    }
    elsif( $ecalpart eq "EBOdd" ){
	
	if( getSMname($fed) =~ /EB(.*)/ ){
	    if( $fed%2 ==1 ) {
		$doprocess=1;
	    }
	}
    }
    elsif( $ecalpart =~ /E(E|B)(\+|-)(\d{1})/ ){
	$testtest=getSMname($fed);
	if(  ${ecalpart} eq "${testtest}" ){
	    $doprocess=1;
	}
	
    } else {
	
	print "Unknown ecalpart: $ecalpart \n";
    }
    
    return $doprocess;
}


sub sendCMSJob{
    
    my $keyword=$_[0];  # header, matacq, laser, testpulse
    my $keyword2=$_[1]; # for ps command
    my $nmaxjobs=$_[2]; 
    my $user=$_[3];
    my $jobdir=$_[4];
    my $scriptdir=$_[5];


    if( $debugMF==1 ){
	print " debug - sendCMSJob 1 : ${cfg} ${log} ${keyword2} ${nmaxjobs} \n";
    }
    my $isItSent = 0;
    my $njobs = 8;
    my $nsub = 8;
    my $ntot = 8;
    my $timewaiting=0;
    my $command="";
    
    if( $keyword2 eq "NoKey" ){
	$command="ps x -u ${user} | grep cmsRun | grep py | grep -v 'ps x -u' |  wc -l"; 
    }else {
	$command="ps x -u ${user} | grep cmsRun | grep py | grep -v 'ps x -u' | grep ${keyword2} |wc -l";
    }
    $njobs =`$command`;
    $njobs =$njobs+0;

    if( $keyword2 eq "NoKey" ){
	$command2="ps x -u ${user} | grep submitJobPy | grep -v 'ps x -u' | wc -l";
    }else{
	$command2="ps x -u ${user} | grep submitJobPy | grep -v 'ps x -u' | grep ${keyword2} | wc -l";
    }
    
    $nsub =`$command2`;
    $nsub =$nsub+0;

    $ntot=$nsub+$njobs;
    
    if( $debugMF==1 ) {
	print "debug - sendCMSJob 2 : njobs= ${njobs}, nsub= ${nsub}, ntot= ${ntot} \n";
    }
    
    my $commandToSend = "${scriptdir}/submitJobPy.csh ${keyword} ${jobdir}";

    while($ntot >=$nmaxjobs){
	
	sleep 5;
	$timewaiting+=5;

	$njobs =`$command`;  
	$njobs =$njobs+0;
	$nsub =`$command2`;  
	$nsub =$nsub+0;
	
	$ntot=$nsub+$njobs;

	if( $debugMF==1 ) {
	    print "debug - sendCMSJob 3 : inside loop njobs= ${njobs}, nsub= ${nsub}, ntot= ${ntot}  \n";
	    
	}
    }

    my $mydate = `date +%s`;
    print " ....... job sent at: ${mydate} after waiting ${timewaiting} seconds \n";
    print " njobs: ${njobs}, nsub: ${nsub}, ntot: ${ntot} \n";
    
    system $commandToSend ;
    
    
    $isItSent=1;
    
    if( $debugMF==1 ) {
	print "debug - sendCMSJob 4 : isItSent: ${isItSent} \n";
	print "debug - sendCMSJob 5 : command: ${commandToSend} \n";
    }
    
    return  $isItSent;
}



sub sendJob{
    
    my $keyword=$_[0];  # header, matacq, laser, testpulse
    my $keyword2=$_[1]; # for ps command
    my $nmaxjobs=$_[2]; 
    my $user=$_[3];
    my $jobdir=$_[4];
    my $scriptdir=$_[5];
    my $donefile=$_[6];
    my $nprocfile=$_[7];
    my $shortjobdir=$_[8];
    my $dormunzip=$_[9];

    if( $debugMF==1 ){
	print " debug - sendJob 1 : ${cfg} ${log} ${keyword2} ${nmaxjobs} \n";
    }
    my $isItSent = 0;
    my $njobs = 8;
    my $nsub = 8;
    my $ntot = 8;
    my $timewaiting=0;
    my $command="";
    
    if( $keyword2 eq "NoKey" ){
	$command="ps x -u ${user} | grep cmsRun | grep py | grep -v 'ps x -u' |  wc -l"; 
    }else {
	$command="ps x -u ${user} | grep cmsRun | grep py | grep -v 'ps x -u' | grep ${keyword2} | wc -l";
    }
    $njobs =`$command`;
    $njobs =$njobs+0;

   if( $keyword2 eq "NoKey" ){
	$command2="ps x -u ${user} | grep submitJobPy2 | grep -v 'ps x -u' | wc -l";
    }else{
	$command2="ps x -u ${user} | grep submitJobPy2 | grep -v 'ps x -u' | grep ${keyword2} | wc -l";
    }
    
    $nsub =`$command2`;
    $nsub =$nsub+0;

    $ntot=$nsub+$njobs;
    
    if( $debugMF==1 ) {
	print "debug - sendJob 2 : njobs= ${njobs}, nsub= ${nsub}, ntot= ${ntot} \n";
    }
    

    while( $nsub>=$nmaxjobs || $njobs>=$nmaxjobs ){
	
	sleep 5;
	$timewaiting+=5;

	$njobs =`$command`;  
	$njobs =$njobs+0;
	$nsub =`$command2`;  
	$nsub =$nsub+0;
	
	$ntot=$nsub+$njobs;
	
	if( $debugMF==1 ) {
	    print "debug - sendJob 3 : inside loop njobs= ${njobs}, nsub= ${nsub}, ntot= ${ntot}  \n";
	    
	}
    }

    my $mydate = `date +%s`;
    print " ....... job sent at: ${mydate} after waiting ${timewaiting} seconds \n";
    print " njobs: ${nsub} ${njobs} \n";
    
    
    # update nprocfile
    my $nprocnew=0;
    if(! -e $nprocfile){
	$nprocnew=1;
    }else{
	my $nproc=`tail -1 ${nprocfile}`;
	$nprocnew=$nproc+1;
    }
    
    my $commandToSend = "${scriptdir}/submitJobPy2.csh ${keyword} ${jobdir} ${donefile} ${shortjobdir} ${nprocnew} ${dormunzip} &";
    
    # send job
    system $commandToSend ;
    $isItSent=1;
    
    system "echo ${nprocnew} >> ${nprocfile}";
    
    if( $debugMF==1 ) {
	print "debug - sendJob 4 : isItSent: ${isItSent} \n";
	print "debug - sendJob 5 : command: ${commandToSend} \n";
    }
    
    return  $isItSent;
}

sub getLMRNumber
{

    my $fed=$_[0];
    my $side=$_[1];

    my $iEEM=0;
    my $iEBM=1;
    my $iEBP=2;
    my $iEEP=3;
    
    
    if($side<0 || $side>1 ){
	print "Wrong side $side \n"<<endl;
	die;
    }
    
    my $fedmax=600;
    my $lmmin=1;
    my $lmmax=54;

    if( $fed > $fedmax ){
	$fed=$fed-$fedmax;
    }
    if( $fed < $lmmin || $fed> $lmmax ){
	print "Wrong dcc num $fed \n"<<endl;
	die;
    }
    

    ##

    my $ireg=0; 
    my $lmr=1;
    
    if( $fed<=9 ){
	$ireg = $iEEM;
    }else {
	$fed -= 9;
	if( $fed<=18 ){
	    $ireg = $iEBM;
	}else
	{
	    $fed -= 18;
	    if( $fed<=18 ){ 
		$ireg = $iEBP;
	    }else
	    {
		$fed -= 18;
		if( $fed<=9 ){
		    $ireg = $iEEP;
		}
		else{
		    die;
		}
	    }
	}
    }
    
    if( $ireg==$iEEM || $ireg==$iEEP )
    {
	if( $side==1 && $fed!=8 ) 
	{
	    return -1;	  
	}
	$lmr = $fed;
	if( $fed==9 ){
	    $lmr++;
	}
	if( $fed==8 && $side==1 ){
	    $lmr++;
	}
    }
    elsif( $ireg==$iEBM || $ireg==$iEBP )
    {
	$lmr = 2*($fed-1) + $side + 1;
    }
    else{
	die;
    }
    if( $ireg==$iEBP ){
	$lmr+=36;
    } elsif( $ireg==$iEEP ){
	$lmr+=72; 
    } elsif( $ireg==$iEEM ){
	$lmr+=82; 
    }

    return $lmr;   
}

sub send2Jobs{
    
    my $cfg1=$_[0];  # header, matacq, laser, testpulse
    my $cfg2=$_[1]; # for ps command
    my $nmaxjobs=$_[2]; 
    my $user=$_[3];
    my $jobdir=$_[4];
    my $scriptdir=$_[5];
    my $donefile=$_[6];
    my $nprocfile=$_[7];
    my $shortjobdir=$_[8];
    my $dormunzip=$_[9];

    if( $debugMF==1 ){
	print " debug - send2Jobs 1 : ${cfg} ${log} ${cfg1} ${cfg2} ${nmaxjobs} \n";
    }
    my $isItSent = 0;
    my $njobs = 8;
    my $nsub = 8;
    my $ntot = 8;
    my $timewaiting=0;
    my $command="";
    
    $command="ps x -u ${user} | grep cmsRun | grep py | grep -v 'ps x -u' |  wc -l"; 

    $njobs =`$command`;
    $njobs =$njobs+0;

    $command2="ps x -u ${user} | grep submitJobPyBoth | grep -v 'ps x -u' | wc -l";
        
    $nsub =`$command2`;
    $nsub =$nsub+0;

    $ntot=$nsub+$njobs;
    
    if( $debugMF==1 ) {
	print "debug - send2Jobs 2 : njobs= ${njobs}, nsub= ${nsub}, ntot= ${ntot} \n";
    }
    
    while( $nsub>=$nmaxjobs || $njobs>=$nmaxjobs ){
	
	sleep 5;
	$timewaiting+=5;

	$njobs =`$command`;  
	$njobs =$njobs+0;
	$nsub =`$command2`;  
	$nsub =$nsub+0;
	
	$ntot=$nsub+$njobs;
	
	if( $debugMF==1 ) {
	    print "debug - sendCMSJob 3 : inside loop njobs= ${njobs}, nsub= ${nsub}, ntot= ${ntot}  \n";
	    
	}
    }

    my $mydate = `date +%s`;
    print " ....... job sent at: ${mydate} after waiting ${timewaiting} seconds \n";
    print " njobs: ${nsub} ${njobs} \n";
    
    
    # update nprocfile
    my $nprocnew=0;
    if(! -e $nprocfile){
	$nprocnew=1;
    }else{
	my $nproc=`tail -1 ${nprocfile}`;
	$nprocnew=$nproc+1;
    }
    
    my $commandToSend = "${scriptdir}/submitJobPyBoth.csh ${cfg1} ${cfg2} ${jobdir} ${donefile} ${shortjobdir} ${nprocnew} ${dormunzip} &";
   
    # send job
    system $commandToSend ;
    $isItSent=1;
    
    system "echo ${nprocnew} >> ${nprocfile}";
    
    if( $debugMF==1 ) {
	print "debug - send2Jobs 4 : isItSent: ${isItSent} \n";
	print "debug - send2Jobs 5 : command: ${commandToSend} \n";
    }
    
    return  $isItSent;
}


sub rootNLSFileName
{
    
    my $fed=$_[0];
    my $side=$_[1];
    my $ts=$_[2];
    my $run=$_[3];
    my $lb=$_[4];
    my $path=$_[5];
    
    my $smname=getSMname($fed);
    my $filename="${path}/${smname}/LMF_${smname}_${side}_NLS_BlueLaser_Run${run}_LB${lb}_TS${ts}.root";
    return $filename;
    
}


sub rootLaserPrimFileName
{
    
    my $fed=$_[0];
    my $side=$_[1];
    my $ts=$_[2];
    my $run=$_[3];
    my $lb=$_[4];
    my $path=$_[5];
    
    my $smname=getSMname($fed);
    my $filename="${path}/${smname}/LMF_${smname}_${side}_BlueLaser_Run${run}_LB${lb}_TS${ts}.root";
    return $filename;
    
}
