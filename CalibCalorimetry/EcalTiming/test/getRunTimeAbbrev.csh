#!/bin/csh
############################################################################
# Simple program, outputs the seconds since Jan 1 1970 for a provided run
# Also, output teh length of the run rounded up to 15 mins.
#---------------------------------------------------------------------------
# This requires parse.pl to be present in the CWD
############################################################################

############################################################################
# Checking Arguments
if ($#argv < 1) then
    echo " scripts needs only 1 input variables GEESH give it the darn run number"
    exit
endif
############################################################################


set RUNNUM = $1
#echo "http://cmsmon.cern.ch/cmsdb/servlet/ECALSummary?RUN=$RUNNUM&RIUNROUP=ECAL"
wget --quiet -nv "http://cmsmon.cern.ch/cmsdb/servlet/ECALSummary?RUN=$RUNNUM&RUNGROUP=ECAL" -O rundate.html

#set rundates = ( `./parse.pl` )
set rundates =  `./parse.pl`
#echo $rundates 

set startdate = $rundates[1]
set enddate = $rundates[2]
#echo "------------------Event Timing for run $RUNNUM------------------"
#echo "Start in Seconds since Jan 1 1970 : " $rundates[1]
#echo "End in Seconds since Jan 1 1970   : " $rundates[2]

#OK OK, I should be shot for writing this in c-shell... oh well.
set runlengths = 0
set runlengthh = 0
set runlengthhp1 = 0
set runlengthhp900 = 0
set runlengthhp1800 = 0
set runlengthhp2700 = 0

set secondsinhour = 3600
@ runlengths =  $enddate - $startdate
@ runlengthh = $runlengths / $secondsinhour 
@ runlengthhp1 = ${runlengths} + 3599
@ runlengthhp1 = ${runlengthhp1} / ${secondsinhour}
@ runlengthhp900 = ${runlengths} + 2699
@ runlengthhp900 = ${runlengthhp900} / ${secondsinhour}
@ runlengthhp1800 = ${runlengths} + 1799  
@ runlengthhp1800 = ${runlengthhp1800} / ${secondsinhour}
@ runlengthhp2700 = ${runlengths} + 899  
@ runlengthhp2700 = ${runlengthhp2700} / ${secondsinhour}

set runlength = 0
if ( $runlengthhp2700 > $runlengthh  ) then
   set runlength = $runlengthhp2700
else if ( $runlengthhp1800 > $runlengthh  ) then
   set runlength = "${runlengthh}.75"
else if ( $runlengthhp900 > $runlengthh  ) then
   set runlength = "${runlengthh}.50"
else if ( $runlengthhp1 > $runlengthh ) then
   set runlength = "${runlengthh}.25"
endif

#echo "The run lasted " $runlengths " seconds"
#echo "The run lasted " $runlength " hours (rounded to 15 mins) "
#echo "--------------------If you use a cfg--------------------------"
#echo "replace ecalCosmicsHists.TimeStampStart = $startdate" 
#echo "replace ecalCosmicsHists.TimeStampLength = $runlength"
#echo "--------------------If you use the script---------------------"
#echo " -rl ${runlength} -st $startdate "
#echo "--------------------------------------------------------------"

echo "export startdate="$startdate
echo "export runlength="$runlength

rm -f rundate.html

#end of file

