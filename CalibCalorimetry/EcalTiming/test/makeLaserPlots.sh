#!/bin/bash

usage='Usage: -r <run number>'

args=`getopt r: -- "$@"`
if test $? != 0
     then
         echo $usage
         exit 1
fi

eval set -- "$args"
for i 
  do
  case "$i" in
      -r) shift; run_num=$2;shift;;
      -at) shift; analy_type=$2;shift;;
  esac      
done

if [ "X"${run_num} == "X" ]
    then
    echo "INVALID RUN NUMBER! Please give a valid run number!"
    echo $usage
    exit 
fi


if [ ${analy_type} == "Timing" ]
    then
    echo " please use makeTimingPlots.sh for timing plots "
    echo " ./makeTimingPlots.sh -r ${run_num} -at Timing"
    exit
fi

if [ "X"${analy_type} == "X" ]
    then
    analy_type="Laser"
    echo " using default analysis type of Laser"
else
    echo "using ${analy_type} type events  "
fi

echo 'Making Laser Webpages for ' ${run_num} 




# specify directories here
#my_cmssw_base='/afs/cern.ch/cms/CAF/CMSCOMM/COMM_ECAL/ccecal/CRAFT_devel_321/src'
my_cmssw_base=$CMSSW_BASE/src
work_dir=${my_cmssw_base}/'CalibCalorimetry/EcalTiming/test'

plots_dir=plots/${analy_type}/$run_num;
mkdir $plots_dir

crab_dir=`\ls -rt1 ${work_dir}/${analy_type}_${run_num} | grep "crab_" | tail -1 | awk '{print $NF}'`;

echo $crab_dir

#root_file=${analy_type}_${run_num}_1.root
root_file=${analy_type}_${run_num}.root
cp ${work_dir}/${analy_type}_${run_num}/${crab_dir}/res/${root_file} ${plots_dir}

echo
echo 'Going to make the plots, by running in ROOT:'
echo
echo '.L '${my_cmssw_base}'/CalibCalorimetry/EcalTiming/test/plotLaser.C'
echo 'DrawLaserPlots("'${plots_dir}'/'${root_file}'",'${run_num}',kTRUE,"png","'${plots_dir}'",kFALSE,"${analy_type}")'
echo

#now I need to make a little python script to make my root plots

cat > ${plots_dir}/plot.py <<EOF

from ROOT import gROOT

#load my macro
gROOT.LoadMacro(  '${my_cmssw_base}/CalibCalorimetry/EcalTiming/test/plotLaser.C')

#get my cute class
from ROOT import DrawLaserPlots

DrawLaserPlots("${plots_dir}/${root_file}",${run_num},True,"png","${plots_dir}",False,"${analy_type}")


EOF

python ${plots_dir}/plot.py -b

rm ${plots_dir}/plot.py

ttreestuff=""

if [ "`ls ${plots_dir} |grep -c EBTIMES`" -gt "0" ]; then 
  ttreestuff="
 
<h4><A name=\"TTree\">${analy_type} Tree Plots</h4>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMES_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMES_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMES_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMES_${run_num}.png\"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMESFILT_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMESFILT_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMESFILT_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMESFILT_${run_num}.png\"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPTIMES_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPTIMES_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMTIMES_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMTIMES_${run_num}.png\"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMEStoAverage_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMEStoAverage_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMEStoAverage_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMEStoAverage_${run_num}.png\"> </A>

<br> 
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMEStoTERR_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMEStoTERR_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMEStoTERR_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMEStoTERR_${run_num}.png\"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMEStoAMP_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMEStoAMP_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMEStoAMP_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMEStoAMP_${run_num}.png\"> </A>

<br>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBEvtTIMEStoAMP_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBEvtTIMEStoAMP_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEEvtTIMEStoAMP_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEEvtTIMEStoAMP_${run_num}.png\"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMESErrtoAMP_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMESErrtoAMP_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMESErrtoAMP_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMESErrtoAMP_${run_num}.png\"> </A>

<br>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBHashed_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBHashed_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEHashed_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEHashed_${run_num}.png\"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBHashedToTime_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBHashedToTime_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEHashedToTime_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEHashedToTime_${run_num}.png\"> </A>
<br>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBCrys_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBCrys_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EECrys_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EECrys_${run_num}.png\"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBCrysToTime_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBCrysToTime_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPCrysToTime_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPCrysToTime_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMCrysToTime_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMCrysToTime_${run_num}.png\"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPTimeToEEMTime_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPTimeToEEMTime_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPDiffEEMTime_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPDiffEEMTime_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPDiffEEMTimeCrys_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPDiffEEMTimeCrys_${run_num}.png\"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBCrysAmp_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBCrysAmp_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EECrysAmp_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EECrysAmp_${run_num}.png\"> </A>


"
fi

cat > ${plots_dir}/index.html <<EOF


<HTML>

<HEAD><TITLE>ECAL ${analy_type} Analysis Run ${run_num}</TITLE></HEAD>
 
<BODY link="Red">
<FONT color="Black">
 
<Center>
<h2><FONT color="Blue"> ECAL ${analy_type} Analysis: </FONT></h2>
</Center>

<Center>
<h3> Run: <A href=http://cmsmon.cern.ch/cmsdb/servlet/RunSummary?RUN=${run_num}>${run_num}</A></h3>
</center>


Jump to:<br>
<FONT color="Blue"> 
<A href="#EB">ECAL Barrel</A><BR>
<A href="#EEM">ECAL Endcap Minus Side</A><BR>
<A href="#EEP">ECAL Endcap Plus Side</A><BR>
<A href="#ALL">ALL ECAL : EB+EE</A><BR>
</FONT>
<br>
<A href=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/Beam10/${run_num}>Link to Calibration Analysis</A>

<h3><A name="EB"><FONT color="Blue">ECAL Barrel</FONT></A><BR></h3>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeCHProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeCHProfile_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeCHProfileRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeCHProfileRel_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTProfile_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTProfileRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTProfileRel_${run_num}.png"> </A>
<br>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeCHAllFEDsEta_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeCHAllFEDsEta_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeCHAllFEDsEtaRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeCHAllFEDsEtaRel_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEta_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEta_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaRel_${run_num}.png"> </A>

<br>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTTTIME_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTTTIME_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBCHTIME_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBCHTIME_${run_num}.png"> </A>



<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_fullAmpProfileEB_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_fullAmpProfileEB_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_OccuCHProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_OccuCHProfile_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_OccuTTProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_OccuTTProfile_${run_num}.png"> </A>


<br>

<h3><A name="EEM"><FONT color="Blue">ECAL Endcap Minus Side</FONT></A><BR></h3>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMtimeCHProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMtimeCHProfile_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMtimeCHProfileRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMtimeCHProfileRel_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMtimeTTProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMtimeTTProfile_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMtimeTTProfileRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMtimeTTProfileRel_${run_num}.png"> </A>

 <br>
 
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaEEM_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaEEM_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaEEMRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaEEMRel_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMTTTIME_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMTTTIME_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMCHTIME_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMCHTIME_${run_num}.png"> </A>



<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_fullAmpProfileEEM_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_fullAmpProfileEEM_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMOccuCHProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMOccuCHProfile_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMOccuTTProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMOccuTTProfile_${run_num}.png"> </A>


<h3><A name="EEP"><FONT color="Blue">ECAL Endcap Plus Side</FONT></A><BR></h3>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPtimeCHProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPtimeCHProfile_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPtimeCHProfileRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPtimeCHProfileRel_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPtimeTTProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPtimeTTProfile_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPtimeTTProfileRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPtimeTTProfileRel_${run_num}.png"> </A>
 <br>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaEEP_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaEEP_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaEEPRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaEEPRel_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPTTTIME_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPTTTIME_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPCHTIME_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPCHTIME_${run_num}.png"> </A>



<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_fullAmpProfileEEP_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_fullAmpProfileEEP_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPOccuCHProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPOccuCHProfile_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPOccuTTProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPOccuTTProfile_${run_num}.png"> </A>



<br>

<h3><A name="ALL"><FONT color="Blue">ALL ECAL EB+EE</FONT></A><BR></h3>

<h4> ${analy_type} </h4>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_XtalsPerEvt_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_XtalsPerEvt_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_laserShift_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_laserShift_${run_num}.png"> </A>

<br>

<h4> Timing from ${analy_type} </h4>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_LM_timing_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_LM_timing_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_LM_timingCorrected_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_LM_timingCorrected_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_SM_timing_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_SM_timing_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_SM_timingCorrected_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_SM_timingCorrected_${run_num}.png"> </A>


<br>


<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TTMeanWithRMS_All_FEDS_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TTMeanWithRMS_All_FEDS_${run_num}.png"> </A>



<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_RelRMS_vs_AbsTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_RelRMS_vs_AbsTime_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_Rel_TimingSigma_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_Rel_TimingSigma_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_Inside_TT_timing_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_Inside_TT_timing_${run_num}.png"> </A>

${ttreestuff}



<h4> ROOT File (download) </h4>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${root_file}> ${root_file} </A>

</FONT>
</BODY>
</HTML>

EOF

exit
