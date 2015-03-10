#!/bin/bash

usage='Usage: -r <run number>'
   
EBAMP=0.6
EEAMP=1.0
EBET=0.3
EEET=0.5

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
      -bxs) shift; BXS=$2;shift;;
      -orbits) shift; ORBITS=$2;shift;;
      -times) shift; TIMES=$2;shift;;
      -lumi) shift; LUMI=$2;shift;;
      -trig) shift; TRIG=$2;shift;;
      -ttrig) shift; TTRIG=$2;shift;;
      -ebemin) shift; EBAMP=$2;shift;;
      -eeemin) shift; EEAMP=$2;shift;;
      -ebetmin) shift; EBET=$2;shift;;
      -eeetmin) shift; EEET=$2;shift;;
  esac       
done

if [ "X"${run_num} == "X" ]
    then
    echo "INVALID RUN NUMBER! Please give a valid run number!"
    echo $usage
    exit 
fi

if [ "X"${analy_type} == "X" ]
    then
    analy_type="Laser"
    echo " using default analysis type of Laser"
else
    echo "using ${analy_type} type events  "
fi

APPENDIX=""
OPTIONS=""

if [ "X"${TRIG} != "X" ]
    then
    APPENDIX=$APPENDIX"_TRIG_"$TRIG
    OPTIONS=$OPTIONS" -trig "$TRIG
fi

if [ "X"${TTRIG} != "X" ]
    then
    APPENDIX=$APPENDIX"_TTRIG_"$TTRIG
    OPTIONS=$OPTIONS" -ttrig "$TTRIG
fi

if [ "X"${LUMI} != "X" ]
    then
    APPENDIX=$APPENDIX"_LUMI_"$LUMI
    OPTIONS=$OPTIONS" -lumi "$LUMI
fi

if [ "X"${BXS} != "X" ]
    then
    APPENDIX=$APPENDIX"_BXS_"$BXS
    OPTIONS=$OPTIONS" -bxs "$BXS
fi

if [ "X"${TIMES} != "X" ]
    then
    APPENDIX=$APPENDIX"_TIMES_"$TIMES
    OPTIONS=$OPTIONS" -times "$TIMES
fi

if [ "X"${ORBITS} != "X" ]
    then
    APPENDIX=$APPENDIX"_ORBITS_"$ORBITS
    OPTIONS=$OPTIONS" -orbits "$ORBITS
fi

OPTIONS=$OPTIONS" -ebemin "$EBAMP" -eeemin "$EEAMP" -ebetmin "$EBET" -eeetmin "$EEET

echo 'Making Laser Webpages for ' ${run_num}  
  

# specify directories here
#my_cmssw_base='/afs/cern.ch/cms/CAF/CMSCOMM/COMM_ECAL/ccecal/CRAFT_devel_321/src'
my_cmssw_base=$CMSSW_BASE/src
work_dir=/castor/cern.ch/user/c/ccecal/Timing
dwork_dir=${my_cmssw_base}/'CalibCalorimetry/EcalTiming/test'
Nrun_num=${run_num}${APPENDIX}

cd $dwork_dir;
eval `scramv1 ru -sh`;


plots_dir=plots/${analy_type}/$Nrun_num;
mkdir $plots_dir

#crab_dir=`\ls -rt1 ${work_dir}/${analy_type}_${run_num} | grep "crab_" | tail -1 | awk '{print $NF}'`;
crab_dir=/castor/cern.ch/user/c/ccecal/Timing
echo $crab_dir

#root_file=${analy_type}_${run_num}_1.root
Nroot_file=${analy_type}_${Nrun_num}.root
root_file=${analy_type}_${run_num}.root
plot_file=${analy_type}_${Nrun_num}_plots.root
#cp ${work_dir}/${analy_type}_${run_num}/${crab_dir}/res/${root_file} ${plots_dir}/${Nroot_file}
rfcp ${work_dir}/${root_file} ${plots_dir}/${Nroot_file}

echo
echo 'Going to make the plots, by running:'
echo  'EcalTimingTTreePlotter '${plots_dir}'/'${Nroot_file}' '${Nrun_num}' 1 png '${plots_dir}' 0 '${analy_type}' '${plot_file}' '$OPTIONS' > '${plots_dir}'/plotting.txt'
#echo '.L '${my_cmssw_base}'/CalibCalorimetry/EcalTiming/test/plotLaser.C'
#echo 'DrawLaserPlots("'${plots_dir}'/'${Nroot_file}'","'${Nrun_num}'",kTRUE,"png","'${plots_dir}'",kFALSE,"${analy_type}","'${plot_file}'")'
echo


EcalTimingTTreePlotter ${plots_dir}/${Nroot_file} ${Nrun_num} 1 png ${plots_dir} 0 ${analy_type} ${plot_file} $OPTIONS > ${plots_dir}/plotting.txt

#now I need to make a little python script to make my root plots

#######cat > ${plots_dir}/plot.py <<EOF

#from ROOT import gROOT
#from ROOT import gSystem
#######from ROOT import *

#load my macro
#######gSystem.Load('libFWCoreFWLite.so')
#######AutoLibraryLoader.enable();
#######gSystem.Load('libDataFormatsFWLite.so')
#######gSystem.Load('libDataFormatsPatCandidates.so')
#######gROOT.LoadMacro(  '${my_cmssw_base}/CalibCalorimetry/EcalTiming/test/plotLaser.C++')
#gSystem.Load('plotLaser_C.so')
#gROOT.ProcessLine(  '.L ${my_cmssw_base}/CalibCalorimetry/EcalTiming/test/plotLaser.C++')

#get my cute class
#######from ROOT import DrawLaserPlots

#######time=DrawLaserPlots("${plots_dir}/${root_file}","${run_num}",True,"png","${plots_dir}",False,"${analy_type}", "${plot_file}")
#######print time

####EOF

#####python ${plots_dir}/plot.py -b > ${plots_dir}/plotting.txt
run_num=${Nrun_num}
mytimel=`tail -n 1 ${plots_dir}/plotting.txt`
mytime=`date -ud @${mytimel}`
echo "The Beginnig of the Run is  $mytime"

#####rm ${plots_dir}/plot.py

ttreestuff=""

if [ "`ls ${plots_dir} |grep -c EBTIMES`" -gt "0" ]; then 
  ttreestuff="
 


<h4><A name=\"TTreeActive\">${analy_type} Active Crystals Plots</h4>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBCrys_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBCrys_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EECrys_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EECrys_${run_num}.png\"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBCrysToTime_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBCrysToTime_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPCrysToTime_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPCrysToTime_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMCrysToTime_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMCrysToTime_${run_num}.png\"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBCrysAmp_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBCrysAmp_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EECrysAmp_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EECrysAmp_${run_num}.png\"> </A>
<br>

<h4><A name=\"MyNewTTree\">${analy_type} TTree BX, Time, Trigger Occupancy, E1/E9, KSwissCross</h4>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_AbsTime_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_AbsTime_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_BX_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_BX_${run_num}.png\"> </A>


<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_Triggers_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_Triggers_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TechTriggers_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TechTriggers_${run_num}.png\"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_E1OE9EB_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_E1OE9EB_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_KSwissCrossEB_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_KSwissCrossEB_${run_num}.png\"> </A>


<h4><A name=\"TTreeAbs\"> Absolute Time to Average Event Timing </h4>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_AbsTimeVsEBPTime_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_AbsTimeVsEBPTime_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_AbsTimeVsEBMTime_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_AbsTimeVsEBMTime_${run_num}.png\"> </A>



<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_AbsTimeVsEEPTime_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_AbsTimeVsEEPTime_${run_num}.png\"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_AbsTimeVsEEMTime_${run_num}.png> <img height=\"200\" src=\"http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_AbsTimeVsEEMTime_${run_num}.png\"> </A>



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
<h3> Run Start Time Used ${mytime} </h3>
</center>


Jump to:<br>
<FONT color="Blue"> 
<A href="#TTree1d">Advanced 1-D Timing Histograms</A> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<A href="#TTree">EE+ to EE- and EB+ to EB- Comparison</A><BR>

<A href="#EB">ECAL Barrel</A>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<A href="#EEM">ECAL Endcap Minus Side</A>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<A href="#EEP">ECAL Endcap Plus Side</A><BR>
<A href="#ALL">ALL ECAL : EB+EE</A><BR>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<A href="#TTreeActive">Number of Active Crystal Plots</A><BR>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<A href="#MyNewTTree">1-D Timing, BX, Trigger Plots</A><BR>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<A href="#TTreeAbs">Abolute Time to Event Timing</A><BR>
<A href="expert.html">Expert Technical Plots</A><BR>

</FONT>
<br>Min Cuts for crystals. 
<br>EB: Amp > 15 ADC && E > ${EBAMP} GeV  && ET > ${EBET} GeV && timing error < 5ns
<br>EE: Amp > 15 ADC && E > ${EEAMP} GeV  && ET > ${EEET} GeV && timing error < 5ns




<h4><A name="TTree1d">Advanced 1-D Timing Histograms</h4>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMES_${run_num}.png> <img height="150" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMES_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBPlusTime_${run_num}.png> <img height="150" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBPlusTime_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBMinusTime_${run_num}.png> <img height="150" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBMinusTime_${run_num}.png"> </A>


<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTimeEtaLess5_${run_num}.png> <img height="150" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTimeEtaLess5_${run_num}.png"> </A>

<br>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMES_${run_num}.png> <img height="150" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMES_${run_num}.png"> </A> 

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPTIMES_${run_num}.png> <img height="150" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPTIMES_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMTIMES_${run_num}.png> <img height="150" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMTIMES_${run_num}.png"> </A>



<h4><A name="TTree">Time Comparison Plots</h4>
 Both EE+ and EE- must be present to fill these plots, or both EB+ and EB- </br>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPTimeToEEMTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPTimeToEEMTime_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPDiffEEMTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPDiffEEMTime_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPDiffEEMTimeCrys_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPDiffEEMTimeCrys_${run_num}.png"> </A>

<br>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPTIMESB_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPTIMESB_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMTIMESB_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMTIMESB_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPTIMESBH_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPTIMESBH_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMTIMESBH_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMTIMESBH_${run_num}.png"> </A>



<br>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBPlus2Minus_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBPlus2Minus_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaRelBHP_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaRelBHP_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaRelBHM_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaRelBHM_${run_num}.png"> </A>


<h4><A name="TTreeProfs">Time Projection Profiles</h4>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeCHAllFEDsEtaRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeCHAllFEDsEtaRel_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaRel_${run_num}.png"> </A>

<br>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaEEMRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaEEMRel_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaEEPRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTAllFEDsEtaEEPRel_${run_num}.png"> </A>


<h3><A name="EB"><FONT color="Blue">ECAL Barrel General Plots</FONT></A><BR></h3>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeCHProfileRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeCHProfileRel_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTProfileRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_timeTTProfileRel_${run_num}.png"> </A>

<br>
    
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_fullAmpProfileEB_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_fullAmpProfileEB_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_OccuCHProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_OccuCHProfile_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_OccuTTProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_OccuTTProfile_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_OccuCHProfileBad_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_OccuCHProfileBad_${run_num}.png"> </A>



<br>

<h3><A name="EEM"><FONT color="Blue">ECAL Endcap Minus Side</FONT></A><BR></h3>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMtimeCHProfileRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMtimeCHProfileRel_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMtimeTTProfileRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMtimeTTProfileRel_${run_num}.png"> </A>

<br>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_fullAmpProfileEEM_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_fullAmpProfileEEM_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMOccuCHProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMOccuCHProfile_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMOccuTTProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMOccuTTProfile_${run_num}.png"> </A>


<h3><A name="EEP"><FONT color="Blue">ECAL Endcap Plus Side</FONT></A><BR></h3>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPtimeCHProfileRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPtimeCHProfileRel_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPtimeTTProfileRel_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPtimeTTProfileRel_${run_num}.png"> </A>

<br>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_fullAmpProfileEEP_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_fullAmpProfileEEP_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPOccuCHProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPOccuCHProfile_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPOccuTTProfile_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPOccuTTProfile_${run_num}.png"> </A>

<h3><A name="XtalTTAVEs"><FONT color="Blue">Crystal and TT Average Times (over all events)</FONT></A><BR></h3>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTTTIME_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTTTIME_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBCHTIME_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBCHTIME_${run_num}.png"> </A>

<br>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMTTTIME_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMTTTIME_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMCHTIME_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEMCHTIME_${run_num}.png"> </A>

<br>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPTTTIME_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPTTTIME_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPCHTIME_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEPCHTIME_${run_num}.png"> </A>


<br>

<h3><A name="ALL"><FONT color="Blue">ALL ECAL EB+EE</FONT></A><BR></h3>

<br>

${ttreestuff}

<br>
<h2><A href="expert.html">Expert Technical Plots</A><BR></h2>

<h4> ROOT File (download) </h4>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${Nroot_file}> ${Nroot_file} </A>
<h4> Plot ROOT File (download) </h4>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${plot_file}> ${plot_file} </A>

</FONT>
</BODY>
</HTML>

EOF

cat > ${plots_dir}/expert.html <<EOF

<HTML>

<HEAD><TITLE>ECAL ${analy_type} Analysis Run ${run_num}</TITLE></HEAD>
 
<BODY link="Red">
<FONT color="Black">
 
<Center>
<h2><FONT color="Blue"> ECAL ${analy_type} Analysis: EXPERT ADDITIONAL PLOTS </FONT></h2>
</Center>

<Center>
<h3> Run: <A href=http://cmsmon.cern.ch/cmsdb/servlet/RunSummary?RUN=${run_num}>${run_num}</A></h3>
<h3> Run Start Time Used ${mytime} </h3>
</center>

<A href="#SMPLOTS">Supermodule and LM Average Timings</A><BR>
<A href="#TTreeOccupancy">Detailed Occupancy Plots</A><BR>
<A href="#TTreeExpert">Expert Timing Plots</A><BR>
<A href="#TTreeOG">Generic Timing Plots</A><BR>
<A href="#TTreeOG2">Crystal Time/Amplitude to Time Errors</A><BR>
<A href="#TTreeExpertTech">Expert Technical Plots</A><BR>
<A href="#TTreeTrigs">Triggers Active By Region</A><BR>
<A href="#TTreeBX">BX By Region</A><BR>
<A href="#TTreeMany">2-D Relationg of BX, Trigger and Absolute Time</A><BR>


<h4><A name="SMPLOTS"> Supermodule (SM) and Light Monitoring Region (LM) Average Timing</A></h4>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_LM_timingCorrected_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_LM_timingCorrected_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_SM_timingCorrected_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_SM_timingCorrected_${run_num}.png"> </A>



<h4><A name="TTreeOccupancy">${analy_type} Occupancy Plots</h4>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBHashed_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBHashed_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEHashed_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEHashed_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBHashedToTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBHashedToTime_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEHashedToTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEHashedToTime_${run_num}.png"> </A>

<h4><A name="TTreeExpert"> Expert Timing Plots </h4>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TTMeanWithRMS_All_FEDS_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TTMeanWithRMS_All_FEDS_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_RelRMS_vs_AbsTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_RelRMS_vs_AbsTime_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_Rel_TimingSigma_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_Rel_TimingSigma_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_Inside_TT_timing_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_Inside_TT_timing_${run_num}.png"> </A>

<h4><A name="TTreeOG"> ${analy_type} </h4>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_XtalsPerEvt_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_XtalsPerEvt_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_laserShift_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_laserShift_${run_num}.png"> </A>


<h4><A name="TTreeOG2"> Relating Average Crystal Time to Individual Crystal times and the timing errors</h4>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMEStoAverage_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMEStoAverage_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMEStoAverage_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMEStoAverage_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMEStoTERR_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMEStoTERR_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMEStoTERR_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMEStoTERR_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMEStoAMP_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMEStoAMP_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMEStoAMP_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMEStoAMP_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMEStoET_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMEStoET_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMEStoET_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMEStoET_${run_num}.png"> </A>

<h4><A name="TTreeExpertTech">Expert Technical Plots, with Amplitude and Time Error and shape</h4>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBEvtTIMEStoAMP_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBEvtTIMEStoAMP_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEEvtTIMEStoAMP_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EEEvtTIMEStoAMP_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMESErrtoAMP_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EBTIMESErrtoAMP_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMESErrtoAMP_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_EETIMESErrtoAMP_${run_num}.png"> </A>

<br>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_E1OverE9VsEBTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_E1OverE9VsEBTime_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_KSwissCrossVsEBTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_KSwissCrossVsEBTime_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_SuperDiscrimVsEBTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_SuperDiscrimVsEBTime_${run_num}.png"> </A>

<h4><A name="TTreeTrigs">Triggers to Event Timing </h4>

<br>Both Active Physics Trigger bits and with respect to Active Technical Trigger Bits are shown. </br>

<br>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TriggerVsEBPTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TriggerVsEBPTime_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TriggerVsEBMTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TriggerVsEBMTime_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TriggerVsEEPTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TriggerVsEEPTime_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TriggerVsEEMTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TriggerVsEEMTime_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TechTriggerVsEBPTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TechTriggerVsEBPTime_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TechTriggerVsEBMTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TechTriggerVsEBMTime_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TechTriggerVsEEPTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TechTriggerVsEEPTime_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TechTriggerVsEEMTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TechTriggerVsEEMTime_${run_num}.png"> </A>

<h4><A name="TTreeBX">BX to Average Event Timings </h4>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_BXVsEBPTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_BXVsEBPTime_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_BXVsEBMTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_BXVsEBMTime_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_BXVsEEPTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_BXVsEEPTime_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_BXVsEEMTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_BXVsEEMTime_${run_num}.png"> </A>

<h4><A name="TTreeMany">Relating the Triggers to BX to Absolute Timings </h4>


<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TriggerVsAbsTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TriggerVsAbsTime_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TechTriggerVsAbsTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TechTriggerVsAbsTime_${run_num}.png"> </A>



<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_hBXVsAbsTime_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_hBXVsAbsTime_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TriggerVsBX_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TriggerVsBX_${run_num}.png"> </A>


<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TechTriggerVsBX_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/${analy_type}Analysis/Beam10/${run_num}/${analy_type}Analysis_TechTriggerVsBX_${run_num}.png"> </A>


EOF

exit
