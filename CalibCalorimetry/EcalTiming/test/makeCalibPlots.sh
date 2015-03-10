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
#      -at) shift; analy_type=$2;shift;;
  esac      
done

if [ "X"${run_num} == "X" ]
    then
    echo "INVALID RUN NUMBER! Please give a valid run number!"
    echo $usage
    exit 
fi

#if [ "X"${analy_type} == "X" ]
#    then
#    analy_type="Laser"
#    echo " using default analysis type of Laser"
#else
#    echo "using ${analy_type} type events  "
#fi

echo 'Making Calib Webpages for ' ${run_num} 


# specify directories here
work_dir=${CMSSW_BASE}'/src/CalibCalorimetry/EcalTiming/test/SCRIPTS'

plots_dir=plots/Calib/$run_num;
mkdir $plots_dir

crab_dir=`\ls -rt1 ${work_dir}/Calib_${run_num} | grep "crab_" | tail -1 | awk '{print $NF}'`;

#root_file=Calib_${run_num}.root
root_file=Calib_${run_num}_1.root
cp ${work_dir}/Calib_${run_num}/${crab_dir}/res/${root_file} ${plots_dir}

echo
echo 'Now making plots, by running these command in ROOT:'
echo
echo '.L '${CMSSW_BASE}'/src/CalibCalorimetry/EcalTiming/test/SCRIPTS/plotCalib.C'
echo 'DrawCalibPlots("'${plots_dir}'/'${root_file}'",'${run_num}',kTRUE,"png","'${plots_dir}'",kFALSE)'
echo

#now I need to make a little python script to make my root plots

cat > ${plots_dir}/plot.py <<EOF

from ROOT import gROOT

#load my macro
gROOT.LoadMacro(  '${CMSSW_BASE}/src/CalibCalorimetry/EcalTiming/test/plotCalib.C')

#get my cute class
from ROOT import DrawCalibPlots

DrawCalibPlots("${plots_dir}/${root_file}",${run_num},True,"png","${plots_dir}",False)


EOF


python ${plots_dir}/plot.py -b

rm ${plots_dir}/plot.py


cat > ${plots_dir}/index.html <<EOF


<HTML>

<HEAD><TITLE>ECAL Calibration Analysis Run ${run_num}</TITLE></HEAD>
 
<BODY link="Red">
<FONT color="Black">
 
<Center>
<h2><FONT color="Blue"> ECAL Calibration Analysis: </FONT></h2>
</Center>

<Center>
<h3> Run: <A href=http://cmsmon.cern.ch/cmsdb/servlet/RunSummary?RUN=${run_num}>${run_num}</A></h3>
</center>

<br>
<A href=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/LaserAnalysis/MWGR/${run_num}>Link to Laser Analysis</A>

<h3><A name="EB"><FONT color="Blue">ECAL Barrel</FONT></A><BR></h3>

<A href=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/Calib_EB.html>link to FED-by-FED plots for EB</A></h3>

<h3><A name="EEM"><FONT color="Blue">ECAL Endcaps</FONT></A><BR></h3>

<A href=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/Calib_EE.html>link to FED-by-FED plots for EE</A></h3>

<h3><A name="ALL"><FONT color="Blue">ALL ECAL EB+EE</FONT></A><BR></h3>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_dccHeaderErrors${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_dccHeaderErrors${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_simpleDataRunType_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_simpleDataRunType_${run_num}.png"> </A>

<br>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_dccHeadRunType_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_dccHeadRunType_${run_num}.png"> </A>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_simpleHeadRunType_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_simpleHeadRunType_${run_num}.png"> </A>

<br>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_LaserFEDCycle_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_LaserFEDCycle_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_PedFEDCycle_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_PedFEDCycle_${run_num}.png"> </A>

<br>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_TestPulseFEDCycle_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_TestPulseFEDCycle_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_UnknownFEDCycle_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_UnknownFEDCycle_${run_num}.png"> </A>

<br>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_RunTypeByCycle_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_RunTypeByCycle_${run_num}.png"> </A>

<h3><A name="ALL"><FONT color="Blue">RunType Vs. BX all FEDs</FONT></A><BR></h3>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_dataRunTypeVsBX_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_dataRunTypeVsBX_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_zoomdataRunTypeVsBX_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_zoomdataRunTypeVsBX_${run_num}.png"> </A>

<br>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_simpledccRunTypeVsBX_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_simpledccRunTypeVsBX_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_zoomsimpledccRunTypeVsBX_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_zoomsimpledccRunTypeVsBX_${run_num}.png"> </A>

<br>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_dccRunTypeVsBX_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_dccRunTypeVsBX_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_zoomdccRunTypeVsBX_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_zoomdccRunTypeVsBX_${run_num}.png"> </A>

<h3><A name="ALL"><FONT color="Blue">RunType Vs. BX for DCC in TCC</FONT></A><BR></h3>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_simpledccInTCCRunTypeVsBX_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_simpledccInTCCRunTypeVsBX_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_zoomsimpledccInTCCRunTypeVsBX_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_zoomsimpledccInTCCRunTypeVsBX_${run_num}.png"> </A>

<br>

<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_dccInTCCRunTypeVsBX_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_dccInTCCRunTypeVsBX_${run_num}.png"> </A>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_zoomdccInTCCRunTypeVsBX_${run_num}.png> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/CalibAnalysis_zoomdccInTCCRunTypeVsBX_${run_num}.png"> </A>


<h4> ROOT File (download) </h4>
<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}/${root_file}> ${root_file} </A>

</FONT>
</BODY>
</HTML>

EOF



# barrel plots

cat > ${plots_dir}/temp1 <<EOF


<HTML>

<HEAD><TITLE>ECAL EB Calibration Analysis Run ${run_num}</TITLE></HEAD>
 
<BODY link="Red">
<FONT color="Black">
 
<Center>
<h2><FONT color="Blue"> ECAL EB Calibration Analysis: </FONT></h2>
</Center>

<Center>
<h3> Run: <A href=http://cmsmon.cern.ch/cmsdb/servlet/RunSummary?RUN=${run_num}>${run_num}</A></h3>
</center>

<br>
<A href=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}>Link back to main Calib Analysis page</A>
<br>
<br>
EOF


for plot in `/bin/ls ${plots_dir} | grep EB- | grep png | grep aveampHist `; do
   echo '<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/'${run_num}'/'${plot}'> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/'${run_num}'/'${plot}'"> </A>' >> ${plots_dir}/temp2
done

for plot in `/bin/ls ${plots_dir} | grep EB+ | grep png | grep aveampHist `; do
   echo '<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/'${run_num}'/'${plot}'> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/'${run_num}'/'${plot}'"> </A>' >> ${plots_dir}/temp2
done


for plot in `/bin/ls ${plots_dir} | grep EB- | grep png | grep DccandDataHist `; do
   echo '<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/'${run_num}'/'${plot}'> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/'${run_num}'/'${plot}'"> </A>' >> ${plots_dir}/temp2
done

for plot in `/bin/ls ${plots_dir} | grep EB+ | grep png | grep DccandDataHist `; do
   echo '<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/'${run_num}'/'${plot}'> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/'${run_num}'/'${plot}'"> </A>' >> ${plots_dir}/temp2
done

cat ${plots_dir}/temp1 ${plots_dir}/temp2 > ${plots_dir}/Calib_EB.html
rm -f ${plots_dir}/temp1 ${plots_dir}/temp2

# endcap plots

cat > ${plots_dir}/temp1 <<EOF


<HTML>

<HEAD><TITLE>ECAL EE Calibration Analysis Run ${run_num}</TITLE></HEAD>
 
<BODY link="Red">
<FONT color="Black">
 
<Center>
<h2><FONT color="Blue"> ECAL EE Calibration Analysis: </FONT></h2>
</Center>

<Center>
<h3> Run: <A href=http://cmsmon.cern.ch/cmsdb/servlet/RunSummary?RUN=${run_num}>${run_num}</A></h3>
</center>

<br>
<A href=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/${run_num}>Link back to main Calib Analysis page</A>
<br>
<br>
EOF


for plot in `/bin/ls ${plots_dir} | grep EE- | grep png | grep aveampHist `; do
   echo '<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/'${run_num}'/'${plot}'> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/'${run_num}'/'${plot}'"> </A>' >> ${plots_dir}/temp2
done

for plot in `/bin/ls ${plots_dir} | grep EE+ | grep png | grep aveampHist `; do
   echo '<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/'${run_num}'/'${plot}'> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/'${run_num}'/'${plot}'"> </A>' >> ${plots_dir}/temp2
done


for plot in `/bin/ls ${plots_dir} | grep EE- | grep png | grep DccandDataHist `; do
   echo '<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/'${run_num}'/'${plot}'> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/'${run_num}'/'${plot}'"> </A>' >> ${plots_dir}/temp2
done

for plot in `/bin/ls ${plots_dir} | grep EE+ | grep png | grep DccandDataHist `; do
   echo '<A HREF=http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/'${run_num}'/'${plot}'> <img height="200" src="http://test-ecal-cosmics.web.cern.ch/test-ecal-cosmics/CalibAnalysis/MWGR/'${run_num}'/'${plot}'"> </A>' >> ${plots_dir}/temp2
done


cat ${plots_dir}/temp1 ${plots_dir}/temp2 > ${plots_dir}/Calib_EE.html

rm -f ${plots_dir}/temp1 ${plots_dir}/temp2
exit
