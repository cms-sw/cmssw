#!/bin/bash

echo 
echo Please use DQMPATCH:04, i.e. type /nfshome0/cmssw/scripts/dqmpatchsetup.sh [-a] 4
echo 

if [ "$#" -lt 1 ] ; then  echo "  Usage $0 [-a ] dqmpatch" ;  exit  ; fi

while getopts a: opt
do
  case "$opt" in
     a) all="true" ; echo "  option a: will check out/update subsystem tags" ;  shift ;;
     h) echo "Option \"h\"" ; shift ;;
     *) all="false" ; break;;
  esac
done

dqmpatch=$1
echo "  dqmpatchversion: $dqmpatch"

if [ -z "$LOCALRT" ]; then echo "  First setup your scramv1 area for CMSSW_1_7_1 - exiting " ; exit ; fi
if [ $CMSSW_VERSION != CMSSW_1_7_1 ] ; then echo "  First setup your area for CMSSW_1_7_1 - exiting " ; exit ; fi
if [ $dqmpatch != "4" ]  ; then echo "  Please use patch version 4 " ; exit ; fi

echo "  installing tags for DQMPATCH:$dqmpatch in your project area:"
if [ $all != "true" ] ; 
then echo "  Note:"
echo "  only DQMServices and Integration packages will be checked out"
echo "  use option -a to checkout subsystem tags"
fi
#

cd $LOCALRT/src
source $CMS_PATH/utils/cmscvsroot.sh CMSSW


if [ $all == "true" ] ; 
then echo "  now checking out subsystem tags  ... " ; sleep 2

cvs co -r V00-05-14 DQMServices/Core
cvs co -r V00-05-12 DQMServices/CoreROOT
cvs co -r V00-05-17 DQMServices/Components
cvs co -r V00-05-11 DQMServices/Examples

cvs co -r V00-00-15 DQM/Integration
cvs co -r V00-00-13 DQM/RenderPlugins
cvs co -r V01-01-04 VisMonitoring/DQMServer

cvs co -r V00-16-05 DQM/DTMonitorModule
cvs co -r V00-03-04 DQM/DTMonitorClient

cvs co -r V00-07-10 DQM/HcalMonitorClient 
cvs co -r V00-07-08 DQM/HcalMonitorTasks
cvs co -r V00-07-09 DQM/HcalMonitorModule
cvs co -r 1.18 DQM/HcalMonitorModule/data/HcalMonitorModule.cfi

cvs co -r HEAD DQM/RPCMonitorDigi       
cvs co -r V01-01-02 Geometry/RPCGeometry

cvs co -r V02-00-03 DQM/CSCMonitorModule

cvs co -r V02-00-20 DQM/L1TMonitor       
cvs co -r V01-00-01 L1Trigger/HardwareValidation
cvs co -r V02-03-03-03 DataFormats/L1Trigger

cvs co -r V00-04-38   DQM/EcalBarrelMonitorClient
cvs co -r V00-04-38   DQM/EcalBarrelMonitorModule
cvs co -r V00-04-38   DQM/EcalBarrelMonitorTasks
cvs co -r V00-04-38   DQM/EcalCommon

cvs co -r V02-07-04        OnlineDB/EcalCondDB
cvs co -r V00-07-06        RecoLocalCalo/EcalRecAlgos
cvs co -r V01-00-08        RecoLocalCalo/EcalRecProducers
cvs co -r V01-01-12        DataFormats/EcalRawData
cvs co -r V00-06-10        EventFilter/EcalRawToDigiDev

cvs co -r V03-06-01 DQM/SiStripCommon
cvs co -r V03-06-02 DQM/SiStripMonitorCluster
cvs co -r V02-00-00 DQM/SiStripMonitorClient 
cvs co -r V01-04-29 DQM/SiStripMonitorPedestals
cvs co -r V00-00-00 DQM/TrackerCommon/test
cvs co -r V00-01-03 CommonTools/TrackerMap
cvs co -r CMSSW_1_7_2 EventFilter/Processor    

cvs co -r V01-08-10 EventFilter/CSCRawToDigi


fi

echo 
echo "  done. Now type scramv1 b to build the tags"

#scramv1 b



