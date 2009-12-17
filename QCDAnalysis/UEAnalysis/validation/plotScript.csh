#!/bin/bash

DIR=$1_`date '+%d%m20%y_%H%M'`
if [ ! -d ${DIR} ]; then
  mkdir ${DIR}
else
  rm -rf ${DIR}/*
fi

#FILENAMEinput_DATA=/data1/livio/CMSSW_3_3_4/src/RecoTracker/TrackProducer/test/123596-rerecoBS.root
#FILENAMEinput_DATA=ntuplizedEvent_123596AllLumi_refitBS.root
#FILENAMEinput_MC=/data2/lucaroni/Tracks/CMSSW_3_3_4/src/RecoTracker/TrackProducer/test/ntuplizedEvent_STARTUP3XV8.root
#FILENAMEinput_DATA=ntuplized_testBS.root
#FILENAMEinput_MC=ntuplized_MCalaAndrea.root
#FILENAMEinput_MC=ntuplized_MCstartupLowStat.root
#FILENAMEinput_MC=ntuplized_MC_10xAPE_vertexAsData.root
#FILENAMEinput_MC=ntuplized_MC_vertexAsData.root
#FILENAMEinput_DATA=ntuplized_123596_lumi69-143.root

#FILENAMEinput_DATA=ntuplized_123596ReRecoed_BSCNOBEAMHALOskim.root
FILENAMEinput_DATA=ntuplized_ReRecoed_BSCNOBEAMHALOskim_clusterInfo.root
#FILENAMEinput_MC=ntuplized_100pTSingleMuon.root
FILENAMEinput_MC=testingCompatibilityCutWithAndreaOnMC.root
beamspot_DATA=-2.64157 
#beamspot_DATA=-2.2444 
#beamspot_MC=0.0330976
beamspot_MC=0.110321

rootmacro=template_ntupleViewer.C

#cp ${FILENAMEinput_DATA} ${DIR}
#cp ${FILENAMEinput_MC} ${DIR}
cp plotScript.csh ${DIR}
cp ${rootmacro} ${DIR}

cd $DIR

echo '~~> real data sequence'
 sed "s#STRINGrealdata#true#"                   < ../${rootmacro} > temp
 sed "s#STRINGbeamspot#${beamspot_DATA}#"       < temp > temp2
 sed "s#STRINGmarkerstyle#22#"                  < temp2 > temp3
 sed "s#STRINGinputfile#../${FILENAMEinput_DATA}#" < temp3 > temp4
 sed "s#STRINGoutfile#histosDATA.root#"         < temp4 > temp5
 sed "s#STRINGhistname#DATA_#"                  < temp5 > ntupleViewer.C 
 root -l -x -b ntupleViewer.C++ -q
 rm -f temp temp2 temp3 temp4 temp5 ntupleViewer.C


echo '~~> MC data sequence'
 sed "s#STRINGrealdata#false#"                  < ../${rootmacro} > temp
 sed "s#STRINGbeamspot#${beamspot_MC}#"         < temp > temp2
 sed "s#STRINGmarkerstyle#25#"                  < temp2 > temp3
 sed "s#STRINGinputfile#../${FILENAMEinput_MC}#"   < temp3 > temp4
 sed "s#STRINGoutfile#histosMC.root#"           < temp4 > temp5
 sed "s#STRINGhistname#MC_#"                    < temp5 > ntupleViewer.C 
## root -l -x -b ntupleViewer.C++ -q
 rm -f temp temp2 temp3 temp4 temp5 ntupleViewer.C

#echo '~~> drawing and saving image files'
#root -l -x -b finalPlots.C -q
