#!/bin/sh

[ "$1" == "" ] && echo -e "\nplease, specify a run number\n" && exit

rm -f out.cmstac*

rm -f badStrips.bin
cp -v `echo /data1/CrabAnalysis/ClusterAnalysis/FNAL_pre6_v17/ClusterAnalysis_TIBTOB_run*$1/res/*run*$1_*HotStrips.bin` badStrips.bin

xterm -T "TIB" -sb -e "ssh cmstac07 'cd  /analysis/sw/StandardAnalysisRelease_DG/Development/ClusterTh/CMSSW_1_3_0_pre6/src/AnalysisExamples/SiStripDetectorPerformance/test/ClusterThr; pwd;   sleep 2;./startRunTIB.sh $1 8; sleep 10' | tee out.cmstac07.TIB.$$" &

xterm -T "TID" -sb -e "ssh cmstac07 'cd  /analysis/sw/StandardAnalysisRelease_DG/Development/ClusterTh/CMSSW_1_3_0_pre6/src/AnalysisExamples/SiStripDetectorPerformance/test/ClusterThr; pwd;   sleep 2;./startRunTID.sh $1 8; sleep 10' | tee out.cmstac07.TID.$$" &

xterm -T "TOB" -sb -e "ssh cmstac12 'cd  /analysis/sw/StandardAnalysisRelease_DG/Development/ClusterTh/CMSSW_1_3_0_pre6/src/AnalysisExamples/SiStripDetectorPerformance/test/ClusterThr; pwd;   sleep 2;./startRunTOB.sh $1 8; sleep 10' | tee out.cmstac12.TOB.$$" &

xterm -T "TEC" -sb -e "ssh cmstac12 'cd  /analysis/sw/StandardAnalysisRelease_DG/Development/ClusterTh/CMSSW_1_3_0_pre6/src/AnalysisExamples/SiStripDetectorPerformance/test/ClusterThr; pwd;   sleep 2;./startRunTEC.sh $1 8; sleep 10' | tee out.cmstac12.TEC.$$" &
