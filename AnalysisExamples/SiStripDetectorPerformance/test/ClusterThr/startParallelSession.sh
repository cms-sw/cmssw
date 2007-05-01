#!/bin/sh

rm -f out.cmstac*

xterm -T "TIB" -sb -e "ssh cmstac07 'cd  /analysis/sw/StandardAnalysisRelease_DG/Development/ClusterTh/CMSSW_1_3_0_pre6/src/AnalysisExamples/SiStripDetectorPerformance/test/ClusterThr; pwd;   sleep 2;./startRunTIB.sh $1 8; sleep 10' | tee out.cmstac07.3" &

xterm -T "TID" -sb -e "ssh cmstac07 'cd  /analysis/sw/StandardAnalysisRelease_DG/Development/ClusterTh/CMSSW_1_3_0_pre6/src/AnalysisExamples/SiStripDetectorPerformance/test/ClusterThr; pwd;   sleep 2;./startRunTID.sh $1 8; sleep 10' | tee out.cmstac07.4" &

xterm -T "TOB" -sb -e "ssh cmstac12 'cd  /analysis/sw/StandardAnalysisRelease_DG/Development/ClusterTh/CMSSW_1_3_0_pre6/src/AnalysisExamples/SiStripDetectorPerformance/test/ClusterThr; pwd;   sleep 2;./startRunTOB.sh $1 8; sleep 10' | tee out.cmstac12.1" &

xterm -T "TEC" -sb -e "ssh cmstac12 'cd  /analysis/sw/StandardAnalysisRelease_DG/Development/ClusterTh/CMSSW_1_3_0_pre6/src/AnalysisExamples/SiStripDetectorPerformance/test/ClusterThr; pwd;   sleep 2;./startRunTEC.sh $1 8; sleep 10' | tee out.cmstac12.2" &
