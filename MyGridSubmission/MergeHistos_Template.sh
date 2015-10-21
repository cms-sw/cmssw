#!/bin/bash
source  /cvmfs/cms.cern.ch/cmsset_default.sh
source /cmshome/piet/slc6_ME0Seg_SLHC_BATCH
 
cd /lustre/home/piet/RunOnNode/ME0Analysis/ME0AnalysisDavid/Merge_SAMPLE/

### Copy All Files  ################################################################################
cp /cmshome/piet/SLC6/ME0_Studies/SLHC26_InTimeOutOfTimePU/CMSSW_6_2_0_SLHC26_patch3/src/MyGridSubmission/MergeList_SAMPLE.txt MergeList.txt
for i in `cat MergeList.txt`
do
  echo "copy /lustre/cms/store/user/piet/ME0Segment_Time/AnalysisDavid/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/SAMPLE/SUBMITDATE/0000/${i}"
  cp /lustre/cms/store/user/piet/ME0Segment_Time/AnalysisDavid/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/SAMPLE/SUBMITDATE/0000/${i} ${i}
done
### Merge All Files ################################################################################
hadd ME0MuonAnalyzerOutput_SAMPLE.root `cat MergeList.txt | awk '{printf $1" "}'`
### Clean Files     ################################################################################
for i in `cat MergeList.txt`
do
  echo "remove ${i}"
  rm -rf ${i}
done
rm -rf MergeList.txt
cp ME0MuonAnalyzerOutput_SAMPLE.root ..
cd ..
rm -rf Merge_SAMPLE
####################################################################################################