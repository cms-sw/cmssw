#!/bin/tcsh
tar -czvf ~cctrack/scratch0/DQM/${CMSSW_VERSION}/output/${1}.tar.gz ../*/${1}
rm -r ../*/${1}
scp /afs/cern.ch/cms/CAF/CMSCOMM/COMM_TRACKER/DQM/SiStrip/jobs/merged/DQM_V0001_SiStrip_${1}-RECO-CAF-CMSSW_2_1_11_Cluster.root cmstacuser@cmstac11:/data3/CAF/Clusters/
scp ~cctrack/scratch0/DQM/${CMSSW_VERSION}/output/${1}.tar.gz  cmstacuser@cmstac11:/data3/CAF/archive/
rm ~cctrack/scratch0/DQM/${CMSSW_VERSION}/output/${1}.tar.gz
