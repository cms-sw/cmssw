#!/bin/tcsh
set CP_PATH=cmstacuser@cmstac11:/data3/CAF
set TAR_BALL=~cctrack/scratch0/DQM/${CMSSW_VERSION}/output/${1}.tar.gz
scp /afs/cern.ch/cms/CAF/CMSCOMM/COMM_TRACKER/DQM/SiStrip/jobs/merged/DQM_V0001_SiStrip_${1}-*.root ${CP_PATH}/Clusters/
tar -czvf ${TAR_BALL} ../*/${1}
scp ${TAR_BALL} ${CP_PATH}/archive/
rm -r ${TAR_BALL} ../*/${1}
