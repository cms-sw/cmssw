#!/bin/csh
echo ${1} ${2} ${3} ${4}
###exit
setenv MYWORKDIR `pwd`/TMP_`date '+%Y-%m-%d_%H_%M_%S'`
mkdir `pwd`/TMP_`date '+%Y-%m-%d_%H_%M_%S'`

### Definitions
setenv SRC ${4}
setenv SCRIPTDIR ${4}
##setenv SCRAM_ARCH slc6_amd64_gcc491
setenv SCRAM_ARCH slc6_amd64_gcc700

### Environment
cd ${SRC}
cmsenv

### Go back to WN
cd ${MYWORKDIR}
cp ${SRC}/PYTHON_${1}/Reco_${2}_${3}_cfg.py .
##exit

### Run CMSSW
cmsRun Reco_${2}_${3}_cfg.py > &log_${2}_${3}
ls -l * >> &log_${2}_${3}

### Copy output files to EOS
#cmsStage -f log_${2}_${3} /store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/CMT/histos/Logs/
#cmsStage -f LogEleMapdb.h /store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/CMT/histos/MAP/LogEleMapdb_${2}_${3}.h
#cmsStage -f Global.root /store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/CMT/histos/Global_${2}_${3}.root

### Copy output files to 
cp Global_${2}_${3}.root /afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/CMSSW_10_4_0/src/RecoHcal/HcalPromptAnalysis/test/SHIFTER_VALIDATION/Global_${2}_${3}.root
## rm all unnesecery
#rm log_${2} 
##rm Global.root
rm Global_${2}_${3}.root


