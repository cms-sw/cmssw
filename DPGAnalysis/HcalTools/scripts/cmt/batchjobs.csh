#!/bin/csh
echo ${1} ${2} ${3}
setenv MYWORKDIR `pwd`
### Definitions
#setenv SRC /afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/CMSSW_5_3_21/src/RecoHcal/HcalPromptAnalysis/test
#setenv SCRIPTDIR ${SRC}/SHIFTER_VALIDATION/ 
setenv SRC ${3}
setenv SCRIPTDIR ${3}
##setenv SCRAM_ARCH slc6_amd64_gcc472
##setenv SCRAM_ARCH slc6_amd64_gcc491
#setenv SCRAM_ARCH slc6_amd64_gcc530
setenv SCRAM_ARCH slc6_amd64_gcc630 

### Environment
cd ${SRC}
cmsenv

### Go back to WN
cd ${MYWORKDIR}
cp ${SRC}/PYTHON_${1}/Reco_${2}_cfg.py .
### Run CMSSW
cmsRun Reco_${2}_cfg.py > &log_${2}
ls -l * >> &log_${2}

### Copy output files to EOS
 
cmsStage -f log_${2} /store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/CMT/histos/Logs/
cmsStage -f LogEleMapdb.h /store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/CMT/histos/MAP/LogEleMapdb_${2}.h
##cp log_${2} /afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/CMSSW_7_4_5_STABLE/src/RecoHcal/HcalPromptAnalysis/test/SHIFTER_VALIDATION
cmsStage -f Global.root /store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/CMT/histos/Global_${2}.root

## rm all unnesecery
#rm log_${2} 
rm Global.root

