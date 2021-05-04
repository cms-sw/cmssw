#!/bin/csh

setenv MYWORKDIR `pwd`

cd /afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/CMSSW_5_3_21/src/RecoHcal/HcalPromptAnalysis/test/

#setenv SCRAM_ARCH slc6_amd64_gcc472
setenv SCRAM_ARCH slc6_amd64_gcc630 
cmsenv

cd ${MYWORKDIR}

cp /afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/CMSSW_5_3_21/src/RecoHcal/HcalPromptAnalysis/test/zzzzzz/PYTHONS/Reco_${1}_cfg.py .

cmsRun Reco_${1}_cfg.py > &log_${1}

ls * >> &log_${1}

cp log_${1} /afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/CMSSW_5_3_21/src/RecoHcal/HcalPromptAnalysis/test/zzzzzz/ZOUT/.


cp Global.root /afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/CMSSW_5_3_21/src/RecoHcal/HcalPromptAnalysis/test/zzzzzz/ZOUT/Global_${1}.root


##rfcp log_${1} /castor/cern.ch/user/k/kodolova/2012/HCAL/PEDESTAL/RUN2012AV1
##rfcp analysis_minbias_Full.root /castor/cern.ch/user/k/kodolova/2012/HCAL/PEDESTAL/RUN2012AV1/analysis_minbias_Full_${1}.root
