#!/bin/csh

### set DAT=`date '+%Y-%m-%d_%H_%M_%S'`
set DAT="2014-03-22_13_49_54"
### Get list of done from RDM webpage ###

set WD='/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/CMSSW_10_4_0/src/RecoHcal/HcalPromptAnalysis/test/RDM'

#${WD}/myselect.csh ${DAT}
set jold = -1
foreach i (`cat ${WD}/LED_LIST/runlist.tmp.${DAT}`)
set iold=`echo ${i} | awk -F _ '{print $1}'`
set jold=`echo ${i} | awk -F _ '{print $2}'`
echo ${iold} ${jold}
ls /afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMweb/LED_${iold}/HB.html
echo ${status}
#${WD}/HcalRemoteMonitoringNew.csh ${iold} ${DAT} ${jold}
end
