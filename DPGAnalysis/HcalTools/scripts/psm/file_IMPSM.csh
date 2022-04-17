#!/bin/tcsh
rm TXT_${1}/*
if (${1} == "") then
echo "No input run list. Please use command: ./file_ALCARECO.csh runsets"
exit
endif

echo "  ***************   myStart file_list  ***************   "


foreach i (`cat $1`)
echo ${i}
##
##dasgoclient --query="file dataset=/HcalNZS/Commissioning2021-HcalCalMinBias-PromptReco-v1/ALCARECO run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i} 

##           dasgoclient --query="file dataset=/HcalNZS/Commissioning2021-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i} 
##
## /HcalNZS/Commissioning2021-HcalCalMinBias-PromptReco-v1/ALCARECO
##
##  /MinimumBias*/Commissioning2021-HcalCalIterativePhiSym-PromptReco-v1/ALCARECO
##
dasgoclient --query="file dataset=/MinimumBias3/Commissioning2021-HcalCalIterativePhiSym-PromptReco-v1/ALCARECO run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i} 
##
end
echo "DONE: file_ALCARECO.csh"
##
