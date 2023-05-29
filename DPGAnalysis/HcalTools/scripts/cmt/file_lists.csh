#!/bin/tcsh
rm TXT_${1}/*
if (${1} == "") then
echo "No input run list. Please use command: ./file_lists.csh runsets"
exit
endif

echo "  ***************   myStart file_list  ***************   "


foreach i (`cat $1`)
echo ${i}
##
##
#dasgoclient --query="file dataset=/Cosmics/Commissioning2018-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file dataset=/MinimumBias/Commissioning2018-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
##

#dasgoclient --query="file dataset= /HcalNZS/Run2022A-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file dataset= /MinimumBias/Run2022A-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file dataset= /ZeroBias/Run2022A-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file site=T2_CH_CERN dataset= /ZeroBias/Run2022B-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file site=T2_CH_CERN dataset= /HcalNZS/Run2022A-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file site=T2_CH_CERN dataset= /HcalNZS/Run2022B-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file site=T2_CH_CERN dataset= /HcalNZS/Run2022C-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file site=T2_CH_CERN dataset= /HcalNZS/Run2022D-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file site=T2_CH_CERN dataset= /ZeroBias/Run2022E-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file site=T2_CH_CERN dataset= /HcalNZS/Run2022E-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file site=T2_CH_CERN dataset= /HcalNZS/Run2022F-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
dasgoclient --query="file site=T2_CH_CERN dataset= /HcalNZS/Run2022G-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file site=T2_CH_CERN dataset= /HcalNZS/Run2022H-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}


#
##

end

echo "DONE: file_list.csh"

##


