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
##
#dasgoclient --query="file dataset=/HcalNZS/Commissioning2017-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}

#dasgoclient --query="file dataset=/Cosmics/Run2017C-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file dataset=/ZeroBias/Run2018A-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file dataset=/ZeroBias/Run2018B-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file dataset=/ZeroBias/Run2018C-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
dasgoclient --query="file dataset=/ZeroBias/Run2018D-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
# this dataset only! for ions:
#dasgoclient --query="file dataset=/HIHcalNZS/HIRun2018A-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
# games with dedicated abort gap dataset  :
#dasgoclient --query="file dataset=/TestEnablesEcalHcal/Run2018D-Express-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#
##
##

end

echo "DONE: file_list.csh"
##
