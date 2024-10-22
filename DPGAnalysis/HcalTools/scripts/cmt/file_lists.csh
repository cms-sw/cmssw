#!/bin/tcsh
rm TXT_${1}/*
if (${1} == "") then
echo "No input run list. Please use command: ./file_lists.csh runsets"
exit
endif

echo "  ***************   myStart file_list  ***************   "


foreach i (`cat $1`)
echo ${i}
dasgoclient --query="file site=T2_CH_CERN dataset= /HcalNZS/Run2023D-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}

end

echo "DONE: file_list.csh"



