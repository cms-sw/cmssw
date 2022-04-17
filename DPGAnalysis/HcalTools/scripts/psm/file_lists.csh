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
#dasgoclient --query="file dataset=/HcalNZS/Commissioning2017-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file dataset=/Cosmics/Run2017C-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#dasgoclient --query="file dataset=/ZeroBias/Run2018A-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
##
##         HcalNZS                         /Commissioning2021  HcalNZS
#dasgoclient --query="file dataset=/HcalNZS/Commissioning2021-HcalCalMinBias-PromptReco-v1/ALCARECO run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}


dasgoclient --query="file dataset=/HcalNZS/Commissioning2021-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i} 

##         HcalNZS                         1
#dasgoclient --query="file dataset=/HcalNZS/Run2018D-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
# runlist11:316636 316982 316991 DONE
#dasgoclient --query="file dataset=/HcalNZS/Run2018A-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
# runlist12:321612  DONE
#dasgoclient --query="file dataset=/HcalNZS/Run2018D-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}

##         MinimumBias                     2
# runlist21:316931 316937 316979 316987  DONE
#dasgoclient --query="file dataset=/MinimumBias/Run2018A-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
# runlist22:318172 319235  DONE
#dasgoclient --query="file dataset=/MinimumBias/Run2018B-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
# runlist23:321612 321668  DONE
#dasgoclient --query="file dataset=/MinimumBias/Run2018D-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}

##         ZeroBias                        3
# runlist31:316580 316944 316972 316991  DONE
#dasgoclient --query="file dataset=/ZeroBias/Run2018A-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
# runlist32:317480 317508 317567 317592 319177 319300  DONE
#dasgoclient --query="file dataset=/ZeroBias/Run2018B-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
# runlist33:321457 321612 321750 323841 323958 324020  DONE
#dasgoclient --query="file dataset=/ZeroBias/Run2018D-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
# runlist34:325283 325338 325396 325418 325458 325476 325553 325644 325700  DONE
#dasgoclient --query="file dataset=/ZeroBias/Run2018E-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}



# this dataset only! for ions:
#dasgoclient --query="file dataset=/HIHcalNZS/HIRun2018A-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
# games with dedicated abort gap dataset  :
#dasgoclient --query="file dataset=/TestEnablesEcalHcal/Run2018D-Express-v1/RAW run=${i}" --limit=0 | sed "s/\/store/\'\/store/g" | sed "s/root/root\',/g"> TXT_${1}/run_${i}
#
## available on eos according runlist_all  (and run 316982 had plots already)
# Global_316636.root Global_316931.root Global_316944.root Global_316972.root Global_316987.root Global_317480.root Global_317508.root Global_317592.root Global_318172.root Global_319177.root 
# Global_319235.root Global_319300.root Global_321457.root Global_321612.root Global_323841.root Global_323958.root Global_324020.root Global_325283.root Global_325338.root Global_325396.root 
# Global_325418.root Global_325476.root
#



##

end

echo "DONE: file_list.csh"
##
