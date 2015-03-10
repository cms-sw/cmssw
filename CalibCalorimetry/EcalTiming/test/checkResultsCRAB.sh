#!/bin/bash

usage='Usage: -r <run number>'

args=`getopt r: -- "$@"`
if test $? != 0
     then
         echo $usage
         exit 1
fi

eval set -- "$args"
for i 
  do
  case "$i" in
      -r) shift; run_num=$2;shift;;
  esac      
done

echo 'Checking CRAB status' ${run_num} 

if [ "X"${run_num} == "X" ]
    then
    echo "INVALID RUN NUMBER! Please give a valid run number!"
    echo $usage
    exit 
fi

# setup crab environment
source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.sh;
eval `scramv1 runtime -sh`;
source /afs/cern.ch/cms/ccs/wm/scripts/Crab/CRAB_2_7_1/crab.sh;
#source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.sh;

cd ${run_num};

nrun=`crab -status | grep -c RUN`;
npend=`crab -status | grep -c PEND`;
ndone=`crab -status | grep -c DONE`;

if [ "${nrun}" == "0" ] && [ "${npend}" == "0" ]
then
    echo "Run "$run_num "is done..." "run:" $nrun "pend:" $npend "done:" $ndone
    crab -get
else
    echo "Run "$run_num "NOT yet done..." "run:" $nrun "pend:" $npend "done:" $ndone
fi

#cd -;
