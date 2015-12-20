#!/bin/bash

if [ "$1" = "-v" ]
then
  echo Initiating test, all logs and test configs will be saved. 
else
  echo Initiating test, run with -v to save all logs and test configs.
fi
rm fail.test_* 

#############################################################
# Step 1: Get the input samples if they're not already here #
#############################################################

if [ -d "samples" ]
then
  # samples are here, no need to grab anything
  echo Samples found.
else
  printf "Grabbing samples."
  wget http://web.mit.edu/mithig/samples/PbPb_DATA_AOD.root &> /dev/null
  printf "."
  wget http://web.mit.edu/mithig/samples/PbPb_MC_AODSIM.root &> /dev/null
  printf "."
  wget http://web.mit.edu/mithig/samples/PbPb_MC_RECODEBUG.root &> /dev/null
  printf "."
  wget http://web.mit.edu/mithig/samples/pp_DATA_AOD.root &> /dev/null
  printf "."
  wget http://web.mit.edu/mithig/samples/pp_DATA_RECO.root &> /dev/null
  printf "."
  wget http://web.mit.edu/mithig/samples/pp_MC_AODSIM.root &> /dev/null
  printf "."
  wget http://web.mit.edu/mithig/samples/pp_MC_RECODEBUG.root &> /dev/null
  mkdir samples
  mv PbPb_DATA_AOD.root PbPb_MC_AODSIM.root PbPb_MC_RECODEBUG.root pp_DATA_AOD.root pp_DATA_RECO.root pp_MC_AODSIM.root pp_MC_RECODEBUG.root samples/
  echo Done.
fi

#############################################################
# Step 2: Setup each runForest on the appropriate sample    #
#############################################################

replacename=`cat runForestAOD_PbPb_DATA_75X.py | grep -v '#' | grep -A10 process.source | grep root | sed 's@,@@g' | awk '{print $1}'`
cat runForestAOD_PbPb_DATA_75X.py | sed 's@'$replacename'@"file:samples/PbPb_DATA_AOD.root"@g' | sed 's@HiForestAOD.root@HiForestAOD_PbPb_DATA_AOD.root@g' > test_runForestAOD_PbPb_DATA_75X.py

replacename=`cat runForestAOD_pp_DATA_75X.py | grep -v '#' | grep -A10 process.source | grep root | sed 's@,@@g' | awk '{print $1}'`
cat runForestAOD_pp_DATA_75X.py | sed 's@'$replacename'@"file:samples/pp_DATA_AOD.root"@g' | sed 's@HiForestAOD.root@HiForestAOD_pp_DATA_AOD.root@g' > test_runForestAOD_pp_DATA_75X.py

replacename=`cat runForestAOD_PbPb_MIX_75X.py | grep -v '#' | grep -A10 process.source | grep root | sed 's@,@@g' | awk '{print $1}'`
cat runForestAOD_PbPb_MIX_75X.py | sed 's@'$replacename'@"file:samples/PbPb_MC_AODSIM.root"@g' | sed 's@HiForestAOD.root@HiForestAOD_PbPb_MC_AODSIM.root@g' > test_runForestAOD_PbPb_MIX_75X.py

replacename=`cat runForestAOD_pp_MC_75X.py | grep -v '#' | grep -A10 process.source | grep root | sed 's@,@@g' | awk '{print $1}'`
cat runForestAOD_pp_MC_75X.py | sed 's@'$replacename'@"file:samples/pp_MC_AODSIM.root"@g' | sed 's@HiForestAOD.root@HiForestAOD_pp_MC_AODSIM.root@g' > test_runForestAOD_pp_MC_75X.py

#############################################################
# Step 3: cmsRun each runForest                             #
#############################################################

printf "Executing cmsRun for each script..."
for i in `ls test_runForestAOD_*.py`
do 
  cmsRun $i &> $i.log &
done
wait
echo Done.
echo
echo
sleep 1

#############################################################
# Step 4: Check for errors                                  #
#############################################################

for i in `ls runForestAOD_*.py`
do
  error=`grep -i "Begin Fatal Exception\|FatalRootError\|Segmentation" test_$i.log`
  if [ -z "$error" ]
  then
    echo -e "\E[32m$i"
    tput sgr0
  else
    echo -e "\E[31m$i"
    tput sgr0
    mv test_$i.log fail.test_$i.log
    mv test_$i fail.test_$i
  fi
done
echo
for i in `ls fail.test_*.log`
do
  echo Please check $i
done
echo
echo

#############################################################
# Step 5: Clean up                                          #
#############################################################

if [ "$1" = "-v" ]
then
  echo Keeping log and test configs. 
else
  echo Removing log and test configs.
  rm test_runForestAOD_*.py test_runForestAOD_*.py.log *.root
fi

echo Test complete. 

