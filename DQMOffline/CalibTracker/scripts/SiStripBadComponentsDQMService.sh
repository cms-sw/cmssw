#!/bin/sh

DQMfilesDir=/home/dqmprolocal/done/

# rootFilesList=(`ls $DQMfilesDir | grep .root`)

# The playback_full is what is used now. It is supposed to change in the future
# when the DQM gui will be able to handle to complete file.
# The list is sorted.
rootFilesList=(`find $DQMfilesDir -name "*.root" | grep Playback_full | sort`)
#rootFilesList=(`echo ${rootFilesList[@]} | grep Playback_full`)

# echo "List of files: $rootFilesList"

if [ -e tempList.txt ]; then
    rm tempList.txt
fi
if [ -e newList.txt ]; then
    rm newList.txt
fi

for file in ${rootFilesList[@]}
do
    echo $file >> tempList.txt
done
cat fullList.txt >> tempList.txt
# Take only unique lines
cat tempList.txt | sort | uniq -u > newList.txt
rm tempList.txt
# Take the difference between the full list and the list of already analyzed files

echo "Setting up the environment"
source /nfshome0/cmssw2/scripts/setup.sh
cd /nfshome0/demattia/CMSSW_3_1_0/src/
echo pwd is `pwd`
cmsenv
echo "Release base is $CMSSW_RELEASE_BASE"
cd $LOCALRT/src/DQMOffline/CalibTracker/scripts
echo pwd is `pwd`

i=0
for file in `cat newList.txt`
do

    echo
    echo "file[$i] = $file"
    runNumber=`echo $file | awk -F/ '{print $NF}' | awk -F_ '{print $5}'`
    # Strip the R at the start of the string
    runNumber=`echo ${runNumber#R}`
    # Strip all the 0s at the start of the runNumber
    # DQM_full_V0001_SiStrip_R000105765_T00000341.root
    runNumber=`echo $runNumber | sed 's/0*//'`
    echo "runNumber = $runNumber"

    # Create the cfg file
    if [ -e temp_cfg.py ]; then
	rm temp_cfg.py
    fi

    cat SiStripBadComponentsDQMServiceTemplate_cfg.py | sed "s/RUNNUMBER/$runNumber/g" | sed "s@FILENAME@$file@" > temp_cfg.py
    cmsRun temp_cfg.py > output_$runNumber.txt
    echo "Job finished with exit code = $?"

    # File processed. Update the list
    echo $file >> fullList.txt

    let i+=1

done

#Cleanup
if [ -e newList.txt ]; then
    rm newList.txt
fi

# Do it every half a hour.
# sleep 1800
