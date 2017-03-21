#!/bin/bash

curdir="$(pwd)"
echo ${curdir}

export PATH=/afs/cern.ch/cms/common:${PATH}
if [[ "$#" == "0" ]]; then
    echo "usage: 'TkMap_script_automatic.sh Cosmics|MinimumBias|StreamExpress|StreamExpressCosmics runNumber1 runNumber2...'";
    exit 1;
fi

FORCE=0
echo $2
if [ "${2}" == "0" ]; then
    FORCE=0
else
    if [ "${2}" == "f" ]; then
        FORCE=1
    fi
fi

export WORKINGDIR=${CMSSW_BASE}/src

cd ${WORKINGDIR}

DataLocalDir=''
DataOflineDir=''

for Run_numb in $@;
do

    if [ $Run_numb -gt 284500 ]; then

        DataLocalDir='Data2016'
        DataOfflineDir='PARun2016'
    else


    if [ "$Run_numb" == "$1" ]; then continue; fi

##2016 data taking period run > 271024
    if [ $Run_numb -gt 271024 ]; then

        DataLocalDir='Data2016'
        DataOfflineDir='Run2016'
    else

#2016 - Commissioning period                                                                                                                               
    if [ $Run_numb -gt 264200 ]; then

        DataLocalDir='Data2016'
        DataOfflineDir='Commissioning2016'
    else

    #Run2015A
    if [ $Run_numb -gt 246907 ]; then
        DataLocalDir='Data2015'
        DataOfflineDir='Run2015'
    else

    #2015 Commissioning period (since January)
    if [ $Run_numb -gt 232881 ]; then
	DataLocalDir='Data2015'
	DataOfflineDir='Commissioning2015'
    else
    #2013 pp run (2.76 GeV)
	if [ $Run_numb -gt 211658 ]; then
	    DataLocalDir='Data2013'
	    DataOfflineDir='Run2013'
	else
    #2013 HI run
	    if [ $Run_numb -gt 209634 ]; then
		DataLocalDir='Data2013'
		DataOfflineDir='HIRun2013'
	    else
		if [ $Run_numb -gt 190450 ]; then
		    DataLocalDir='Data2012'
		    DataOfflineDir='Run2012'
		fi
	    fi
	fi
    fi
    fi
    fi
    fi
    fi
    #loop over datasets
    #if Cosmics, do StreamExpressCosmics as well

    datasets=${1}
    if [ "${1}" == "Cosmics" ]; then
	prefix=`echo $Run_numb | awk '{print substr($0,0,3)}'` 
	checkdir='/data/users/event_display/'${DataLocalDir}'/Cosmics/'${prefix}'/'${Run_numb}'/StreamExpressCosmics'
	if [ ! -d $checkdir ]; then
	    datasets=$datasets' StreamExpressCosmics'
	    echo "Running on datasets "$datasets
	fi
    fi
    
    for thisDataset in $datasets
      do
      echo "Processing "$thisDataset "in "${DataOfflineDir}"..."

      nnn=`echo $Run_numb | awk '{print substr($0,0,4)}'` 

      echo 'Directory to fetch the DQM file from: https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/'${DataOfflineDir}'/'$thisDataset'/000'${nnn}'xx/'
  
    curl -k --cert /data/users/cctrkdata/current/auth/proxy/proxy.cert --key /data/users/cctrkdata/current/auth/proxy/proxy.cert -X GET 'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/'${DataOfflineDir}'/'$thisDataset'/000'${nnn}'xx/' > index.html
    dqmFileNames=`cat index.html | grep ${Run_numb} | egrep "_DQM.root|_DQMIO.root" | egrep "Prompt|Express|22Jan2013" | sed 's/.*>\(.*\)<\/a.*/\1/' `
    dqmFileName=`expr "$dqmFileNames" : '\(DQM[A-Za-z0-9_/.\-]*root\)'`
    echo ' dqmFileNames = '$dqmFileNames
    echo ' dqmFileName = ['$dqmFileName']'
    curl -k --cert /data/users/cctrkdata/current/auth/proxy/proxy.cert --key /data/users/cctrkdata/current/auth/proxy/proxy.cert -X GET https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/$DataOfflineDir/$thisDataset/000${nnn}xx/${dqmFileName} > /tmp/${dqmFileName}
    checkFile=`ls /tmp/${dqmFileName} | grep ${Run_numb}`

##check if the full run is fully processed in GUI (Info/Run summary/ProvInfo/ runIsComplete flag == 1? 
##if not, throw a warning

    file_path="/tmp/"

    echo "FORCE is " ${FORCE}
    ## check if run is complete - LG
    echo "get the run status from DQMFile"
    runStatus=-1
    runStatus="$(${pathTools}getRunStatusFromDQMFile.py ${file_path}/$dqmFileName $Run_numb runIsComplete | wc -l)"
    if [[ ${runStatus} == 0 ]] 
	then 
	echo ${Run_numb} >> ${curdir}/runsNotComplete_tmp.txt
        if [ ${FORCE} == 0] 
	then 
	    continue; 
	fi
    fi
    ## LG end

    if [ $FORCE == 0 ]; then
	check_runcomplete ${file_path}/$dqmFileName
	if [ $? -ne 0 ]; then continue; fi
    fi

    echo Process ${file_path}/$dqmFileName

    cd /tmp

    nnn=`echo $Run_numb | awk '{print substr($0,0,4)}'` 
  

    echo ' dqmFileName = ['$dqmFileName']'

     if [[ "${dqmFileName}" == "" ]] 
     then
 	echo "Run ${Run_numb} not yet ready"
 	continue
     fi
   
    cd ${WORKINGDIR}    

    [ -e $Run_numb ] || mkdir $Run_numb;
    [ -e $Run_numb/$thisDataset ] || mkdir $Run_numb/$thisDataset;
    echo "Run ${Run_numb}"

    cd $Run_numb/$thisDataset
    echo `pwd`
    rm -f *.png
    rm -f *.xml
    rm -f *.log
    rm -f *.txt
    rm -f *.html
    rm -f *.root


    cp ${WORKINGDIR}/DQM/SiStripMonitorClient/scripts/DeadROCCounter.py .

# Determine the GlobalTag name used to process the data and the DQM

    GLOBALTAG=`getGTfromDQMFile.py ${file_path}/$dqmFileName $Run_numb globalTag_Step1`
    
    if [[ "${GLOBALTAG}" == "" ]]
        then
        echo " No GlobalTag found: trying from DAS.... "
        GLOBALTAG=`getGTscript.sh $dqmFileName $Run_numb`
        fi
    if [[ "${GLOBALTAG}" == "" ]]
    then
        echo " No GlobalTag found: skipping this run.... "
        continue
    fi

#Temporary fix to remove hidden ASCII characters
    GLOBALTAG=`echo $GLOBALTAG | cut -c 9-${#GLOBALTAG}`
#    GLOBALTAG=`sed -i 's/[\d128-\d255]//g' <<< "${GLOBALTAG}"`
#    GLOBALTAG=`echo $GLOBALTAG | sed 's/[\d128-\d255]//'`
#    echo `expr length $GLOBALTAG`

    echo " Creating the TrackerMap.... "

    detIdInfoFileName=`echo "file://TkDetIdInfo_Run${Run_numb}_${thisDataset}.root"`

    #cmsRun ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/test/SiStripDQM_OfflineTkMap_Template_cfg_DB.py print globalTag=${GLOBALTAG} runNumber=${Run_numb} dqmFile=${file_path}/$dqmFileName  # update GlobalTag
    cmsRun ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/test/SiStripDQM_OfflineTkMap_Template_cfg_DB.py print globalTag=${GLOBALTAG} runNumber=${Run_numb} dqmFile=${file_path}/$dqmFileName  detIdInfoFile=${detIdInfoFileName} # update GlobalTag

# rename bad module list file

     mv QTBadModules.log QualityTest_run${Run_numb}.txt

    if [ $thisDataset == "Cosmics" ]; then  # should I add StreamExpressCosmics too
	cat ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/data/index_template_TKMap_cosmics.html | sed -e "s@RunNumber@$Run_numb@g" > index.html
    else
	cat ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/data/index_template_TKMap.html | sed -e "s@RunNumber@$Run_numb@g" > index.html
    fi
    cp ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/data/fedmap.html fedmap.html
    cp ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/data/psumap.html psumap.html

    echo " Check TrackerMap on $Run_numb/$thisDataset folder"

    nnn=`echo ${Run_numb} | awk '{print substr($0,0,3)}'`

## Producing the list of bad modules
    echo " Creating the list of bad modules "
    
    listbadmodule ${file_path}/$dqmFileName PCLBadComponents.log
   if [ "$thisDataset" != "StreamExpress" ] ; then
       sefile=QualityTest_run${Run_numb}.txt

       if [ "$thisDataset" == "Cosmics" ]; then
           python ../../DQM/SiStripMonitorClient/scripts/findBadModT9.py -p $sefile -s /data/users/event_display/${DataLocalDir}/Cosmics/${nnn}/${Run_numb}/StreamExpressCosmics/${sefile}
       else

           python ../../DQM/SiStripMonitorClient/scripts/findBadModT9.py -p $sefile -s /data/users/event_display/${DataLocalDir}/Beam/${nnn}/${Run_numb}/StreamExpress/${sefile}

       fi
   fi

#    mv QualityTest*txt $Run_numb/$thisDataset

## Producing the run certification by lumisection
    echo " Creating the lumisection certification:"

    if [ $thisDataset == "MinimumBias" -o $thisDataset == "StreamExpress" ]; then	
	ls_cert 0.95 0.95 ${file_path}/$dqmFileName
#	mv Certification_run_* $Run_numb/$thisDataset
    fi

## Producing the PrimaryVertex/BeamSpot quality test by LS..
    if [ "$thisDataset" != "Cosmics" ]  &&  [ "$thisDataset" != "StreamExpress" ]  &&  [ "$thisDataset" != "StreamExpressCosmics" ]; then
	echo " Creating the BeamSpot Calibration certification summary:"

	lsbs_cert  ${file_path}/$dqmFileName

#	mv Certification_BS_run_* $Run_numb/$thisDataset
    fi
## .. and harvest the bad beamspot LS with automatic emailing (if in period and if bad LS found)
#    bs_bad_ls_harvester $Run_numb/$thisDataset $Run_numb
    bs_bad_ls_harvester . $Run_numb

## Producing the Module difference for ExpressStream
    if [ $thisDataset == "StreamExpress" -o $thisDataset == "StreamExpressCosmics" ]; then	
	echo " Creating the Module Status Difference summary:"

#	modulediff $Run_numb ${1}
#	./modulediff_summary $Run_numb
#	mv ModuleDifference_${Run_numb}.txt $Run_numb/$thisDataset
    fi

    dest=Beam
    if [ $thisDataset == "Cosmics" -o $thisDataset == "StreamExpressCosmics" ]; then dest="Cosmics"; fi

# overwrite destination for tests
# dest=FinalTest

## create merged list of BadComponent from (PCL, RunInfo and FED Errors)
    cmsRun ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/test/mergeBadChannel_Template_cfg.py globalTag=${GLOBALTAG} runNumber=${Run_numb} dqmFile=${file_path}/$dqmFileName
    mv MergedBadComponents.log MergedBadComponents_run${Run_numb}.txt

    rm -f *.xml
    rm -f *svg

    ssh cctrack@vocms061 "mkdir -p /data/users/event_display/TkCommissioner_runs/${DataLocalDir}/${dest} 2> /dev/null"
    scp *.root cctrack@vocms061:/data/users/event_display/TkCommissioner_runs/${DataLocalDir}/${dest}
    rm *.root

    echo "countig dead pixel ROCs" 
    ./DeadROCCounter.py ${file_path}/$dqmFileName

#    mkdir -p /data/users/event_display/${DataLocalDir}/${dest}/${nnn}/${Run_numb}/$thisDataset #2> /dev/null
#    cp -r ${Run_numb}/$thisDataset /data/users/event_display/Data2011/${dest}/${nnn}/${Run_numb}/
#    cp -r ${Run_numb}/$thisDataset /data/users/event_display/${DataLocalDir}/${dest}/${nnn}/${Run_numb}/$thisDataset 
    ssh cctrack@vocms061 "mkdir -p /data/users/event_display/${DataLocalDir}/${dest}/${nnn}/${Run_numb}/$thisDataset 2> /dev/null"
    scp -r * cctrack@vocms061:/data/users/event_display/${DataLocalDir}/${dest}/${nnn}/${Run_numb}/$thisDataset

     rm ${file_path}/$dqmFileName

     cd ${WORKINGDIR}    
     rm -rf $Run_numb
     rm -rf index.html

#done with loop over thisDataset
done

done
