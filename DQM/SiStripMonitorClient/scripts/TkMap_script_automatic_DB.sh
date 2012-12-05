#!/bin/bash

export PATH=/afs/cern.ch/cms/common:${PATH}
if [[ "$#" == "0" ]]; then
    echo "usage: 'TkMap_script_automatic.sh Cosmics|MinimumBias|StreamExpress|StreamExpressCosmics runNumber1 runNumber2...'";
    exit 1;
fi

export WORKINGDIR=${CMSSW_BASE}/src
#export WORKINGDIR=/afs/cern.ch/user/c/cctrack/scratch0/TKMap/AndreaTests

#echo " Moving to CMSSW release"
#cd /afs/cern.ch/user/c/cctrack/scratch0/TKMap/CMSSW_4_2_3/src
#export SCRAM_ARCH=slc5_amd64_gcc434
#eval `/afs/cern.ch/cms/common/scram runtime -sh`

cd ${WORKINGDIR}

for Run_numb in $@;
do

    if [ "$Run_numb" == "$1" ]; then continue; fi

# copy of the file

    #loop over datasets
    #if Cosmics, do StreamExpressCosmics as well

    datasets=${1}
    if [ "${1}" == "Cosmics" ]; then
	prefix=`echo $Run_numb | awk '{print substr($0,0,3)}'` 
	checkdir='/data/users/event_display/Data2012/Cosmics/'${prefix}'/'${Run_numb}'/StreamExpressCosmics'
	if [ ! -d $checkdir ]; then
	    datasets=$datasets' StreamExpressCosmics'
	    echo "Running on datasets "$datasets
	fi
    fi
    
    for thisDataset in $datasets
      do
      echo "Processing "$thisDataset"..."
      
      nnn=`echo $Run_numb | awk '{print substr($0,0,4)}'` 

    curl -k --cert /data/users/cctrkdata/current/auth/proxy/proxy.cert --key /data/users/cctrkdata/current/auth/proxy/proxy.cert -X GET 'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/Run2012/'$thisDataset'/000'${nnn}'xx/' > index.html
#    wget --no-check-certificate 'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/HIRun2011/'${1}'/000'${nnn}'xx/' -O index.html
    dqmFileNames=`cat index.html | grep ${Run_numb} | grep "_DQM.root" | sed 's/.*>\(.*\)<\/a.*/\1/' `
    dqmFileName=`expr "$dqmFileNames" : '\(DQM[A-Za-z0-9_/.\-]*root\)'`
    echo ' dqmFileNames = '$dqmFileNames
    echo ' dqmFileName = ['$dqmFileName']'
    curl -k --cert /data/users/cctrkdata/current/auth/proxy/proxy.cert --key /data/users/cctrkdata/current/auth/proxy/proxy.cert -X GET https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/Run2012/$thisDataset/000${nnn}xx/${dqmFileName} > /tmp/${dqmFileName}
    checkFile=`ls /tmp/${dqmFileName} | grep ${Run_numb}`

##check if the full run is completely saved (Info/Run summary/ProvInfo/ runIsComplete flag == 1? 
##if not, throw a warning

    file_path="/tmp/"

#    check_runcomplete $Run_numb  ${1}
    check_runcomplete ${file_path}/$dqmFileName
    #if [ $? -ne  0 ]; then continue; fi

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

# Determine the GlobalTag name used to process the data and the DQM

    GLOBALTAG=`python ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/getGTfromDQMFile.py ${file_path}/$dqmFileName $Run_numb globalTag_Step1`
    if [[ "${GLOBALTAG}" == "" ]]
        then
        GLOBALTAG=`getGTscript.sh $dqmFileName $Run_numb`
        fi
    if [[ "${GLOBALTAG}" == "" ]]
    then
        echo " No GlobalTag found: skipping this run.... "
        continue
    fi
    echo "The GlobalTag is $GLOBALTAG"

    echo " Creating the TrackerMap.... "


    cmsRun ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/test/SiStripDQM_OfflineTkMap_Template_cfg_DB.py print globalTag=${GLOBALTAG} runNumber=${Run_numb} dqmFile=${file_path}/$dqmFileName  # update GlobalTag

# rename bad module list file

     mv QTBadModules.log QualityTest_run${Run_numb}.txt
#    mv *.png $Run_numb/$thisDataset
#    mv *.xml $Run_numb/$thisDataset
#    mv PCLBadComponents.log $Run_numb/$thisDataset

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

#    mv QualityTest*txt $Run_numb/$thisDataset

## Producing the run certification by lumisection
    echo " Creating the lumisection certification:"

    if [ $thisDataset == "MinimumBias" -o $thisDataset == "StreamExpress" ]; then	
	ls_cert 0.95 0.95 ${file_path}/$dqmFileName
#	mv Certification_run_* $Run_numb/$thisDataset
    fi

## Producing the PrimaryVertex/BeamSpot quality test by LS..
    if [ $thisDataset == "MinimumBias" -o $thisDataset == "Jet" ]; then	
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

#    dest=FinalTest

#    ssh cmstacuser@cmstac05 "mkdir -p /storage/data2/SiStrip/event_display/Data2011/${dest}/${nnn}/${Run_numb} 2> /dev/null"
 mkdir -p /data/users/event_display/Data2012/${dest}/${nnn}/${Run_numb}/$thisDataset 2> /dev/null
     rm -f *.xml
     rm -f *svg

#    scp -r ${Run_numb}/$thisDataset cmstacuser@cmstac05:/storage/data2/SiStrip/event_display/Data2011/${dest}/${nnn}/${Run_numb}/
#    cp -r ${Run_numb}/$thisDataset /data/users/event_display/Data2011/${dest}/${nnn}/${Run_numb}/
     ssh cctrack@vocms01 "mkdir -p /data/users/event_display/Data2012/${dest}/${nnn}/${Run_numb}/$thisDataset 2> /dev/null"
#     scp -r ${Run_numb}/$thisDataset cctrack@vocms01:/data/users/event_display/Data2012/${dest}/${nnn}/${Run_numb}/
     scp -r * cctrack@vocms01:/data/users/event_display/Data2012/${dest}/${nnn}/${Run_numb}/$thisDataset

     rm ${file_path}/$dqmFileName

     cd ${WORKINGDIR}    
     rm -rf $Run_numb

#done with loop over thisDataset
done

done