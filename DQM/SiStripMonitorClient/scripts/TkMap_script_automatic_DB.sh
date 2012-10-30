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
    nnn=`echo $Run_numb | awk '{print substr($0,0,4)}'` 
    curl -k --cert /data/users/cctrkdata/current/auth/proxy/proxy.cert --key /data/users/cctrkdata/current/auth/proxy/proxy.cert -X GET 'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/Run2012/'${1}'/000'${nnn}'xx/' > index.html
#    wget --no-check-certificate 'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/HIRun2011/'${1}'/000'${nnn}'xx/' -O index.html
    dqmFileNames=`cat index.html | grep ${Run_numb} | grep "_DQM.root" | sed 's/.*>\(.*\)<\/a.*/\1/' `
    dqmFileName=`expr "$dqmFileNames" : '\(DQM[A-Za-z0-9_/.\-]*root\)'`
    echo ' dqmFileName = ['$dqmFileName']'
    curl -k --cert /data/users/cctrkdata/current/auth/proxy/proxy.cert --key /data/users/cctrkdata/current/auth/proxy/proxy.cert -X GET https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/Run2012/${1}/000${nnn}xx/${dqmFileName} > /tmp/${dqmFileName}
    checkFile=`ls /tmp/${dqmFileName} | grep ${Run_numb}`

##check if the full run is completely saved (Info/Run summary/ProvInfo/ runIsComplete flag == 1? 
##if not, throw a warning

    file_path="/tmp/"

#    check_runcomplete $Run_numb  ${1}
    check_runcomplete ${file_path}/$dqmFileName
    if [ $? -ne  0 ]; then continue; fi

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
    [ -e $Run_numb/$1 ] || mkdir $Run_numb/$1;
    echo "Run ${Run_numb}"

    cd $Run_numb/$1
    echo `pwd`
    rm -f *.png
    rm -f *.xml
    rm -f *.log
    rm -f *.txt
    rm -f *.html

# Determine the GlobalTag name used to process the data and the DQM

    GLOBALTAG=`getGTscript.sh $dqmFileName $Run_numb` 
#    GLOBALTAG="GR_P_V40::All"
    echo "The GlobalTag is $GLOBALTAG"
    if [[ "${GLOBALTAG}" == "" ]]
    then
       GLOBALTAG="GR_P_V42::All"
    fi

    echo " Creating the TrackerMap.... "


    cmsRun ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/test/SiStripDQM_OfflineTkMap_Template_cfg_DB.py print globalTag=${GLOBALTAG} runNumber=${Run_numb} dqmFile=${file_path}/$dqmFileName  # update GlobalTag

#    mv *.png $Run_numb/$1
#    mv *.xml $Run_numb/$1
#    mv PCLBadComponents.log $Run_numb/$1

    if [ "${1}" == "Cosmics" ]; then  # should I add StreamExpressCosmics too
	cat ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/data/index_template_TKMap_cosmics.html | sed -e "s@RunNumber@$Run_numb@g" > index.html
    else
	cat ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/data/index_template_TKMap.html | sed -e "s@RunNumber@$Run_numb@g" > index.html
    fi
    
    echo " Check TrackerMap on $Run_numb/$1 folder"

    nnn=`echo ${Run_numb} | awk '{print substr($0,0,3)}'`

## Producing the list of bad modules
    echo " Creating the list of bad modules "
    
    listbadmodule ${file_path}/$dqmFileName PCLBadComponents.log

#    mv QualityTest*txt $Run_numb/$1

## Producing the run certification by lumisection
    echo " Creating the lumisection certification:"

    if [ "${1}" == "MinimumBias" -o "${1}" == "StreamExpress" ]; then	
	ls_cert 0.95 0.95 ${file_path}/$dqmFileName
#	mv Certification_run_* $Run_numb/$1
    fi

## Producing the PrimaryVertex/BeamSpot quality test by LS..
    if [ "${1}" == "MinimumBias" -o "${1}" == "Jet" ]; then	
	echo " Creating the BeamSpot Calibration certification summary:"

	lsbs_cert  ${file_path}/$dqmFileName

#	mv Certification_BS_run_* $Run_numb/$1
    fi
## .. and harvest the bad beamspot LS with automatic emailing (if in period and if bad LS found)
#    bs_bad_ls_harvester $Run_numb/$1 $Run_numb
    bs_bad_ls_harvester . $Run_numb

## Producing the Module difference for ExpressStream
    if [ "${1}" == "StreamExpress" -o "${1}" == "StreamExpressCosmics" ]; then	
	echo " Creating the Module Status Difference summary:"

#	modulediff $Run_numb ${1}

#	./modulediff_summary $Run_numb
#	mv ModuleDifference_${Run_numb}.txt $Run_numb/$1
    fi

    dest=Beam
    if [ "${1}" == "Cosmics" -o "${1}" == "StreamExpressCosmics" ]; then dest="Cosmics"; fi

# overwrite destination for tests

#    dest=AndreaTests

#    ssh cmstacuser@cmstac05 "mkdir -p /storage/data2/SiStrip/event_display/Data2011/${dest}/${nnn}/${Run_numb} 2> /dev/null"
 mkdir -p /data/users/event_display/Data2012/${dest}/${nnn}/${Run_numb}/$1 2> /dev/null
     rm -f *.xml
     rm -f *svg

#    scp -r ${Run_numb}/$1 cmstacuser@cmstac05:/storage/data2/SiStrip/event_display/Data2011/${dest}/${nnn}/${Run_numb}/
#    cp -r ${Run_numb}/$1 /data/users/event_display/Data2011/${dest}/${nnn}/${Run_numb}/
     ssh cctrack@vocms01 "mkdir -p /data/users/event_display/Data2012/${dest}/${nnn}/${Run_numb}/$1 2> /dev/null"
#     scp -r ${Run_numb}/$1 cctrack@vocms01:/data/users/event_display/Data2012/${dest}/${nnn}/${Run_numb}/
     scp -r * cctrack@vocms01:/data/users/event_display/Data2012/${dest}/${nnn}/${Run_numb}/$1

     rm ${file_path}/$dqmFileName

done
