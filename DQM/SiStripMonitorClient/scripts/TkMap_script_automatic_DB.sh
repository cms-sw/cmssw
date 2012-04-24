#!/bin/bash

export PATH=/afs/cern.ch/cms/common:${PATH}
if [[ "$#" == "0" ]]; then
    echo "usage: 'TkMap_script_automatic.sh Cosmics|MinimumBias|StreamExpress|StreamExpressCosmics runNumber1 runNumber2...'";
    exit 1;
fi

export WORKINGDIR=/afs/cern.ch/user/c/cctrack/scratch0/TKMap

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
    dqmFileName=`cat index.html | grep ${Run_numb} | grep "_DQM.root" | sed 's/.*>\(.*\)<\/a.*/\1/' `
    echo ' dqmFileName = ['$dqmFileName']'
    curl -k --cert /data/users/cctrkdata/current/auth/proxy/proxy.cert --key /data/users/cctrkdata/current/auth/proxy/proxy.cert -X GET https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/Run2012/${1}/000${nnn}xx/${dqmFileName} > /tmp/${dqmFileName}
    checkFile=`ls /tmp/${dqmFileName} | grep ${Run_numb}`

##check if the full run is completely saved (Info/Run summary/ProvInfo/ runIsComplete flag == 1? 
##if not, throw a warning
# root macro to be moved in the CMSSW package

#    root.exe -l -b -q   "check_runcomplete.C+( $Run_numb , \"${1}\")"

    check_runcomplete $Run_numb  ${1}

    echo Process $Run_numb

    cd /tmp

    nnn=`echo $Run_numb | awk '{print substr($0,0,4)}'` 
  
file_path="/tmp/"

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

#    ##Build the cfg files:
#    [ -e SiStripDQM_OfflineTkMap_cfg_$Run_numb.py ] && rm SiStripDQM_OfflineTkMap_cfg_$Run_numb.py
#    cat SiStripDQM_OfflineTkMap_Template_cfg_DB.py | sed -e "s@RunNum@$Run_numb@g" | sed -e "s@DQMfile_to_process.root@${file_path}/$dqmFileName@g" > SiStripDQM_OfflineTkMap_cfg_$Run_numb.py  
#
    echo " Creating the TrackerMap.... "


#    cmsRun SiStripDQM_OfflineTkMap_cfg_${Run_numb}.py &> logfile_${Run_numb}.txt
 
    cmsRun ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/test/SiStripDQM_OfflineTkMap_Template_cfg_DB.py print globalTag=GR_P_V32::All runNumber=${Run_numb} dqmFile=${file_path}/$dqmFileName

    mv *.png $Run_numb/$1

    cat /afs/cern.ch/user/c/cctrack/scratch0/TKMap/scripts/index_template_TKMap.html | sed -e "s@RunNumber@$Run_numb@g" > $Run_numb/$1/index.html

    echo " Check TrackerMap on $Run_numb/$1 folder"

    nnn=`echo ${Run_numb} | awk '{print substr($0,0,3)}'`

## Producing the list of bad modules
    echo " Creating the list of bad modules "
#    root.exe -l -b -q   "listbadmodule.C+(\"${file_path}/$dqmFileName\")"
    
    listbadmodule ${file_path}/$dqmFileName

    mv QualityTest*txt $Run_numb/$1

## Producing the run certification by lumisection
    echo " Creating the lumisection certification:"
##    root.exe -l -b -q   "ls_cert.C+(0.95,0.95,\"${file_path}/$dqmFileName\")"

    ls_cert 0.95 0.95 ${file_path}/$dqmFileName

    mv Certification_run_* $Run_numb/$1

## Producing the PrimaryVertex/BeamSpot quality test by LS..
    if [ "${1}" == "MinimumBias" -o "${1}" == "Jet" ]; then	
	echo " Creating the BeamSpot Calibration certification summary:"
##	root.exe -l -b -q   "lsbs_cert.C+(\"${file_path}/$dqmFileName\")"

	lsbs_cert  ${file_path}/$dqmFileName

	mv Certification_BS_run_* $Run_numb/$1
    fi
## .. and harvest the bad beamspot LS with automatic emailing (if in period and if bad LS found)
    bs_bad_ls_harvester $Run_numb/$1 $Run_numb

## Producing the Module difference for ExpressStream
    if [ "${1}" == "StreamExpress" -o "${1}" == "StreamExpressCosmics" ]; then	
	echo " Creating the Module Status Difference summary:"
##	root.exe -l -b -q   "modulediff.C+( $Run_numb , \"${1}\")"

#	modulediff $Run_numb ${1}

#	./modulediff_summary $Run_numb
#	mv ModuleDifference_${Run_numb}.txt $Run_numb/$1
    fi

    dest=Beam
    if [ "${1}" == "Cosmics" -o "${1}" == "StreamExpressCosmics" ]; then dest="Cosmics"; fi

# overwrite destination for tests

#    dest=AndreaTests

#    ssh cmstacuser@cmstac05 "mkdir -p /storage/data2/SiStrip/event_display/Data2011/${dest}/${nnn}/${Run_numb} 2> /dev/null"
 mkdir -p /data/users/event_display/Data2012/${dest}/${nnn}/${Run_numb} 2> /dev/null

#    scp -r ${Run_numb}/$1 cmstacuser@cmstac05:/storage/data2/SiStrip/event_display/Data2011/${dest}/${nnn}/${Run_numb}/
#    cp -r ${Run_numb}/$1 /data/users/event_display/Data2011/${dest}/${nnn}/${Run_numb}/
    ssh cctrack@vocms01 "mkdir -p /data/users/event_display/Data2012/${dest}/${nnn}/${Run_numb} 2> /dev/null"
    scp -r ${Run_numb}/$1 cctrack@vocms01:/data/users/event_display/Data2012/${dest}/${nnn}/${Run_numb}/

    rm -f *svg
#    rm -rf ${Run_numb}
#    rm -f SiStripDQM_OfflineTkMap_cfg_$Run_numb.py
#    rm -f logfile_${Run_numb}.txt

#    rm ${file_path}/$dqmFileName

done
