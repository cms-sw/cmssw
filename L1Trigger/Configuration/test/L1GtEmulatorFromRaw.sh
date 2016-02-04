#!/bin/sh
#
# V.M. Ghete 2010-06-10
#

# minimum number of required arguments
NR_ARG=1

if [[ $1 == "-help" || $1 == "--help" || $# == 0 ]]; then
    echo
    echo "Run the L1 GT emulator starting from a RAW file, with input from unpacked GCT and GMT products"

    echo
    echo "Usage:              "
    echo "   cmsrel CMSSW_4_X_Y"
    echo "   cd CMSSW_4_X_Y/src"
    echo "   cmsenv"
    echo "   addpkg L1Trigger/Configuration"  
    echo "   cd L1Trigger/Configuration/test"  
    echo "   ./L1GtEmulatorFromRaw.sh EventSampleType Global_Tag"
    echo

    echo "Event sample type follow the cmsDriver convention: " 
    echo "   data"
    echo "   mc"
    echo

    echo "Global tag must be given in one of the following formats: " 
    echo "   auto:mc"
    echo "   auto:startup"
    echo "   auto:com10"
    echo "   MC_42_V13" 
    echo "   START42_V13" 
    echo "   GR_R_42_V18" 
    echo
    echo "Default:"
    echo "   data: auto:com10"
    echo "   mc: auto:mc"
    echo
    

    if [[ $# < ${NR_ARG} ]]; then
      echo -e "\n $# arguments available, while ${NR_ARG} are required. \n Check command again."
    fi

    exit 1    
fi

EventSampleType=$1
GlobalTag=$2

#

if [[ ${EventSampleType} == "data" ]]; then

    if [[ ${GlobalTag} == '' ]]; then
        GlobalTag='auto:com10'
        echo "No global tag given. Using by default: ${GlobalTag}"
    fi

    if [[ `echo ${GlobalTag} | grep auto` ]]; then
        gTag=${GlobalTag}
    else
        gTag=FrontierConditions_GlobalTag,${GlobalTag}::All
    fi
      
    cmsDriver.py l1GtEmulatorFromRaw -s RAW2DIGI,L1 -n 100 \
        --conditions ${gTag} \
        --datatier 'DIGI-RECO' \
        --eventcontent FEVTDEBUGHLT \
        --data \
        --filein /store/data/Run2011A/MinimumBias/RAW/v1/000/165/514/28C65E11-E584-E011-AED9-0030487CD700.root,/store/data/Run2011A/MinimumBias/RAW/v1/000/165/514/44C0FC26-EE84-E011-B657-003048F1C424.root \
        --customise=L1Trigger/Configuration/customise_l1GtEmulatorFromRaw \
        --processName='L1EmulRaw' \
        --no_exec

        exit 0  
         
elif [[ ${EventSampleType} == "mc" ]]; then

    if [[ ${GlobalTag} == '' ]]; then
        GlobalTag='auto:mc'
        echo "No global tag given. Using by default: ${GlobalTag}"
    fi

    if [[ `echo ${GlobalTag} | grep auto` ]]; then
        gTag=${GlobalTag}
    else
        gTag=FrontierConditions_GlobalTag,${GlobalTag}::All
    fi
      

    cmsDriver.py l1GtEmulatorFromRaw -s RAW2DIGI,L1 -n 100 \
        --conditions ${gTag} \
        --datatier 'DIGI-RECO' \
        --eventcontent FEVTDEBUGHLT \
        --filein /store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0068/BC61B16D-647C-E011-9972-0030486791BA.root,/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0063/FE440F3F-847B-E011-8E8F-0018F3D096CA.root \
        --customise=L1Trigger/Configuration/customise_l1GtEmulatorFromRaw \
        --processName='L1EmulRaw' \
        --no_exec

        exit 0   
        
else
    echo "Option for sample type ${EventSampleType} not valid."
    echo "Valid options:"
    echo "   data"
    echo "   mc"

    exit 1

fi


    
    