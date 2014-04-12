#!/bin/sh
#
# V.M. Ghete 2010-06-10
#

# minimum number of required arguments
NR_ARG=1

if [[ $1 == "-help" || $1 == "--help" || $# == 0 ]]; then
    echo
    echo "Run the L1 emulator starting from a RAW file"

    echo
    echo "Usage:              "
    echo "   cmsrel CMSSW_4_X_Y"
    echo "   cd CMSSW_4_X_Y/src"
    echo "   cmsenv"
    echo "   addpkg L1Trigger/Configuration"  
    echo "   cd L1Trigger/Configuration/test"  
    echo "   ./L1EmulatorFromRaw.sh EventSampleType Global_Tag"
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
    echo "   mc: auto:startup"
    echo
    echo "NOTE: the sample and the global tag must be consistent"

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

    cmsDriver.py l1EmulatorFromRaw -s RAW2DIGI,L1 -n 100 \
        --conditions ${gTag} \
        --datatier 'DIGI-RECO' \
        --eventcontent FEVTDEBUGHLT \
        --data \
        --filein /store/data/Run2011A/MinimumBias/RAW/v1/000/165/514/28C65E11-E584-E011-AED9-0030487CD700.root,/store/data/Run2011A/MinimumBias/RAW/v1/000/165/514/44C0FC26-EE84-E011-B657-003048F1C424.root \
        --customise=L1Trigger/Configuration/customise_l1EmulatorFromRaw \
        --processName='L1EmulRaw' \
        --no_exec

        exit 0   
        
elif [[ ${EventSampleType} == "mc" ]]; then

    if [[ ${GlobalTag} == '' ]]; then
        GlobalTag='auto:startup'
        echo "No global tag given. Using by default: ${GlobalTag}"
    fi

    if [[ `echo ${GlobalTag} | grep auto` ]]; then
        gTag=${GlobalTag}
    else
        gTag=FrontierConditions_GlobalTag,${GlobalTag}::All
    fi

    cmsDriver.py l1EmulatorFromRaw -s RAW2DIGI,L1 -n 100 \
        --conditions ${gTag} \
        --datatier 'DIGI-RECO' \
        --eventcontent FEVTDEBUGHLT \
        --mc \
        --filein /store/relval/CMSSW_5_2_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/02B4D46B-BB51-E111-A789-003048678A76.root,/store/relval/CMSSW_5_2_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/0EAB51E9-BD51-E111-8C43-003048679228.root \
        --customise=L1Trigger/Configuration/customise_l1EmulatorFromRaw \
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


    
    