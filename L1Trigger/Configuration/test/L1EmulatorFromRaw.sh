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
    echo "   cmsrel CMSSW_3_X_Y"
    echo "   cd CMSSW_3_X_Y/src"
    echo "   cmsenv"
    echo "   addpkg L1Trigger/Configuration"  
    echo "   cd L1Trigger/Configuration/test"  
    echo "   ./L1EmulatorFromRaw.sh Global_Tag EventSampleType"

    echo "Global tag must be given in one of the following formats: " 
    echo "   auto:mc"
    echo "   auto:startup"
    echo "   auto:com10"
    echo "   MC_3XY_V22" 
    echo "   START3X_V22" 
    echo "   GR10_P_V3" 
    echo
    echo "Default: auto:com10"
    echo
    
    echo "Event sample type follow the cmsDriver convention: " 
    echo "   data"
    echo "   mc"
    echo "   empty string default to mc"  

    if [[ $# < ${NR_ARG} ]]; then
      echo -e "\n $# arguments available, while ${NR_ARG} are required. \n Check command again."
    fi

    exit 1    
fi

GlobalTag=$1
EventSampleType=$2

# global tag manipulation
if [[ ${GlobalTag} == '' ]]; then
    GlobalTag='auto:com10'
    echo "No global tag given. Using by default: ${GlobalTag}"
fi

      
if [[ `echo ${GlobalTag} | grep auto` ]]; then
    gTag=${GlobalTag}
else
    gTag=FrontierConditions_GlobalTag,${GlobalTag}::All
fi

#

if [[ ${EventSampleType} == "data" ]]; then
    cmsDriver.py l1EmulatorFromRaw -s RAW2DIGI,L1 -n 100 \
        --conditions ${gTag} \
        --datatier 'DIGI-RECO' \
        --eventcontent FEVTDEBUGHLT \
        --data \
        --filein '/store/data/Commissioning10/Cosmics/RAW/v3/000/127/715/FCB12D5F-6C18-DF11-AB4B-000423D174FE.root' \
        --customise=L1Trigger/Configuration/customise_l1EmulatorFromRaw \
        --processName='L1EmulRaw' \
        --no_exec

        exit 0   
else
    cmsDriver.py l1EmulatorFromRaw -s RAW2DIGI,L1 -n 100 \
        --conditions ${gTag} \
        --datatier 'DIGI-RECO' \
        --eventcontent FEVTDEBUGHLT \
        --filein '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/FAA58A57-3D1E-DF11-87A5-001731A283DF.root' \
        --customise=L1Trigger/Configuration/customise_l1EmulatorFromRaw \
        --processName='L1EmulRaw' \
        --no_exec

        exit 0   
fi


    
    