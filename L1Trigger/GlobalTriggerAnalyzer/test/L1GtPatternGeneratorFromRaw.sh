#!/bin/sh
#
# V.M. Ghete 2011-02-10
#

# minimum number of required arguments
NR_ARG=1

if [[ $1 == "-help" || $1 == "--help" || $# == 0 ]]; then
    echo
    echo "Run the L1 GT pattern generatot starting from a RAW file"

    echo
    echo "Usage:              "
    echo "   cmsrel CMSSW_3_X_Y"
    echo "   cd CMSSW_3_X_Y/src"
    echo "   cmsenv"
    echo "   addpkg L1Trigger/Configuration"  
    echo "   addpkg L1Trigger/GlobalTriggerAnalyzer"  
    echo "   cd L1Trigger/GlobalTriggerAnalyzer/test"  
    echo "   ./L1GtPatternGeneratorFromRaw.sh Global_Tag EventSampleType"

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
    cmsDriver.py l1PatternGeneratorFromRaw -s RAW2DIGI,L1 -n 3564 \
        --conditions ${gTag} \
        --datatier 'DIGI-RECO' \
        --eventcontent FEVTDEBUGHLT \
        --data \
        --filein '/store/data/Run2010A/MinimumBias/RAW/v1/000/143/657/00FB1636-91AE-DF11-B177-001D09F248F8.root' \
                 '/store/data/Run2010A/MinimumBias/RAW/v1/000/143/657/023EB128-51AE-DF11-96D3-001D09F24682.root' \
        --customise=L1Trigger/GlobalTriggerAnalyzer/customise_l1GtPatternGeneratorFromRaw \
        --processName='L1GtPatternGenerator' \
        --no_exec

        exit 0   
else
    cmsDriver.py l1PatternGeneratorFromRaw -s RAW2DIGI,L1 -n 3564 \
        --conditions ${gTag} \
        --datatier 'DIGI-RECO' \
        --eventcontent FEVTDEBUGHLT \
        --filein '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/FEC54C30-612B-E011-9836-00261894386E.root'\
                 '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/FC3DBA1B-652B-E011-82F4-00261894392B.root'\
                 '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/FA15AF21-622B-E011-B577-0018F3D096F8.root' \
        --customise=L1Trigger/GlobalTriggerAnalyzer/customise_l1GtPatternGeneratorFromRaw \
        --processName='L1GtPatternGenerator' \
        --no_exec

        exit 0   
fi


    
    