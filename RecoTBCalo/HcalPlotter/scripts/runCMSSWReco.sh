#!/bin/bash

if (( ${#LOCALRT} < 4 ))
then
    if (( ${#HCALDAQ_SW_LOC} > 3  && ${#HCAL_CMSSW_RELEASE} > 3 ))
    then
	pushd $HCALDAQ_SW_LOC/src/$HCAL_CMSSW_RELEASE/src >/dev/null
	eval `scramv1 runtime -sh`
	popd >/dev/null
    fi
fi

if (( ${#LOCALRT} < 4 ))
then
    echo Please setup your runtime environment!
    exit
fi

ARG1=$1
OUTPUTFILE=$2

if [[ -e ./reco_setup.rc ]] 
then
    source ./reco_setup.rc
fi

if (( ${#DOCALIB} > 1 ))
then
    UNPACKCALIB=true
else
    UNPACKCALIB=false
fi

# ARG1 determines the file selection mode
#
if [[ "${ARG1}" == *[[:alpha:]]* ]]
then
    # Filename mode
    FILE=$ARG1
else
    # Run Number mode
    FILE=`printf "${FORMAT}" ${ARG1}`
fi

echo $FILE

if (( ${#EVENTLIMIT} == 0 )) 
then
    EVENTLIMIT="-1";
fi

if (( ${#MAPFILE} < 10 )) 
then
    echo "Must have a map file in the reco_setup.rc or config file"
    exit 1
fi

#### common head part of Config File
### create the file
CFGFILE=/tmp/runCMSSWReco_${USER}.cfg
cat > ${CFGFILE}<<EOF
process RECO = {
 service = MessageLogger
        {
        untracked vstring destinations = { "cout" }
        untracked vstring categories = {  "DDLParser", "FwkJob", "FwkReport"}
        untracked PSet cout = {untracked string threshold = "INFO"
                               untracked PSet INFO        = { untracked int32 limit = 10000 }
                               untracked PSet DDLParser = { untracked int32 limit = 0 }
                               untracked PSet FwkJob = { untracked int32 limit =10 }
                               untracked PSet FwkReport = { untracked int32 limit = 20 }
                              }
        }
EOF

### Mode-dependent part

if [[ "$MODE" == "TESTSTAND" || "${FILE}" == *"HTB_"* ]]
then
# STREAMS: comma-separated list of strings, like: " 'HCAL_Trigger','HCAL_DCC020'  "
    STREAMS=","${FEDS}
    STREAMS=${STREAMS/20/020} # special case for teststand files
    STREAMS=${STREAMS//,/\',\'HCAL_DCC}
    STREAMS="'HCAL_Trigger"${STREAMS}\'

    if [[ "$MODE" == "TESTSTAND" && "$FIRSTFED" == "" ]]
    then
	FIRSTFED=20
    fi
    EXTRAPREPATH="tbunpacker,"
cat >> ${CFGFILE}<<EOF
    // Loads the events from testbeam files
    source = HcalTBSource { 
                untracked vstring fileNames = { "file:${FILE}" }
                untracked int32 maxEvents = ${EVENTLIMIT}
                untracked vstring streams = { ${STREAMS} }
    }
    module tbunpacker = HcalTBObjectUnpacker {
           untracked int32 HcalTriggerFED  = 1
           untracked int32 HcalSlowDataFED = -1
           untracked int32 HcalTDCFED      = -1
           untracked int32 HcalSourcePosFED = -1
           untracked bool IncludeUnmatchedHits = false
           untracked string ConfigurationFile='configQADCTDC.txt'
    }
    module hcaldigi = HcalRawToDigi {
	int32 firstSample = 0
	int32 lastSample = 9
	untracked bool UnpackCalib = ${UNPACKCALIB}
	bool FilterDataQuality = true
	untracked int32 HcalFirstFED = ${FIRSTFED}
	untracked vint32 FEDs = { ${FEDS} }
    }
EOF
# MTCC mode unavailable for now
# elif [[ "$MODE" == "MTCC" ]] 
#     then
#
#     if [[ -z "$FIRSTFED" ]]
#     then
#	 FIRSTFED=700
#     fi
# 
#     EXT=${FILE##*.}
#     if [[ "$EXT" == "dat" ]]
# 	then
# 	
# cat >> ${CFGFILE}<<EOF
#   source = NewEventStreamFileReader
#   {
#     string fileName = "${FILE}"
#     untracked int32 maxEvents = ${EVENTLIMIT}
#     int32 max_event_size = 7000000
#     int32 max_queue_depth = 5
#   }
# EOF
#     elif [[ "$EXT" == "root" ]]
# 	then
# 
# 	PREF=${FILE#*:}
# 	if (( ${#PREF} == ${#FILE} ))
# 	    then
# 	    FILE="file:$FILE"
# 	fi
# 
# 
# cat >> ${CFGFILE}<<EOF
#         source = PoolConvert { 
#                 untracked vstring fileNames = { '${FILE}' }
#                 untracked int32 maxEvents = ${EVENTLIMIT}
#         }
# EOF
#     fi
elif [[ "$MODE" == "TB06" ]] 
    then

    if [[ -z "$FIRSTFED" ]]
    then
	FIRSTFED=700
    fi

    PREF=${FILE#*:}
    if (( ${#PREF} == ${#FILE} ))
	then
	FILE="file:$FILE"
    fi

    EXTRAPREPATH="tbunpacker,"
cat >> ${CFGFILE}<<EOF
    source = PoolSource { 
                untracked vstring fileNames = { "${FILE}" }
                untracked int32 maxEvents = ${EVENTLIMIT}
    }

         module tbunpacker = HcalTBObjectUnpacker {
                untracked int32 HcalTriggerFED  = 1
                untracked int32 HcalSlowDataFED = -1  # 3
                untracked int32 HcalTDCFED      = -1  # 8
                untracked int32 HcalQADCFED      = -1 # 8
                untracked int32 HcalSourcePosFED = -1
                untracked bool IncludeUnmatchedHits = false
#               untracked string ConfigurationFile='configQADCTDC.txt'
         }
	 module hcaldigi = HcalRawToDigi {
	     int32 firstSample = 0
	     int32 lastSample = 9
	     untracked bool UnpackCalib = ${UNPACKCALIB}
	     bool FilterDataQuality = true
	     untracked int32 HcalFirstFED = ${FIRSTFED}
	     untracked vint32 FEDs = { ${FEDS} }
	 }
EOF
elif [[ "$MODE" == "USC" ]] 
    then
    EXTRAPREPATH="tbunpacker,"
cat >> ${CFGFILE}<<EOF
    // Loads the events from testbeam files
    source = HcalTBSource { 
                untracked vstring fileNames = { "file:${FILE}" }
                untracked int32 maxEvents = ${EVENTLIMIT}
                untracked vstring streams = { 'HCAL_Trigger',
		    'HCAL_DCC700','HCAL_DCC701','HCAL_DCC702','HCAL_DCC703',
		    'HCAL_DCC704','HCAL_DCC705','HCAL_DCC706','HCAL_DCC707',
		    'HCAL_DCC708','HCAL_DCC709','HCAL_DCC710','HCAL_DCC711',
		    'HCAL_DCC712','HCAL_DCC713','HCAL_DCC714','HCAL_DCC715',
		    'HCAL_DCC716','HCAL_DCC717','HCAL_DCC718','HCAL_DCC719',
		    'HCAL_DCC720','HCAL_DCC721','HCAL_DCC722','HCAL_DCC723',
		    'HCAL_DCC724','HCAL_DCC725','HCAL_DCC726','HCAL_DCC727',
		    'HCAL_DCC728','HCAL_DCC729','HCAL_DCC730','HCAL_DCC731' }
        }

    module tbunpacker = HcalTBObjectUnpacker {
           untracked int32 HcalTriggerFED  = 1
           untracked int32 HcalSlowDataFED = -1
           untracked int32 HcalTDCFED      = -1
           untracked int32 HcalSourcePosFED = -1
           untracked bool IncludeUnmatchedHits = false
           untracked string ConfigurationFile='configQADCTDC.txt'
       }

    module hcaldigi = HcalRawToDigi {
	int32 firstSample = 0
	int32 lastSample = 9
	untracked bool UnpackCalib = ${UNPACKCALIB}
	bool FilterDataQuality = true
	// untracked int32 HcalFirstFED = ${FIRSTFED} # use default!
	untracked vint32 FEDs = {}
    }
EOF
else
  echo Unknown mode '$MODE'
  exit
fi    

#### common tail part of Config File
cat >> ${CFGFILE}<<EOF99
   module hbhereco = HcalSimpleReconstructor {
    /// Indicate which digi time sample to start with when
    /// integrating the signal
    int32 firstSample = 1
    /// Indicate how many digi time samples to integrate over
    int32 samplesToAdd = 8
    /// Indicate whether to apply energy-dependent time-slew corrections
    bool correctForTimeslew = true
    /// Indicate whether to apply corrections for pulse containment in the summing window
    bool correctForPhaseContainment = true
    /// Nanosecond phase for pulse containment correction (default of 13 ns appropriate for simulation)
    double correctionPhaseNS = 13.0
    /// Indicate which subdetector to reconstruct for.
    string Subdetector = 'HBHE'
    /// Give the label associated with the HcalRawToDigi unpacker module.
    /// NOTE: cross-dependency here.
    InputTag digiLabel = hcaldigi
  }

   module horeco = HcalSimpleReconstructor {
    /// Indicate which digi time sample to start with when
    /// integrating the signal
    int32 firstSample = 1
    /// Indicate how many digi time samples to integrate over
    int32 samplesToAdd = 8
    /// Indicate whether to apply energy-dependent time-slew corrections
    bool correctForTimeslew = false
    /// Indicate whether to apply corrections for pulse containment in the summing window
    bool correctForPhaseContainment = true
    /// Nanosecond phase for pulse containment correction (default of 13 ns appropriate for simulation)
    double correctionPhaseNS = 13.0
    /// Indicate which subdetector to reconstruct for.
    string Subdetector = 'HO'
    /// Give the label associated with the HcalRawToDigi unpacker module.
    /// NOTE: cross-dependency here.
    InputTag digiLabel = hcaldigi
  }

   module hfreco = HcalSimpleReconstructor {
    /// Indicate which digi time sample to start with when
    /// integrating the signal
    int32 firstSample = 1
    /// Indicate how many digi time samples to integrate over
    int32 samplesToAdd = 4
    /// Indicate whether to apply energy-dependent time-slew corrections
    bool correctForTimeslew = true
    /// Indicate whether to apply corrections for pulse containment in the summing window (not in HF or ZDC)
    bool correctForPhaseContainment = false
    /// Nanosecond phase for pulse containment correction (ignored if correction is not used)
    double correctionPhaseNS = 0.0
    /// Indicate which subdetector to reconstruct for.
    string Subdetector = 'HF'
    /// Give the label associated with the HcalRawToDigi unpacker module.
    /// NOTE: cross-dependency here.
    InputTag digiLabel = hcaldigi
  }

  module plotanal = HcalQLPlotAnal {
     untracked InputTag hbheRHtag = hbhereco
     untracked InputTag hoRHtag   = horeco
     untracked InputTag hfRHtag   = hfreco
     untracked InputTag hcalDigiTag = hcaldigi
     untracked InputTag hcalTrigTag = tbunpacker
     untracked string outputFilename = "${OUTPUTFILE}"
     untracked bool     doCalib   = ${UNPACKCALIB} // false is the default

//   untracked double calibFC2GeV = 0.2   // 0.2 is the default

     PSet HistoParameters = 
     {
        double pedGeVlo   = ${PED_E_GEV_LO}
        double pedGeVhi   = ${PED_E_GEV_HI}
        double pedADClo   = ${PED_E_ADC_LO}
        double pedADChi   = ${PED_E_ADC_HI}
        double ledGeVlo   = ${LED_E_GEV_LO}
        double ledGeVhi   = ${LED_E_GEV_HI}
        double laserGeVlo = ${LASER_E_GEV_LO}
        double laserGeVhi = ${LASER_E_GEV_HI}
        double otherGeVlo = ${OTHER_E_GEV_LO}
        double otherGeVhi = ${OTHER_E_GEV_HI}
        double beamGeVlo  = ${BEAM_E_GEV_LO}
        double beamGeVhi  = ${BEAM_E_GEV_HI}
        double timeNSlo   = ${TIME_NS_LO}
        double timeNShi   = ${TIME_NS_HI}
     }
  }

  path p = { ${EXTRAPREPATH} hcaldigi, hbhereco, horeco, hfreco, plotanal }


// stuff for the calibration system
  es_module = HcalDbProducer {}
EOF99

# Stuff related to the setup

HARDCODED="\"PedestalWidths\", \"GainWidths\", \"QIEShape\", \"QIEData\", \"ChannelQuality\""
TEXT="{\nstring object=\"ElectronicsMap\"\nFileInPath file=\"${MAPFILE}\"\n}\n"

if (( ${#PEDESTALFILE} > 1 )) 
then
    TEXT=${TEXT}",{\nstring object=\"Pedestals\"\nFileInPath file=\"${PEDESTALFILE}\"\n}\n"
else
    HARDCODED=${HARDCODED}", \"Pedestals\""
fi

if (( ${#GAINSFILE} > 1 )) 
then
    TEXT=${TEXT}",{\nstring object=\"Gains\"\nFileInPath file=\"${GAINSFILE}\"\n}\n"
else
    HARDCODED=${HARDCODED}", \"Gains\""
fi

echo "   es_source es_hardcode = HcalHardcodeCalibrations { untracked vstring toGet= { ${HARDCODED}  } }" >> ${CFGFILE}

echo "   es_source es_ascii = HcalTextCalibrations { VPSet input = {" >> ${CFGFILE}
printf "${TEXT}" >> ${CFGFILE}
echo "   } }" >> ${CFGFILE}

echo "}" >> ${CFGFILE}

# run cmsRun
cmsRun ${CFGFILE}
