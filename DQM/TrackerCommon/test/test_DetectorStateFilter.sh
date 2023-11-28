#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

function countEvents() {
    local FILE="$1"
    NUMBER=$(echo `edmFileUtil $FILE | grep events` | awk '{print $6}')
    echo $NUMBER
}

echo "TESTING detectorStateFilter ..."

# Take as input file an express FEVT file from 2021 commissioning, Run 343498
# https://cmsoms.cern.ch/cms/runs/report/fullscreen/4637?cms_run=343498&cms_run_sequence=GLOBAL-RUN
# It has Pixels and Strips HV ON
# Also used to feed RelVal Matrix wfs 138.1, 138.2

### LS2 - MWGR 2021 - CSC, DAQ, DCS, DQM, DT, ECAL, GEM, HCAL, L1SCOUT, PIXEL, RPC, TCDS, TRACKER, TRG ###
#INPUTFILE="/store/express/Commissioning2021/ExpressCosmics/FEVT/Express-v1/000/343/498/00000/004179ae-ac29-438a-bd2d-ea98139c21a7.root"

# Take as input file an express FEVT file from 2021 commissioning, Run 338714
# https://cmsoms.cern.ch/cms/runs/report/fullscreen/4637?cms_run=338714&cms_run_sequence=GLOBAL-RUN
# It has Pixels HV OFF but Strips HV ON
# It was used to feed RelVal Matrix wfs 138.1, 138.2

 ### LS2 - MWGR#5 2020 - CSC, DAQ, DCS, DQM, DT, ECAL, GEM, HCAL, RPC, TCDS, TRACKER, TRG ###
INPUTFILE="/store/express/Commissioning2020/ExpressCosmics/FEVT/Express-v1/000/338/714/00000/515CD930-9BBA-1945-87C3-AD555E25F301.root"

# test Pixel
printf "TESTING Pixels ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/test_DetectorStateFilter_cfg.py maxEvents=10 isStrip=False inputFiles=$INPUTFILE outputFile=outPixels.root || die "Failure filtering on pixels" $?

# test Strips
printf "TESTING Strips ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/test_DetectorStateFilter_cfg.py maxEvents=10 isStrip=True inputFiles=$INPUTFILE outputFile=outStrips.root || die "Failure filtering on strips" $?

# count events
pixelCounts=`countEvents outPixels_numEvent10.root`
stripCounts=`countEvents outStrips_numEvent10.root`

if [[ $pixelCounts -eq 0 ]]
then
  echo "The number of events in the pixel filter file matches expectations ($pixelCounts)."
else 
  echo "WARNING!!! The number of events in the pixel filter file ($pixelCounts) does NOT match expectations (0)."
  exit 1
fi

if [[ $stripCounts -eq 10 ]]
then
  echo "The number of events in the strip filter file matches expectations ($stripCounts)."
else 
  echo "WARNING!!! The number of events in the strip filter file ($stripCounts) does NOT match expectations (10)."
  exit 1
fi
