#!/bin/bash

if (( ${#LOCALRT} < 4 ))
then
    echo Please setup your runtime environment!
    exit
fi

ABFILE=$1
POOLFILE=${ABFILE##*/}
POOLFILE=${POOLFILE#AB-}

echo "making ${POOLFILE}"

rm -f ${POOLFILE}

RUN=${POOLFILE##*-}
RUN=${RUN%.root}

echo "for run: ${RUN}"


EVENTLIMIT="-1";


### create the file
CFGFILE=/tmp/runLaserAnalyzer_${USER}.cfg
cat > ${CFGFILE}<<EOF
process ANALYZE = {

// Loads the events from testbeam files
        source = PoolSource { 
                untracked vstring fileNames = { 'file:${ABFILE}' }
                untracked int32 maxEvents = ${EVENTLIMIT}
        }

module ecalLaserAnalyze = EcalLaserAnalyzer {
 
        untracked string hitCollection = "EcalUncalibRecHitsEB"
        untracked string hitProducer = "ecaluncalibrechit"
        untracked string PNdigiCollection = ""
        untracked string digiProducer = "ecalEBunpacker"
        untracked string outFileName = "${POOLFILE}"
        untracked string SM = "SM22"
        untracked string Run = "${RUN}"
 
}
 
path p = { ecalLaserAnalyze }

}
EOF

# Stuff related to the setup


# run cmsRun
SMLOG=SM-LOG.txt

cmsRun ${CFGFILE} >& ${SMLOG} &

wait

echo "LASER ANALYZE DONE"

exit
