#!/bin/bash

if (( ${#LOCALRT} < 4 ))
then
    echo Please setup your runtime environment!
    exit
fi

FILE=$1

RUN=${FILE##*/}
RUN=${RUN#h4b.}
RUN=${RUN%.A.0.0.root}

echo "for run: ${RUN}"
SM="SM22"

ABFILE=AB-${SM}-${RUN}.root

echo "making rechits in: ${ABFILE}"

LPFILE=LP-${SM}-${RUN}.root

echo "making laser shape file: ${LPFILE}"

LPTXT=LP-${SM}-${RUN}.txt

#exit

EVENTLIMIT="-1";


### create the file
CFGFILE1=/tmp/runLaserSHAPE_${USER}.cfg
cat > ${CFGFILE1}<<EOF
process LASERSHAPE = {

  untracked PSet maxEvents = {untracked int32 input = ${EVENTLIMIT}}

// Loads the events from testbeam files
        source = PoolSource { 
                untracked vstring fileNames = { 'file:${FILE}' }
//                untracked int32 maxEvents = ${EVENTLIMIT}
        }

module ecalEBunpacker = EcalDCCUnpackingModule {}

module ecalLaser = EcalLaserShapeTools {
  untracked int32 verbosity = 0
  string hitCollection = "ecalEBuncalibFixedAlphaBetaRecHits"
  string digiCollection = ""
  string hitProducer = "uncalibHitMaker"
  string digiProducer = "ecalEBunpacker"
  string pndiodeProducer = "ecalEBunpacker"
  untracked string HistogramOutFile = "histos_devel.root"
  untracked string rootOutFile = "${LPFILE}"
  untracked string txtOutFile = "${LPTXT}"
} 

path p = { ecalEBunpacker, ecalLaser }

}
EOF




# run cmsRun
LPLOG=LP-LOG.txt
rm -f ${LPLOG}
cmsRun ${CFGFILE1} >& ${LPLOG} &

wait

echo "LASER SHAPE DONE"


CFGFILE2=/tmp/runLaserRECO_${USER}.cfg
cat > ${CFGFILE2}<<EOF
process LASERRECO = {

include "CalibCalorimetry/EcalTrivialCondModules/data/EcalTrivialCondRetriever.cfi"

untracked PSet maxEvents = {untracked int32 input = ${EVENTLIMIT} }

source = PoolSource {
  untracked vstring fileNames = { 'file:${FILE}' }
  untracked bool debugFlag = false
}

module ecalEBunpacker = EcalDCCUnpackingModule {}

#PNlook module has been removed. read directly in laser analyzer

module ecaluncalibrechit = EcalFixedAlphaBetaFitUncalibRecHitProducer {
  InputTag EBdigiCollection = ecalEBunpacker:
  InputTag EEdigiCollection = ecalEBunpacker:eeDigis
  string EBhitCollection = "EcalUncalibRecHitsEB"
  string EEhitCollection = "EcalUncalibRecHitsEE"
  untracked string AlphaBetaFilename = "${LPTXT}"
}

module out = PoolOutputModule {
  untracked string fileName = "${ABFILE}"
}

path p = { ecalEBunpacker, ecaluncalibrechit }

endpath ep = { out }

}
EOF

RECOLOG=RECO-LOG.txt
rm -f ${RECOLOG}
cmsRun ${CFGFILE2} >& ${RECOLOG} &

wait

echo "DIGI+RECO DONE"

exit





