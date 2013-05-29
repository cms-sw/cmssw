#!/bin/bash

eval `scramv1 r -sh`
#/afs/cern.ch/cms/sdt/web/qa/igprof/data/<architecture>/<release>/<candle>___<tiers>___<pileup>___<global-tag>___<process>___<counter>___<number-of-events>.sql

CANDLE="SingleMuPt10"
TIERS="VALIDATION"
TIERSBIN="VALIDATION-BINREPORT"
PILEUP="NoPileUp"
GT="AUTOMC"
PROCESS="VALIDATION"
NUMEV=10
DESTDIR="/afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/rovere/dqm-bins-igprofiles/data/${SCRAM_ARCH}/${CMSSW_VERSION}"

# Prepare destination directory for the reports, so that we can avoid to have the check done for every single report.

if [ ! -e ${DESTDIR} ]; then
  mkdir -p ${DESTDIR}
fi

function doReport {
    igprof-analyse -g -d -v -p -r $1   -s IgProf.$2.gz   | sqlite3 igreport_$2_$1.sql3
    cp igreport_$2_$1.sql3 ${DESTDIR}/${CANDLE}___${TIERS}___${PILEUP}___${GT}___${PROCESS}___$1___$2.sql3
#    igprof-analyse -g -d -v -p -r $1   -s IgProfRECO.$2.gz   | sqlite3 igreportRECO_$2_$1.sql3
}

function doCumulative {
    igprof-analyse -g -d -v -p -r $1   -s IgProf$2.gz   | sqlite3 igreport_$2_$1.sql3
    cp igreport_$2_$1.sql3 ${DESTDIR}/${CANDLE}___${TIERS}___${PILEUP}___${GT}___${PROCESS}___$1___$2.sql3
#    igprof-analyse -g -d -v -p -r $1   -s IgProfRECO$2.gz   | sqlite3 igreportRECO_$2_$1.sql3
}

doReport MEM_LIVE 1
doReport MEM_LIVE 6
doReport MEM_TOTAL 1
doReport MEM_TOTAL 6
doCumulative MEM_LIVE Cumulative_10
doCumulative MEM_TOT Cumulative_10

sqlite3 < dqm-bin-stats.sql dqm-bin-stats.sql3
cp dqm-bin-stats.sql3 ${DESTDIR}/${CANDLE}___${TIERSBIN}___${PILEUP}___${GT}___${PROCESS}___MEM_LIVE___${NUMEV}.sql3
