#!/bin/bash

eval `scramv1 r -sh`
#/afs/cern.ch/cms/sdt/web/qa/igprof/data/<architecture>/<release>/<candle>___<tiers>___<pileup>___<global-tag>___<process>___<counter>___<number-of-events>.sql

CANDLE="MultiJet"
DQMSEQUENCE=DQM
TIERS=${DQMSEQUENCE}
TIERSBIN="${DQMSEQUENCE}-BINCOUNTS"
PILEUP="RealData"
GT="AUTOCOM"
PROCESS="RECO"
NUMEV=100
TNUM=26
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
doReport MEM_LIVE 26
doReport MEM_LIVE 51
doReport MEM_LIVE 76
doReport MEM_TOTAL 1
doReport MEM_TOTAL 26
doReport MEM_TOTAL 51
doReport MEM_TOTAL 76
doCumulative MEM_LIVE Cumulative_100
doCumulative MEM_TOT Cumulative_100

sqlite3 < dqm-bin-stats.sql dqm-bin-stats.sql3
cp dqm-bin-stats.sql3 ${DESTDIR}/${CANDLE}___${TIERSBIN}___${PILEUP}___${GT}___${PROCESS}___MEM_LIVE___${NUMEV}.sql3
python ../../py2html_new.py test_${TNUM}.py igreport_76_MEM_LIVE.sql3
mkdir -p /afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/rovere/DQMSequences/pp_${CMSSW_VERSION}_${DQMSEQUENCE}_test${TNUM}/step2
cp -pr html /afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/rovere/DQMSequences/pp_${CMSSW_VERSION}_${DQMSEQUENCE}_test${TNUM}/step2
sed -i -e "s#\(.*<!-- PLACEHOLDER_pp_TEST18 -->\)#  <li> <a href=\"./pp_${CMSSW_VERSION}_${DQMSEQUENCE}_test${TNUM}/step2/html/index.html\" >${CMSSW_VERSION} - Step2 - ${DQMSEQUENCE} </a> </li> \n\1#" /afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/rovere/DQMSequences/index.html
