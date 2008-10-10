#!/bin/csh

date

setenv WORKDIR `pwd`/RESULT
mkdir ${WORKDIR}

setenv MYDIR /afs/cern.ch/user/a/anikiten/scratch0/CMSSW_2_1_9/src/JetMETCorrections/JetPlusTrack/test

setenv REDIR /afs/cern.ch/user/a/anikiten/scratch0/CMSSW_2_1_9/src/JetMETCorrections/JetPlusTrack/test/RESULT

cd ${MYDIR}

eval `scramv1 runtime -csh`

cp ${MYDIR}/JPTanalyzer_cfg.py ${WORKDIR}/.

cd ${WORKDIR}

ls ${WORKDIR}

cmsRun -p JPTanalyzer_cfg.py

cp analysis.root ${REDIR}/.
