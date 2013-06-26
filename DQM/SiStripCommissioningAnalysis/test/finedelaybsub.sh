#!/bin/sh
#trap -p "echo got a signal" SIGABRT SIGBUS SIGILL SIGINT SIGKILL SIGQUIT SIGSEGV SIGSTOP 
echo preparing environment
WORKDIR=`pwd`
cd $2
eval `scramv1 runtime -sh`
cd $WORKDIR
export SCRATCH=`pwd`
cp $2/finedelayAnalysis_cfg.template finedelayAnalysis_cfg.py
echo -n 's/INPUTFILE/' > theinput
echo -n $1 | sed 's/\//\\\//g' >> theinput
echo '/g' >> theinput
sed -i -f theinput finedelayAnalysis_cfg.py 
rm theinput
echo using config file:
cat finedelayAnalysis_cfg.py
echo running in $SCRATCH
set +e 
cmsRun finedelayAnalysis_cfg.py || true
echo copying back result
cp SiStripCommissioning*.root $2
set -e
