#! /bin/bash

# $1 : CMSSW config file
# $2 : output directory
# $3 : output DQM directory (optional but recommended)
source /afs/cern.ch/cms/caf/setup.sh
curdir="$(pwd)"
#workdir="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/HIP/bonato/DEVEL/HIPWorkflow/CMSSW_3_2_4/src/"
workdir="<MYCMSSW>"
dqmdir="MICKEY"

if [ $# == 3 ]
then 
dqmdir="$3"
else
dqmdir="${curdir}/MONITORING/DQM/"
fi


# set up the CMS environment (choose your release and working area):
echo Setting up CMSSW environment in $workdir
cd $workdir
eval `scramv1 runtime -sh`
#export STAGE_SVCCLASS=cmscafuser
cd $curdir

#prepare log files
rfcp $1 /castor/cern.ch/cms/$2/logfiles/
BASE_JOBNAME=$(basename "$1" .py)
LOGFILE="${BASE_JOBNAME}.log"
OUTFILE="${BASE_JOBNAME}.out"
echo "Running the prescaling in $curdir..."
echo "Logfile in $LOGFILE"

#do the thing
time cmsRun $1 &> $LOGFILE
echo ""
echo "---------"
echo "File list in $(pwd): "
ls -lh 
echo "---------"
echo ""
echo ""
echo "Copying to /castor/cern.ch/cms/$2" 
# copy files to their final destination
#export STAGE_SVCCLASS=cmscafuser

STATUScp=0
for outROOT in $( ls  ALCA*resc*.root )
do
cmsStageOut $outROOT "$2/"

if [ $? -ne 0 ]
then
echo "Error in copying the .root file to CASTOR !"
fi
let STATUScp=$STATUScp+$?
done
echo "Copying to /castor/cern.ch/cms/$2/logfiles/" 
STATUScp=0
for outLOG in $( ls  *.out )
do
cp $outLOG "${dqmdir}/../logfiles/"
let STATUScp=$STATUScp+$?
cmsStageOut $outLOG "$2/"
let STATUScp=$STATUScp+$?
done

for outLOG in $( ls  *.log )
do
cmsStageOut $outLOG "$2/"
let STATUScp=$STATUScp+$?
cp $outLOG "${dqmdir}/../logfiles/"
let STATUScp=$STATUScp+$?

if [ $STATUcp -ne 0 ]
then
echo "Error in copying the .log file to CASTOR !"
fi
done




if [ $STATUScp -ne 0 ]
then
echo "Error in copying the logfiles (status=${STATUScp})! Parallel job $1" > tmpmess.txt
echo >> tmpmess.txt
echo >> tmpmess.txt
cat tmpmess.txt |  mail -s "--- ERROR in copying files during ALCAPrescale!  ---" ${USER}@mail.cern.ch
fi


### Clean up
rm -f  *.root
rm -f *.log
rm -f *.out

for logfile in $( ls ${curdir}/MONITORING/logfiles/*.log ) 
do
gzip  $logfile
rm -f $logfile
done 
