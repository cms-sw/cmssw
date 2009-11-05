#! /bin/bash

# $1 : CMSSW config file
# $2 : output directory
# $3 : output DQM directory (optional but recommended)

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
export STAGE_SVCCLASS=cmscaf
cd $curdir

#prepare log files
rfcp $1 $2/logfiles/
BASE_JOBNAME=$(basename "$1" .py)
LOGFILE="${BASE_JOBNAME}.log"
OUTFILE="${BASE_JOBNAME}.out"
echo "Running the prescaling in $curdir..."


#do the thing
time cmsRun $1 &> $LOGFILE
echo ""
echo "---------"
echo "File list in $(pwd): "
ls -lh 
echo "---------"
echo ""
echo "Copying to $2/logfiles/" 
# copy files to their final destination
export STAGE_SVCCLASS=cmscafuser

rfcp ALCA*resc*.root "$2/"
if [ $? -ne 0 ]
then
echo "Error in copying the .root file to CASTOR !"
fi
let STATUScp=$STATUScp+$?

STATUScp=0
rfcp *.log "$2/logfiles/"
let STATUScp=$STATUScp+$?
rfcp *.out "$2/logfiles/"
let STATUScp=$STATUScp+$?
rfcp *.log "${dqmdir}/../logfiles/"
let STATUScp=$STATUScp+$?
rfcp *.out "${dqmdir}/../logfiles/$OUTFILE"
let STATUScp=$STATUScp+$?

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
##rm -f $logfile
done 
