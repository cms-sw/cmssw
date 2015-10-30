#!/bin/tcsh

# working dir
set workdir = $1

# The batch job directory (will vanish after job end):
set curdir = `pwd`

cp <ODIR>/../main/IOIteration*.root $curdir/.
cp <ODIR>/../main/IOAlignedPositions*.root $curdir/. 

# printing
echo Setting up CMSSW environment in $workdir
echo Running in $curdir...

# set up the CMS environment (choose your release and working area):
cd $workdir
eval `scramv1 runtime -csh`
setenv STAGE_SVCCLASS cmscaf
rehash

cd $curdir
# Execute
time cmsRun <ODIR>/<JOBTYPE>
echo  $? >  <ODIR>/DONE

if(`cat <ODIR>/DONE | cut -b1` == 0) then 
exit 0
else
exit 1
endif  
