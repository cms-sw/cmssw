#!/bin/tcsh

# cmssw wrapper for job submission
# run cmssw and copy output and log to destdir 
# 
# $Id: cmssw-job.csh,v 1.3 2009/01/16 08:40:05 argiro Exp $

set conffile = $1 
set logfile  = $2
set errfile  = $3
set workdir  = $4
set destdir  = $5

set lsfdir=`pwd`

cd $workdir
eval `scramv1 runtime -csh`
cd $lsfdir

limit coredumpsize 0
#setenv STAGE_SVCCLASS cmscaf
#here to keep  log files
echo "Running cmsRun  >& $logfile"  $workdir/$conffile 
cmsRun $workdir/$conffile >& $lsfdir/$logfile
rm -f $workdir/$conffile
ls

setenv logfilebase `basename $logfile`
setenv logfiledir  `dirname $logfile`

setenv protocol `echo $destdir | cut -d ':' -f1 `
setenv destpath `echo $destdir | cut -d ':' -f2 `

echo "copying to $destdir"

if ($protocol == 'castor') then
   rfcp $logfile $destpath
else
   scp * $destdir
endif


