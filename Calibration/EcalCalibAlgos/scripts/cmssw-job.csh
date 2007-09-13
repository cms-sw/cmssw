#!/bin/tcsh

set conffile = $1 
set logfile  = $2
set errfile  = $3
set workdir  = $4
set outfile  = $5
set destdir  = $6

cd $workdir
eval `scramv1 runtime -csh`

limit coredumpsize 0

#here to keep  log files
echo "Running cmsRun  >& $logfile" $conffile 
cmsRun $conffile >& $logfile

rm -f $conffile

setenv logfilebase `basename $logfile`
setenv logfiledir  `dirname $logfile`

setenv protocol `echo $destdir | cut -d ':' -f1 `
setenv destpath `echo $destdir | cut -d ':' -f2 `

echo "copying $outfile to $destdir"

if ($protocol == 'castor') then
   rfcp $outfile $destpath
   rfcp $logfile $destpath
else
   scp $outfile $destdir 
   scp $logfile $destdir 
endif


rm $outfile 
rm $logfile 
