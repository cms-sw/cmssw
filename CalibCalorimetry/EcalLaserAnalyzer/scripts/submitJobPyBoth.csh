#!/bin/tcsh

eval `scramv1 runtime -csh`
#set here = `pwd`

#${cfg1} ${cfg2} ${jobdir} ${donefile} ${shortjobdir} ${nprocnew} ${dorezip}

cd $3
rm -f $1.log
rm -f $2.log

set datestart1=`date +%F`
set datestart2=`date +%R:%S`

# send jobs:
#============
cmsRun $1.py >& $1.log 
cmsRun $2.py >& $2.log 

set date1=`date +%F`
set date2=`date +%R:%S`

# write sendfile:
#=================
echo "$5 START=${datestart1}_${datestart2} STOP=${date1}_${date2} NPROC=${6}" >> $4


# eventually remove unzipped file:
#=================================

if( $7 == 1) then
  rm input.lmf
endif
