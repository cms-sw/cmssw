#!/bin/tcsh

eval `scramv1 runtime -csh`
#set here = `pwd`
cd $2
rm -f $1.log
cmsRun $1.py >& $1.log
#cd $here
