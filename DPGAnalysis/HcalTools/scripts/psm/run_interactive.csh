#!/bin/csh
###############################################################################
###############
# to create dir. with py files, use: ./mkcfg_new120.csh runlist_run
# and then, if PYTHON_runlist_run exist, and there NN py-files, use: 
#    ./run_interactive.csh runlist_run 1 NN 
#         from main dir.
#  razbivaya po ~ 15 jobs per PC, 1-15, 15-30, ... NN
#set nn1=1
#set nn2=3
set nn1=${2}
set nn2=${3}
#set nn1=38
#set nn2=40
echo ${nn1} ${nn2} 
echo "Start..."
###############
#./mkcfg_new120.csh ${1}
#ls PYTHON_${1}/*py
###############
################################################################ loop:
set nn=0
if( ${status} == "0" ) then
foreach i (`ls PYTHON_${1}`)

set j=`echo ${i} | awk -F _ '{print $2}'`
set k=`echo ${i} | awk -F _ '{print $3}'`


@ nn = ${nn} + "1"
#echo ${i} ${j} ${k} ${1} ${nn} 
#echo ${k} ${nn} 

if( ${nn} >= ${nn1} && ${nn} <= ${nn2}   ) then
#if( ${k} >= ${nn1} && ${k} <= ${nn2}   ) then
echo ${nn} ${k}
cmsRun PYTHON_${1}/Reco_${j}_${k}_cfg.py > CMTLOG/log_${j}_${k} &
endif

################################################################
end
else
##echo "Problem: No jobs are created: check PYTHON_${1} directory: Notify developpers"
endif

