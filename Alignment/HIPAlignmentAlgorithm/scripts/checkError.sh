#! /bin/bash

#$1 = login name
STATUS=0

for logfile in $( ls collect*out* )
do
#echo
echo
#echo "LOGFILE : $logfile"
iter=${logfile##co*ct} 
iter=`expr match "$iter"  '\([0-9]*\)'`
echo "Checking ITER $iter"

filename="collect.out.gz"

#echo "Suffix is ${filename:(-2)}"

ISZIPPED=0

if [ "${filename:(-2)}" == "gz" ]
then
    filename=$(basename $logfile .gz)
    gunzip $logfile
    ISZIPPED=1
else
    filename="$logfile"
fi

           
             
if [ $ISZIPPED -lt 3 ]    #this overrides the check and will send an email 
                           #also for the previous iterations with problems. Redundant.
#if [ $ISZIPPED -eq 0 ]
then
    if ( cat $filename | grep -i 'error'  | grep -i 'unzip')
	then 
#echo "===================="
#echo "--- $filename "
	let STATUS=$STATUS+1
	echo "error from $(hostname): RUnzip erro in collect at iter $iter " | mail -s "--- RUNZIP ERROR in $filename ---" ${1}@mail.cern.ch
    fi
fi

if [ $ISZIPPED -gt 0 ]
then
 gzip $filename
fi

done
#echo Returning $STATUS
exit $STATUS
