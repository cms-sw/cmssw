#! /bin/bash

#$1 = login name
STATUS=0

for logfile in $( ls collect*out* )
do

filename=$(basename $logfile .gz)
gunzip $logfile
if ( cat $filename | grep -i error  | grep -i unzip)
then 
#echo "===================="
#echo "--- $filename "
let STATUS=$STATUS+1
echo "error" | mail -s "--- RUNZIP ERROR in $filename ---" ${1}@mail.cern.ch
#echo ""
fi
gzip $filename
done
#echo Returning $STATUS
exit $STATUS
