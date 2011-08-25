#!/bin/sh

echo "Updating runs"

year=`date | awk '{print $6}'`

dbsql "find run where run.createdate = ${year} and run.starttime != 0" > times_${year}.txt

lastRun=`tail -n 2 full_run.js | head -n 1 | awk -F"[" '{print $NF}' | awk -F"]" '{print $1}' | awk -F, '{print $3}'`
# echo "lastRun = $lastRun"
cat times_${year}.txt | sort > sortedRuns.txt

# awk -v var=${lastRun} '/$var/{n++}{print >"out" n ".txt" }' sortedRuns.txt

lineNum=`cat sortedRuns.txt | grep -n ${lastRun} | awk -F: '{print $1}'`
totalLines=`wc -l sortedRuns.txt | awk '{print $1}'`
let "tailLines = $totalLines - $lineNum"
# echo $tailLines
tail -n $tailLines sortedRuns.txt > runsToUpdate_temp.txt

rm -f runsToUpdate.txt
touch runsToUpdate.txt
cat runsToUpdate_temp.txt | while read line; do
    if [[ ${line} != "" ]] && [[ ${line} != *-* ]] && [[ ${line} != U* ]] && [[ ${line} != r* ]]; then
	if grep -q ${line} full_run.js; then
	    a=1
	else
	    echo ${line} >> runsToUpdate.txt
	fi
    fi
done
rm runsToUpdate_temp.txt

./StartTime.py

./MergeFiles.py
mv full_run.js full_run_old.js
cp full_run_updated.js full_run.js
python DCSLastRun.py

cp full_run.js /afs/cern.ch/cms/tracker/sistrcalib/WWW/DCSTrend
cp oneMonth_run.js /afs/cern.ch/cms/tracker/sistrcalib/WWW/DCSTrend
