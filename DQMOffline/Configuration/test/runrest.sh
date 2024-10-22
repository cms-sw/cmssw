#!/bin/bash -ex
ERR=0
PYTHONUNBUFFERED=1 cmsswSequenceInfo.py --runTheMatrix --steps DQM,VALIDATION --infile $1 --offset $2 --dbfile sequences$2.db --threads 1 >run.log 2>&1 || ERR=1
cat run.log
seqs=$(grep 'Analyzing [0-9][0-9]* seqs' run.log | sed 's|.*Analyzing *||;s| .*||')
echo "Sequences run by final DQMOfflineConfiguration: $seqs"
if [ "$seqs" -gt 0 ] ; then
  echo "Final DQMOfflineConfiguration should not run any sequences."
  echo "Please update parameters for TestDQMOfflineConfiguration unittest to run the extra sequences."
  exit 1
fi
exit $ERR
