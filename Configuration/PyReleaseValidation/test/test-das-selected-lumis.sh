#!/bin/bash -ex
export CMS_BOT_USE_DASGOCLIENT=true
QUERY="lumi,file dataset=/HIHardProbes/HIRun2018A-v1/RAW run=326479"
dasgoclient --limit 0 --query "${QUERY}" --format json | das-selected-lumis.py 1,23 | grep '^/store/' > original.txt
cat original.txt
if [ $(cat original.txt | wc -l) -eq 0 ] ; then
  exit 1
fi
