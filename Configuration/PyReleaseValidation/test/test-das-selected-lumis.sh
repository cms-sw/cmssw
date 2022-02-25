#!/bin/bash -ex
export CMS_BOT_USE_DASGOCLIENT=true
QUERY="lumi,file dataset=/HIHardProbes/HIRun2018A-v1/RAW run=326479"
dasgoclient --limit 0 --query "${QUERY}" --format json | das-selected-lumis.py 1,23 | sort > original.txt
export CMS_BOT_USE_DASGOCLIENT=false
dasgoclient --limit 0 --query "${QUERY}" --format json | das-selected-lumis.py 1,23 | sort > cached.txt
diff -u original.txt cached.txt

