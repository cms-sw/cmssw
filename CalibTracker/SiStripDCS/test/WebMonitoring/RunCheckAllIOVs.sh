#!/bin/bash

# Everything in the output files is in local time. As long as the scripts
# are run on a CERN machine it is fine.

lastIOV=`cat lastIOV.txt`

echo Starting from $lastIOV

echo "Running: ./CheckAllIOVs.py oracle://cms_orcoff_prod/CMS_COND_31X_STRIP SiStripDetVOff_v1_prompt "${lastIOV}
./CheckAllIOVs.py oracle://cms_orcoff_prod/CMS_COND_31X_STRIP SiStripDetVOff_v1_prompt "${lastIOV}"

# Cleanup of unneded output
rm DetVOffPrint*
rm DetVOffReaderDebug__FROM_*

# Produce the json file for the full database content
echo "Producing the json file for the full database content"
python DCSTrend.py
cp full.js /afs/cern.ch/cms/tracker/sistrcalib/WWW/DCSTrend

# Produce the json file for the last month
echo "Producing the json file for the last month"
python DCSLastTrend.py
cp oneMonth.js /afs/cern.ch/cms/tracker/sistrcalib/WWW/DCSTrend
