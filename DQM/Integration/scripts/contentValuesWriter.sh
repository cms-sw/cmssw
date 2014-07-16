#!/bin/bash

from=120000

for f in `bash -c "python contentValuesFiles.py --from=$from 2>contentValuesFiles.err"`; do

  python contentValuesToDBS.py --url http://pccmsdqm04.cern.ch:1099 -f "('.*','DQM OFFLINE','Summary')" $f -d

  shift='online'
  if [ "`echo $f | grep '/Online/'`" == "" ]; then
    shift='offline'
  fi
  python contentValuesToRR.py  --url http://pccmsdqm04.cern.ch/runregistry/xmlrpc -s $shift $f -d

done
