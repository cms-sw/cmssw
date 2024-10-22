#! /bin/bash

[ "${LOCALTOP}" ] || LOCALTOP=$CMSSW_BASE

cmsRun ${LOCALTOP}/src/FWCore/Modules/test/testGenericConsumer.py 2>&1 | grep '^TrigReport' | \
  awk 'BEGIN { KEEP = 0; } /Module Summary/ { KEEP = 1; } { if (! KEEP) next; print; } /\<thing\>|\<otherThing\>|\<anotherThing\>/ { if ($3 == 0) exit 1; } /\<notRunningThing\>/ { if ($3 != 0) exit 1; }'
