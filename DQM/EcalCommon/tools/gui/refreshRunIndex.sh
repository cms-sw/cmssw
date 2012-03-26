#!/bin/sh

if [ $# -lt 1 ]; then
  echo 'Usage: refreshRunIndex.sh NUMBER-OF-FILES'
  exit
fi

N=$1

ulimit -c 0

cd ${HOME}/DQM/dqm-gui

. $PWD/current/apps/dqmgui/etc/profile.d/env.sh

if [ -e /tmp/updateRunIndex.lock ]; then
  echo "Lock file is present, exit"
  exit 1
fi

touch /tmp/updateRunIndex.lock

echo "Index refresh: begin"

FILES=`find /data/ecalod-disk01/dqm-data/root/ -name 'DQM_V*.root' -mtime -1 | xargs -r ls -tr | tail -$N`

for F in $FILES; do
  echo "Add: "$F
  visDQMIndex add /data/ecalod-disk01/dqm-gui/state/dqmgui/ecal/ix $F
done

echo "Index refresh: end"

rm /tmp/updateRunIndex.lock

exit 0

