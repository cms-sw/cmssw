#!/bin/sh

ulimit -c 0

cd ${HOME}/DQM/dqm-gui

. $PWD/current/apps/dqmgui/etc/profile.d/env.sh

if [ -e /tmp/createRunIndex.lock ]; then
  echo "Lock file is present, exit"
  exit 1
fi

touch /tmp/createRunIndex.lock

mv /data/ecalod-disk01/dqm-gui/state/dqmgui/ecal/ix /data/ecalod-disk01/dqm-gui/state/dqmgui/ecal/ix.old 

echo "Index create: begin"

visDQMIndex create /data/ecalod-disk01/dqm-gui/state/dqmgui/ecal/ix

echo "Index create: end"
echo "Old index directory moved to /data/ecalod-disk01/dqm-gui/state/dqmgui/ecal/ix.old"

rm /tmp/createRunIndex.lock

exit 0

