#!/bin/sh

ulimit -c 0

cd ~/DQM/dqm-GUI

LD_LIBRARY_PATH=
source rpms/cmsset_default.sh
source rpms/slc4_ia32_gcc345/cms/dqmgui/4.6.0/etc/profile.d/env.sh

if [ -e /tmp/createNewRunDb.lock ]; then
  echo "Lock file is present, exit"
  exit 1;
fi

touch /tmp/createNewRunDb.lock

rm -f /data/ecalod-disk01/dqm-GUI/db/tmp-new.db*

find /data/ecalod-disk01/dqm-data/root/ -mtime -150 -name 'DQM*.root' | sort | xargs -n 1 -r visDQMRegisterFile /data/ecalod-disk01/dqm-GUI/db/tmp-new.db "/Global/Online/ALL" "Global run"

mv /data/ecalod-disk01/dqm-GUI/db/tmp-new.db /data/ecalod-disk01/dqm-GUI/db/dqm-new.db

rm /tmp/createNewRunDb.lock

exit 0

