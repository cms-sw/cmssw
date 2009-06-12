#!/bin/sh

ulimit -c 0

cd ~/DQM/dqm-GUI

LD_LIBRARY_PATH=
source rpms/cmsset_default.sh
source rpms/slc4_ia32_gcc345/cms/dqmgui/4.6.0/etc/profile.d/env.sh

if [ -e /tmp/updateNewRunDb.lock ]; then
  echo "Lock file is present, exit"
  exit 1
fi

touch /tmp/updateNewRunDb.lock

rm -f /data/ecalod-disk01/dqm-GUI/db/tmp-new.db*

echo "Moving dqm-new.db in tmp-new.db"

mv /data/ecalod-disk01/dqm-GUI/db/dqm-new.db /data/ecalod-disk01/dqm-GUI/db/tmp-new.db

echo "Updating tmp-new.db"

find /data/ecalod-disk01/dqm-data/root/ -mtime -240 -name 'DQM*.root' | sort | xargs -n 1 -r visDQMRegisterFile /data/ecalod-disk01/dqm-GUI/db/tmp-new.db "/Global/Online/ALL" "Global run"

find /data/ecalod-disk01/dqm-data/root/ -mtime -1 -name 'DQM*.root' | sort | xargs -n 1 -r visDQMRegisterFile /data/ecalod-disk01/dqm-GUI/db/tmp-new.db "/Global/Online/ALL" "Global run"

echo "Moving back tmp-new.db in dqm-new.db"

mv /data/ecalod-disk01/dqm-GUI/db/tmp-new.db /data/ecalod-disk01/dqm-GUI/db/dqm-new.db

rm /tmp/updateNewRunDb.lock

exit 0

