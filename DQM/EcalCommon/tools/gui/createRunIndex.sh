#!/bin/sh

ulimit -c 0

cd ${HOME}/DQM/dqm-GUI

LD_LIBRARY_PATH=
source sw/cmsset_default.sh
source sw/slc4_ia32_gcc345/cms/dqmgui/5.0.2/etc/profile.d/env.sh

if [ -e /tmp/createRunIndex.lock ]; then
  echo "Lock file is present, exit"
  exit 1
fi

touch /tmp/createRunIndex.lock

rm -fr /data/ecalod-disk01/dqm-GUI/idx

visDQMIndex create /data/ecalod-disk01/dqm-GUI/idx

echo "Index create: begin"

./visDQMImport /data/ecalod-disk01/dqm-GUI/idx /data/ecalod-disk01/dqm-data/root

echo "Index create: end"

rm /tmp/createRunIndex.lock

exit 0

