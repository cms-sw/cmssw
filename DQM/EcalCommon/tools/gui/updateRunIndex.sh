#!/bin/sh

ulimit -c 0

cd ${HOME}/DQM/dqm-GUI

LD_LIBRARY_PATH=
source sw/cmsset_default.sh
source sw/slc4_ia32_gcc345/cms/dqmgui/5.0.1/etc/profile.d/env.sh

if [ -e /tmp/updateRunIndex.lock ]; then
  echo "Lock file is present, exit"
  exit 1
fi

touch /tmp/updateRunIndex.lock

[ -e /data/ecalod-disk01/dqm-GUI/idx/generation ] || visDQMIndex create /data/ecalod-disk01/dqm-GUI/idx

echo "Index update: begin"

./visDQMImport /data/ecalod-disk01/dqm-GUI/idx /data/ecalod-disk01/dqm-data/root

echo "Index update: end"

rm /tmp/updateRunIndex.lock

exit 0

