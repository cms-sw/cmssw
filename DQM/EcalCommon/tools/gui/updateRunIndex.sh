#!/bin/sh

ulimit -c 0

cd ${HOME}/DQM/dqm-gui

LD_LIBRARY_PATH=
VO_CMS_SW_DIR=${PWD}/rpms
SCRAM_ARCH=slc5_amd64_gcc434
source ${VO_CMS_SW_DIR}/cmsset_default.sh
source ${VO_CMS_SW_DIR}/${SCRAM_ARCH}/cms/dqmgui/5.4.0b/etc/profile.d/env.sh

if [ -e /tmp/updateRunIndex.lock ]; then
  echo "Lock file is present, exit"
  exit 1
fi

touch /tmp/updateRunIndex.lock

echo "Index update: begin"

./visDQMImport /data/ecalod-disk01/dqm-gui/idx /data/ecalod-disk01/dqm-data/root

echo "Index update: end"

rm /tmp/updateRunIndex.lock

exit 0

