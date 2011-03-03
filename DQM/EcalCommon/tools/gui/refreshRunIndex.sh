#!/bin/sh

if [ $# -lt 1 ]; then
  echo 'Usage: updateRunIndex.sh NUMBER-OF-FILES'
  exit
fi

N=$1

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

echo "Index refresh: begin"

FILES=`find /data/ecalod-disk01/dqm-data/root/ -name 'DQM_V*.root' -mtime -1 | xargs -r ls -tr | tail -$N`

for F in $FILES; do
  echo "Add: "$F
  visDQMIndex add --dataset /Global/Online/ALL /data/ecalod-disk01/dqm-gui/idx $F
done

echo "Index refresh: end"

rm /tmp/updateRunIndex.lock

exit 0

