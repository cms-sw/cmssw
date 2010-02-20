#!/bin/sh

if [ ${HOSTNAME} != "vmepcS2F19-25" ]; then
  echo "This is not vmepcS2F19-25 !!!"
  exit 1
fi

ulimit -c 0

cd ${HOME}/DQM/dqm-GUI

LD_LIBRARY_PATH=
VO_CMS_SW_DIR=${PWD}/rpms
SCRAM_ARCH=slc5_amd64_gcc434
source ${VO_CMS_SW_DIR}/cmsset_default.sh
source ${VO_CMS_SW_DIR}/${SCRAM_ARCH}/cms/dqmgui/5.1.7b/etc/profile.d/env.sh

rm -f gui/*/blacklist.txt

visDQMControl $1 all from config/server-conf-ecal.py

exit 0
