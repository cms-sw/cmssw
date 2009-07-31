#!/bin/sh

if [ ${HOSTNAME} != "vmepcS2F19-25" ]; then
  echo "This is not vmepcS2F19-25 !!!"
  exit 1
fi

ulimit -c 0

cd ${HOME}/DQM/dqm-GUI

LD_LIBRARY_PATH=
source sw/cmsset_default.sh
source sw/slc4_ia32_gcc345/cms/dqmgui/5.0.2/etc/profile.d/env.sh

rm -f gui/*/blacklist.txt

visDQMControl $1 all from config/server-conf-ecal.py

exit 0
