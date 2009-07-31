#!/bin/sh

ulimit -c 0

cd ~/work/cms/dqm-GUI

LD_LIBRARY_PATH=
source sw/cmsset_default.sh
source sw/slc4_ia32_gcc345/cms/dqmgui/5.0.0/etc/profile.d/env.sh

if [ -e /tmp/updateRunDb.lock ]; then
  echo "Lock file is present, exit"
  exit 1
fi

touch /tmp/updateRunDb.lock

[ -d ~/work/cms/dqm-GUI/idx ] || visDQMIndex create ~/work/cms/dqm-GUI/idx

echo "Index update: begin"

./visDQMImport ~/work/cms/dqm-GUI/idx ~/work/cms/CMSSW_3_2_0/src/DQM/Ecal*MonitorModule/test/python

echo "Index update: end"

rm /tmp/updateRunDb.lock

exit 0

