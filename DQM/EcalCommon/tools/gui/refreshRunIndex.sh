#!/bin/sh

if [ $# -lt 1 ]; then
  echo 'Usage: updateRunIndex.sh NUMBER-OF-FILES'
  exit
fi

N=$1

ulimit -c 0

cd ${HOME}/work/cms/dqm-GUI

LD_LIBRARY_PATH=
source sw/cmsset_default.sh
source sw/slc4_ia32_gcc345/cms/dqmgui/5.0.0/etc/profile.d/env.sh

if [ -e /tmp/updateRunIndex.lock ]; then
  echo "Lock file is present, exit"
  exit 1
fi

touch /tmp/updateRunIndex.lock

echo "Index refresh: begin"

#find /data/ecalod-disk01/dqm-data/root/ -name 'DQM_V*.root' | xargs -r ls -tr | tail -$N | xargs -n 1 -r visDQMIndex -d add --dataset /Global/Online/ALL /data/ecalod-disk01/dqm-GUI/idx

find ${HOME}/work/cms/CMSSW_3_2_0 -name 'DQM_V*.root' | xargs -r ls -tr | tail -$N | xargs -n 1 -r visDQMIndex -d add --dataset /Global/Online/ALL ${HOME}/work/cms/dqm-GUI/idx

echo "Index refresh: end"

rm /tmp/updateRunIndex.lock

exit 0

