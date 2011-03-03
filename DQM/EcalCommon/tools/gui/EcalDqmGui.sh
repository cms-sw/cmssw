#!/bin/sh

if [ ${HOSTNAME} != "srv-S2F19-29" ]; then
  echo "This is not ecalod-disk01 !!!"
  exit 1
fi

ulimit -c 0

cd ${HOME}/DQM/dqm-gui

LD_LIBRARY_PATH=
VO_CMS_SW_DIR=${PWD}/rpms
SCRAM_ARCH=slc5_amd64_gcc434
source ${VO_CMS_SW_DIR}/cmsset_default.sh
source ${VO_CMS_SW_DIR}/${SCRAM_ARCH}/cms/dqmgui/5.4.0b/etc/profile.d/env.sh

rm -f gui/*/blacklist.txt

visDQMControl $1 all from config/server-conf-ecal.py

case "$1" in
    start)
        DQMCollector --listen 9090 > collector/collector.out 2>&1 </dev/null &
        ;;
    stop)
        killall -9 DQMCollector
        ;;
    restart)
        killall -9 DQMCollector
        DQMCollector --listen 9090 > collector/collector.out 2>&1 </dev/null &
        ;;
esac

exit 0
