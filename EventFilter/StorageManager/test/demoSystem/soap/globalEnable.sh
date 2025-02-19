#!/bin/sh

isPlayback=`echo $STMGR_CONFIG | grep -c 'sm_playback.xml'`
if [[ $isPlayback -eq 0 ]] ; then
    echo ""
    echo "========================================"
    echo "Setting run numbers..."
    echo "========================================"
    ./setRunNumbers.sh
    echo ""
fi

./sendCmdToAllApps.sh Enable
