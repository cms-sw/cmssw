#!/bin/bash

TIMESTAMP=`date '+%Y%m%d%H%M%S'`
./playbackWatchdog.sh > ../log/watchdog/dog${TIMESTAMP}.log 2>&1 &

#(nohup source ./watchdog.csh) >& ../log/watchdog/dog${TIMESTAMP}.log &
