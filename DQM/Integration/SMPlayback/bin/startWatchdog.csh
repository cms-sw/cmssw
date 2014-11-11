#!/usr/bin/env csh

set TIMESTAMP = `date '+%Y%m%d%H%M%S'`
(nohup source ./watchdog.csh) >& ../log/watchdog/dog${TIMESTAMP}.log &
