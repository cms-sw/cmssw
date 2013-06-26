#!/bin/sh
RUNNR=$1
CMSDISK0_MTCC_DIR="/data0/mtcc_0_9_0";
echo $CMSDISK0_MTCC_DIR

case $# in
0)
        ssh `whoami`@cmsdisk0.cern.ch "ls -lh ${CMSDISK0_MTCC_DIR}";
	;;
1)
        RUNNR=$1;
        ssh `whoami`@cmsdisk0.cern.ch "ls -lh ${CMSDISK0_MTCC_DIR}" | grep ${RUNNR} | grep '\.dat'
        ;;
*)
        ;;
esac

