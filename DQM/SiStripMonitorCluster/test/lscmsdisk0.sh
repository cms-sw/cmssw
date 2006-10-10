#!/bin/sh
RUNNR=$1
CMSDISK0_MTCC_DIR="/data0/mtcc_test/";

case $# in
0)
        ssh `whoami`@cmsdisk0.cern.ch "ls ${CMSDISK0_MTCC_DIR}";
	;;
1)
        RUNNR=$1;
        ssh `whoami`@cmsdisk0.cern.ch "ls ${CMSDISK0_MTCC_DIR}" | grep ${RUNNR} | grep '\.root'
        ;;
*)
        ;;
esac

