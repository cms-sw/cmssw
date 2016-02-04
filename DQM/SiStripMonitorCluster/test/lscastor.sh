#!/bin/sh
RUNNR=$1
CASTOR_MTCC_DIR="/castor/cern.ch/cms/MTCC/data/"
echo $CASTOR_MTCC_DIR

case $# in
0)
	rfdir $CASTOR_MTCC_DIR
	;;
1)
        RUNNR=$1;
        rfdir ${CASTOR_MTCC_DIR}/0000${RUNNR}/A
        ;;
*)
        ;;
esac

