#!/bin/sh

source /nfshome0/cmssw2/scripts/setup.sh
eval `scramv1 run -sh`
export TNS_ADMIN=/nfshome0/xiezhen/conddb

sflag=0
nflag=0
while getopts 'snh' OPTION
do
  case $OPTION in
      s) sflag=1
	  ;;
      n) nflag=1
	  ;;
      h) echo "Usage: [-s] [-n]"
          echo "  -s: write to sqlite file instead of ORCON"
          echo "  -n: do not run init.py"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

if [ ${sflag} -ne 0 ]
    then
    echo "Setting up sqlite_file:l1config.db"

    if [ ${nflag} -eq 0 ]
	then
        # setup schema
	./bootstrap.com
    fi
fi

if [ ${nflag} -eq 0 ]
    then    
    if [ ! -d gen ]
	then
	mkdir gen
    fi

    # write empty L1TriggerKeyList
    cmsRun init.py
fi

exit
