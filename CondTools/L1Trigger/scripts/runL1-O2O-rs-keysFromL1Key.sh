#!/bin/sh

xflag=0
oflag=0
pflag=0
gflag=0
fflag=0
while getopts 'xopgfh' OPTION
  do
  case $OPTION in
      x) xflag=1
          ;;
      o) oflag=1
          ;;
      p) pflag=1
          ;;
      g) gflag=1
	  ;;
      f) fflag=1
	  ;;
      h) echo "Usage: [-x] tsckey runnum"
          echo "  -x: write to ORCON instead of sqlite file"
	  echo "  -o: overwrite keys"
          echo "  -p: centrally installed release, not on local machine"
	  echo "  -g: GT RS records only"
	  echo "  -f: force IOV update"
          exit
	  ;;
  esac
done
shift $(($OPTIND - 1))

runnum=$1
l1Key=$2

if [ ${pflag} -eq 0 ]
    then
    export SCRAM_ARCH=""
    export VO_CMS_SW_DIR=""
    source /opt/cmssw/cmsset_default.sh
else
    source /nfshome0/cmssw2/scripts/setup.sh
fi
eval `scramv1 run -sh`
export TNS_ADMIN=/nfshome0/popcondev/conddb

if [ ${oflag} -eq 1 ]
    then
    overwrite="overwriteKeys=1"
fi

if [ ${gflag} -eq 1 ]
    then
    rsflag="-g"
else
    rsflag="-r"
fi

if [ ${fflag} -eq 1 ]
    then
    forceUpdate="forceUpdate=1"
fi

if [ ${xflag} -eq 0 ]
    then
    echo "Writing to sqlite_file:l1config.db instead of ORCON."
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSOnline_cfg.py runNumber=${runnum} outputDBConnect=sqlite_file:l1config.db outputDBAuth=. ${overwrite} ${forceUpdate} logTransactions=0 `$CMSSW_BASE/src/CondTools/L1Trigger/scripts/getKeys.sh ${rsflag} ${l1Key}` keysFromDB=0 print
    o2ocode=$?
    if [ ${o2ocode} -ne 0 ]
	then
	if [ ${o2ocode} -eq 66 ]
	    then
	    echo "L1-O2O-ERROR: unable to connect to OMDS or ORCON.  Check that /nfshome0/centraltspro/secure/authentication.xml is up to date (OMDS)."
	    echo "L1-O2O-ERROR: unable to connect to OMDS or ORCON.  Check that /nfshome0/centraltspro/secure/authentication.xml is up to date (OMDS)." 1>&2
	else
	    if [ ${o2ocode} -eq 65 ]
		then
		echo "L1-O2O-ERROR: problem writing object to ORCON."
		echo "L1-O2O-ERROR: problem writing object to ORCON." 1>&2
	    fi
	fi
    fi
else
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSOnline_cfg.py runNumber=${runnum} outputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T outputDBAuth=/nfshome0/popcondev/conddb_taskWriters/L1T ${overwrite} ${forceUpdate} `$CMSSW_BASE/src/CondTools/L1Trigger/scripts/getKeys.sh ${rsflag} ${l1Key}` keysFromDB=0 print
    o2ocode=$?
    if [ ${o2ocode} -ne 0 ]
	then
	if [ ${o2ocode} -eq 66 ]
	    then
	    echo "L1-O2O-ERROR: unable to connect to OMDS or ORCON.  Check that /nfshome0/centraltspro/secure/authentication.xml is up to date (OMDS)."
	    echo "L1-O2O-ERROR: unable to connect to OMDS or ORCON.  Check that /nfshome0/centraltspro/secure/authentication.xml is up to date (OMDS)." 1>&2
	else
	    if [ ${o2ocode} -eq 65 ]
		then
		echo "L1-O2O-ERROR: problem writing object to ORCON."
		echo "L1-O2O-ERROR: problem writing object to ORCON." 1>&2
	    fi
	fi
    fi
fi

exit ${o2ocode}
