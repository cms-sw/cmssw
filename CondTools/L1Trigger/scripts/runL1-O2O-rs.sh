
xflag=0
gflag=0

CMS_OPTIONS=""

while getopts 'oxfgh' OPTION
  do
  case $OPTION in
      o) CMS_OPTIONS=$CMS_OPTIONS" overwriteKeys=1"
          ;;
      x) xflag=1
          ;;
      f) CMS_OPTIONS=$CMS_OPTIONS" forceUpdate=1"
	  ;;
      h) echo "Usage: [-x] runnum l1key"
	  echo "  -o: overwrite keys"
          echo "  -x: write to ORCON instead of sqlite file"
	  echo "  -f: force IOV update"
          exit
	  ;;
  esac
done
shift $(($OPTIND - 1))

runnum=$1
l1Key=$2


echo "INFO: ADDITIONAL CMS OPTIONS:  " $CMS_OPTIONS

if [ ${xflag} -eq 0 ]
then
    echo "Writing to sqlite_file:l1config.db instead of ORCON."
    DB_OPTIONS="outputDBConnect=sqlite_file:l1config.db outputDBAuth=." 
else
    echo "Cowardly refusing to write to the online database"
    DB_OPTIONS="outputDBConnect=oracle://cms_orcoff_prep/CMS_CONDITIONS outputDBAuth=." 
    # WHEN READY FOR PRIME TIME:
    # DB_OPTIONS= "outputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T outputDBAuth=."
fi


cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSOnline_cfg.py runNumber=${runnum} ${DB_OPTIONS} ${CMS_OPTIONS} logTransactions=0 print
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
else
    echo "runL1-o2o-rs-keysFromL1Key.sh ran successfully."
fi
exit ${o2ocode}

