#!/bin/sh

# Run dummy cmsRun job to prevent startup latency

pflag=0
while getopts 'ph' OPTION
  do
  case $OPTION in
      p) pflag=1
	  ;;
      h) echo "Usage: [-p]"
          echo "  -p: centrally installed release, not on local machine"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

release=CMSSW_4_2_3_ONLINE
version=012

echo "`date` : o2o-dummy.sh" >> /nfshome0/popcondev/L1Job/o2o-dummy-${version}.log
echo "`uptime`" >> /nfshome0/popcondev/L1Job/o2o-dummy-${version}.log
START=$(date +%s)

# set up environment variables
cd /cmsnfshome0/nfshome0/popcondev/L1Job/${release}/o2o

if [ ${pflag} -eq 0 ]
    then
    export SCRAM_ARCH=""
    export VO_CMS_SW_DIR=""
    source /opt/cmssw/cmsset_default.sh
else
    source /nfshome0/cmssw2/scripts/setup.sh
    centralRel="-p"
fi
eval `scramv1 run -sh`

# run script; args are run key
rm -f tmp.log
echo "`date`" >& tmp.log

$CMSSW_BASE/src/CondTools/L1Trigger/scripts/runL1-dummy.sh >> tmp.log 2>&1
o2ocode=$?

cat tmp.log >> /nfshome0/popcondev/L1Job/o2o-dummy-${version}.log
rm -f tmp.log

echo "dummy status ${o2ocode}" >> /nfshome0/popcondev/L1Job/o2o-dummy-${version}.log
echo "`date` : o2o-dummy.sh finished" >> /nfshome0/popcondev/L1Job/o2o-dummy-${version}.log

END=$(date +%s)
DIFF=$(( $END - $START ))
if [ ${DIFF} -gt 60 ]
    then
    echo "O2O SLOW: `date`, ${DIFF} seconds for ${run} ${l1Key}" >> /nfshome0/popcondev/L1Job/o2o-dummy-${version}.log
else
    echo "Time elapsed: ${DIFF} seconds" >> /nfshome0/popcondev/L1Job/o2o-dummy-${version}.log
fi
echo "" >> /nfshome0/popcondev/L1Job/o2o-dummy-${version}.log

tail -4 /nfshome0/popcondev/L1Job/o2o-dummy-${version}.log >> /nfshome0/popcondev/L1Job/o2o.summary

exit ${o2ocode}
