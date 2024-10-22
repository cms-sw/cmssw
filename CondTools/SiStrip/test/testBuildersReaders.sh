#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

if [ "${SCRAM_TEST_NAME}" != "" ] ; then
  mkdir ${SCRAM_TEST_NAME}
  cd ${SCRAM_TEST_NAME}
fi
if test -f "SiStripConditionsDBFile.db"; then
    echo "cleaning the local test area"
    rm -fr SiStripConditionsDBFile.db  # builders test
    rm -fr modifiedSiStrip*.db         # miscalibrator tests
    rm -fr gainManipulations.db        # rescaler tool
fi
pwd
echo " testing CondTools/SiStrip"

# do the builders first (need the input db file)
for entry in "${SCRAM_TEST_PATH}/"SiStrip*Builder_cfg.py
do
  echo "===== Test \"cmsRun $entry \" ===="
  (cmsRun $entry) || die "Failure using cmsRun $entry" $?
done

echo -e " Done with the writers \n\n"

## do the readers
for entry in "${SCRAM_TEST_PATH}/"SiStrip*Reader_cfg.py
do
  echo "===== Test \"cmsRun $entry \" ===="
  (cmsRun $entry) || die "Failure using cmsRun $entry" $?
done

echo -e " Done with the readers \n\n"

sleep 5

## do the builders from file
for entry in "${SCRAM_TEST_PATH}/"SiStrip*FromASCIIFile_cfg.py
do
  echo "===== Test \"cmsRun $entry \" ===="
  (cmsRun $entry) || die "Failure using cmsRun $entry" $?
done

echo -e " Done with the builders from file \n\n"

## do the miscalibrators
for entry in "${SCRAM_TEST_PATH}/"SiStrip*Miscalibrator_cfg.py
do
  echo "===== Test \"cmsRun $entry \" ===="
  (cmsRun $entry) || die "Failure using cmsRun $entry" $?
done

echo -e " Done with the miscalibrators \n\n"

## do the scaler

# copy all the necessary conditions in order to run the miscalibration tool
G1TAG=SiStripApvGain_GR10_v1_hlt
G2TAG=SiStripApvGain_FromParticles_GR10_v1_express
OLDG1since=325642
NEWG1since=343828
OLDG2since=336067

echo -e "\n\n Copying IOVs $OLDG1since and $NEWG1since from $G1TAG and IOV $OLDG2since from $G2TAG"

conddb --yes --db pro copy ${G1TAG} G1_old --from ${OLDG1since} --to $((++OLDG1since)) --destdb gainManipulations.db
conddb --yes --db pro copy ${G1TAG} G1_new --from ${NEWG1since} --to $((++NEWG1since)) --destdb gainManipulations.db
conddb --yes --db pro copy ${G2TAG} G2_old --from ${OLDG2since} --to $((++OLDG2since)) --destdb gainManipulations.db

sqlite3 gainManipulations.db "update IOV SET SINCE=1 where SINCE=$((--OLDG1since))" # sets the IOV since to 1
sqlite3 gainManipulations.db "update IOV SET SINCE=1 where SINCE=$((--NEWG1since))" # sets the IOV since to 1
sqlite3 gainManipulations.db "update IOV SET SINCE=1 where SINCE=$((--OLDG2since))" # sets the IOV since to 1

echo -e "\n\n Checking the content of the local conditions file \n\n"
sqlite3 gainManipulations.db "select * from IOV"

(cmsRun ${SCRAM_TEST_PATH}/SiStripApvGainRescaler_cfg.py additionalConds=sqlite_file:${PWD}/gainManipulations.db) || die "Failure using cmsRun SiStripApvGainRescaler_cfg.py)" $?
echo -e " Done with the gain rescaler \n\n"

## do the visualization code

(cmsRun "${SCRAM_TEST_PATH}/"SiStripCondVisualizer_cfg.py) || die "Failure using cmsRun SiStripCondVisualizer_cfg.py" $?
(python3 "${SCRAM_TEST_PATH}/"db_tree_dump_wrapper.py) || die "Failure running python3 db_tree_dump_wrapper.py" $?
echo -e " Done with the visualization \n\n"

echo -e "\n\n Testing the conditions patching code: \n\n"
## do the bad components patching code
myFEDs=""
for i in {51..490..2}; do
    myFEDs+="${i},"
done
myFEDs+=491
echo " masking the following FEDs: "${myFEDs}

(cmsRun "${SCRAM_TEST_PATH}/"SiStripBadChannelPatcher_cfg.py isUnitTest=True FEDsToAdd=${myFEDs} outputTag=OddFEDs) || die "Failure using cmsRun SiStripBadChannelPatcher_cfg.py when adding FEDs" $?
(cmsRun "${SCRAM_TEST_PATH}/"SiStripBadChannelPatcher_cfg.py isUnitTest=True inputConnection=sqlite_file:outputDB.db inputTag=OddFEDs FEDsToRemove=${myFEDs} outputTag=OddFEDsRemoved) || die "Failure using cmsRun SiStripBadChannelPatcher_cfg.py when removing FEDs" $?

echo -e " Done with the bad components patching \n\n"
