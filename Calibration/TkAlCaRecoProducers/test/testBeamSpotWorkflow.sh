#! /bin/bash
function die { echo $1: status $2 ; exit $2; }
if [ "${SCRAM_TEST_NAME}" != "" ] ; then
  mkdir ${SCRAM_TEST_NAME}
  cd ${SCRAM_TEST_NAME}
fi

echo "TESTING  AlcaBeamSpotProducer ..."
cmsRun ${SCRAM_TEST_PATH}/Alca_BeamFit_Workflow.py || die "Failure running Alca_BeamFit_Workflow.py" $?

echo "TESTING AlcaBeamSpotHarvester ..."
cmsRun ${SCRAM_TEST_PATH}/Alca_BeamSpot_Harvester.py || die "Failure running Alca_BeamSpot_Harvester.py" $?
