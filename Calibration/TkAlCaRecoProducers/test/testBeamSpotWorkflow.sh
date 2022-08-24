#! /bin/bash
function die { echo $1: status $2 ; exit $2; }
if [ "${SCRAM_TEST_NAME}" != "" ] ; then
  mkdir ${SCRAM_TEST_NAME}
  cd ${SCRAM_TEST_NAME}
fi

echo "TESTING  AlcaBeamSpotProducer ..."
cmsRun ${LOCAL_TEST_DIR}/Alca_BeamFit_Workflow.py || die "Failure running Alca_BeamFit_Workflow.py" $?

echo "TESTING AlcaBeamSpotHarvester ..."
cmsRun ${LOCAL_TEST_DIR}/Alca_BeamSpot_Harvester.py || die "Failure running Alca_BeamSpot_Harvester.py" $?
