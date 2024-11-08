#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo " testing CalibTracker/StandalonTrackerTopology"

cmsRun ${SCRAM_TEST_PATH}/testStandaloneTrackerTopology_cfg.py || die "Failure using cmsRun testPixelTopologyMapTest_cfg.py (Phase-0 test)" $?
cmsRun ${SCRAM_TEST_PATH}/testStandaloneTrackerTopology_cfg.py runNumber=300000 || die "Failure using cmsRun testPixelTopologyMapTest_cfg.py (Phase-1 test)" $?
cmsRun ${SCRAM_TEST_PATH}/testStandaloneTrackerTopology_cfg.py isPhase2=True || die "Failure using cmsRun testPixelTopologyMapTest_cfg.py (Phase-2 test)" $?
