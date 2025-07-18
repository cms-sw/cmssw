#!/bin/env bash

function die { echo $1: status $2 ;  exit $2; }

echo cmsRun test_shallowClustersProducer_cfg.py
cmsRun ${SCRAM_TEST_PATH}/test_shallowClustersProducer_cfg.py || die "Failure using test_shallowClustersProducer_cfg.py" $?

echo cmsRun test_shallowDigisProducer_cfg.py
cmsRun ${SCRAM_TEST_PATH}/test_shallowDigisProducer_cfg.py || die "Failure using test_shallowDigisProducer_cfg.py" $?

echo cmsRun test_shallowEventDataProducer_cfg.py
cmsRun ${SCRAM_TEST_PATH}/test_shallowEventDataProducer_cfg.py || die "Failure using test_shallowEventDataProducer_cfg.py" $?

echo cmsRun test_shallowGainCalibration_cfg.py
cmsRun ${SCRAM_TEST_PATH}/test_shallowGainCalibration_cfg.py || die "Failure using test_shallowGainCalibration_cfg.py" $?

echo cmsRun test_shallowRechitClustersProducer_cfg.py
cmsRun ${SCRAM_TEST_PATH}/test_shallowRechitClustersProducer_cfg.py || die "Failure using test_shallowRechitClustersProducer_cfg.py" $?

#echo cmsRun test_shallowSimTracksProducer_cfg.py #fails due to missing product
#cmsRun ${SCRAM_TEST_PATH}/test_shallowSimTracksProducer_cfg.py || die "Failure using test_shallowSimTracksProducer_cfg.py" $?
## Looking for type: reco::TrackToTrackingParticleAssociator
## Looking for module label: trackAssociatorByHits

#echo cmsRun test_shallowSimhitClustersProducer_cfg.py #fails due to missing product
#cmsRun ${SCRAM_TEST_PATH}/test_shallowSimhitClustersProducer_cfg.py || die "Failure using test_shallowSimhitClustersProducer_cfg.py" $?
## Looking for type: std::vector<PSimHit>
## Looking for module label: g4SimHits

echo cmsRun test_shallowTrackClustersProducer_cfg.py
cmsRun ${SCRAM_TEST_PATH}/test_shallowTrackClustersProducer_cfg.py || die "Failure using test_shallowTrackClustersProducer_cfg.py" $?

echo cmsRun test_shallowTracksProducer_cfg.py
cmsRun ${SCRAM_TEST_PATH}/test_shallowTracksProducer_cfg.py || die "Failure using test_shallowTracksProducer_cfg.py" $?
