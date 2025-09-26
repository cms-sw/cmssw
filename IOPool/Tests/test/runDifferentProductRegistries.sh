#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

# Set the directory holding the configuration files
LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

# Run first write job (events starting at 1)
cmsRun ${LOCAL_TEST_DIR}/testDifferentProductRegistriesWrite_cfg.py --startEvent 1 --outputFile write1.root || die "cmsRun write1" $?

# Run second write job (events starting at 4)
cmsRun ${LOCAL_TEST_DIR}/testDifferentProductRegistriesWrite_cfg.py --startEvent 4 --outputFile write2.root || die "cmsRun write2" $?

# Run read job with both output files as input
cmsRun ${LOCAL_TEST_DIR}/testDifferentProductRegistriesRead_cfg.py --inputFiles file:write1.root file:write2.root || die "cmsRun read 1 then 2" $?

cmsRun ${LOCAL_TEST_DIR}/testDifferentProductRegistriesRead_cfg.py --inputFiles file:write2.root file:write1.root || die "cmsRun read 2 then 1" $?
