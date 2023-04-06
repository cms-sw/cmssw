#!/bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "===== Test MF map regressions  ===="

#Note: Only the most relevant tests are left uncommented since the full set is highly redundant and time consuming.

#Run II, 3.8T
(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=static_DDD)      || die "MF regression failure, from xml, DDD, 3.8T" $?
(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=fromDB)          || die "MF regression failure, from DB, DDD, 3.8T" $?
(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=static_DD4hep)          || die "MF regression failure, from xml, DD4hep, 3.8T" $?
(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=fromDB_DD4hep)   || die "MF regression failure, from DB, DD4hep, 3.8T" $?

#Run I, 3.8T
#(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=static_DDD era=RunI)      || die "MF regression failure, from xml, DDD, 3.8T RunI" $?
(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=fromDB era=RunI)          || die "MF regression failure, from DB, DDD, 3.8T RunI" $?
#(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=static_DD4hep era=RunI)          || die "MF regression failure, from xml, DD4hep, 3.8T RunI" $?
#(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=fromDB_DD4hep era=RunI)   || die "MF regression failure, from DB, DD4hep, 3.8T RunI" $?

#Run II, 3.5T
export CURRENT=16730
#(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=static_DDD current=$CURRENT)    || die "MF regression failure, from xml, DDD, 3.5T" $?
#(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=fromDB current=$CURRENT)        || die "MF regression failure, from DB, DDD, 3.5T" $?
(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=static_DD4hep current=$CURRENT)        || die "MF regression failure, from xml, DD4hep, 3.5T" $?
#(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=fromDB_DD4hep current=$CURRENT)  || die "MF regression failure, from DB, DD4hep, 3.5T" $?

#Run II, 3T
export CURRENT=14340
(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=static_DDD current=$CURRENT)    || die "MF regression failure, from xml, DDD, 3T" $?
#(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=fromDB current=$CURRENT)        || die "MF regression failure, from DB, DDD, 3T" $?
#(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=static_DD4hep current=$CURRENT)        || die "MF regression failure, from xml, DD4hep, 3T" $?
#(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=fromDB_DD4hep current=$CURRENT)  || die "MF regression failure, from DB, DD4hep, 3T" $?

#Run II, 2T
export CURRENT=9500
#(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=static_DDD current=$CURRENT)    || die "MF regression failure, from xml, DDD, 2T" $?
#(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=fromDB current=$CURRENT)        || die "MF regression failure, from DB, DDD, 2T" $?
(cmsRun ${SCRAM_TEST_PATH}/regression.py producerType=fromDB_DD4hep current=$CURRENT) || die "MF regression failure, from DB, DD4hep, 2T" $?
