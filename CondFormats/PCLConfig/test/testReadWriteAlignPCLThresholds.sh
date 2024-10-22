#!/bin/bash
function die { echo $1: status $2; exit $2; }
# test High Granularity
cmsRun ${SCRAM_TEST_PATH}/AlignPCLThresholdsWriter_cfg.py || die 'failed running AlignPCLThresholdsWriter_cfg.py' $?
cmsRun ${SCRAM_TEST_PATH}/AlignPCLThresholdsReader_cfg.py || die 'failed running AlignPCLThresholdsReader_cfg.py' $?

# test Low Granularity
(cmsRun ${SCRAM_TEST_PATH}/AlignPCLThresholdsWriter_cfg.py writeLGpayload=True) || die 'failed running AlignPCLThresholdsWriter_cfg.py writeLGpayload=True' $?
(cmsRun ${SCRAM_TEST_PATH}/AlignPCLThresholdsReader_cfg.py readLGpayload=True) || die 'failed running AlignPCLThresholdsReader_cfg.py readLGpayload=True' $?
