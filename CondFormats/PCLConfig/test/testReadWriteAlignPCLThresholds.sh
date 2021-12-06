#!/bin/bash
function die { echo $1: status $2; exit $2; }
cmsRun ${LOCAL_TEST_DIR}/AlignPCLThresholdsWriter_cfg.py || die 'failed running AlignPCLThresholdsWriter_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/AlignPCLThresholdsReader_cfg.py || die 'failed running AlignPCLThresholdsReader_cfg.py' $?
