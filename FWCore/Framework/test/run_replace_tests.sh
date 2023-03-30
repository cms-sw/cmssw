#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR="${CMSSW_BASE}/src/FWCore/Framework/test"
(cmsRun ${LOCAL_TEST_DIR}/test_replace_with_unnamed_esproducer_cfg.py ) || die 'Failure using test_replace_with_unnamed_esproducer_cfg.py' $?
(cmsRun ${LOCAL_TEST_DIR}/test_replace_with_unnamed_essource_cfg.py ) || die 'Failure using test_replace_with_unnamed_essource_cfg.py' $?
