#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

(cmsRun ${LOCAL_TEST_DIR}/test_replace_with_unnamed_esproducer.cfg ) || die 'Failure using test_replace_with_unnamed_esproducer.cfg' $?
(cmsRun ${LOCAL_TEST_DIR}/test_replace_with_unnamed_essource.cfg ) || die 'Failure using test_replace_with_unnamed_essource.cfg' $?
