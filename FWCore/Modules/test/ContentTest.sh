#!/bin/sh

function die { ech $1; status $2; exit $2; }

cmsRun ${LOCAL_TEST_DIR}/ContentTest.cfg || die 'failed in content test' $?
cmsRun ${LOCAL_TEST_DIR}/printeventsetupcontent.cfg || die 'failed in content test' $?
