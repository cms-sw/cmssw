#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

XMLPATH=${SCRAM_TEST_PATH}/stubs/
LIBFILE=${LOCALTOP}/lib/${SCRAM_ARCH}/libFWCoreReflectionTestObjects.so

edmCheckClassVersion -l ${LIBFILE} -x ${XMLPATH}/classes_def.xml || die "edmCheckClassVersion failed" $?
