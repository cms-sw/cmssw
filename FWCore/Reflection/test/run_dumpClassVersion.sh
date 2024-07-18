#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

XMLPATH=${SCRAM_TEST_PATH}/stubs/
LIBFILE=${LOCALTOP}/lib/${SCRAM_ARCH}/libFWCoreReflectionTestObjects.so

edmDumpClassVersion -l ${LIBFILE} -x ${XMLPATH}/classes_def.xml -o dump.json || die "edmDumpClassVersion failed" $?
diff -u ${SCRAM_TEST_PATH}/dumpClassVersion_reference.json dump.json || die "Unexpected class version dump" $?
