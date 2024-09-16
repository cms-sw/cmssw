#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

XMLPATH=${SCRAM_TEST_PATH}/stubs
LIBFILE=libFWCoreReflectionTestObjects.so

edmCheckClassVersion -l ${LIBFILE} -x ${XMLPATH}/classes_def.xml || die "edmCheckClassVersion failed" $?

function runFailure {
    edmCheckClassVersion -l ${LIBFILE} -x ${XMLPATH}/$1 > log.txt && die "edmCheckClassVersion for $1 did not fail" 1
    grep -q "$2" log.txt
    RET=$?
    if [ "$RET" != "0" ]; then
        echo "edmCheckClassVersion for $1 did not contain '$2', log is below"
        cat log.txt
        exit 1
    fi
}

runFailure test_def_nameMissing.xml "There is an element 'class' without 'name' attribute"
runFailure test_def_ClassVersionMissingInClass.xml "Class element for type 'edmtest::reflection::IntObject' contains a 'version' element, but 'ClassVersion' attribute is missing from the 'class' element"
runFailure test_def_ClassVersionMissingInVersion.xml "Version element for type 'edmtest::reflection::IntObject' is missing 'ClassVersion' attribute"
runFailure test_def_checksumMissingInVersion.xml "Version element for type 'edmtest::reflection::IntObject' is missing 'checksum' attribute"
