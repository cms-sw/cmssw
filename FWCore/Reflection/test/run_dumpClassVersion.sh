#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

XMLPATH=${SCRAM_TEST_PATH}/stubs
LIBFILE=libFWCoreReflectionTestObjects.so

edmDumpClassVersion -l ${LIBFILE} -x ${XMLPATH}/classes_def.xml -o dump.json || die "edmDumpClassVersion failed" $?
diff -u ${SCRAM_TEST_PATH}/dumpClassVersion_reference.json dump.json || die "Unexpected class version dump" $?

function runFailure() {
    edmDumpClassVersion -l ${LIBFILE} -x ${XMLPATH}/$1 > log.txt && die "edmDumpClassVersion for $1 did not fail" 1
    grep -q "$2" log.txt
    RET=$?
    if [ "$RET" != "0" ]; then
        echo "edmDumpClassVersion for $1 did not contain '$2', log is below"
        cat log.txt
        exit 1
    fi
}

runFailure test_def_nameMissing.xml "There is an element 'class' without 'name' attribute"
runFailure test_def_ClassVersionMissingInClass.xml "Class element for type 'edmtest::reflection::IntObject' contains a 'version' element, but 'ClassVersion' attribute is missing from the 'class' element"
runFailure test_def_ClassVersionMissingInVersion.xml "Version element for type 'edmtest::reflection::IntObject' is missing 'ClassVersion' attribute"
runFailure test_def_checksumMissingInVersion.xml "Version element for type 'edmtest::reflection::IntObject' is missing 'checksum' attribute"
