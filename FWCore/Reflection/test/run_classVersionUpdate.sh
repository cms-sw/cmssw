#!/bin/bash -e

SCRAM_TEST_NAME=TestFWCoreReflectionClassVersionUpdate
rm -rf $SCRAM_TEST_NAME
mkdir $SCRAM_TEST_NAME
cd $SCRAM_TEST_NAME

# Create a new CMSSW dev area and build modified DataFormats/TestObjects in it
NEW_CMSSW_BASE=$(/bin/pwd -P)/$CMSSW_VERSION
scram -a $SCRAM_ARCH project $CMSSW_VERSION
pushd $CMSSW_VERSION/src

# Copy FWCore/Reflection code to be able to edit it to make ROOT header parsing to fail
for DIR in ${CMSSW_BASE} ${CMSSW_RELEASE_BASE} ${CMSSW_FULL_RELEASE_BASE} ; do
    if [ -d ${DIR}/src/FWCore/Reflection ]; then
        mkdir FWCore
        cp -Lr ${DIR}/src/FWCore/Reflection FWCore/
        break
    fi
done
if [ ! -e FWCore/Reflection ]; then
    echo "Failed to symlink FWCore/Reflection from local or release area"
    exit 1;
fi

# The original src/ tree is protected from writes in PR tests
chmod -R u+w FWCore/Reflection/test/stubs

# Modify the IntObject class to trigger a new version
#
# Just setting USER_CXXFLAGS for scram is not sufficient,because
# somehow ROOT (as used by edmCheckClassVersion) still picks up the
# version 3 of the class
echo "#define FWCORE_REFLECTION_TEST_INTOBJECT_V4" | cat - FWCore/Reflection/test/stubs/TestObjects.h > TestObjects.h.tmp
mv TestObjects.h.tmp FWCore/Reflection/test/stubs/TestObjects.h


#Set env and build in sub-shel
(eval $(scram run -sh) ; SCRAM_NOEDM_CHECKS=yes scram build -j $(nproc))

popd

# Prepend NEW_CMSSW_BASE's lib/src paths in to LD_LIBRARY_PATH and ROOT_INCLUDE_PATH
export LD_LIBRARY_PATH=${NEW_CMSSW_BASE}/lib/${SCRAM_ARCH}:${LD_LIBRARY_PATH}
export ROOT_INCLUDE_PATH=${NEW_CMSSW_BASE}/src:${ROOT_INCLUDE_PATH}

# Make the actual test
echo "Initial setup complete, now for the actual test"
XMLPATH=${SCRAM_TEST_PATH}/stubs
LIBFILE=libFWCoreReflectionTestObjects.so

function die { echo Failure $1: status $2 ; exit $2 ; }
function runFailure {
    $1 -l ${LIBFILE} -x ${XMLPATH}/$2 > log.txt && die "$1 for $2 did not fail" 1
    grep -q "$3" log.txt
    RET=$?
    if [ "$RET" != "0" ]; then
        echo "$1 for $2 did not contain '$3', log is below"
        cat log.txt
        exit 1
    fi
}

echo "edmCheckClassVersion tests"

runFailure edmCheckClassVersion classes_def.xml "error: class 'edmtest::reflection::IntObject' has a different checksum for ClassVersion 3. Increment ClassVersion to 4 and assign it to checksum 2954816125"
runFailure edmCheckClassVersion test_def_v4.xml "error: for class 'edmtest::reflection::IntObject' ROOT says the ClassVersion is 3 but classes_def.xml says it is 4. Are you sure everything compiled correctly?"

edmCheckClassVersion -l ${LIBFILE} -x ${XMLPATH}/classes_def.xml -g || die "edmCheckClassVersion -g failed" $?
diff -u ${XMLPATH}/test_def_v4.xml classes_def.xml.generated || die "classes_def.xml.generated differs from expectation" $?


echo "edmDumpClassVersion tests"

edmDumpClassVersion -l ${LIBFILE} -x ${XMLPATH}/classes_def.xml -o dump.json
diff -u ${SCRAM_TEST_PATH}/dumpClassVersion_reference_afterUpdate.json dump.json || die "Unexpected class version dump" $?
