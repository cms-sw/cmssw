#!/bin/bash -e

function die { echo $1: status $2 ;  exit $2; }

SCRAM_TEST_NAME=TestIOPoolInputNoParentDictionary
rm -rf $SCRAM_TEST_NAME
mkdir $SCRAM_TEST_NAME
cd $SCRAM_TEST_NAME

# Create a new CMSSW dev area and build modified DataFormats/TestObjects in it
NEW_CMSSW_BASE=$(/bin/pwd -P)/$CMSSW_VERSION
scram -a $SCRAM_ARCH project $CMSSW_VERSION
pushd $CMSSW_VERSION/src

# Copy DataFormats/TestObjects code to be able to edit it to make ROOT header parsing to fail
for DIR in ${CMSSW_BASE} ${CMSSW_RELEASE_BASE} ${CMSSW_FULL_RELEASE_BASE} ; do
    if [ -d ${DIR}/src/DataFormats/TestObjects ]; then
        mkdir DataFormats
        cp -Lr ${DIR}/src/DataFormats/TestObjects DataFormats/
        break
    fi
done
if [ ! -e DataFormats/TestObjects ]; then
    echo "Failed to copy DataFormats/TestObjects from local or release area"
    exit 1;
fi

# Enable TransientIntParentT dictionaries
cat DataFormats/TestObjects/test/BuildFile_extra.xml >> DataFormats/TestObjects/test/BuildFile.xml
#Set env and build in sub-shel
(eval $(scram run -sh) ; scram build -j $(nproc))

popd

# Prepend NEW_CMSSW_BASE's lib/src paths in to LD_LIBRARY_PATH and ROOT_INCLUDE_PATH
export LD_LIBRARY_PATH=${NEW_CMSSW_BASE}/lib/${SCRAM_ARCH}:${LD_LIBRARY_PATH}
export ROOT_INCLUDE_PATH=${NEW_CMSSW_BASE}/src:${ROOT_INCLUDE_PATH}

echo "Produce a file with TransientIntParentT<1> product"
cmsRun ${LOCALTOP}/src/IOPool/Input/test/PoolNoParentDictionaryTestStep1_cfg.py || die 'Failed cmsRun PoolNoParentDictionaryTestStep1_cfg.py' $?


# Then make attempt to load TransientIntParentT<1> to fail
echo "PREVENT_HEADER_PARSING" >> ${NEW_CMSSW_BASE}/src/DataFormats/TestObjects/interface/ToyProducts.h
rm ${NEW_CMSSW_BASE}/lib/${SCRAM_ARCH}/DataFormatsTestObjectsParent1_xr_rdict.pcm ${NEW_CMSSW_BASE}/lib/${SCRAM_ARCH}/libDataFormatsTestObjectsParent1.so
sed -i -e 's/libDataFormatsTestObjectsParent1.so/libDataFormatsTestObjectsParent2.so/' ${NEW_CMSSW_BASE}/lib/${SCRAM_ARCH}/DataFormatsTestObjectsParent1_xr.rootmap

echo "Read the file without TransientIntParentT<1> dictionary"
cmsRun ${LOCALTOP}/src/IOPool/Input/test/PoolNoParentDictionaryTestStep2_cfg.py || die 'Failed cmsRun PoolNoParentDictionaryTestStep2_cfg.py' $?
