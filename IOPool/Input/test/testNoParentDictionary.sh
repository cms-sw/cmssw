#!/bin/bash -e

function die { echo $1: status $2 ;  exit $2; }

SCRAM_TEST_NAME=TestIOPoolInputNoParentDictionary
rm -rf $SCRAM_TEST_NAME
mkdir $SCRAM_TEST_NAME
cd $SCRAM_TEST_NAME

# Create a new CMSSW dev area
OLD_CMSSW_BASE=${CMSSW_BASE}
LD_LIBRARY_PATH_TO_OLD=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep "^${CMSSW_BASE}/" | tr '\n' ':')
scram -a $SCRAM_ARCH project $CMSSW_VERSION
pushd $CMSSW_VERSION/src
eval `scram run -sh`

# Copy DataFormats/TestObjects code to be able to edit it to make ROOT header parsing to fail
for DIR in ${OLD_CMSSW_BASE} ${CMSSW_RELEASE_BASE} ${CMSSW_FULL_RELEASE_BASE} ; do
    if [ -d ${DIR}/src/DataFormats/TestObjects ]; then
        mkdir DataFormats
        cp -r ${DIR}/src/DataFormats/TestObjects DataFormats/
        break
    fi
done
if [ ! -e DataFormats/TestObjects ]; then
    echo "Failed to copy DataFormats/TestObjects from local or release area"
    exit 1;
fi

# Enable TransientIntParentT dictionaries
cat DataFormats/TestObjects/test/BuildFile_extra.xml >> DataFormats/TestObjects/test/BuildFile.xml
scram build -j $(nproc)

popd

# Add OLD_CMSSW_BASE in between CMSSW_BASE and CMSSW_RELEASE_BASE for
# LD_LIBRARY_PATH and ROOT_INCLUDE_PATH
export LD_LIBRARY_PATH=$(echo -n ${LD_LIBRARY_PATH} | sed -e "s|${CMSSW_BASE}/external/${SCRAM_ARCH}/lib:|${CMSSW_BASE}/external/${SCRAM_ARCH}/lib:${LD_LIBRARY_PATH_TO_OLD}:|")
export ROOT_INCLUDE_PATH=$(echo -n ${ROOT_INCLUDE_PATH} | sed -e "s|${CMSSW_BASE}/src:|${CMSSW_BASE}/src:${OLD_CMSSW_BASE}/src:|")

echo "Produce a file with TransientIntParentT<1> product"
cmsRun ${LOCALTOP}/src/IOPool/Input/test/PoolNoParentDictionaryTestStep1_cfg.py || die 'Failed cmsRun PoolNoParentDictionaryTestStep1_cfg.py' $?


# Then make attempt to load TransientIntParentT<1> to fail
echo "PREVENT_HEADER_PARSING" >> ${CMSSW_BASE}/src/DataFormats/TestObjects/interface/ToyProducts.h
rm ${CMSSW_BASE}/lib/${SCRAM_ARCH}/DataFormatsTestObjectsParent1_xr_rdict.pcm ${CMSSW_BASE}/lib/${SCRAM_ARCH}/libDataFormatsTestObjectsParent1.so
sed -i -e 's/libDataFormatsTestObjectsParent1.so/libDataFormatsTestObjectsParent2.so/' ${CMSSW_BASE}/lib/${SCRAM_ARCH}/DataFormatsTestObjectsParent1_xr.rootmap

echo "Read the file without TransientIntParentT<1> dictionary"
cmsRun ${LOCALTOP}/src/IOPool/Input/test/PoolNoParentDictionaryTestStep2_cfg.py || die 'Failed cmsRun PoolNoParentDictionaryTestStep2_cfg.py' $?
