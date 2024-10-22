#!/bin/bash -ex
cd $CMSSW_BASE
rm -rf test-data-file-deletion
mkdir -p test-data-file-deletion/src/Geometry/CMSCommonData/data/dir-for-data-file
#Create symlink for datafile to point to the directory. cmsRun should fail to open this datafile via FileInPath mechanism
ln -s dir-for-data-file test-data-file-deletion/src/Geometry/CMSCommonData/data/materials.xml
ERR=0
pushd test-data-file-deletion
  CMSSW_SEARCH_PATH=$(pwd)/src:${CMSSW_SEARCH_PATH} cmsRun $CMSSW_BASE/src/CondTools/Geometry/test/writehelpers/geometryxmlwriter.py || ERR=$?
popd
edm_exception_header="${CMSSW_BASE}/src/FWCore/Utilities/interface/EDMException.h"
if [ ! -f ${edm_exception_header} ] ; then
  edm_exception_header=$CMSSW_RELEASE_BASE/src/FWCore/Utilities/interface/EDMException.h
fi
let FILE_IN_PATH_ERROR=$(grep ' FileInPathError *= *' ${edm_exception_header} | sed 's|.*= *||;s|,.*$||')%256
rm -rf test-data-file-deletion
if [ "$ERR" = "$FILE_IN_PATH_ERROR" ] ; then
  exit 0
fi

echo "ERROR: This tests should have failed with FileInPath error."
exit 1
