#!/bin/bash

pushd ${LOCAL_TMP_DIR}

cfiFiles=Geometry/CMSCommonData/cmsIdealGeometryXML_cfi
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometryXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2015XML_cfi"

status=0

for cfiFile in ${cfiFiles}
do
  echo "run_DDErrorReport.py $cfiFile"
  ${LOCAL_TEST_DIR}/run_DDErrorReport.py $cfiFile
  if [ $? -ne 0 ]
  then
    status=1
  fi
done

popd
exit $status
