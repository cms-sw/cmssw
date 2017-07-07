#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

rm -f run_DOMCount.log
echo "Normal output of DOMCount is written to file tmp/${SCRAM_ARCH}/run_DOMCount.log"

# Each of these cfi files contains a list of xml files
# We will run DOMCount on each of the xml files
cfiFiles=Geometry/CMSCommonData/cmsIdealGeometryXML_cfi
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsIdealGeometry2015XML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsIdealGeometry2015devXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsIdealGeometryAPD1XML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsIdealGeometryGFlashXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsIdealGeometryHFLibraryXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsIdealGeometryHFParametrizeXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsIdealGeometryNoAPDXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsIdealGeometryTotemT1XML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2015CastorMeasuredXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2015CastorSystMinusXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2015CastorSystPlusXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2015MuonGEMDevXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2015PilotXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2015XML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2015ZeroMaterialXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2015devCastorMeasuredXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2015devCastorSystMinusXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2015devCastorSystPlusXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2015devXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2016XML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2017XML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2019XML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2023D17XML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2023D10XML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometry2023D11XML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometryGFlashXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometryHFLibraryNoCastorXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometryHFLibraryXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometryHFParametrizeXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometryNoCastorXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometryTest2014XML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometryXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometryZeroMaterialXML_cfi"

for cfiFile in ${cfiFiles}
do
  echo "run_DOMCount.py $cfiFile" | tee -a run_DOMCount.log
  ${LOCAL_TEST_DIR}/run_DOMCount.py $cfiFile >> run_DOMCount.log 2>&1 || die "run_DOMCount.py $cfiFile" $?
done

# Errors in the xml files and also missing xml or schema files will
# show up in the log file. (if the python script above actually
# exits with a nonzero status it probably means the python test
# script has a bug in it)
errorCount=`(grep --count "Error" run_DOMCount.log)`

if [ $errorCount -eq 0 ]
then
    echo "No XML Schema violations in xml files."
else
    echo "Test failed. Here are the errors from tmp/${SCRAM_ARCH}/run_DOMCount.log:"
    cat run_DOMCount.log | grep -v '\.xml:.*elems\.$'
    popd
    exit 1
fi

popd
exit 0
