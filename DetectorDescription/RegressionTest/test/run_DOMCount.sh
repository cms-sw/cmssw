#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

rm -f run_DOMCount.log
echo "Normal output of DOMCount is written to file tmp/${SCRAM_ARCH}/run_DOMCount.log"

# Each of these cfi files contains a list of xml files
# We will run DOMCount on each of the xml files
cfiFiles=Geometry/CMSCommonData/cmsIdealGeometryXML_cfi
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometryXML_cfi"
cfiFiles="${cfiFiles} Geometry/CMSCommonData/cmsExtendedGeometryLiMaxXML_cfi"

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
