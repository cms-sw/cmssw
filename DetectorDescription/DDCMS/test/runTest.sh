#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo " testing DetectorDescription/DDCMS"

export tmpdir=${PWD}
echo "===== Test \"python UnitsCheck.py testUnits.xml\" ===="
python ${SCRAM_TEST_PATH}/python/UnitsCheck.py ${SCRAM_TEST_PATH}/data/testUnits.xml

for entry in "${SCRAM_TEST_PATH}/python"/test*
do
  # Skip local DB test
  if ! expr $entry : '.*TGeoIteratorLocalDB.*' > /dev/null ; then
    echo "===== Test \"cmsRun $entry \" ===="
    (cmsRun $entry) || die "Failure using cmsRun $entry" $?
  fi
done

