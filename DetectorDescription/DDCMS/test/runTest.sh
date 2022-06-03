#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo " testing DetectorDescription/DDCMS"

export tmpdir=${LOCAL_TMP_DIR:-/tmp}
echo "===== Test \"python UnitsCheck.py testUnits.xml\" ===="
python ${LOCAL_TEST_DIR}/python/UnitsCheck.py ${LOCAL_TEST_DIR}/data/testUnits.xml

for entry in "${LOCAL_TEST_DIR}/python"/test*
do
  # Skip local DB test
  if ! expr $entry : '.*TGeoIteratorLocalDB.*' > /dev/null ; then
    echo "===== Test \"cmsRun $entry \" ===="
    (cmsRun $entry) || die "Failure using cmsRun $entry" $?
  fi
done

