#!/bin/sh -ex

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

python_cmd="python3"

for file in ${CMSSW_BASE}/src/FWCore/PythonFramework/python/*.py
do
  bn=`basename $file`
  if [ "$bn" != "__init__.py" ]; then
     ${python_cmd} "$file" || die "unit tests for $bn failed" $?
  fi
done

echo "running test_producer.py"
${python_cmd} ${SCRAM_TEST_PATH}/test_producer.py || die "test_producer.py failed" $?
