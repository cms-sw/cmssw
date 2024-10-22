 #! /bin/bash -e

if ! [ "${LOCALTOP}" ]; then
  export LOCALTOP=${CMSSW_BASE}
  cd ${CMSSW_BASE}
fi

mkdir -p testHeterogeneousCoreCUDATestWriteRead
cd testHeterogeneousCoreCUDATestWriteRead
rm -f test.root
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/HeterogeneousCore/CUDATest/test/writer.py"
echo
cmsRun ${LOCALTOP}/src/HeterogeneousCore/CUDATest/test/writer.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/HeterogeneousCore/CUDATest/test/reader.py"
echo
cmsRun ${LOCALTOP}/src/HeterogeneousCore/CUDATest/test/reader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ edmDumpEventContent test.root"
echo
edmDumpEventContent test.root || exit $?
