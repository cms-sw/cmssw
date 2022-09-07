 #! /bin/bash -e

if ! [ "${LOCALTOP}" ]; then
  export LOCALTOP=${CMSSW_BASE}
  cd ${CMSSW_BASE}
fi

mkdir -p testHeterogeneousCoreAlpakaTestWriteRead
cd testHeterogeneousCoreAlpakaTestWriteRead
rm -f test.root
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/HeterogeneousCore/AlpakaTest/test/writer.py"
echo
cmsRun ${LOCALTOP}/src/HeterogeneousCore/AlpakaTest/test/writer.py
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/HeterogeneousCore/AlpakaTest/test/reader.py"
echo
cmsRun ${LOCALTOP}/src/HeterogeneousCore/AlpakaTest/test/reader.py
echo
echo "--------------------------------------------------------------------------------"
echo "$ edmDumpEventContent test.root"
echo
edmDumpEventContent test.root
