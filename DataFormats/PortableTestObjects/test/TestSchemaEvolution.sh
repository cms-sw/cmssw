 #! /bin/bash -e

if ! [ "${LOCALTOP}" ]; then
  export LOCALTOP=${CMSSW_BASE}
  cd ${CMSSW_BASE}
fi

echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/DataFormats/PortableTestObjects/test/SchemaEvolutionZeroReader.py"
echo
cmsRun ${LOCALTOP}/src/DataFormats/PortableTestObjects/test/SchemaEvolutionZeroReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/DataFormats/PortableTestObjects/test/SchemaEvolutionOneReader.py"
echo
cmsRun ${LOCALTOP}/src/DataFormats/PortableTestObjects/test/SchemaEvolutionOneReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/DataFormats/PortableTestObjects/test/SchemaEvolutionTwoReader.py"
echo
cmsRun ${LOCALTOP}/src/DataFormats/PortableTestObjects/test/SchemaEvolutionTwoReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/DataFormats/PortableTestObjects/test/SchemaEvolutionThreeReader.py"
echo
cmsRun ${LOCALTOP}/src/DataFormats/PortableTestObjects/test/SchemaEvolutionThreeReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/DataFormats/PortableTestObjects/test/SchemaEvolutionFourReader.py"
echo
cmsRun ${LOCALTOP}/src/DataFormats/PortableTestObjects/test/SchemaEvolutionFourReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/DataFormats/PortableTestObjects/test/SchemaEvolutionFiveReader.py"
echo
cmsRun ${LOCALTOP}/src/DataFormats/PortableTestObjects/test/SchemaEvolutionFiveReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"