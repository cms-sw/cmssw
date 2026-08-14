 #! /bin/bash -e

if ! [ "${LOCALTOP}" ]; then
  export LOCALTOP=${CMSSW_BASE}
  cd ${CMSSW_BASE}
fi

echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionZeroReader.py"
echo
cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionZeroReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionOneReader.py"
echo
cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionOneReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionTwoReader.py"
echo
cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionTwoReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionThreeReader.py"
echo
cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionThreeReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionFourReader.py"
echo
cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionFourReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionFiveReader.py"
echo
cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionFiveReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
