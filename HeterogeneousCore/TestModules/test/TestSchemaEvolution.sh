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


echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionAoSZeroReader.py"
echo
cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionAoSZeroReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionAoSOneReader.py"
echo
cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionAoSOneReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionAoSTwoReader.py"
echo
cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionAoSTwoReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionAoSThreeReader.py"
echo
cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionAoSThreeReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionAoSFourReader.py"
echo
cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionAoSFourReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"
echo "$ cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionAoSFiveReader.py"
echo
cmsRun ${LOCALTOP}/src/HeterogeneousCore/TestModules/test/SchemaEvolutionAoSFiveReader.py || exit $?
echo
echo "--------------------------------------------------------------------------------"