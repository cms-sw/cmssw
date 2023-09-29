#!/bin/bash

# Pass in name and status
function die {
  printf "\n%s: status %s\n" "$1" "$2"
  exit $2
}

if [ -z "${SCRAM_TEST_PATH}" ]; then
  printf "\n%s\n" "ERROR -- environment variable SCRAM_TEST_PATH not defined"
  printf "%s\n"   "         (hint: see readme file in the directory of this script)"
  exit 1
fi

# run test job
inputFileList="${SCRAM_TEST_PATH}"/testAccessToEDMInputsOfHLTTests_filelist.txt

if [ ! -f "${inputFileList}" ]; then
  printf "\n%s\n" "ERROR -- invalid path to file listing EDM input files:"
  printf "%s\n"   "         ${inputFileList}"
  exit 1
fi

for inputFile in $(cat "${inputFileList}"); do
  cmsRun "${SCRAM_TEST_PATH}"/testAccessToEDMInputsOfHLTTests_cfg.py inputFiles="${inputFile}" \
    || die "Failure running testAccessToEDMInputsOfHLTTests_cfg.py" $?
done
unset inputFile
