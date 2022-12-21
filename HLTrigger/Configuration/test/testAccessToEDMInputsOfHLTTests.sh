#!/bin/bash

# Pass in name and status
function die {
  printf "\n%s: status %s\n" "$1" "$2"
  exit $2
}

# run test job
TESTDIR="${LOCALTOP}"/src/HLTrigger/Configuration/test

inputFileList="${TESTDIR}"/testAccessToEDMInputsOfHLTTests_filelist.txt

if [ ! -f "${inputFileList}" ]; then
  printf "\n%s\n" "ERROR -- invalid path to file listing EDM input files:"
  printf "%s\n"   "         ${inputFileList}"
  exit 1
fi

for inputFile in $(cat "${inputFileList}"); do
  cmsRun "${TESTDIR}"/testAccessToEDMInputsOfHLTTests_cfg.py inputFiles="${inputFile}" \
    || die "Failure running testAccessToEDMInputsOfHLTTests_cfg.py" $?
done
unset inputFile
