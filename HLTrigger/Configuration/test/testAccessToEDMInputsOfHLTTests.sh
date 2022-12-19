#!/bin/bash

# Pass in name and status
function die {
  echo $1: status $2
  echo === Log file ===
  cat ${3:-/dev/null}
  echo === End log file ===
  exit $2
}

# run test job
TESTDIR="${LOCALTOP}"/src/HLTrigger/Configuration/test

inputFileList="${TESTDIR}"/testAccessToEDMInputsOfHLTTests_filelist.txt

if [ ! -f "${inputFileList}" ]; then
  printf "%s\n" "ERROR -- invalid path to file listing EDM input files: ${inputFileList}"
  exit 1
fi

LOGFILE=log_testAccessToEDMInputsOfHLTTests

rm -f "${LOGFILE}"
for inputFile in $(cat "${inputFileList}"); do
  cmsRun "${TESTDIR}"/testAccessToEDMInputsOfHLTTests_cfg.py inputFiles="${inputFile}" &>> "${LOGFILE}" \
    || die "Failure running testAccessToEDMInputsOfHLTTests_cfg.py" $? "${LOGFILE}"
done
unset inputFile
