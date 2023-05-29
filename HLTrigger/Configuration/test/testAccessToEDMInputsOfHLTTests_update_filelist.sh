#!/bin/bash -e

# This script updates the file "${CMSSW_BASE}"/src/HLTrigger/Configuration/test/testAccessToEDMInputsOfHLTTests_filelist.txt
# with the list of EDM files potentially used by HLT tests in the main release cycles of CMSSW (i.e. branches named CMSSW_\d_\d_X).

# path to output file
outputFile="${CMSSW_BASE}"/src/HLTrigger/Configuration/test/testAccessToEDMInputsOfHLTTests_filelist.txt

# path to directory hosting this script
TESTDIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# ensure that directory hosting this script corresponds to ${CMSSW_BASE}/src/HLTrigger/Configuration/test
if [ "${TESTDIR}" != "${CMSSW_BASE}"/src/HLTrigger/Configuration/test ]; then
  printf "\n%s\n" "ERROR -- the directory hosting testAccessToHLTTestInputs.sh [1] does not correspond to \${CMSSW_BASE}/src/HLTrigger/Configuration/test [2]"
  printf "%s\n"   "         [1] ${TESTDIR}"
  printf "%s\n\n" "         [2] ${CMSSW_BASE}/src/HLTrigger/Configuration/test"
  exit 1
fi

# files in CMSSW using EDM inputs for HLT tests
cmsswFiles=(
  HLTrigger/Configuration/test/cmsDriver.csh
  Configuration/HLT/python/addOnTestsHLT.py
  Utilities/ReleaseScripts/scripts/addOnTests.py
)

# list of CMSSW branches to be checked
# official-cmssw is the default name of the remote corresponding to the central CMSSW repository
cmsswBranches=($(git branch -a | grep 'remotes/official-cmssw/CMSSW_[0-9]*_[0-9]*_X$'))
cmsswBranches+=("HEAD") # add HEAD to include updates committed locally

# create 1st temporary file (list of EDM input files used by HLT tests, incl. duplicates)
TMPFILE1=$(mktemp)

# grep from base directory
cd "${CMSSW_BASE}"/src

printf "%s\n" "-------------------------"
printf "%s\n" "Finding list of EDM files used by HLT tests in CMSSW (branches: '^CMSSW_[0-9]*_[0-9]*_X$')..."

# loop over CMSSW branches to be grep-d
for cmsswBranch in "${cmsswBranches[@]}"; do
  foo=($(git grep -h "[='\" ]/store/.*.root" ${cmsswBranch} -- ${cmsswFiles[*]} 2> /dev/null |
    sed 's|=/store/| /store/|g' | sed "s|'| |g" | sed 's|"| |g' |
    awk '{ for(i=1;i<=NF;i++) if ($i ~ /\/store\/.*.root/) print $i }'))
  printf "\n  %s\n" "${cmsswBranch}"
  for bar in "${foo[@]}"; do
    printf "    %s\n" "${bar}"
    echo "${bar}" >> "${TMPFILE1}"
  done
  unset foo bar
done; unset cmsswBranch

# create 2nd temporary file (list of available EDM input files used by HLT tests, without duplicates)
TMPFILE2=$(mktemp)

# edmFileIsAvailable:
#  use LFN to check if a EDM file is in the ibeos cache,
#  or can be accessed remotely via global redirector
function edmFileIsAvailable() {
  [ $# -eq 1 ] || return 1
  # check access to ibeos cache
  edmFileUtil -f root://eoscms.cern.ch//eos/cms/store/user/cmsbuild"${1}" &> /dev/null
  [ $? -ne 0 ] || return 0
  # check remote access via global redirector
  edmFileUtil -f root://cms-xrd-global.cern.ch/"${1}" &> /dev/null
  return $?
}

printf "%s\n" "-------------------------"
printf "%s\n" "Checking availability of EDM files..."
printf "%s\n" "(checks whether the file is in the ibeos cache, or it can be accessed remotely via the redirector cms-xrd-global.cern.ch)"

for inputFile in $(cat "${TMPFILE1}" | sort -u); do
  printf '\e[1;34m%-20s\e[m %s\033[0K\r' "[Checking...]" "${inputFile}"
  if ! edmFileIsAvailable "${inputFile}"; then
    printf '\e[1;31m%-20s\e[m %s\n' "[File not available]" "${inputFile}"
    continue
  fi
  printf '\e[1;32m%-20s\e[m %s\n' "[File available]" "${inputFile}"
  echo "${inputFile}" >> "${TMPFILE2}"
done
unset inputFile

# create/update output file
cat "${TMPFILE2}" | sort -u > "${outputFile}"

printf "%s\n" "-------------------------"
printf "%s\n" "File updated: ${outputFile}"
printf "%s\n" "-------------------------"

# return to test/ directory
cd "${TESTDIR}"
