#!/bin/bash

# Pass in name and status
function die {
  printf "\n%s: status %s\n" "$1" "$2"
  exit $2
}

check_for_success() {
    "${@}" && echo -e "\n ---> Passed test of '${@}'\n\n" || exit 1
}

check_for_failure() {
    "${@}" && exit 1 || echo -e "\n ---> Passed test of '${@}'\n\n"
}

inputfile=/store/data/Run2024C/EphemeralHLTPhysics0/RAW/v1/000/379/416/00000/e8dd5e3c-216f-4545-acb6-ab86c9161085.root

echo "========================================"
echo "Testing convertToRaw in ${SCRAM_TEST_PATH}."
echo "----------------------------------------"
echo

echo "========================================"
echo "testing help function "
echo "----------------------------------------"
echo

convertToRaw --help  || die "Failure running convertToRaw --help" $?

echo "========================================"
echo "testing successful conversion"
echo "----------------------------------------"
echo

check_for_success convertToRaw -f 1 -l=1 -v $inputfile

echo "========================================"
echo "testing failing conversion"
echo "----------------------------------------"
echo

check_for_failure convertToRaw -f 1 -l=-1 -s rawDataRepacker $inputfile
