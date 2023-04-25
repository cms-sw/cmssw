#!/bin/zsh

cd $CMSSW_BASE/src/Alignment/OfflineValidation/test

echo "Printing help"
validateAlignments.py -h

echo "Running over YAML"
validateAlignments.py -v -f -d unit_test.yaml

echo "Running over JSON"
validateAlignments.py -v -d -f unit_test.json
