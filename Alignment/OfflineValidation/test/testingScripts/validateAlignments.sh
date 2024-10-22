#!/bin/zsh

echo "Printing help"
validateAlignments.py -h

echo "Running over YAML"
validateAlignments.py -v -f -d ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/unit_test.yaml

echo "Running over JSON"
validateAlignments.py -v -d -f ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/unit_test.json
