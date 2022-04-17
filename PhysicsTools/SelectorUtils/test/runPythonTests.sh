#!/bin/sh

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

python3 ${CMSSW_BASE}/src/PhysicsTools/SelectorUtils/test/test_vid_selectors.py
