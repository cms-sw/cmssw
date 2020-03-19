#! /bin/bash -e

# move to the .../src directory
cd $CMSSW_BASE/src/

# check out all packages containing .cu files
git ls-files --full-name | grep '.*\.cu$' | cut -d/ -f-2 | sort -u | xargs git cms-addpkg

# rebuild all checked out packages
scram b -j
