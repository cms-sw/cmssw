#!/bin/bash

cp -r $CMSSW_BASE/src/RecoTracker/LSTCore ./
find LSTCore -type d | xargs chmod +w
cd LSTCore/standalone
source setup.sh
echo "Building LST CPU backend..."
lst_make_tracklooper -mCs
