#!/bin/sh

cmsRun runReco_PbPb_MC_53X.py
cmsRun runForest_PbPb_MIX_53X.py

root -l quickResponse.C

