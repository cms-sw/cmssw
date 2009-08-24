#!/bin/bash

$CMSSW_RELEASE_BASE/src/CondTools/SiStrip/scripts/CreatingTables.sh sqlite_file:SiStripLorentzAngle_CalibrationSmear.db a a

cmsRun $CMSSW_BASE/src/CalibTracker/SiStripLorentzAngle/python/SQLiteEnsembleGenerator_cfg.py
cmsRun $CMSSW_BASE/src/CalibTracker/SiStripLorentzAngle/python/SQLiteCheck_cfg.py

cmscond_list_iov -c sqlite_file:SiStripLorentzAngle_CalibrationSmear.db -t SiStripLorentzAngle_CalibrationSmear_31X

