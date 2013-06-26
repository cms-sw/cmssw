#!/bin/sh
# This file is used for making all constants in a single SQLite file. That file is for use with the CSC Validation package to test new constants.  
cmsRun CSCDBCrosstalkPopCon_cfg.py
mv DBCrossTalk.db DBNoiseMatrix.db

cmsRun CSCDBNoiseMatrixPopCon_cfg.py
mv DBNoiseMatrix.db DBPedestals.db

cmsRun CSCDBPedestalsPopCon_cfg.py
mv DBPedestals.db DBGains.db

cmsRun CSCDBGainsPopCon_cfg.py
mv DBGains.db NewConstantsTest.db 

