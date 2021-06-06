The validation of the ECAL Reco geometry in the old DDD way and new DD4HEP way is done with two dumpers present in Geometry/EcalAlgo/test/python :
1) runEcalCellDumpDDD_cfg.py for DDD
2) runEcalCellDumpDD4Hep_cfg.py for DD4HEP

These dump the position, backface, dimensions and the 8 corner-coordinates of each ECAL crystal in EB and EE.

For this two analyzers are used ecalBarrelCellParameterDump.cc and ecalEndcapCellParameterDump.cc which are present in: Geometry/EcalAlgo/test 

To dump these into a file execute: cmsRun PATH_TO_CONFIG/runEcalCellDumpDD4Hep_cfg.py > ECAL_cellParams_DD4HEP_Dump.txt and similarly for the DDD config.
The dump will be found to be identical.
