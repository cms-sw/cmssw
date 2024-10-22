The validation of the ECAL Reco geometry in the old DDD way and new DD4hep way is done with two dumpers present in Geometry/EcalAlgo/test/python :
1) runEcalCellDumpDDD_cfg.py for DDD
2) runEcalCellDumpDD4hep_cfg.py for DD4hep

These dump the position, backface, dimensions and the 8 corner-coordinates of each ECAL crystal in EB and EE.

For this two analyzers are used ecalBarrelCellParameterDump.cc and ecalEndcapCellParameterDump.cc which are present in: Geometry/EcalAlgo/test 

To dump these into a file execute: cmsRun PATH_TO_CONFIG/runEcalCellDumpDD4hep_cfg.py > ECAL_cellParams_DD4hep_Dump.txt and similarly for the DDD config.
The dump will be found to be identical.
