
#########################
For Analysis of CSCTF Distributions and Efficiencies in MC:

cmsRun csctfEfficiencySim_cfg.py

The primary histos are on canvases in the main cscTFEfficiency folder.

For analysis of CSCTF Distributions in Data:

cmsRun csctfAnaData_cfg.py

In Data, the useful histos are in the TrackFinder folder (kinimatic
distributions), and the TFMultiplicity folder, which can give rate histograms.

##########################
For Analysis of LCT/Track Stub rates per sector/subsector and 
distributions in phi/eta

cmsRun lctOccupancies_cfg.py

For basic printouts of LCT vars:

cmsRun lctPrinter_cfg.py

##########################
To Make Single Muon Data for CSCTF:

Run:
cmsDriver.py L1Trigger/CSCTrackFinder/python/PtGun_cfi.py -s GEN,SIM,DIGI -n 10000 --conditions auto:mc --no_exec

Then replace the outputCommands line with this in the OutputModule:
	outputCommands = cms.untracked.vstring(
	"keep PSimHits_g4SimHits_MuonDTHits_*",
 	"keep PSimHits_g4SimHits_MuonCSCHits_*",
 	"keep SimTracks_g4SimHits_*_*",
 	"keep CSCDetIdCSCComparatorDigiMuonDigiCollection_*_*_*",
 	"keep CSCDetIdCSCStripDigiMuonDigiCollection_*_*_*",
 	"keep CSCDetIdCSCWireDigiMuonDigiCollection_*_*_*",
 	"keep DTLayerIdDTDigiMuonDigiCollection_*_*_*",
 	"keep DTLayerIdDTDigiSimLinkMuonDigiCollection_*_*_*"
	),
