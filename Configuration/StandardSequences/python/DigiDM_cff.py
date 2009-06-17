import FWCore.ParameterSet.Config as cms

# Start with Standard Digitization:

from Configuration.StandardSequences.Digi_cff import *

# If we are going to run this with the DataMixer to follow adding
# detector noise, turn this off for now:

##### #turn off noise in all subdetectors
simHcalUnsuppressedDigis.doNoise = False
simEcalUnsuppressedDigis.doNoise = False
ecal_electronics_sim.doNoise = False
es_electronics_sim.doESNoise = False
simSiPixelDigis.AddNoise = False
simSiStripDigis.Noise = False
simMuonCSCDigis.strips.doNoise = False
simMuonCSCDigis.wires.doNoise = False
#DTs are strange - no noise flag - only use true hits?
#simMuonDTDigis.IdealModel = True
simMuonDTDigis.onlyMuHits = True
simMuonRPCDigis.Noise = False
