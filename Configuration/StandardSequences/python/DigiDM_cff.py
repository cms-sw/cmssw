import FWCore.ParameterSet.Config as cms

# Start with Standard Digitization:

from Configuration.StandardSequences.Digi_cff import *

# If we are going to run this with the DataMixer to follow adding
# detector noise, turn this off for now:

##### #turn off noise in all subdetectors
simHcalUnsuppressedDigis.doNoise = False
simEcalUnsuppressedDigis.doNoise = False
#simEcalUnsuppressedDigis.doESNoise = False
simSiPixelDigis.AddNoise = False
simSiStripDigis.Noise = False
simMuonCSCDigis.strips.doNoise = False
simMuonCSCDigis.wires.doNoise = False
#DTs are strange - no noise flag - only use true hits?
#simMuonDTDigis.IdealModel = True
simMuonDTDigis.onlyMuHits = True
simMuonRPCDigis.Noise = False

# remove unnecessary modules from 'pdigi' sequence
pdigi.remove(simEcalTriggerPrimitiveDigis)
pdigi.remove(simEcalDigis)
pdigi.remove(simEcalPreshowerDigis)
pdigi.remove(simHcalDigis)
pdigi.remove(simHcalTTPDigis)
