import FWCore.ParameterSet.Config as cms

# Start with Standard Digitization:

from Configuration.StandardSequences.Digi_cff import *

#from SimGeneral.MixingModule.mixNoPU_cfi import *

# If we are going to run this with the DataMixer to follow adding
# detector noise, turn this off for now:

##### #turn off noise in all subdetectors
#simHcalUnsuppressedDigis.doNoise = False
#mix.digitizers.hcal.doNoise = False
#simEcalUnsuppressedDigis.doNoise = False
#mix.digitizers.ecal.doNoise = False
#simEcalUnsuppressedDigis.doESNoise = False
#simSiPixelDigis.AddNoise = False
#mix.digitizers.pixel.AddNoise = False
#simSiStripDigis.Noise = False
#mix.digitizers.strip.AddNoise = False
#simMuonCSCDigis.strips.doNoise = False
#simMuonCSCDigis.wires.doNoise = False
#DTs are strange - no noise flag - only use true hits?
#simMuonDTDigis.IdealModel = True
#simMuonDTDigis.onlyMuHits = True
#simMuonRPCDigis.Noise = False

# for PreMixing, to first approximation, allow noise in Muon system

# remove unnecessary modules from 'pdigi' sequence - run after DataMixing
# standard mixing module now makes unsuppressed digis for calorimeter
pdigi.remove(simEcalTriggerPrimitiveDigis)
pdigi.remove(simEcalDigis)  # does zero suppression
pdigi.remove(simEcalPreshowerDigis)  # does zero suppression
pdigi.remove(simHcalDigis)
pdigi.remove(simHcalTTPDigis)
