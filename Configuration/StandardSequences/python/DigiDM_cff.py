import FWCore.ParameterSet.Config as cms

# Start with Standard Digitization:

from Configuration.StandardSequences.Digi_cff import *

_simMuonCSCDigis_orig = simMuonCSCDigis.clone()
_simMuonDTDigis_orig = simMuonDTDigis.clone()
_simMuonRPCDigis_orig = simMuonRPCDigis.clone()

#from SimGeneral.MixingModule.mixNoPU_cfi import *

# If we are going to run this with the DataMixer to follow adding
# detector noise, turn this off for now:

# In premixing stage2 muon digis are produced after PreMixingModule
# The simMuon*Digis modules get used in DataMixerPreMix_cff, so better
# leave them untouched here.
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2

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
(~premix_stage2).toModify(simMuonCSCDigis,
    strips = dict(doNoise = False),
    wires  = dict(doNoise = False)
)
#DTs are strange - no noise flag - only use true hits?
#simMuonDTDigis.IdealModel = True
(~premix_stage2).toModify(simMuonDTDigis, onlyMuHits = True)
(~premix_stage2).toModify(simMuonRPCDigis, Noise = False)

# remove unnecessary modules from 'pdigi' sequence - run after DataMixing
# standard mixing module now makes unsuppressed digis for calorimeter
pdigiTask.remove(simEcalTriggerPrimitiveDigis)
pdigiTask.remove(simEcalEBTriggerPrimitiveDigis) # phase2
pdigiTask.remove(simEcalDigis)  # does zero suppression
pdigiTask.remove(simEcalPreshowerDigis)  # does zero suppression
pdigiTask.remove(simHcalDigis)
pdigiTask.remove(simHcalTTPDigis)
