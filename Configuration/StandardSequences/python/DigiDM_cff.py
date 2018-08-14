import FWCore.ParameterSet.Config as cms

# Start with Standard Digitization:

from Configuration.StandardSequences.Digi_cff import *

_simMuonCSCDigis_orig = simMuonCSCDigis.clone()
_simMuonDTDigis_orig = simMuonDTDigis.clone()
_simMuonRPCDigis_orig = simMuonRPCDigis.clone()

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
simMuonCSCDigis.strips.doNoise = False
simMuonCSCDigis.wires.doNoise = False
#DTs are strange - no noise flag - only use true hits?
#simMuonDTDigis.IdealModel = True
simMuonDTDigis.onlyMuHits = True
simMuonRPCDigis.Noise = False

# remove unnecessary modules from 'pdigi' sequence - run after DataMixing
# standard mixing module now makes unsuppressed digis for calorimeter
pdigi.remove(simEcalTriggerPrimitiveDigis)
pdigi.remove(simEcalEBTriggerPrimitiveDigis) # phase2
pdigi.remove(simEcalDigis)  # does zero suppression
pdigi.remove(simEcalPreshowerDigis)  # does zero suppression
pdigi.remove(simHcalDigis)
pdigi.remove(simHcalTTPDigis)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
# for PreMixing, to first approximation, allow noise in Muon system
premix_stage2.toReplaceWith(simMuonCSCDigis, _simMuonCSCDigis_orig)
premix_stage2.toReplaceWith(simMuonDTDigis , _simMuonDTDigis_orig )
premix_stage2.toReplaceWith(simMuonRPCDigis, _simMuonRPCDigis_orig)

# premixing stage2 runs addPileupInfo after DataMixer (configured in DataMixerPreMix_cff)
premix_stage2.toReplaceWith(pdigi, pdigi.copyAndExclude([addPileupInfo, genPUProtons]))

# genPUProtons, on the other hand, is an EDAlias. In principle it is
# already loaded with digitizers_cfi, but in practice that gets
# overwritten by the EDProducer by loading this file. So we hack
# around by adding a ProcessModifier with overwrites it back to
# EDAlias (because we can't toReplaceWith() an EDProducer with an
# EDAlias).
def _loadPremixStage2Alias(process):
    import SimGeneral.MixingModule.aliases_PreMix_cfi as _aliases
    process.genPUProtons = _aliases.genPUProtons
modifyDigiDM_loadPremixStage2Alias = premix_stage2.makeProcessModifier(_loadPremixStage2Alias)
