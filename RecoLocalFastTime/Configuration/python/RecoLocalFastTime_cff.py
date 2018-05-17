import FWCore.ParameterSet.Config as cms

from RecoLocalFastTime.FTLRecProducers.ftlUncalibratedRecHits_cfi import ftlUncalibratedRecHits
from RecoLocalFastTime.FTLRecProducers.ftlRecHits_cfi import ftlRecHits

fastTimingLocalReco = cms.Sequence(ftlUncalibratedRecHits*ftlRecHits)

from RecoLocalFastTime.FTLRecProducers.mtdUncalibratedRecHits_cfi import mtdUncalibratedRecHits
_phase2_timing_layer_new_fastTimingLocalReco = cms.Sequence(mtdUncalibratedRecHits*ftlRecHits)
from Configuration.Eras.Modifier_phase2_timing_layer_new_cff import phase2_timing_layer_new
phase2_timing_layer_new.toReplaceWith(fastTimingLocalReco, _phase2_timing_layer_new_fastTimingLocalReco)
phase2_timing_layer_new.toModify(ftlRecHits, barrelUncalibratedRecHits = cms.InputTag('mtdUncalibratedRecHits:FTLBarrel'), 
                                 endcapUncalibratedRecHits = cms.InputTag('mtdUncalibratedRecHits:FTLEndcap') )
