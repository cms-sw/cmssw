import FWCore.ParameterSet.Config as cms

from RecoLocalFastTime.FTLRecProducers.ftlUncalibratedRecHits_cfi import ftlUncalibratedRecHits
from RecoLocalFastTime.FTLRecProducers.ftlRecHits_cfi import ftlRecHits

fastTimingLocalReco = cms.Sequence(ftlUncalibratedRecHits*ftlRecHits)

from RecoLocalFastTime.FTLRecProducers.mtdUncalibratedRecHits_cfi import mtdUncalibratedRecHits
from RecoLocalFastTime.FTLRecProducers.mtdRecHits_cfi import mtdRecHits
_phase2_timing_layer_new_fastTimingLocalReco = cms.Sequence(mtdUncalibratedRecHits*mtdRecHits)
from Configuration.Eras.Modifier_phase2_timing_layer_new_cff import phase2_timing_layer_new
phase2_timing_layer_new.toReplaceWith(fastTimingLocalReco, _phase2_timing_layer_new_fastTimingLocalReco)
phase2_timing_layer_new.toModify(mtdRecHits, barrelUncalibratedRecHits = cms.InputTag('mtdUncalibratedRecHits:FTLBarrel'), 
                                 endcapUncalibratedRecHits = cms.InputTag('mtdUncalibratedRecHits:FTLEndcap') )
