import FWCore.ParameterSet.Config as cms

from RecoLocalFastTime.FTLRecProducers.ftlUncalibratedRecHits_cfi import ftlUncalibratedRecHits
from RecoLocalFastTime.FTLRecProducers.ftlRecHits_cfi import ftlRecHits

fastTimingLocalRecoTask = cms.Task(ftlUncalibratedRecHits,ftlRecHits)
fastTimingLocalReco = cms.Sequence(fastTimingLocalRecoTask)


from RecoLocalFastTime.FTLRecProducers.mtdUncalibratedRecHits_cfi import mtdUncalibratedRecHits
from RecoLocalFastTime.FTLRecProducers.mtdRecHits_cfi import mtdRecHits
from RecoLocalFastTime.FTLRecProducers.mtdTrackingRecHits_cfi import mtdTrackingRecHits
from RecoLocalFastTime.FTLClusterizer.mtdClusters_cfi import mtdClusters

from RecoLocalFastTime.FTLClusterizer.MTDCPEESProducers_cff import *
from RecoLocalFastTime.FTLRecProducers.MTDTimeCalibESProducers_cff import *

_phase2_timing_layer_fastTimingLocalRecoTask = cms.Task(mtdUncalibratedRecHits,mtdRecHits,mtdClusters,mtdTrackingRecHits)
_phase2_timing_layer_fastTimingLocalReco = cms.Sequence(_phase2_timing_layer_fastTimingLocalRecoTask)

from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer

phase2_timing_layer.toReplaceWith(fastTimingLocalRecoTask, _phase2_timing_layer_fastTimingLocalRecoTask)

phase2_timing_layer.toModify(mtdRecHits, 
                              barrelUncalibratedRecHits = 'mtdUncalibratedRecHits:FTLBarrel', 
                              endcapUncalibratedRecHits = 'mtdUncalibratedRecHits:FTLEndcap')
