import FWCore.ParameterSet.Config as cms

from RecoLocalFastTime.FTLRecProducers.mtdUncalibratedRecHits_cfi import mtdUncalibratedRecHits, btlBaseRecHitsSoA, etlBaseRecHitsSoA
from RecoLocalFastTime.FTLRecProducers.btlDigiSoAProducer_cfi import btlDigiSoAProducer
from RecoLocalFastTime.FTLRecProducers.etlDigiSoAProducer_cfi import etlDigiSoAProducer
from RecoLocalFastTime.FTLRecProducers.mtdRecHits_cfi import mtdRecHits, btlRecHitsSoA, etlRecHitsSoA
from RecoLocalFastTime.FTLRecProducers.mtdTrackingRecHits_cfi import mtdTrackingRecHits
from RecoLocalFastTime.FTLClusterizer.mtdClusters_cfi import mtdClusters
from RecoLocalFastTime.FTLClusterizer.mtdMergedClusters_cfi import mtdMergedClusters

from RecoLocalFastTime.FTLClusterizer.MTDCPEESProducers_cff import *
from RecoLocalFastTime.FTLRecProducers.MTDTimeCalibESProducers_cff import *

fastTimingLocalRecoTask = cms.Task(mtdUncalibratedRecHits,mtdRecHits,btlDigiSoAProducer,btlBaseRecHitsSoA,btlRecHitsSoA,etlDigiSoAProducer,etlBaseRecHitsSoA,etlRecHitsSoA,mtdClusters,mtdMergedClusters,mtdTrackingRecHits)
fastTimingLocalReco = cms.Sequence(fastTimingLocalRecoTask)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(btlDigiSoAProducer, btlDigiCollection = "mixData:MTDBarrel")
premix_stage2.toModify(etlDigiSoAProducer, etlDigiCollection = "mixData:MTDEndcap")

# with the mtdRawData process modifier the BTL SoA local reconstruction reads the digis of the BTL unpacker,
# which replaces the conversion of the simulated digis to SoA
from Configuration.ProcessModifiers.mtdRawData_cff import mtdRawData
mtdRawData.toModify(btlBaseRecHitsSoA, digi = "btlRawToDigi")
mtdRawData.toReplaceWith(fastTimingLocalRecoTask, fastTimingLocalRecoTask.copyAndExclude([btlDigiSoAProducer]))
