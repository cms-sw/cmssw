from FastSimulation.Configuration.mixNoPU_cfi import *
from FastSimulation.PileUpProducer.PileUpProducer_cff import *
# PileupSummaryInfo
from SimGeneral.PileupInformation.AddPileupSummary_cfi import *
addPileupInfo.PileupMixingLabel = 'famosPileUp'
addPileupInfo.simHitLabel = 'famosSimHits'

famosMixing = cms.Sequence(
    famosPileUp+
    addPileupInfo
)

    

