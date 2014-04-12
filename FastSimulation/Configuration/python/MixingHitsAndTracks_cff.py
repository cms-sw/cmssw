#from FastSimulation.Configuration.mixNoPU_cfi import * 

from SimGeneral.PileupInformation.AddPileupSummary_cfi import *
addPileupInfo.PileupMixingLabel = 'mix'
addPileupInfo.simHitLabel = 'g4SimHits'

from FastSimulation.Configuration.mixHitsAndTracksWithPU_cfi import *
mixHitsAndTracks = cms.Sequence(
    mix+
    addPileupInfo
    )
    
