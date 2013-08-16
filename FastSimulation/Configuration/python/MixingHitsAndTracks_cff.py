#from FastSimulation.Configuration.mixNoPU_cfi import * 

from SimGeneral.PileupInformation.AddPileupSummary_cfi import *
#addPileupInfo.PileupMixingLabel = 'mixSimCaloHits'
addPileupInfo.PileupMixingLabel = 'mix'
addPileupInfo.simHitLabel = 'g4SimHits'

from FastSimulation.Configuration.mixHitsAndTracksWithPU_cfi import *
mixHitsAndTracks = cms.Sequence(
    mix+
    addPileupInfo
    )
    

#from FastSimulation.Configuration.mixHitsWithPU_cfi import *
#mixHits = cms.Sequence(
#    mixSimCaloHits+
#    addPileupInfo
#    )
    
#from FastSimulation.Configuration.mixTracksWithPU_cfi import *
#mixTracks = cms.Sequence(
#    mixRecoTracks
#    )
    
