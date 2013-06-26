#from FastSimulation.Configuration.mixNoPU_cfi import * 

# PileupSummaryInfo is taken from mixHits because this is the one that can handle OOT pileup
from SimGeneral.PileupInformation.AddPileupSummary_cfi import *
addPileupInfo.PileupMixingLabel = 'mixSimCaloHits'
addPileupInfo.simHitLabel = 'g4SimHits'

from FastSimulation.Configuration.mixHitsWithPU_cfi import *
mixHits = cms.Sequence(
    mixSimCaloHits+
    addPileupInfo
    )
    
from FastSimulation.Configuration.mixTracksWithPU_cfi import *
mixTracks = cms.Sequence(
    mixRecoTracks
    )
    
