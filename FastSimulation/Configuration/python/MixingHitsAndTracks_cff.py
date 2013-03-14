#from FastSimulation.Configuration.mixNoPU_cfi import * 

from FastSimulation.Configuration.mixHitsWithPU_cfi import *
mixHits = cms.Sequence(
    mixSimCaloHits
    )
    
from FastSimulation.Configuration.mixTracksWithPU_cfi import *
mixTracks = cms.Sequence(
    mixRecoTracks
    )
    
