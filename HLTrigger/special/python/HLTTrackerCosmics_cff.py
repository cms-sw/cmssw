import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
level1seedHLTTrackerCosmics = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescaleHLTTrackerCosmics = copy.deepcopy(hltPrescaler)
#
#addition for CR.xT runs.
#
# create all strip rechits ###########
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
from HLTrigger.special.CosmicsCoTF_ForHLT_cff import *
from HLTrigger.special.CosmicsCTF_ForHLT_cff import *
from HLTrigger.special.CosmicsRS_ForHLT_cff import *
hltcosmicsRPHIrechitfilter = cms.EDFilter("HLTCountNumberOfSingleRecHit",
    src = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    MaxN = cms.int32(50),
    MinN = cms.int32(-1)
)

hltcosmicsSTEREOrechitfilter = cms.EDFilter("HLTCountNumberOfSingleRecHit",
    src = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    MaxN = cms.int32(50),
    MinN = cms.int32(-1)
)

hltcosmicsMATCHEDrechitfilter = cms.EDFilter("HLTCountNumberOfMatchedRecHit",
    src = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    MaxN = cms.int32(50),
    MinN = cms.int32(-1)
)

hltTrackerCosmics = cms.Sequence(level1seedHLTTrackerCosmics+prescaleHLTTrackerCosmics)
hltTrackerCosmicsRecHits = cms.Sequence(cms.SequencePlaceholder("doLocalTracker")+siStripMatchedRecHits)
hltTrackerCosmicsRecHitsFilter = cms.Sequence(hltcosmicsRPHIrechitfilter+hltcosmicsSTEREOrechitfilter+hltcosmicsMATCHEDrechitfilter)
hltTrackerCosmicsCoTF = cms.Sequence(hltTrackerCosmicsRecHits+hltTrackerCosmicsRecHitsFilter+hltTrackerCosmicsSeedsCoTF+hltTrackerCosmicsSeedsFilterCoTF+hltTrackerCosmicsTracksCoTF+hltTrackerCosmicsTracksFilterCoTF)
hltTrackerCosmicsRecoCoTF = cms.Sequence(hltTrackerCosmicsRecHits+hltTrackerCosmicsSeedsCoTF+hltTrackerCosmicsTracksCoTF)
hltTrackerCosmicsCTF = cms.Sequence(hltTrackerCosmicsRecHits+hltTrackerCosmicsRecHitsFilter+hltTrackerCosmicsSeedsCTF+hltTrackerCosmicsSeedsFilterCTF+hltTrackerCosmicsTracksCTF+hltTrackerCosmicsTracksFilterCTF)
hltTrackerCosmicsRecoCTF = cms.Sequence(hltTrackerCosmicsRecHits+hltTrackerCosmicsSeedsCTF+hltTrackerCosmicsTracksCTF)
hltTrackerCosmicsRS = cms.Sequence(hltTrackerCosmicsRecHits+hltTrackerCosmicsRecHitsFilter+hltTrackerCosmicsSeedsRS+hltTrackerCosmicsSeedsFilterRS+hltTrackerCosmicsTracksRS+hltTrackerCosmicsTracksFilterRS)
hltTrackerCosmicsRecoRS = cms.Sequence(hltTrackerCosmicsRecHits+hltTrackerCosmicsSeedsRS+hltTrackerCosmicsTracksRS)
level1seedHLTTrackerCosmics.L1TechTriggerSeeding = True
level1seedHLTTrackerCosmics.L1SeedsLogicalExpression = 28
prescaleHLTTrackerCosmics.prescaleFactor = 1
siStripMatchedRecHits.Regional = True

