import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfPileUp_cfi  import *
from CommonTools.ParticleFlow.TopProjectors.pfNoPileUp_cfi import *
from CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi import *

pfPileUpJME = pfPileUp.clone( PFCandidates='particleFlowPtrs',
                              Vertices = 'goodOfflinePrimaryVertices',
                              checkClosestZVertex = False )
pfNoPileUpJME = pfNoPileUp.clone( topCollection = 'pfPileUpJME',
                                  bottomCollection = 'particleFlowPtrs' )



pfNoPileUpJMESequence = cms.Sequence(
    goodOfflinePrimaryVertices +
    pfPileUpJME +
    pfNoPileUpJME
    )
