import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

### correlations/flow condensed track information

allTracks = cms.EDProducer("ConcreteChargedCandidateProducer",
                           src = cms.InputTag("hiSelectedTracks"),
                           particleType = cms.string('pi+')
                           )
