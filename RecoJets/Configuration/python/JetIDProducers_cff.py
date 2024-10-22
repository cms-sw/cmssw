import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4JetID_cfi import *

recoJetIdsTask = cms.Task( ak4JetID )
recoJetIds = cms.Sequence(recoJetIdsTask)
recoAllJetIds = cms.Sequence(recoJetIdsTask)
