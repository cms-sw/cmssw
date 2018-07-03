import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4JetID_cfi import *

recoAllJetIds = cms.Sequence( ak4JetID )

recoJetIds = cms.Sequence( ak4JetID
			  )
