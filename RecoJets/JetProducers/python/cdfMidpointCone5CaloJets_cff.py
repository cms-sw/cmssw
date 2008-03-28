import FWCore.ParameterSet.Config as cms

# $Id: cdfMidpointCone5CaloJets.cff,v 1.2 2007/08/02 21:58:22 fedor Exp $
from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.MconeJetParameters_cfi import *
cdfMidpointCone5CaloJets = cms.EDProducer("CDFMidpointJetProducer",
    MconeJetParameters,
    FastjetNoPU,
    CaloJetParameters,
    coneRadius = cms.double(0.5),
    JetPtMin = cms.double(0.0),
    alias = cms.untracked.string('CDFMC5CaloJet')
)


