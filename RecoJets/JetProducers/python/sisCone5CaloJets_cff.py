import FWCore.ParameterSet.Config as cms

# $Id: sisCone5CaloJets.cff,v 1.1 2007/08/02 21:58:23 fedor Exp $
from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.SISConeJetParameters_cfi import *
sisCone5CaloJets = cms.EDProducer("SISConeJetProducer",
    CaloJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    alias = cms.untracked.string('SISC5CaloJet'),
    coneRadius = cms.double(0.5)
)


