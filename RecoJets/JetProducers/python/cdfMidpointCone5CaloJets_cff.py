import FWCore.ParameterSet.Config as cms

# $Id: cdfMidpointCone5CaloJets_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.MconeJetParameters_cfi import *
cdfMidpointCone5CaloJets = cms.EDProducer("CDFMidpointJetProducer",
    MconeJetParameters,
    FastjetNoPU,
    CaloJetParameters,
    coneRadius = cms.double(0.5),
    
    alias = cms.untracked.string('CDFMC5CaloJet')
)


