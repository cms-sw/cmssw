import FWCore.ParameterSet.Config as cms

# $Id: cdfMidpointCone5GenJetsNoNuBSM.cff,v 1.1 2007/08/02 21:58:22 fedor Exp $
from RecoJets.JetProducers.GenJetParametersNoNuBSM_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.MconeJetParameters_cfi import *
cdfMidpointCone5GenJetsNoNuBSM = cms.EDProducer("CDFMidpointJetProducer",
    MconeJetParameters,
    FastjetNoPU,
    GenJetParametersNoNuBSM,
    coneRadius = cms.double(0.5),
    JetPtMin = cms.double(0.0),
    alias = cms.untracked.string('CDFMC5GenJetNoNuBSM')
)


