import FWCore.ParameterSet.Config as cms

# $Id: sisCone5GenJetsNoNuBSM.cff,v 1.1 2007/08/02 21:58:23 fedor Exp $
from RecoJets.JetProducers.GenJetParametersNoNuBSM_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.SISConeJetParameters_cfi import *
sisCone5GenJetsNoNuBSM = cms.EDProducer("SISConeJetProducer",
    GenJetParametersNoNuBSM,
    SISConeJetParameters,
    FastjetNoPU,
    alias = cms.untracked.string('SISC5GenJetNoNuBSM'),
    coneRadius = cms.double(0.5)
)


