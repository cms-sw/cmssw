import FWCore.ParameterSet.Config as cms

# $Id: cdfMidpointCone5GenJetsNoNuBSM_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.GenJetParametersNoNuBSM_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.MconeJetParameters_cfi import *
cdfMidpointCone5GenJetsNoNuBSM = cms.EDProducer("CDFMidpointJetProducer",
    MconeJetParameters,
    FastjetNoPU,
    GenJetParametersNoNuBSM,
    coneRadius = cms.double(0.5),
    
    alias = cms.untracked.string('CDFMC5GenJetNoNuBSM')
)


