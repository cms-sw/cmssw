import FWCore.ParameterSet.Config as cms

# $Id: cdfMidpointCone5GenJetsNoNuBSM_cff.py,v 1.2 2008/04/21 03:28:16 rpw Exp $
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


