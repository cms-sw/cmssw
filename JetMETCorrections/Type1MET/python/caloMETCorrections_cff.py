import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from JetMETCorrections.Configuration.JetCorrectionServices_cff import *

##____________________________________________________________________________||
caloJetMETcorr = cms.EDProducer("CaloJetMETcorrInputProducer",
    src = cms.InputTag('ak4CaloJets'),
    jetCorrLabel = cms.string("ak4CaloL2L3"), # NOTE: use "ak4CaloL2L3" for MC / "ak4CaloL2L3Residual" for Data
    jetCorrEtaMax = cms.double(9.9),
    type1JetPtThreshold = cms.double(20.0),
    skipEM = cms.bool(True),
    skipEMfractionThreshold = cms.double(0.90),
    srcMET = cms.InputTag('corMetGlobalMuons')
)

##____________________________________________________________________________||
muonCaloMETcorr = cms.EDProducer("MuonMETcorrInputProducer",
    src = cms.InputTag('muons'),
    srcMuonCorrections = cms.InputTag('muonMETValueMapProducer', 'muCorrData')
)

##____________________________________________________________________________||
from JetMETCorrections.Type1MET.caloMETCorrections_Old_cff import *

##____________________________________________________________________________||
