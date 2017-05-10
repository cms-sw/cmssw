import FWCore.ParameterSet.Config as cms

#-----------------------------------------------------------
#AlCaReco Filtering for HO calibration using cosmicMuon/StandAlonMuon
#----------------------------------------------------------- 
#process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
from Configuration.StandardSequences.Reconstruction_Data_cff import *

hoCalibProducer = cms.EDProducer("AlCaHOCalibProducer",
    hbheInput = cms.InputTag("hbhereco"),
    hotime = cms.untracked.bool(False),
    hbinfo = cms.untracked.bool(False),
    sigma = cms.untracked.double(1.0),
    hoInput = cms.InputTag("horeco"),
    towerInput = cms.InputTag("towerMaker"),
    RootFileName = cms.untracked.string('test.root'),
    m_scale = cms.untracked.double(4.0),
    debug = cms.untracked.bool(False),
    muons = cms.untracked.InputTag("muons"),
    #muons = cms.untracked.InputTag("cosmicMuons"),
    #muons = cms.untracked.InputTag("standAloneMuons"),
    vertexTags = cms.InputTag("offlinePrimaryVertices"),
    lumiTags = cms.InputTag("lumiProducer")
)


