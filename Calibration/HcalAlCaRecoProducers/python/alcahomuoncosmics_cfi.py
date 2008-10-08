import FWCore.ParameterSet.Config as cms

#-----------------------------------------------------------
#AlCaReco Filtering for HO calibration using cosmicMuon/StandAlonMuon
#----------------------------------------------------------- 
hoCalibCosmicsProducer = cms.EDProducer("AlCaHOCalibProducer",
    hbheInput = cms.InputTag("hbhereco"),
    lastTS = cms.untracked.int32(8),
    hotime = cms.untracked.bool(False),
    hbinfo = cms.untracked.bool(False),
    sigma = cms.untracked.double(1.0),
    hoInput = cms.InputTag("horeco"),
    hltInput = cms.InputTag("TriggerResults"),
    l1Input = cms.InputTag("l1extraParticleMap"),
    towerInput = cms.InputTag("towerMaker"),
    digiInput = cms.untracked.bool(False),
    RootFileName = cms.untracked.string('test.root'),
    m_scale = cms.untracked.double(4.0),
    debug = cms.untracked.bool(False),
    muons = cms.untracked.InputTag("cosmicMuons"),
    #muons = cms.untracked.InputTag("standAloneMuons"),
    firstTS = cms.untracked.int32(5),
    PedestalFile = cms.untracked.string('peds_mtcc2_4333.log')
)


