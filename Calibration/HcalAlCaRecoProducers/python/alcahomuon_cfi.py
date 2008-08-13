import FWCore.ParameterSet.Config as cms

#-----------------------------------------------------------
#AlCaReco Filtering for HO calibration using cosmicMuon/StandAlonMuon
#----------------------------------------------------------- 
hoCalibProducer = cms.EDProducer("AlCaHOCalibProducer",
    hbheInput = cms.InputTag("hbhereco"),
    l1Input = cms.InputTag("l1extraParticleMap"),
    sigma = cms.untracked.double(1.0),
    hotime = cms.untracked.bool(False),
    towerInput = cms.InputTag("towerMaker"),
    hbinfo = cms.untracked.bool(False),
    hoInput = cms.InputTag("horeco"),
    PedestalFile = cms.untracked.string('peds_mtcc2_4333.log'),
    digiInput = cms.untracked.bool(False),
    RootFileName = cms.untracked.string('test.root'),
    lastTS = cms.untracked.int32(8),
    debug = cms.untracked.bool(False),
    m_scale = cms.untracked.double(4.0),
    firstTS = cms.untracked.int32(5),
    hltInput = cms.InputTag("TriggerResults"),
    #        untracked InputTag muons =cosmicMuons     # standAloneMuons
    muons = cms.untracked.InputTag("standAloneMuons")
)



