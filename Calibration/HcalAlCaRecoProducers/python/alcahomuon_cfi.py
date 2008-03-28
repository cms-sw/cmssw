import FWCore.ParameterSet.Config as cms

#-----------------------------------------------------------
#AlCaReco Filtering for HO calibration using cosmicMuon/StandAlonMuon
#----------------------------------------------------------- 
hoCalibProducer = cms.EDProducer("AlCaHOCalibProducer",
    lastTS = cms.untracked.int32(8),
    hotime = cms.untracked.bool(False),
    hbinfo = cms.untracked.bool(False),
    sigma = cms.untracked.double(1.0),
    digiInput = cms.untracked.bool(False),
    RootFileName = cms.untracked.string('test.root'),
    m_scale = cms.untracked.double(4.0),
    debug = cms.untracked.bool(False),
    #        untracked InputTag muons =cosmicMuons     # standAloneMuons
    muons = cms.untracked.InputTag("standAloneMuons"),
    firstTS = cms.untracked.int32(5),
    #        untracked string PedestalFile = "/afs/cern.ch/user/m/majumder/scratch0/anal/CMSSW_1_6_4/src/Calibration/HcalAlCaRecoProducers/test/peds_mtcc2_4333.log"
    PedestalFile = cms.untracked.string('peds_mtcc2_4333.log')
)


