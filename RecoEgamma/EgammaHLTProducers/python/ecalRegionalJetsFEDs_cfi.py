import FWCore.ParameterSet.Config as cms

# -- needed for regional unpacking:
# from L1TriggerConfig.L1GeometryProducers.l1CaloGeometry_cfi import *
# from L1TriggerConfig.L1GeometryProducers.l1CaloGeomRecordSource_cff import *
#  es_source l1CaloGeomRecordSource = EmptyESSource {
#    string recordName = "L1CaloGeometryRecord"
#    vuint32 firstValid = { 1 }
#    bool iovIsRunNotTime = false
#  }
ecalRegionalJetsFEDs = cms.EDProducer("EcalListOfFEDSProducer",
    Muon = cms.untracked.bool(False),
    JETS_regionEtaMargin = cms.untracked.double(1.0),
    JETS_doForward = cms.untracked.bool(True),
    JETS_doTau = cms.untracked.bool(True),
    OutputLabel = cms.untracked.string(''),
    JETS_doCentral = cms.untracked.bool(True),
    ForwardSource = cms.untracked.InputTag("l1extraParticles","Forward"),
    TauSource = cms.untracked.InputTag("l1extraParticles","Tau"),
    CentralSource = cms.untracked.InputTag("l1extraParticles","Central"),
    # untracked double JETS_regionEtaMargin = 10.
    # untracked double JETS_regionPhiMargin = 3.1416
    Ptmin_jets = cms.untracked.double(50.0),
    debug = cms.untracked.bool(False),
    EGamma = cms.untracked.bool(False),
    JETS_regionPhiMargin = cms.untracked.double(1.0),
    Jets = cms.untracked.bool(True)
)


