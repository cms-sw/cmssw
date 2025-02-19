import FWCore.ParameterSet.Config as cms

# -- needed for regional unpacking:
# from L1TriggerConfig.L1GeometryProducers.l1CaloGeometry_cfi import *
# from L1TriggerConfig.L1GeometryProducers.l1CaloGeomRecordSource_cff import *
#  es_source l1CaloGeomRecordSource = EmptyESSource {
#    string recordName = "L1CaloGeometryRecord"
#    vuint32 firstValid = { 1 }
#    bool iovIsRunNotTime = false
#  }
ecalRegionalMuonsFEDs = cms.EDProducer("EcalListOfFEDSProducer",
    Muon = cms.untracked.bool(True),
    MuonSource = cms.untracked.InputTag("l1extraParticles"),
    MU_regionPhiMargin = cms.untracked.double(1.0),
    OutputLabel = cms.untracked.string(''),
    Jets = cms.untracked.bool(False),
    # untracked double MU_regionEtaMargin = 1.0
    # untracked double MU_regionPhiMargin = 1.0
    Ptmin_muon = cms.untracked.double(0.0),
    debug = cms.untracked.bool(False),
    EGamma = cms.untracked.bool(False),
    MU_regionEtaMargin = cms.untracked.double(1.0)
)


