import FWCore.ParameterSet.Config as cms

# -- needed for regional unpacking:
# from L1TriggerConfig.L1GeometryProducers.l1CaloGeometry_cfi import *
# from L1TriggerConfig.L1GeometryProducers.l1CaloGeomRecordSource_cff import *
#  es_source l1CaloGeomRecordSource = EmptyESSource {
#    string recordName = "L1CaloGeometryRecord"
#    vuint32 firstValid = { 1 }
#    bool iovIsRunNotTime = false
#  }
ecalRegionalEgammaFEDs = cms.EDProducer("EcalListOfFEDSProducer",
    Muon = cms.untracked.bool(False),
    Ptmin_noniso = cms.untracked.double(5.0),
    EM_l1TagIsolated = cms.untracked.InputTag("l1extraParticles","Isolated"),
    OutputLabel = cms.untracked.string(''),
    EM_regionEtaMargin = cms.untracked.double(0.25),
    Jets = cms.untracked.bool(False),
    EM_doNonIsolated = cms.untracked.bool(True),
    EM_doIsolated = cms.untracked.bool(True),
    EM_l1TagNonIsolated = cms.untracked.InputTag("l1extraParticles","NonIsolated"),
    debug = cms.untracked.bool(False),
    EM_regionPhiMargin = cms.untracked.double(0.4),
    Ptmin_iso = cms.untracked.double(5.0),
    EGamma = cms.untracked.bool(True)
)


