import FWCore.ParameterSet.Config as cms

SiStripRegionConnectivity = cms.ESProducer("SiStripRegionConnectivity",
    EtaDivisions = cms.untracked.uint32(20),
    PhiDivisions = cms.untracked.uint32(20),
    EtaMax = cms.untracked.double(2.5)
)


