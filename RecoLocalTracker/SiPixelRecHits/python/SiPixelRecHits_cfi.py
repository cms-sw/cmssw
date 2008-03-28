import FWCore.ParameterSet.Config as cms

siPixelRecHits = cms.EDFilter("SiPixelRecHitConverter",
    eff_charge_cut_lowY = cms.untracked.double(0.0),
    src = cms.InputTag("siPixelClusters"),
    # dfehling
    eff_charge_cut_lowX = cms.untracked.double(0.0),
    eff_charge_cut_highX = cms.untracked.double(1.0),
    eff_charge_cut_highY = cms.untracked.double(1.0),
    size_cutY = cms.untracked.double(3.0),
    size_cutX = cms.untracked.double(3.0),
    # untracked string ClusterCollLabel   = "siPixelClusters"
    CPE = cms.string('PixelCPEGeneric'),
    VerboseLevel = cms.untracked.int32(0),
    TanLorentzAnglePerTesla = cms.double(0.106),
    Alpha2Order = cms.bool(True),
    speed = cms.int32(0)
)


