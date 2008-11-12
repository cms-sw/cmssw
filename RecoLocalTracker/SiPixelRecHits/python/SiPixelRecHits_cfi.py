import FWCore.ParameterSet.Config as cms

siPixelRecHits = cms.EDFilter("SiPixelRecHitConverter",
    src = cms.InputTag("siPixelClusters"),
    # untracked string ClusterCollLabel   = "siPixelClusters"
    CPE = cms.string('PixelCPEGeneric'),
    VerboseLevel = cms.untracked.int32(0),
    TanLorentzAnglePerTesla = cms.double(0.106),
    Alpha2Order = cms.bool(True),
    speed = cms.int32(0)

    # Allows cuts to be optimized
#    eff_charge_cut_lowX = cms.double(0.0),
 #   eff_charge_cut_lowY = cms.double(0.0),
  #  eff_charge_cut_highX = cms.double(1.0),
   # eff_charge_cut_highY = cms.double(1.0),
  #  size_cutX = cms.double(3.0),
   # size_cutY = cms.double(3.0)
)


