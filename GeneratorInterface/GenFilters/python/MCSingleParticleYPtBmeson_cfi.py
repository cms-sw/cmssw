import FWCore.ParameterSet.Config as cms

#filter major b-meson "in the acceptance of the detector", only one match is enough, anti-particles are checked by default
bmesonFilter = cms.EDFilter("MCSingleParticleYPt",
    ParticleID = cms.untracked.vint32(511,521,531,541),
    MinPt = cms.untracked.vdouble(1.5,1.5,1.5,1.5),
    MinY = cms.untracked.vdouble(-3.,-3.,-3.,-3.),
    MaxY = cms.untracked.vdouble(3.,3.,3.,3.)
)
