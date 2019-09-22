import FWCore.ParameterSet.Config as cms

SiPhase2RecHitMatcherESProducer = cms.ESProducer("SiPhase2RecHitMatcherESProducer",
     ComponentName = cms.string('SiPhase2VectorHitMatcher'),
     Clusters = cms.string('siPhase2Clusters'),
     offlinestubs = cms.string('vectorHits'),
     maxVectorHits = cms.int32(999999999),
     maxVectorHitsInAStack = cms.int32(999),
     Algorithm = cms.string('VectorHitBuilderAlgorithm'),
     BarrelCut = cms.vdouble( 0., 0.05, 0.06, 0.08, 0.09, 0.12, 0.2), #layers are 6
     EndcapCut = cms.vdouble( 0., 0.1, 0.1, 0.1, 0.1, 0.1), #disks are 5
     CPE = cms.ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE"),
     Phase2CPE_name = cms.string('Phase2StripCPE')
#    NSigmaInside = cms.double(3.0)
)


