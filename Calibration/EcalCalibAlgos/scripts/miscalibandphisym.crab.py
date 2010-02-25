#
# Python config file to drive phi symmetry et sum accumulation
# Apply a miscalibration from xml and run phisym
#

import FWCore.ParameterSet.Config as cms


process=cms.Process("PHISYM")
process.load('Configuration/StandardSequences/GeometryPilot2_cff')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('')
)




process.CaloMiscalibTools = cms.ESSource("CaloMiscalibTools",
  fileName = cms.untracked.string('smearedsmscale0.03.xml')


)

process.prefer("CaloMiscalibTools")


process.miscalrechit = cms.EDFilter("EcalRecHitRecalib",
    # Reco labels                                
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    ecalRecHitsProducer = cms.string('ecalRecHit'),
    RecalibBarrelHitCollection = cms.string('EcalRecHitsEB'),
    RecalibEndcapHitCollection = cms.string('EcalRecHitsEE')
)






process.phisymcalib = cms.EDAnalyzer("PhiSymmetryCalibration",
                                     
    ecalRecHitsProducer = cms.string("miscalrechit"),
    barrelHitCollection = cms.string("EcalRecHitsEB"),
    endcapHitCollection = cms.string("EcalRecHitsEE"),
    eCut_barrel = cms.double(0.250),
    eCut_endcap = cms.double(0.000),
    eventSet = cms.int32(1)
  )

process.p = cms.Path(process.miscalrechit*process.phisymcalib)
