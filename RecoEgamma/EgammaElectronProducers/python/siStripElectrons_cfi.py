import FWCore.ParameterSet.Config as cms

#
# $ Id: $
# Author: Jim Pivarski, Cornell 3 Aug 2006
#
siStripElectrons = cms.EDProducer("SiStripElectronProducer",
    siStereoHitCollection = cms.string('stereoRecHit'),
    maxHitsOnDetId = cms.int32(4),
    minHits = cms.int32(5),
    trackCandidatesLabel = cms.string(''),
    superClusterProducer = cms.string('correctedHybridSuperClusters'),
    phiBandWidth = cms.double(0.01), ## radians

    siStripElectronsLabel = cms.string(''),
    siRphiHitCollection = cms.string('rphiRecHit'),
    siHitProducer = cms.string('siStripMatchedRecHits'),
    maxReducedChi2 = cms.double(10000.0), ## might not work yet

    originUncertainty = cms.double(15.0), ## cm

    maxNormResid = cms.double(10.0),
    siMatchedHitCollection = cms.string('matchedRecHit'),
    superClusterCollection = cms.string('')
)


