import FWCore.ParameterSet.Config as cms

process = cms.Process("DISPLAY")


#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")


### Expects test.root in current directory.
#     fileNames=cms.untracked.vstring('root://xrootd.t2.ucsd.edu//tas-5/cerati/TTbarEventsPU/step2_ttbar_45PU25ns.root')
process.source = cms.Source(
    "PoolSource",  
 fileNames=cms.untracked.vstring('file:/home/alja/cms-dev/tracking.root')
)

process.POurFilter = cms.EDFilter("RandomFilter",
                                    acceptRate = cms.untracked.double(1.0))



from SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi import *
process.load("SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi")

# process.xxx = cms.Sequence(simHitTPAssocProducer)
# process.yyy = cms.Path(process.xxx)
# process.schedule = cms.Schedule(process.yyy)


process.xxx=cms.EDProducer('SimHitTPAssociationProducer',
simHitSrc = cms.VInputTag(cms.InputTag('g4SimHits','MuonDTHits'),
cms.InputTag('g4SimHits','MuonCSCHits'),
cms.InputTag('g4SimHits','MuonRPCHits'),
cms.InputTag('g4SimHits','TrackerHitsTIBLowTof'),
cms.InputTag('g4SimHits','TrackerHitsTIBHighTof'),
cms.InputTag('g4SimHits','TrackerHitsTIDLowTof'),
cms.InputTag('g4SimHits','TrackerHitsTIDHighTof'),
cms.InputTag('g4SimHits','TrackerHitsTOBLowTof'),
cms.InputTag('g4SimHits','TrackerHitsTOBHighTof'),
cms.InputTag('g4SimHits','TrackerHitsTECLowTof'),
cms.InputTag('g4SimHits','TrackerHitsTECHighTof'),
cms.InputTag( 'g4SimHits','TrackerHitsPixelBarrelLowTof'),
cms.InputTag('g4SimHits','TrackerHitsPixelBarrelHighTof'),
cms.InputTag('g4SimHits','TrackerHitsPixelEndcapLowTof'),
cms.InputTag('g4SimHits','TrackerHitsPixelEndcapHighTof') ),
trackingParticleSrc = cms.InputTag('mix', 'MergedTrackTruth'),
simHitTpMapTag = cms.InputTag("simHitTPAssocProducer")
)


process.mypath = cms.Path(process.xxx)


