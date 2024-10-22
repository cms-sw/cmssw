import FWCore.ParameterSet.Config as cms


from Configuration.Eras.Era_Phase2_cff import Phase2
process = cms.Process('MuonClassif',Phase2)

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.source = cms.Source("PoolSource",
 fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_9_1_1/RelValZMM_14/GEN-SIM-RECO/PU25ns_91X_upgrade2023_realistic_v1_D17PU200-v1/10000/003FC7CB-EB3F-E711-92D1-0025905A6076.root'
 ),
 secondaryFileNames = cms.untracked.vstring(
 )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5) )    

process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

## ==== PAT used to parse the muon selection
process.load("MuonAnalysis.MuonAssociators.patMuonsWithTrigger_cff")
from MuonAnalysis.MuonAssociators.patMuonsWithTrigger_cff import *

## ==== Classification by Hits
process.load("MuonAnalysis.MuonAssociators.muonClassificationByHitsTP_cfi")
#
from MuonAnalysis.MuonAssociators.muonClassificationByHitsTP_cfi import addUserData as addClassByHits
addClassByHits(process.patMuonsWithoutTrigger, extraInfo=True)

# test output
process.output = cms.OutputModule("PoolOutputModule",
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('output_test.root'),
    splitLevel = cms.untracked.int32(0)
)

process.muonClassifier = cms.Path(process.muonClassificationByHits)

process.output_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(process.muonClassifier)


# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

######

process.MessageLogger.cerr = cms.untracked.PSet(
    noTimeStamps = cms.untracked.bool(True),

    threshold = cms.untracked.string('WARNING'),

    MuonToTrackingParticleAssociatorEDProducer = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    MuonToTrackingParticleAssociatorByHits = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    MuonToTrackingParticleAssociatorByHitsImpl = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    MuonAssociatorByHitsHelper = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    TrackerMuonHitExtractor = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    )
)

process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    noTimeStamps = cms.untracked.bool(True),
    threshold = cms.untracked.string('INFO'),

    default = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    MuonToTrackingParticleAssociatorEDProducer = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    ),
    MuonToTrackingParticleAssociatorByHits = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    ),
    MuonToTrackingParticleAssociatorByHitsImpl = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    ),
    MuonAssociatorByHitsHelper = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    ),
    TrackerMuonHitExtractor = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    MuonMCClassifier = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    ),
    FwkReport = cms.untracked.PSet(
        reportEvery = cms.untracked.int32(1),
        limit = cms.untracked.int32(10000000)
    ),
    FwkSummary = cms.untracked.PSet(
        reportEvery = cms.untracked.int32(1),
        limit = cms.untracked.int32(10000000)
    ),
    FwkJob = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    Root_NoDictionary = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    )
)
