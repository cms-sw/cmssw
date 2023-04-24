import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECO', eras.Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
# process.load('Configuration.StandardSequences.Reconstruction_cff')



process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(
        '/store/data/Run2022F/EGamma/RAW/v1/000/362/154/00000/01df3a20-4381-4103-8b98-5fd77e266823.root',
    ),
)


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data', '')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

#-----------------------------------------
# CMSSW/Hcal non-DQM Related Module import
#-----------------------------------------
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("RecoLocalCalo.Configuration.hcalLocalReco_cff")
# process.load("RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff")
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")
process.load("RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi")

# load both cpu plugins
process.load("RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalCPUUncalibRecHitProducer_cfi")

##
## force HLT configuration for ecalMultiFitUncalibRecHit
##

process.ecalDigis = process.ecalEBunpacker.clone()
process.ecalDigis.InputLabel = cms.InputTag('rawDataCollector')

process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("test_uncalib.root")
)

process.finalize = cms.EndPath(process.out)

process.bunchSpacing = cms.Path(
    process.bunchSpacingProducer
)

process.digiPath = cms.Path(
    process.ecalDigis
)

process.recoPath = cms.Path(
    process.ecalMultiFitUncalibRecHit
    *process.ecalCPUUncalibRecHitProducer
)

process.schedule = cms.Schedule(
    process.bunchSpacing,
    process.digiPath,
    process.recoPath,
    process.finalize
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(8),
    numberOfStreams = cms.untracked.uint32(8),
    SkipEvent = cms.untracked.vstring('ProductNotFound'),
    wantSummary = cms.untracked.bool(True)
)


