import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing

from Configuration.Eras.Era_Run2_25ns_cff import Run2_25ns
process = cms.Process('CTPPS2',Run2_25ns)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.load("CondFormats.PPSObjects.CTPPSPixelDAQMappingESSourceXML_cfi")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(10000)
        )

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('ERROR')
    )
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",ignoreTotal = cms.untracked.int32(1) )
process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring(
 '/store/data/Run2017A/ZeroBias2/AOD/PromptReco-v1/000/294/736/00000/44589413-F73F-E711-9E8D-02163E014337.root',
 '/store/data/Run2017A/ZeroBias2/AOD/PromptReco-v1/000/294/736/00000/4493383A-F63F-E711-A67F-02163E011870.root',
 '/store/data/Run2017A/ZeroBias2/AOD/PromptReco-v1/000/294/736/00000/5C6E42A5-0640-E711-9C3E-02163E011B62.root',
 '/store/data/Run2017A/ZeroBias2/AOD/PromptReco-v1/000/294/736/00000/98F7DD15-F93F-E711-87E1-02163E019D62.root',
 '/store/data/Run2017A/ZeroBias2/AOD/PromptReco-v1/000/294/736/00000/A01D8813-0A40-E711-A300-02163E01A211.root',
 '/store/data/Run2017A/ZeroBias2/AOD/PromptReco-v1/000/294/736/00000/B83C33DB-F63F-E711-82FC-02163E019DF3.root',
 '/store/data/Run2017A/ZeroBias2/AOD/PromptReco-v1/000/294/736/00000/D6706DD8-F93F-E711-9F6D-02163E019D7F.root',
 '/store/data/Run2017A/ZeroBias2/AOD/PromptReco-v1/000/294/736/00000/E206A2BF-FA3F-E711-8E72-02163E01369C.root'
),
duplicateCheckMode = cms.untracked.string("checkEachFile")
)


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_hlt_relval', '')

process.load("RecoPPS.Configuration.recoCTPPS_cff")

#process.patternProd = cms.EDProducer("patternProducer",
 #                                    label=cms.untracked.string("ctppsPixelRecHits"),
  #                                   RPixVerbosity = cms.untracked.int32(2)

#)
#process.load("RecoPPS.Local.patternProd_cfi")
#process.patternProd.RPixVerbosity = cms.untracked.int32(0)

############
process.o1 = cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring('drop *',
                                               'keep CTPPSPixelClusteredmDetSetVector_ctppsPixelClusters_*_*',
                                               'keep CTPPSPixelRecHitedmDetSetVector_ctppsPixelRecHits_*_*',
                                               'keep CTPPSPixelLocalTrackedmDetSetVector_ctppsPixelLocalTracks_*_*',
 'keep CTPPSLocalTrackLites_ctppsLocalTrackLiteProducer_*_*'
        ),
        fileName = cms.untracked.string("RPix_RecoLocalTrack.root")
       )


process.mixedigi_step = cms.Path(
process.recoCTPPS
#*process.patternProd
)

process.outpath = cms.EndPath(process.o1)

process.schedule = cms.Schedule(process.mixedigi_step,process.outpath)

# filter all path with the production filter sequence
for path in process.paths:
  #  getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq
    getattr(process,path)._seq =  getattr(process,path)._seq


