import FWCore.ParameterSet.Config as cms

process = cms.Process("MUSCLEFITMUONPRODUCER")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      "file:/home/demattia/3C83C26B-8B91-DF11-9CE6-90E6BAE8CC13.root"
    )
)

process.poolDBESSource = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dummyScale.db'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('MuScleFitDBobjectRcd'),
        tag = cms.string('JPsi_1_3_invNb_innerTrack')
    ))
)

process.MuScleFitMuonProducer = cms.EDProducer(
    'MuScleFitMuonProducer',
    MuonLabel = cms.InputTag("muons")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myOutputFile.root')
)
  
process.p = cms.Path(process.MuScleFitMuonProducer)

process.e = cms.EndPath(process.out)
