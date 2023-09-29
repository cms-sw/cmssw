### CMSSW command line parameter parser                                                                                                                                                               
import FWCore.ParameterSet.Config as cms


process = cms.Process("test")

# Message Logger settings
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 250
process.load("RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff")

process.maxEvents = cms.untracked.PSet( 
    input = cms.untracked.int32(100)
)

process.options = cms.untracked.PSet( 
    allowUnscheduled = cms.untracked.bool(True),
    wantSummary      = cms.untracked.bool(True),
    numberOfThreads  = cms.untracked.uint32(8),
    numberOfStreams = cms.untracked.uint32(0)
)

process.source = cms.Source("PoolSource", 
    fileNames = cms.untracked.vstring("/store/mc/RunIISummer20UL18MiniAODv2/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/130000/25BF763A-BF41-E242-86A2-5E0BE8EF605C.root")                            
)

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('TrackingTools.TransientTrack.TransientTrackBuilder_cfi')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')


from PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cfi import updatedPatJets
process.slimmedJetsUpdated = updatedPatJets.clone(
    jetSource = "slimmedJetsPuppi",
    addJetCorrFactors = False,
    discriminatorSources = cms.VInputTag(
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probb"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probc"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probuds"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probg"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probmu"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probele"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaup1h0p"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaup1h1p"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaup1h2p"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaup3h0p"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaup3h1p"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaum1h0p"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaum1h1p"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaum1h2p"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaum3h0p"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaum3h1p"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:ptcorr"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:ptnu"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:ptreshigh"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:ptreslow"),

        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiForwardJetTags:probq"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiForwardJetTags:probg"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiForwardJetTags:ptcorr"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiForwardJetTags:ptnu"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiForwardJetTags:ptreshigh"),
        cms.InputTag("pfParticleNetFromMiniAODAK4PuppiForwardJetTags:ptreslow"),

    )
)


process.path = cms.Path(
    process.slimmedJetsUpdated,process.pfParticleNetFromMiniAODAK4PuppiTask
)

process.output = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "test.root" ),
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(4),
    eventAutoFlushCompressedSize = cms.untracked.int32(31457280),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string( 'RECO' ),
        filterName = cms.untracked.string( '' )
    ),
    overrideBranchesSplitLevel = cms.untracked.VPSet(),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_*slimmedJetsUpdated*_*_*',
    )
)

process.endpath = cms.EndPath(process.output);
process.schedule = cms.Schedule(process.path,process.endpath);

