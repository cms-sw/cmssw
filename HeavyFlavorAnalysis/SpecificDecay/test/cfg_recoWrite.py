import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask

process = cms.Process("bphAnalysis")

patAlgosToolsTask = getPatAlgosToolsTask(process)

#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
#filename = open('/lustre/cmswork/ronchese/pat_ntu/cmssw910/chkMini02/src/PDAnalysis/EDM/prod/datafiles_Vmini/mc/BsToJpsiPhi_BMuonFilter_TuneCUEP8M1_13TeV-pythia8-evtgen_mini910/BsToJpsiPhi_BMuonFilter_TuneCUEP8M1_13TeV-pythia8-evtgen_mini910_020.list', 'r')
#fileList = cms.untracked.vstring( filename.readlines() )

process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
patAlgosToolsTask.add(process.MEtoEDMConverter)
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# for BPH skim input
process.CandidateSelectedTracks = cms.EDProducer( "ConcreteChargedCandidateProducer",
                src=cms.InputTag("oniaSelectedTracks::RECO"),
                particleType=cms.string('pi+')
)

from PhysicsTools.PatAlgos.producersLayer1.genericParticleProducer_cfi import patGenericParticles
process.patSelectedTracks = patGenericParticles.clone(src=cms.InputTag("CandidateSelectedTracks"))
patAlgosToolsTask.add(process.patSelectedTracks)
# end BPH skim input

process.source = cms.Source("PoolSource",fileNames = cms.untracked.vstring(
#
### use this to access the nearest copy of the input file, querying the catalog
#
    '/store/data/Run2016E/Charmonium/USER/BPHSkim-PromptReco-v2/000/276/831/00000/00FD1519-714D-E611-B686-FA163E321AE0.root'
### use this to access the input file if by any reason you want to specify 
### the data server
#    'root://xrootd-cms.infn.it//store/data/Run2016E/Charmonium/USER/BPHSkim-PromptReco-v2/000/276/831/00000/00FD1519-714D-E611-B686-FA163E321AE0.root'
#
### use this to access an input file locally available
#    'file:/...complete_file_path.../XXXX.root'
))

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

from HeavyFlavorAnalysis.SpecificDecay.recoSelectForWrite_cfi import recoSelect

process.bphWriteSpecificDecay = cms.EDProducer('BPHWriteSpecificDecay',
    pVertexLabel = cms.string('offlinePrimaryVertices::RECO'),

# for BPH skim input
    gpCandsLabel = cms.string('patSelectedTracks'),
    ccCandsLabel = cms.string('onia2MuMuPAT::RECO'),
# for MiniAOD input
#    patMuonLabel = cms.string('slimmedMuons'),
#    pcCandsLabel = cms.string('packedPFCandidates::RECO'),
#    kSCandsLabel = cms.string('slimmedKshortVertices::RECO'),
#    lSCandsLabel = cms.string('slimmedLambdaVertices::RECO'),

    oniaName = cms.string('oniaFitted'),
    sdName   = cms.string('kx0Cand'),
    ssName   = cms.string('phiCand'),
    buName   = cms.string('buFitted'),
    bdName   = cms.string('bdFitted'),
    bsName   = cms.string('bsFitted'),
    k0Name   = cms.string('k0Fitted'),
    l0Name   = cms.string('l0Fitted'),
    b0Name   = cms.string('b0Fitted'),
    lbName   = cms.string('lbFitted'),
    bcName     = cms.string('bcFitted'),
    x3872Name  = cms.string('x3872Fitted'),
    writeVertex   = cms.bool( True ),
    writeMomentum = cms.bool( True ),
    recoSelect = cms.VPSet(recoSelect)
)

process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('reco.root'),
    outputCommands = cms.untracked.vstring(
# for BPH skim input
      "keep *",
      "keep *_bphWriteSpecificDecay_*_*",
      "drop *_patSelectedTracks_*_*",
      "drop *_CandidateSelectedTracks_*_*",
      "drop *_TriggerResults_*_bphAnalysis",
      "drop *_random*_*_bphAnalysis"
# for MINIAOD input
#      "keep *_bphWriteSpecificDecay_*_*",
#      "keep *_TriggerResults_*_HLT",
#      "keep *_offlineSlimmedPrimaryVertices_*_*"
    ),
)

process.p = cms.Path(
# for BPH skim input
    process.CandidateSelectedTracks *
# end BPH skim input
    process.bphWriteSpecificDecay
)

process.e = cms.EndPath(process.out, patAlgosToolsTask)

