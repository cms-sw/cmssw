import FWCore.ParameterSet.Config as cms

process = cms.Process("bckAnalysis")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.source = cms.Source("PoolSource",fileNames = cms.untracked.vstring(
    'file:reco.root'
))

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

process.TFileService = cms.Service('TFileService',
  fileName = cms.string('his.root'),
  closeFileFast = cms.untracked.bool(True)
)

process.bphHistoSpecificDecay = cms.EDAnalyzer('BPHHistoSpecificDecay',
    trigResultsLabel  = cms.string('TriggerResults::HLT'), 
    oniaCandsLabel = cms.string('bphWriteSpecificDecay:oniaFitted:bphAnalysis'),
    sdCandsLabel = cms.string('bphWriteSpecificDecay:kx0Cand:bphAnalysis'),
    ssCandsLabel = cms.string('bphWriteSpecificDecay:phiCand:bphAnalysis'),
    buCandsLabel = cms.string('bphWriteSpecificDecay:buFitted:bphAnalysis'),
    bdCandsLabel = cms.string('bphWriteSpecificDecay:bdFitted:bphAnalysis'),
    bsCandsLabel = cms.string('bphWriteSpecificDecay:bsFitted:bphAnalysis'),
    k0CandsLabel = cms.string('bphWriteSpecificDecay:k0Fitted:bphAnalysis'),
    l0CandsLabel = cms.string('bphWriteSpecificDecay:l0Fitted:bphAnalysis'),
    b0CandsLabel = cms.string('bphWriteSpecificDecay:b0Fitted:bphAnalysis'),
    lbCandsLabel = cms.string('bphWriteSpecificDecay:lbFitted:bphAnalysis'),
    bcCandsLabel = cms.string('bphWriteSpecificDecay:bcFitted:bphAnalysis'),
    x3872CandsLabel = cms.string('bphWriteSpecificDecay:x3872Fitted:bphAnalysis')
)

process.p = cms.Path(
    process.bphHistoSpecificDecay
)


