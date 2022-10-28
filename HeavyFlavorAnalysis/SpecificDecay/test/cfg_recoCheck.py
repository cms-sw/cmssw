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

process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.source = cms.Source("PoolSource",fileNames = cms.untracked.vstring(
    'file:reco.root'
))

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

process.checkBPHWriteDecay = cms.EDAnalyzer('CheckBPHWriteDecay',
    fileName = cms.untracked.string('out'),
### to dump only one event
#    runNumber = cms.uint32( 275371 ),
#    evtNumber = cms.uint32( 783544498 ),
    writePtr = cms.untracked.bool(False),
    candsLabel = cms.vstring('bphWriteSpecificDecay:oniaFitted:bphAnalysis'
                            ,'bphWriteSpecificDecay:kx0Cand:bphAnalysis'
                            ,'bphWriteSpecificDecay:phiCand:bphAnalysis'
                            ,'bphWriteSpecificDecay:buFitted:bphAnalysis'
                            ,'bphWriteSpecificDecay:bdFitted:bphAnalysis'
                            ,'bphWriteSpecificDecay:bsFitted:bphAnalysis'
                            ,'bphWriteSpecificDecay:k0Fitted:bphAnalysis'
                            ,'bphWriteSpecificDecay:l0Fitted:bphAnalysis'
                            ,'bphWriteSpecificDecay:b0Fitted:bphAnalysis'
                            ,'bphWriteSpecificDecay:lbFitted:bphAnalysis'
                            ,'bphWriteSpecificDecay:bcFitted:bphAnalysis'
                            ,'bphWriteSpecificDecay:x3872Fitted:bphAnalysis')
)

process.p = cms.Path(
    process.checkBPHWriteDecay
)


