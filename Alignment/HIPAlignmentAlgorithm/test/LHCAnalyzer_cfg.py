import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpGENSIMRECO

process = cms.Process("Demo")

###################################################################
# Messages
###################################################################
process.load("FWCore.MessageService.MessageLogger_cfi")
MessageLogger = cms.Service("MessageLogger",
                            cout = cms.untracked.PSet(
                                threshold = cms.untracked.string('WARNING')
                            ),
                            destinations = cms.untracked.vstring('cout')
                            )
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

###################################################################
# Conditions
###################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')

###################################################################
# Event source
###################################################################
process.source = cms.Source("PoolSource",
                            fileNames = filesRelValTTbarPileUpGENSIMRECO)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

###################################################################
# Analyzer
###################################################################
process.LhcTrackAnalyzer = cms.EDAnalyzer("LhcTrackAnalyzer",
                                          TrackCollectionTag = cms.InputTag("generalTracks"),
                                          #TrackCollectionTag = cms.InputTag("ALCARECOTkAlMinBias"),
                                          PVtxCollectionTag = cms.InputTag("offlinePrimaryVertices"),
                                          acceptedBX        = cms.vuint32(), # (51,2724)
                                          OutputFileName    = cms.string("AnalyzerOutput_1.root"),
                                          Debug = cms.bool(False)
                                          )

process.p = cms.Path(process.LhcTrackAnalyzer)
