import FWCore.ParameterSet.Config as cms

process = cms.Process("L1SKIM")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.MessageLogger.categories.append('L1GtTrigReport')
process.MessageLogger.categories.append('HLTrigReport')

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

#process.load(INPUTFILELIST)
process.source = cms.Source("PoolSource",fileNames =cms.untracked.vstring(
     '/store/mc/Summer10/DiPion_E1to300/GEN-SIM-RECO/START36_V9_S09-v1/0024/4CEE3150-E581-DF11-B9C4-001A92971BDC.root'
    
    ))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(50) )

##################### digi-2-raw plus L1 emulation #########################

process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff')

#################### Conditions and L1 menu ################################

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run1_mc']

process.load('Calibration.IsolatedParticles.isolatedTracksHcalScale_cfi')
process.isolatedTracksHcal.MaxDxyPV  = 10.
process.isolatedTracksHcal.MaxDzPV   = 10.
process.isolatedTracksHcal.Verbosity = 1

process.primaryVertexFilter = cms.EDFilter("GoodVertexFilter",
                                           vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                           minimumNDOF      = cms.uint32(4) ,
                                           maxAbsZ          = cms.double(20.0),
                                           maxd0            = cms.double(10.0)
                                           )


process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('IsolatedTracksHcalScale.root')
                                   )

# define an EndPath to analyze all other path results
process.hltTrigReport = cms.EDAnalyzer( 'HLTrigReport',
      #HLTriggerResults = cms.InputTag( 'TriggerResults','','REDIGI36X')
      HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT') 
)

process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")
#process.l1GtTrigReport.L1GtRecordInputTag = 'simGtDigis'
process.l1GtTrigReport.L1GtRecordInputTag = 'gtDigis'
process.l1GtTrigReport.PrintVerbosity = 0
#=============================================================================

#process.p1 = cms.Path(process.primaryVertexFilter*process.isolatedTracksHcal)
process.p1 = cms.Path( process.isolatedTracksHcal )
