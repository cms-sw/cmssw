import FWCore.ParameterSet.Config as cms

process = cms.Process("L1SKIM")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100000
#process.MessageLogger.categories.append('L1GtTrigReport')
#process.MessageLogger.categories.append('HLTrigReport')

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

#process.load(INPUTFILELIST)
process.source = cms.Source("PoolSource",fileNames =cms.untracked.vstring(
 #   '/store/mc/Winter10/QCD_Pt_30to50_TuneZ2_7TeV_pythia6/GEN-SIM-RECODEBUG/E7TeV_ProbDist_2010Data_BX156_START39_V8-v1/0000/FC29B1D0-4F0D-E011-81EC-00261894393D.root'
    '/store/mc/Winter10/QCD_Pt_30to50_TuneZ2_7TeV_pythia6/AODSIM/E7TeV_ProbDist_2010Data_BX156_START39_V8-v1/0010/FCC19F72-200F-E011-9FBB-001A928116E8.root'
#    '/store/mc/Summer10/DiPion_E1to300/GEN-SIM-RECO/START36_V9_S09-v1/0024/4CEE3150-E581-DF11-B9C4-001A92971BDC.root'
#'/store/mc/Winter10/QCD_Pt_80to120_TuneZ2_7TeV_pythia6/AODSIM/E7TeV_ProbDist_2010Data_BX156_START39_V8-v1/0000/30E36146-CD0D-E011-A157-003048D4603C.root'
    ))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5) )

##################### digi-2-raw plus L1 emulation #########################
process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('TrackingTools/TrackAssociator/DetIdAssociatorESProducer_cff')

#################### Conditions and L1 menu ################################

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'START3X_V25B::All'
#process.GlobalTag.globaltag = 'START3X_V27::All'
process.GlobalTag.globaltag = 'START311_V2::All'

process.primaryVertexFilter = cms.EDFilter("GoodVertexFilter",
                                           vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                           minimumNDOF      = cms.uint32(4) ,
                                           maxAbsZ          = cms.double(20.0),
                                           maxd0            = cms.double(10.0)
                                           )

process.isotracksGen = cms.EDAnalyzer("IsolatedParticlesGeneratedJets",
                                      Debug            = cms.untracked.bool(True),
                                      JetSource        = cms.InputTag("ak5GenJets"),
                                      ParticleSource   = cms.InputTag("genParticles"),
                                      )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('IsolatedTracksGenParticles.root')
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

process.p1 = cms.Path(process.primaryVertexFilter*process.isotracksGen)
#process.p1 = cms.Path( process.isotracksGen )
