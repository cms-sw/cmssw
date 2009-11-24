import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP3X_V12::All')

process.source = cms.Source (
    "PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_4_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3XY_V9-v1/0003/FA7139E8-97BD-DE11-A3E2-002618943935.root',
    '/store/relval/CMSSW_3_4_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3XY_V9-v1/0003/BC3224A5-9ABD-DE11-A625-002354EF3BDB.root',
    '/store/relval/CMSSW_3_4_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3XY_V9-v1/0003/8C578DA3-C0BD-DE11-9DEA-0017312A250B.root',
    '/store/relval/CMSSW_3_4_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3XY_V9-v1/0003/7A29EA77-9DBD-DE11-A3BC-0026189438ED.root',
    '/store/relval/CMSSW_3_4_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3XY_V9-v1/0003/3EA8A506-10BE-DE11-BB21-0018F3D09704.root',
    '/store/relval/CMSSW_3_4_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3XY_V9-v1/0003/04383FF7-9EBD-DE11-8511-0018F3D09616.root',
    )
    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.load("JetMETCorrections.Configuration.ZSPJetCorrections219_cff")
process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")
process.load("PhysicsTools.PatAlgos.patSequences_cff")

from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection

addJetCollection( process,
                  cms.InputTag('JetPlusTrackZSPCorJetIcone5'),
                  'IC5JPT',
                  doJTA        = True,
                  doBTagging   = True,
                  jetCorrLabel = None,
                  doType1MET   = False,
                  doL1Cleaning = True,
                  doL1Counters = True,
                  genJetCollection = cms.InputTag("iterativeCone5GenJets"),
                  doJetID      = False,
                  jetIdLabel   = "",
                  )

addJetCollection( process,
                  cms.InputTag('JetPlusTrackZSPCorJetSiscone5'),
                  'SC5JPT',
                  doJTA        = True,
                  doBTagging   = True,
                  jetCorrLabel = None,
                  doType1MET   = False,
                  doL1Cleaning = True,
                  doL1Counters = True,
                  genJetCollection = cms.InputTag("sisCone5GenJets"),
                  doJetID      = False,
                  jetIdLabel   = "",
                  )

addJetCollection( process,
                  cms.InputTag('JetPlusTrackZSPCorJetAntiKt5'),
                  'AK5JPT',
                  doJTA        = True,
                  doBTagging   = True,
                  jetCorrLabel = None,
                  doType1MET   = False,
                  doL1Cleaning = True,
                  doL1Counters = True,
                  genJetCollection = cms.InputTag("ak5GenJets"),
                  doJetID      = False,
                  jetIdLabel   = "",
                  )

#process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.Path (
#    process.dump *
    process.ZSPJetCorrectionsIcone5 *
    process.JetPlusTrackCorrectionsIcone5 *
    process.ZSPJetCorrectionsSisCone5 *
    process.JetPlusTrackCorrectionsSisCone5 *
    process.ZSPJetCorrectionsAntiKt5 *
    process.JetPlusTrackCorrectionsAntiKt5 *
    process.patDefaultSequence
    )

process.o = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('test.root'),
    outputCommands = cms.untracked.vstring(
    'keep *',
    )
    )
process.e = cms.EndPath( process.o )
