import FWCore.ParameterSet.Config as cms

process = cms.Process("ZMuMuSubskim")

# Setup PAT
from PhysicsTools.PatAlgos.patTemplate_cfg import *
from PhysicsTools.PatAlgos.tools.coreTools import *
from PhysicsTools.PatAlgos.tools.pfTools import *

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
#process.options.SkipEvent = cms.untracked.vstring('ProductNotFound')
process.options.FailPath = cms.untracked.vstring('ProductNotFound')


process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 100


# Input files
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/2010/HLTMu15_run2010B_Nov4ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/c82cb8a47fc34b62fedf9d7879ccf9bf/testZMuMuSubskim_100_1_Luy.root",
#"file:/scratch1/cms/data/fall10/mc/zmumu_powheg/00F38C7F-49CB-DF11-9110-00237DF1FFB0.root",

    )
)
#import os
#dirname = "/tmp/degrutto/MinBiasMC/"
#dirlist = os.listdir(dirname)
#basenamelist = os.listdir(dirname + "/")
#for basename in basenamelist:
#                process.source.fileNames.append("file:" + dirname + "/" + basename)
#                print "Number of files to process is %s" % (len(process.source.fileNames))

                


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START38_V12::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

### Subskim

############
## to run on data or without MC truth uncomment the following
process.load("ElectroWeakAnalysis.Skimming.zMuMu_SubskimPaths_cff")
############

# output module configuration
process.load("ElectroWeakAnalysis.Skimming.zMuMuSubskimOutputModule_cfi")

############
## to run the MC truth uncomment the following
## Look also at python/ZMuMuAnalysisSchedules_cff.py
##process.load("ElectroWeakAnalysis.Skimming.zMuMu_SubskimPathsWithMCTruth_cff")
##process.zMuMuSubskimOutputModule.outputCommands.extend(process.mcEventContent.outputCommands)
####

process.zMuMuSubskimOutputModule.fileName = 'file:testZMuMuSubskim_oneshot_Test.root'

process.outpath = cms.EndPath(process.zMuMuSubskimOutputModule)

### Here set the HLT Path for trigger matching
#process.dimuonsHLTFilter.HLTPaths = ["HLT_Mu9"]
#process.muonTriggerMatchHLTMuons.pathNames = cms.vstring( 'HLT_Mu9' )
#process.userDataMuons.hltPath = cms.string("HLT_Mu9")
##process.userDataDimuons.hltPath = cms.string("HLT_Mu9")
#process.userDataDimuonsOneTrack.hltPath = cms.string("HLT_Mu9")

############

### Analysis
from ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesSequences_cff import *



isData = True
removeMCMatching(process, ['All'])


switchJetCollection(process,cms.InputTag('ak5PFJets'),
                                                                doJTA        = True,
                                                                doBTagging   = True,
                                                                jetCorrLabel = ('AK5PF', cms.vstring(['L2Relative', 'L3Absolute', 'L2L3Residual'])),
                                                                doType1MET   = True,
                                                             #   genJetCollection=cms.InputTag("ak5GenJets"),
                                                                doJetID      = True
                                                            )

if isData!=True:
                switchJetCollection(process,cms.InputTag('ak5PFJets'),
                                                                                                                            doJTA        = True,
                                                                                                                            doBTagging   = True,
                                                                                                                            jetCorrLabel = ('AK5PF', cms.vstring(['L2Relative', 'L3Absolute' ])),
                                                                                                                            doType1MET   = True,
                                                                                                                            genJetCollection=cms.InputTag("ak5GenJets"),
                                                                                                                            doJetID      = True

                                                                                                                        )

# Clean the Jets from the seleted leptons, and apply loose(???) btag cuts and loose id cuts!
process.cleanPatJets = cms.EDProducer("PATJetCleaner",
                                      src = cms.InputTag("patJets"),
                                      preselection = cms.string('( (neutralEmEnergy/energy < 0.99) &&  (neutralHadronEnergy/energy < 0.99) && numberOfDaughters>1) '),
                                      checkOverlaps = cms.PSet(
    muons = cms.PSet(
    src       = cms.InputTag("userDataMuons"),
    algorithm = cms.string("byDeltaR"),
    preselection        = cms.string(""),
    deltaR              = cms.double(0.5),
    checkRecoComponents = cms.bool(False),
    pairCut             = cms.string(""),
    requireNoOverlaps   = cms.bool(True),
    ),
    ),
                                      ###applying 2 loose cutson the b-tagged jets
                                      
                                      finalCut = cms.string('')

                                      
                                      )
                                      



                
process.jetPath = cms.Path(
                         process.makePatJets *
                         process.cleanPatJets
                              )
                


process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string('ewkZMuMuCategories_oneshot_Test.root')
)


### vertexing
#process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesVtxed_cff")
#process.vtxedNtuplesOut.fileName = cms.untracked.string('file:VtxedNtupleLoose_test.root')

### 3_5_X reprocessed MC: to process REDIGI HLT tables uncomment the following
#process.patTrigger.processName = "REDIGI"
#process.patTriggerEvent.processName = "REDIGI"
#process.patTrigger.triggerResults = cms.InputTag( "TriggerResults::REDIGI" )
#process.patTrigger.triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::REDIGI" )

### 3_6_X reprocessed MC: to process REDIGI HLT tables uncomment the following
#process.dimuonsHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","REDIGI36X")
#process.patTrigger.processName = "REDIGI36X"
##process.patTriggerEvent.processName = "REDIGI36X"
#process.patTrigger.triggerResults = cms.InputTag( "TriggerResults::REDIGI36X" )
#process.patTrigger.triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::REDIGI36X" )

### plots
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesPlots_cff")

### ntuple
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuAnalysisNtupler_cff")
process.ntuplesOut.fileName = cms.untracked.string('file:Ntuple_oneshot_dym20_ttbarCheck.root')

###
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuAnalysisSchedules_cff") 

### Here set the HLT Path for trigger matching
process.dimuonsHLTFilter.HLTPaths = ["HLT_Mu15_v1"]
process.muonTriggerMatchHLTMuons.pathNames = cms.vstring( 'HLT_Mu15_v1' )
process.userDataMuons.hltPath = cms.string("HLT_Mu15_v1")
process.userDataDimuons.hltPath = cms.string("HLT_Mu15_v1")
process.userDataDimuonsOneTrack.hltPath = cms.string("HLT_Mu15_v1")
############

### 3_8_X reprocessed MC: to process REDIGI HLT tables uncomment the following
#process.dimuonsHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","REDIGI38X")
#process.patTrigger.processName = "REDIGI38X"
#process.patTriggerEvent.processName = "REDIGI38X"
#process.patTrigger.triggerResults = cms.InputTag( "TriggerResults::REDIGI38X" )
#process.patTrigger.triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::REDIGI38X" )

