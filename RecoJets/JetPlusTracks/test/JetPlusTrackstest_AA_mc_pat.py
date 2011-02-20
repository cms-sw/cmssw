# Auto generated configuration file
# using: 
# Revision: 1.232.2.6 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step2 -s RAW2DIGI,L1Reco,RECO,ALCA:HcalCalMinBias --datatier RECO --eventcontent ALCARECO --geometry DB --conditions=FrontierConditions_GlobalTag,GR_R_39X_V2::All --filein=/store/hidata/HIRun2010/HICorePhysics/RAW/v1/000/150/476/7AD12282-DDEB-DF11-9574-001D09F2AD4D.root -n 10 --scenario HeavyIons --data
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO2')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
#process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')
#process.load("HeavyIonsAnalysis.Configuration.collisionEventSelection_cff")

from RecoJets.JetPlusTracks.JetPlusTrackCorrectionsAA_cff import *
JetPlusTrackZSPCorJetIconePu5.UseZSP = cms.bool(False)
JetPlusTrackZSPCorJetAntiKtPu5.UseZSP = cms.bool(False)

process.load("RecoJets.Configuration.RecoJPTJetsHIC_cff")

#process.load('PhysicsTools.PatAlgos.patHeavyIonSequences_cff')
#from PhysicsTools.PatAlgos.tools.heavyIonTools import *
##configureHeavyIons(process)

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.232.2.6 $'),
    annotation = cms.untracked.string('step2 nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
##    input = cms.untracked.int32(10)
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#    '/store/hidata/HIRun2010/HICorePhysics/RECO/PromptReco-v3/000/151/353/F0C3D4E3-37F2-DF11-9285-003048D2BCA2.root'
#     'file:/tmp/irina/Pat_hiGoodMergedTracks_v1_56.root'
     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_103.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_240.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_104.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_105.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_113.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_114.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_122.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_123.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_132.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_133.root'
##       'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_141.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_142.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_150.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_151.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_16.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_160.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_161.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_17.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_170.root'
###     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_18.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_199.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_2.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_203.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_204.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_212.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_213.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_222.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_223.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_231.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_232.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_240.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_241.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_25.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_250.root'
####---     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_251.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_26.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_260.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_27.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_28.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_3.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_35.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_36.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_37.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_4.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_44.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_45.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_46.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_53.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_54.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_55.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_63.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_64.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_65.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_72.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_73.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_74.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_81.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_82.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_83.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_90.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_91.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_92.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_93.root'
##     'file:/tmp/irina/Pat_hiGoodMergedTracks_VtxPatch_v1_1.root'
    )
)

process.options = cms.untracked.PSet(

)

process.HeavyIonGlobalParameters = cms.PSet(
    centralitySrc = cms.InputTag("hiCentrality"),
    centralityVariable = cms.string("HFhits"),
    nonDefaultGlauberModel = cms.string("")
    )
from CmsHi.Analysis2010.CommonFunctions_cff import *
overrideCentrality(process)

# Other statements
#process.GlobalTag.globaltag = 'GR_R_39X_V4::All'

process.GlobalTag.globaltag = 'GR10_P_V12::All'

# Path and EndPath definitions

#### Choose techical bits 40 or 41 and coincidence with BPTX (0)
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('35 OR 40')
####
###process.eventsel = cms.Path(process.collisionEventSelection)

process.primaryVertexFilter = cms.EDFilter("VertexSelector",
    src = cms.InputTag("hiSelectedVertex"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events, instead making an empty vertex collection
)

process.myjetplustrack = cms.EDAnalyzer("PatJetPlusTrackAnalysis",
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_56_WithTrackQualityCut.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_56.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_103.root'),
    HistOutFile = cms.untracked.string('JetAnalysis_v1_103_WithTrackQualityCut.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_240.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_104.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_105.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_113.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_114.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_122.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_123.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_132.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_133.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_141.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_142.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_150.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_151.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_16.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_160.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_161.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_17.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_170.root'),
###    HistOutFile = cms.untracked.string('JetAnalysis_v1_18.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_199.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_2.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_203.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_204.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_212.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_213.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_222.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_223.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_231.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_232.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_240.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_241.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_25.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_250.root'),
####    HistOutFile = cms.untracked.string('JetAnalysis_v1_251.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_26.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_260.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_27.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_28.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_3.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_35.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_36.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_37.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_4.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_44.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_45.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_46.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_53.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_54.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_55.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_63.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_64.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_65.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_72.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_73.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_74.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_81.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_82.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_83.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_90.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_91.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_92.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_93.root'),
#    HistOutFile = cms.untracked.string('JetAnalysis_v1_1.root'),


    src2 = cms.InputTag("iterativeCone5HiGenJets"),
    src22 = cms.InputTag("iterativeCone5HiGenJets"),
    src3 = cms.InputTag("jpticPu5patJets"),
    src4 = cms.InputTag("JetPlusTrackZSPCorJetIconePu5"),
    src1 = cms.InputTag("iterativeConePu5CaloJets"),
    src11 = cms.InputTag("iterativeConePu5CaloJets"),
    Data = cms.int32(0),
    jetsID = cms.string('ak5JetID'),
    jetsID2 = cms.string('ak5JetID'),
    Cone1 = cms.double(0.5),
    Cone2 = cms.double(0.5),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HFRecHitCollectionLabel = cms.InputTag("hfreco"),
    HORecHitCollectionLabel = cms.InputTag("horeco"),
    HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
##    inputTrackLabel = cms.untracked.string('hiSelectedTracks')
    inputTrackLabel = cms.untracked.string('hiGoodMergedTracks')
)


process.p1 = cms.Path(process.myjetplustrack)
process.jptbg = cms.Path(process.recoJPTJetsHIC)
#process.p2 = cms.Path(process.hltLevel1GTSeed)
#process.p2 = cms.Path(process.primaryVertexFilter)

# Schedule definition
#process.schedule = cms.Schedule(process.p2,process.jptbg,process.p1)
process.schedule = cms.Schedule(process.jptbg,process.p1)
