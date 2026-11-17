### HiForest Configuration
# Collisions: pp
# Type: Data
# Input: miniAOD

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_2024_ppRef_cff import Run3_2024_ppRef
process = cms.Process('HiForest', Run3_2024_ppRef)
process.options = cms.untracked.PSet()

#####################################################################################
# HiForest labelling info
#####################################################################################

process.load("HeavyIonsAnalysis.EventAnalysis.HiForestInfo_cfi")
# process.HiForestInfo.info = cms.vstring("HiForest, miniAOD, 150X, data")
process.HiForestInfo.info = cms.vstring("HiForest, miniAOD, 141X, data")

#####################################################################################
# Input source
#####################################################################################

process.source = cms.Source("PoolSource",
    duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = cms.untracked.vstring(
        #'/store/data/Run2024J/PPRefHardProbes3/MINIAOD/PromptReco-v1/000/387/409/00000/dae85526-3df2-42ee-809c-0d6d17d09e6b.root'
         '/store/data/Run2024J/PPRefHardProbes1/MINIAOD/PromptReco-v1/000/387/520/00000/1b53abb6-8910-44f2-ab43-b7f87085aa45.root',
        # '/store/data/Run2024J/PPRefZeroBiasPlusForward0/MINIAOD/PromptReco-v1/000/387/322/00000/936c40b6-61bf-491e-b050-bf06db4e6141.root',
        # '/store/data/Run2024J/PPRefZeroBiasPlusForward0/MINIAOD/PromptReco-v1/000/387/396/00000/159e5131-22f0-4dcb-b3bb-ad2303f0b0dd.root'
    )
)

# Number of events we want to process, -1 = all events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

#####################################################################################
# Load Global Tag, Geometry, etc.
#####################################################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

#GlobalTag used in Prompt RECO
#https://cms-conddb.cern.ch/cmsDbBrowser/list/Prod/gts/132X_dataRun3_Prompt_v3
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '141X_dataRun3_Prompt_v3', '')
process.HiForestInfo.GlobalTagLabel = process.GlobalTag.globaltag

# FIXME: Old calibration here, might need to update
# Commenting out until understood
#process.GlobalTag.toGet.extend([
#    cms.PSet(record = cms.string("BTagTrackProbability3DRcd"),
#             tag = cms.string("JPcalib_MC94X_2017pp_v2"),
#             connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
#
#         )
#      ])

#####################################################################################
# Define tree output
#####################################################################################

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("HiForestMiniAOD.root"))

#####################################################################################
# Additional Reconstruction and Analysis: Main Body
#####################################################################################

#############################
# Jets
#############################
process.load("HeavyIonsAnalysis.JetAnalysis.ak4PFJetSequence_ppref_data_cff")
#####################################################################################

############################
# Event Analysis
############################
# use data version to avoid PbPb MC
process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_data_cfi')
process.hiEvtAnalyzer.Vertex = cms.InputTag("offlineSlimmedPrimaryVertices")
process.hiEvtAnalyzer.doCentrality = cms.bool(False)
process.hiEvtAnalyzer.doEvtPlane = cms.bool(False)
process.hiEvtAnalyzer.doEvtPlaneFlat = cms.bool(False)
#Turn off MC info
process.hiEvtAnalyzer.doMC = cms.bool(False) # general MC info
process.hiEvtAnalyzer.doHiMC = cms.bool(False) # HI specific MC info
process.hiEvtAnalyzer.doHFfilters = cms.bool(False) # Disable HF filters for ppRef

process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.hltobject_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.l1object_cfi')

process.load('HeavyIonsAnalysis.EventAnalysis.skimanalysis_cfi')

process.load('HeavyIonsAnalysis.EventAnalysis.particleFlowAnalyser_cfi')

# FIXME: Do we have an updated trigger list?
#from HeavyIonsAnalysis.EventAnalysis.hltobject_cfi import trigger_list_data_2023_skimmed
#process.hltobject.triggerNames = trigger_list_data_2023_skimmed

#####################################################################################

#########################
# Track Analyzer
#########################
process.load('HeavyIonsAnalysis.TrackAnalysis.TrackAnalyzers_cff')
# process.ppTracks.trackEtaMax = cms.untracked.double(3.0)
# process.ppTracks.trackPtMin = cms.untracked.double(0.3)
#####################################################################################

#####################
# Event rho for the optional ppRef L1FastJet correction
#####################
# Keep the existing EEC input path ggHiNtuplizer/EventTree/rho.
# With object output disabled, this tree contains run, event, lumis and rho.
process.load('HeavyIonsAnalysis.EGMAnalysis.ggHiNtuplizer_cfi')
process.ggHiNtuplizer.rhoSrc = cms.InputTag("fixedGridRhoFastjetAll")
process.ggHiNtuplizer.doGenParticles = cms.bool(False)
process.ggHiNtuplizer.doSuperClusters = cms.bool(False)
process.ggHiNtuplizer.doElectrons = cms.bool(False)
process.ggHiNtuplizer.doPhotons = cms.bool(False)
process.ggHiNtuplizer.doMuons = cms.bool(False)
process.ggHiNtuplizer.doPfIso = cms.bool(False)
process.ggHiNtuplizer.useValMapIso = cms.bool(False)
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

####################################################################################

# muons
process.load("HeavyIonsAnalysis.MuonAnalysis.unpackedMuons_cfi")
process.unpackedMuons.muonSelectors = cms.vstring()
process.load("HeavyIonsAnalysis.MuonAnalysis.muonAnalyzer_cfi")

#########################
# ZDC RecHit Producer && Analyzer
#########################
# to prevent crash related to HcalSeverityLevelComputerRcd record
process.load("RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi")
process.load('HeavyIonsAnalysis.ZDCAnalysis.ZDCAnalyzersPP_cff')

###############################################################################

#########################
# Main analysis list
#########################

process.forest = cms.Path(
    process.HiForestInfo +
    process.hltanalysis *
    process.hiEvtAnalyzer +
#    process.hltobject +
    # process.l1object 
    process.ggHiNtuplizer +
    # process.zdcSequencePP +
    process.trackSequencePP
    # process.particleFlowAnalyser +
    # process.unpackedMuons +
    # process.muonAnalyzer
)

#####################################################################################
# Select the types of jets filled
matchJets = True             # Enables q/g and heavy flavor jet identification in MC 
jetPtMin = 15
jetAbsEtaMax = 2.5

# Choose which additional information is added to jet trees
doHIJetID = True             # Fill jet ID and composition information branches
doWTARecluster = True        # Add jet phi and eta for WTA axis
doBtagging = False           # EEC does not consume b-tag discriminator branches

# 0 means use original mini-AOD jets, otherwise use R value, e.g., 3,4,8
# Add all the values you want to process to the list
jetLabels = ["4", "8"]

# add candidate tagging for all selected jet radii
from HeavyIonsAnalysis.JetAnalysis.setupJets_ppRef_cff import candidateBtaggingMiniAOD

for jetLabel in jetLabels:
    candidateBtaggingMiniAOD(process, isMC = False, jetPtMin = jetPtMin, jetCorrLevels = ['L2Relative', 'L3Absolute'], doBtagging = doBtagging, labelR = jetLabel)

    # setup jet analyzer
    setattr(process,"ak"+jetLabel+"PFJetAnalyzer",process.ak4PFJetAnalyzer.clone())
    getattr(process,"ak"+jetLabel+"PFJetAnalyzer").jetTag = "selectedUpdatedPatJetsAK"+jetLabel+"PFCHSBtag"
    getattr(process,"ak"+jetLabel+"PFJetAnalyzer").jetName = 'ak'+jetLabel+'PF'
    getattr(process,"ak"+jetLabel+"PFJetAnalyzer").matchJets = matchJets
    getattr(process,"ak"+jetLabel+"PFJetAnalyzer").matchTag = 'patJetsAK'+jetLabel+'PFUnsubJets'
    getattr(process,"ak"+jetLabel+"PFJetAnalyzer").doBtagging = doBtagging
    getattr(process,"ak"+jetLabel+"PFJetAnalyzer").doHiJetID = doHIJetID
    getattr(process,"ak"+jetLabel+"PFJetAnalyzer").doWTARecluster = doWTARecluster
    getattr(process,"ak"+jetLabel+"PFJetAnalyzer").jetPtMin = jetPtMin
    getattr(process,"ak"+jetLabel+"PFJetAnalyzer").jetAbsEtaMax = cms.untracked.double(jetAbsEtaMax)
    getattr(process,"ak"+jetLabel+"PFJetAnalyzer").rParam = 0.4 if jetLabel=='0' else float(jetLabel)*0.1
    if doBtagging:
        getattr(process,"ak"+jetLabel+"PFJetAnalyzer").pfJetProbabilityBJetTag = cms.untracked.string("pfJetProbabilityBJetTagsAK"+jetLabel+"PFCHSBtag")
        getattr(process,"ak"+jetLabel+"PFJetAnalyzer").pfUnifiedParticleTransformerAK4JetTags = cms.untracked.string("pfUnifiedParticleTransformerAK4JetTagsAK"+jetLabel+"PFCHSBtag")
    process.forest += getattr(process,"ak"+jetLabel+"PFJetAnalyzer")


# Schedule definition
process.pAna = cms.EndPath(process.skimanalysis)

# process.primaryVertexFilter = cms.EDFilter("VertexSelector",
#     src = cms.InputTag("offlineSlimmedPrimaryVertices"),
#     cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), #in miniADO trackSize()==0, however there is no influence.
#     filter = cms.bool(True), # otherwise it won't filter the event
# )

from HeavyIonsAnalysis.TrackAnalysis.unpackedTracksAndVertices_cfi import *
process.unpackedTracksAndVertices = unpackedTracksAndVertices
process.noscraping = cms.EDFilter("FilterOutScraping",
    applyfilter = cms.untracked.bool(True),
    debugOn = cms.untracked.bool(False),
    numtrack = cms.untracked.uint32(10),
    thresh = cms.untracked.double(0.25),
    src = cms.untracked.InputTag("unpackedTracksAndVertices") # generalTracks collection not found
)


process.ppNoScrapingSequence = cms.Sequence(
    process.unpackedTracksAndVertices +
    process.noscraping)
process.ppNoScrapingFilter = cms.Path(process.ppNoScrapingSequence)

process.load('HeavyIonsAnalysis.EventAnalysis.hffilterPF_cfi')
# process.load('HeavyIonsAnalysis.EventAnalysis.hffilter_cfi')
process.OOphfCoincFilterPF2Th4 = cms.Path(process.phfCoincFilterPF2Th4)
process.load('HeavyIonsAnalysis.EventAnalysis.collisionEventSelection_cff')
process.pclusterCompatibilityFilter = cms.Path(process.clusterCompatibilityFilter)
process.pprimaryVertexFilter = cms.Path(process.primaryVertexFilter)

# Save the EEC input contract directly during forest production.
from HeavyIonsAnalysis.Configuration.eecSlimContent_cff import configureEECSlimForest
configureEECSlimForest(process, isMC=False, isOO=False)
