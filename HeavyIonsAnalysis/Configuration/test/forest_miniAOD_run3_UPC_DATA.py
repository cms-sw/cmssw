### HiForest Configuration
# Input: miniAOD
# Type: data

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_2025_UPC_cff import Run3_2025_UPC
process = cms.Process('HiForest', Run3_2025_UPC)

###############################################################################

# HiForest info
process.load("HeavyIonsAnalysis.EventAnalysis.HiForestInfo_cfi")
process.HiForestInfo.info = cms.vstring("HiForest, miniAOD, 151X, data")

# import subprocess, os
# version = subprocess.check_output(
#     ['git', '-C', os.path.expandvars('$CMSSW_BASE/src'), 'describe', '--tags'])
# if version == '':
#     version = 'no git info'
# process.HiForestInfo.HiForestVersion = cms.string(version)

###############################################################################

# input files
process.source = cms.Source("PoolSource",
    duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = cms.untracked.vstring(
        '/store/hidata/HIRun2025A/HIForward9/MINIAOD/PromptReco-v1/000/400/401/00000/17810c72-10b7-434a-a587-b23cf3df10eb.root'
    ), 
)

# number of events to process, set to -1 to process all events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
    )

###############################################################################

# load Global Tag, geometry, etc.
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')


from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag, '151X_dataRun3_Prompt_v1', '')
process.HiForestInfo.GlobalTagLabel = process.GlobalTag.globaltag

###############################################################################

# root output
process.TFileService = cms.Service("TFileService",
    fileName = cms.string("HiForestMiniAOD.root"))

# # edm output for debugging purposes
# process.output = cms.OutputModule(
#     "PoolOutputModule",
#     fileName = cms.untracked.string('HiForestEDM.root'),
#     outputCommands = cms.untracked.vstring(
#         'keep *',
#         )
#     )

# process.output_path = cms.EndPath(process.output)

###############################################################################

# event analysis
process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_data_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.skimanalysis_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.hltobject_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.l1object_cfi')

# No centrality binning for UPC
process.hiEvtAnalyzer.doCentrality = cms.bool(False)
process.hiEvtAnalyzer.doHFfilters = cms.bool(False)

# FIXME: Do we have an updated trigger list?
#from HeavyIonsAnalysis.EventAnalysis.hltobject_cfi import trigger_list_data_2023_skimmed
#process.hltobject.triggerNames = trigger_list_data_2023_skimmed
process.hltobject.triggerNames = cms.vstring()

process.load('HeavyIonsAnalysis.EventAnalysis.particleFlowAnalyser_cfi')
################################
# electrons, photons, muons
process.load('HeavyIonsAnalysis.EGMAnalysis.ggHiNtuplizer_cfi')
process.ggHiNtuplizer.doGenParticles = cms.bool(False)
process.ggHiNtuplizer.doMuons = cms.bool(False)
process.ggHiNtuplizer.useValMapIso = cms.bool(False) # True here causes seg fault
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
################################
# jet reco sequence
process.load('HeavyIonsAnalysis.JetAnalysis.ak4PFJetSequence_ppref_data_cff')
################################
# tracks
process.load("HeavyIonsAnalysis.TrackAnalysis.TrackAnalyzers_cff")
process.ppTracks.dedxEstimators = cms.VInputTag(["dedxEstimator:dedxAllLikelihood"])
process.ppTracks.trackEtaMax = cms.untracked.double(3.0)
# muons (FTW)
process.load("HeavyIonsAnalysis.MuonAnalysis.unpackedMuons_cfi")
process.unpackedMuons.muonSelectors = cms.vstring()
process.load("HeavyIonsAnalysis.MuonAnalysis.muonAnalyzer_cfi")
process.unpackedMuons.muonSelectors = cms.vstring()
###############################################################################

#########################
# ZDC RecHit Producer && Analyzer
#########################
# to prevent crash related to HcalSeverityLevelComputerRcd record
process.load("RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi")
#process.load('HeavyIonsAnalysis.ZDCAnalysis.PPSAnalyzers_cff')
#process.load('HeavyIonsAnalysis.ZDCAnalysis.ZDCAnalyzersPbPb_cff')
process.load('HeavyIonsAnalysis.ZDCAnalysis.ZDCAnalyzersPP_cff')

###############################################################################
# main forest sequence
process.forest = cms.Path(
    process.HiForestInfo +
    process.hiEvtAnalyzer +
    process.hltanalysis +
    #process.hltobject +
    process.l1object +
    process.trackSequencePP +
    process.particleFlowAnalyser +
    process.ggHiNtuplizer +
    process.zdcSequencePP +
    #process.ppsSequence +
    process.unpackedMuons +
    process.muonAnalyzer
    )

#customisation

#####################################################################################
# Select the types of jets filled
matchJets = True             # Enables q/g and heavy flavor jet identification in MC
jetPtMin = 15
jetAbsEtaMax = 2.5

# Choose which additional information is added to jet trees
doHIJetID = True             # Fill jet ID and composition information branches
doWTARecluster = False        # Add jet phi and eta for WTA axis
doBtagging  =  False         # Note that setting to True increases computing time a lot

# 0 means use original mini-AOD jets, otherwise use R value, e.g., 3,4,8
# Add all the values you want to process to the list
jetLabels = ["0"]

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


#########################
# Event Selection -> add the needed filters here
#########################

process.load('HeavyIonsAnalysis.EventAnalysis.collisionEventSelection_cff')
process.pclusterCompatibilityFilter = cms.Path(process.clusterCompatibilityFilter)
process.pprimaryVertexFilter = cms.Path(process.primaryVertexFilter)
process.load('HeavyIonsAnalysis.EventAnalysis.hffilterPF_cfi')
process.load('HeavyIonsAnalysis.ZDCAnalysis.HiZDCfilter_cfi')
process.pAna = cms.EndPath(process.skimanalysis)

# from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel
# process.hltfilter = hltHighLevel.clone(
#    HLTPaths = [
#        "HLT_HIMinimumBias_v2",
#    ]
# )
# process.filterSequence = cms.Sequence(
#     process.hltfilter *
#     process.primaryVertexFilter *
#     (process.zdcrecoRun3 + process.zdcEnergyFilter0nOr)
# )

# process.superFilterPath = cms.Path(process.filterSequence)
# process.skimanalysis.superFilters = cms.vstring("superFilterPath")

# for path in process.paths:
#    getattr(process, path)._seq = process.filterSequence * getattr(process,path)._seq

