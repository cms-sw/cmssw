### HiForest Configuration
# Input: miniAOD
# Type: mc

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_pp_on_PbPb_2026_cff import Run3_pp_on_PbPb_2026
process = cms.Process('HiForest', Run3_pp_on_PbPb_2026)

###############################################################################

# HiForest info
process.load("HeavyIonsAnalysis.EventAnalysis.HiForestInfo_cfi")
process.HiForestInfo.info = cms.vstring("HiForest, miniAOD, 161X, mc")

###############################################################################

# input files
process.source = cms.Source("PoolSource",
    duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = cms.untracked.vstring(
        'root://cmsxrootd.fnal.gov//store/user/fdamas/PbPb2026/RunPrepMC/HydjetMB_1610pre3/PATwith161pre4_151X_mcRun3_2025_realistic_HI_v5/260416_175046/0000/step4_PAT_1.root'
    ),
)

# number of events to process, set to -1 to process all events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
    )

###############################################################################

# load Global Tag, geometry, etc.
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '151X_mcRun3_2025_realistic_HI_v5', '')
process.HiForestInfo.GlobalTagLabel = process.GlobalTag.globaltag
process.GlobalTag.snapshotTime = cms.string("9999-12-31 23:59:59.000")
process.GlobalTag.toGet.extend([
    cms.PSet(record = cms.string("BTagTrackProbability3DRcd"),
             tag = cms.string("JPcalib_MC103X_2018PbPb_v4"),
             connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
         )
])

# Define centrality binning
process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")
process.centralityBin.Centrality = cms.InputTag("hiCentrality")
process.centralityBin.centralityVariable = cms.string("HFtowers")

###############################################################################

# root output
process.TFileService = cms.Service("TFileService",
    fileName = cms.string("HiForestMiniAODMC.root"))

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

#############################
# Gen Analyzer
#############################
process.load('HeavyIonsAnalysis.EventAnalysis.HiGenAnalyzer_cfi')
#process.HiGenParticleAna.ptMin = cms.untracked.double(0.7) # default is 5
#process.HiGenParticleAna.etaMax = cms.untracked.double(2.6) # default is 2.5

# event analysis
process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.particleFlowAnalyser_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_mc_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.skimanalysis_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.hltobject_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.l1object_cfi')

from HeavyIonsAnalysis.EventAnalysis.hltobject_cfi import trigger_list_data_2025
process.hltobject.triggerNames = trigger_list_data_2025

################################
# electrons, photons, muons
process.load('HeavyIonsAnalysis.EGMAnalysis.ggHiNtuplizer_cfi')
process.ggHiNtuplizer.doGenParticles = cms.bool(True)
process.ggHiNtuplizer.doMuons = cms.bool(False)
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
################################
# jet reco sequence
process.load('HeavyIonsAnalysis.JetAnalysis.akCs4PFJetSequence_pponPbPb_mc_cff')
################################
# tracks
process.load("HeavyIonsAnalysis.TrackAnalysis.TrackAnalyzers_cff")
#muons
process.load("HeavyIonsAnalysis.MuonAnalysis.unpackedMuons_cfi")
process.load("HeavyIonsAnalysis.MuonAnalysis.muonAnalyzer_cfi")
process.muonAnalyzer.doGen = cms.bool(True)

###############################################################################

#########################                                                                                                                                                 
# ZDC RecHit Producer && Analyzer                                                                                                                                         
#########################                                                                                                                                                 
# to prevent crash related to HcalSeverityLevelComputerRcd record                                                                                                         
process.load("RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi")
process.load('HeavyIonsAnalysis.ZDCAnalysis.ZDCAnalyzersPbPb_cff')

###############################################################################
# main forest sequence
process.forest = cms.Path(
    process.HiForestInfo +
    process.centralityBin +
    process.hltanalysis +
#    process.hltobject +
#    process.l1object +
    process.trackSequencePbPb +
#    process.particleFlowAnalyser +
    process.hiEvtAnalyzer +
    process.HiGenParticleAna +
    process.ggHiNtuplizer +
    process.zdcSequencePbPb
#    process.unpackedMuons +
#    process.muonAnalyzer
    )

#customisation

# Select the types of jets filled
matchJets = True             # Enables q/g and heavy flavor jet identification in MC
jetPtMin = 15
jetAbsEtaMax = 2.5

# Choose which additional information is added to jet trees
doHIJetID = True             # Fill jet ID and composition information branches
doWTARecluster = False        # Add jet phi and eta for WTA axis
doBtagging  =  False         # Note that setting to True increases computing time a lot

# Configuration for jet flow subtraction
iterativeFlow = True         # Iterative jetty region exclusion. Default = True
pfCandidateEtaCut = 2        # Eta range for PF candidates used in flow fit. Default = 2
minPfCandidatesPerEvent = 60 # Minimum number of PF candidates to make the flow fit. Default = 60
minPfCandidatePt = 0.3       # Minimum pT for PF candidates in flow fit. Default = 0.3
maxPfCandidatePt = 3         # Maximum pT for PF candidates in flow fit. Default = 3
minFitQuality = 0            # Minimum flow fit quality score. Default = 0
maxFitQuality = 1            # Maximum flow fit quality score. Default = 1
firstFittedVn = 2            # First fitted vn component. Default = 2
lastFittedVn = 3             # Last fitted vn component. Default = 3

# 0 means use original mini-AOD jets, otherwise use R value, e.g., 3,4,8
# Generator level jets in original miniAOD jets contain neutrinos
# You will need to do reclustering with R-value to get generator level jets without neutrinos
# Add all the values you want to process to the list
# These will create collections of CS subtracted jets (only eta dependent background)
jetLabelsCS = ["4"]

# For this list, give the R-values for flow subtracted CS jets (eta and phi dependent background)
jetLabelsFlowCS = ["4"]

# Combine the two lists such that all selected jets can be easily looped over
# Also add "Flow" tag for the flow jets to distinguish them from non-flow jets
allJetLabels = jetLabelsCS + [flowR + "Flow" for flowR in jetLabelsFlowCS]

# add candidate tagging, copy/paste to add other jet radii
from HeavyIonsAnalysis.JetAnalysis.setupJets_PbPb_cff import candidateBtaggingMiniAOD

for jetLabel in allJetLabels:
    candidateBtaggingMiniAOD(process, isMC = True, jetPtMin = jetPtMin, jetCorrLevels = ['L2Relative', 'L3Absolute'], doBtagging = doBtagging, labelR = jetLabel)

    # setup jet analyzer
    setattr(process,"akCs"+jetLabel+"PFJetAnalyzer",process.akCs4PFJetAnalyzer.clone())
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").jetTag =  "selectedUpdatedPatJetsAK"+jetLabel+"PFBtag"
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").jetName = 'akCs'+jetLabel+'PF'
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").matchJets = matchJets
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").matchTag = 'patJetsAK'+jetLabel+'PFUnsubJets'
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").doBtagging = doBtagging
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").doHiJetID = doHIJetID
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").doWTARecluster = doWTARecluster
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").jetPtMin = jetPtMin
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").jetAbsEtaMax = cms.untracked.double(jetAbsEtaMax)
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").rParam = 0.4 if jetLabel=="0" else  float(jetLabel.replace("Flow",""))*0.1
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").jetFlavourInfos = "ak"+jetLabel+"PFUnsubJetFlavourInfos"
    if jetLabel != "0": getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").genjetTag = "ak"+jetLabel+"GenJetsReclusterNoNu"
    if doBtagging:
        getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").pfJetProbabilityBJetTag = cms.untracked.string("pfJetProbabilityBJetTagsAK"+jetLabel+"PFBtag") 
        getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").pfUnifiedParticleTransformerAK4JetTags = cms.untracked.string("pfUnifiedParticleTransformerAK4JetTagsAK"+jetLabel+"PFBtag")
    process.forest += getattr(process,"akCs"+jetLabel+"PFJetAnalyzer")

# Configuration for the flow fit
for jetLabel in [flowR + "Flow" for flowR in jetLabelsFlowCS]:

    getattr(process, "rhoModulationAkCs"+jetLabel+"PFJets").pfCandidateEtaCut = pfCandidateEtaCut
    getattr(process, "rhoModulationAkCs"+jetLabel+"PFJets").minPfCandidatesPerEvent = minPfCandidatesPerEvent
    getattr(process, "rhoModulationAkCs"+jetLabel+"PFJets").firstFittedVn = firstFittedVn
    getattr(process, "rhoModulationAkCs"+jetLabel+"PFJets").lastFittedVn = lastFittedVn
    getattr(process, "rhoModulationAkCs"+jetLabel+"PFJets").pfCandidateMinPtCut = minPfCandidatePt
    getattr(process, "rhoModulationAkCs"+jetLabel+"PFJets").pfCandidateMaxPtCut = maxPfCandidatePt
    getattr(process, "akCs"+jetLabel+"PFJets").minFlowChi2Prob = minFitQuality
    getattr(process, "akCs"+jetLabel+"PFJets").maxFlowChi2Prob = maxFitQuality

    if iterativeFlow:
        getattr(process, "rhoModulationIterAkCs"+jetLabel+"PFJets").pfCandidateEtaCut = pfCandidateEtaCut
        getattr(process, "rhoModulationIterAkCs"+jetLabel+"PFJets").minPfCandidatesPerEvent = minPfCandidatesPerEvent
        getattr(process, "rhoModulationIterAkCs"+jetLabel+"PFJets").firstFittedVn = firstFittedVn
        getattr(process, "rhoModulationIterAkCs"+jetLabel+"PFJets").lastFittedVn = lastFittedVn
        getattr(process, "rhoModulationIterAkCs"+jetLabel+"PFJets").pfCandidateMinPtCut = minPfCandidatePt
        getattr(process, "rhoModulationIterAkCs"+jetLabel+"PFJets").pfCandidateMaxPtCut = maxPfCandidatePt

#########################
# Event Selection -> add the needed filters here
#########################

process.load('HeavyIonsAnalysis.EventAnalysis.collisionEventSelection_cff')
process.pclusterCompatibilityFilter = cms.Path(process.clusterCompatibilityFilter)
process.pprimaryVertexFilter = cms.Path(process.primaryVertexFilter)
process.load('HeavyIonsAnalysis.EventAnalysis.hffilter_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.hffilterPF_cfi')
process.pAna = cms.EndPath(process.skimanalysis)
