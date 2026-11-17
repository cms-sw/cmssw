### HiForest Configuration
# Input: miniAOD
# Type: data

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_2025_OXY_cff import Run3_2025_OXY
process = cms.Process('HiForest', Run3_2025_OXY)

###############################################################################

# HiForest info
process.load("HeavyIonsAnalysis.EventAnalysis.HiForestInfo_cfi")
#process.HiForestInfo.info = cms.vstring("HiForest, miniAOD, 150X, data")

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
        # '/store/hidata/OORun2025/IonPhysics0/MINIAOD/PromptReco-v1/000/394/154/00000/14792428-42d1-4d08-9578-eed4891a4594.root'
        # '/store/hidata/HIRun2024A/HIMinimumBias0/MINIAOD/PromptReco-v1/000/387/757/00000/b5701796-9e14-4abd-9645-dd8569c47d94.root'
        '/store/hidata/OORun2025/IonPhysics0/MINIAOD/PromptReco-v1/000/394/175/00000/c44c983c-4ce8-4a59-9670-9e51587a10c3.root'
    ), 
)

# number of events to process, set to -1 to process all events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
    )

###############################################################################

# load Global Tag, geometry, etc.
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '150X_dataRun3_Prompt_v1', '')
process.HiForestInfo.GlobalTagLabel = process.GlobalTag.globaltag

## --> only use this starting from 388000
process.es_prefer = cms.ESPrefer('HcalTextCalibrations','es_ascii')
process.es_ascii = cms.ESSource('HcalTextCalibrations',
   input = cms.VPSet(
      cms.PSet(
         object = cms.string('Gains'),
         file   = cms.FileInPath('HeavyIonsAnalysis/Configuration/data/ZDCConditions_1400V/DumpGainsForUpload_AllChannels.txt')
      ),
      cms.PSet(
        object = cms.string('TPChannelParameters'),
        file   = cms.FileInPath('HeavyIonsAnalysis/Configuration/data/ZDCConditions_1400V/DumpTPChannelParameters_Run387473.txt')
      ),
   )
)
## <--
###############################################################################

# Define centrality binning
process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")
process.centralityBin.Centrality = cms.InputTag("hiCentrality")
process.centralityBin.centralityVariable = cms.string("HFtowers")

###############################################################################

# root output
process.TFileService = cms.Service("TFileService",
    fileName = cms.string("L1_object_data.root"))

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
process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_data_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.skimanalysis_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.hltobject_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.l1object_cfi')



#process.hiEvtAnalyzer.doCentrality = cms.bool(False)
process.hiEvtAnalyzer.doHFfilters = cms.bool(False)

# FIXME: Do we have an updated trigger list?
#from HeavyIonsAnalysis.EventAnalysis.hltobject_cfi import trigger_list_data_2023_skimmed
#process.hltobject.triggerNames = trigger_list_data_2023_skimmed

process.load('HeavyIonsAnalysis.EventAnalysis.particleFlowAnalyser_cfi')
################################
# electrons, photons, muons
process.load('HeavyIonsAnalysis.EGMAnalysis.ggHiNtuplizer_cfi')
process.ggHiNtuplizer.doMuons = cms.bool(False)
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
################################
# jet reco sequence
process.load('HeavyIonsAnalysis.JetAnalysis.akCs4PFJetSequence_pponPbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.akPu4CaloJetSequence_pponPbPb_data_cff')
process.akPu4CaloJetAnalyzer.doHiJetID = True
################################
# tracks
process.load("HeavyIonsAnalysis.TrackAnalysis.TrackAnalyzers_cff")
# muons (FTW)
process.load("HeavyIonsAnalysis.MuonAnalysis.unpackedMuons_cfi")
process.load("HeavyIonsAnalysis.MuonAnalysis.muonAnalyzer_cfi")
###############################################################################

#########################
# ZDC RecHit Producer && Analyzer
#########################
# to prevent crash related to HcalSeverityLevelComputerRcd record
process.load("RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi")
process.load('HeavyIonsAnalysis.ZDCAnalysis.ZDCAnalyzersPbPb_cff')
process.load('HeavyIonsAnalysis.ZDCAnalysis.FSCAnalyzers_cff')


from HeavyIonsAnalysis.TrackAnalysis.unpackedTracksAndVertices_cfi import *
process.unpackedTracksAndVertices = unpackedTracksAndVertices


###############################################################################
# main forest sequence
process.forest = cms.Path(
    process.HiForestInfo +
    process.centralityBin +
    process.hiEvtAnalyzer +
    process.hltanalysis +
    # process.hltobject +
    # process.l1object
    process.trackSequencePbPb #+
    #process.particleFlowAnalyser    #+
    # process.ggHiNtuplizer +
    # process.zdcSequencePbPb +
    # process.fscSequence +
    # process.unpackedTracksAndVertices +
    # process.unpackedMuons +
    # process.muonAnalyzer 
    # process.akPu4CaloJetAnalyzer
    )

#customisation

# Select the types of jets filled
matchJets = False             # Enables q/g and heavy flavor jet identification in MC 
jetPtMin = 15
jetAbsEtaMax = 2.5

# Choose which additional information is added to jet trees
doHIJetID = True             # Fill jet ID and composition information branches
doWTARecluster = True        # Add jet phi and eta for WTA axis
doBtagging = False           # EEC does not consume b-tag discriminator branches

# 0 means use original mini-AOD jets, otherwise use R value, e.g., 3,4,8
# Add all the values you want to process to the list
# These will create collections of CS subtracted jets (only eta dependent background)
jetLabelsCS = ["4", "8"]

# For this list, give the R-values for flow subtracted CS jets (eta and phi dependent background)
jetLabelsFlowCS = []

# Combine the two lists such that all selected jets can be easily looped over
# Also add "Flow" tag for the flow jets to distinguish them from non-flow jets
allJetLabels = jetLabelsCS + [flowR + "Flow" for flowR in jetLabelsFlowCS]

# add candidate tagging
from HeavyIonsAnalysis.JetAnalysis.setupJets_PbPb_cff import candidateBtaggingMiniAOD

for jetLabel in allJetLabels:
    candidateBtaggingMiniAOD(process, isMC = False, jetPtMin = jetPtMin, jetCorrLevels = ['L2Relative', 'L2L3Residual'], doBtagging = doBtagging, labelR = jetLabel)

    # setup jet analyzer
    setattr(process,"akCs"+jetLabel+"PFJetAnalyzer",process.akCs4PFJetAnalyzer.clone())
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").jetTag = "selectedUpdatedPatJetsAK"+jetLabel+"PFBtag"
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").jetName = 'akCs'+jetLabel+'PF'
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").matchJets = matchJets
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").matchTag = 'patJetsAK'+jetLabel+'PFUnsubJets'
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").doBtagging = doBtagging
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").doHiJetID = doHIJetID
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").doWTARecluster = doWTARecluster
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").jetPtMin = jetPtMin
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").jetAbsEtaMax = cms.untracked.double(jetAbsEtaMax)
    getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").rParam = 0.4 if jetLabel=="0" else float(jetLabel.replace("Flow",""))*0.1
    if doBtagging:
        getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").pfJetProbabilityBJetTag = cms.untracked.string("pfJetProbabilityBJetTagsAK"+jetLabel+"PFBtag")
        getattr(process,"akCs"+jetLabel+"PFJetAnalyzer").pfUnifiedParticleTransformerAK4JetTags = cms.untracked.string("pfUnifiedParticleTransformerAK4JetTagsAK"+jetLabel+"PFBtag")
    process.forest += getattr(process,"akCs"+jetLabel+"PFJetAnalyzer")


#########################
# Event Selection -> add the needed filters here
#########################

process.load('HeavyIonsAnalysis.EventAnalysis.collisionEventSelection_cff')
process.pclusterCompatibilityFilter = cms.Path(process.clusterCompatibilityFilter)
process.pprimaryVertexFilter = cms.Path(process.primaryVertexFilter)

from HeavyIonsAnalysis.TrackAnalysis.unpackedTracksAndVertices_cfi import *
process.unpackedTracksAndVertices = unpackedTracksAndVertices
process.load('HeavyIonsAnalysis.VertexAnalysis.pileupvertexfilter_cfi')
process.pileupvertexfilter.doOO = True
process.pileupvertexfilter.doNeNe = False

process.PAcollisionEventSelection = cms.Sequence(
    # process.phfCoincFilterPF2Th4 *
    # process.PAprimaryVertexFilter *
    # process.clusterCompatibilityFilter *
    process.unpackedTracksAndVertices *
    process.pileupvertexfilter
    )

process.pileupVertexFilter = cms.Path(process.PAcollisionEventSelection)
process.load('HeavyIonsAnalysis.EventAnalysis.hffilterPF_cfi')
# process.load('HeavyIonsAnalysis.EventAnalysis.hffilter_cfi')
process.OOpfCoincFilterPF2Th4 = cms.Path(process.phfCoincFilterPF2Th4)
process.OOpfCoincFilterPF2Th10 = cms.Path(process.phfCoincFilterPF2Th10) #bug!!!!!
process.OOpfCoincFilterPF2Th13 = cms.Path(process.phfCoincFilterPF2Th13) #bug!!!!
# process.pphfCoincFilter4Th2 = cms.Path(process.phfCoincFilter4Th2)
# process.pphfCoincFilter1Th3 = cms.Path(process.phfCoincFilter1Th3)
# process.pphfCoincFilter2Th3 = cms.Path(process.phfCoincFilter2Th3)
# process.pphfCoincFilter3Th3 = cms.Path(process.phfCoincFilter3Th3)
# process.pphfCoincFilter4Th3 = cms.Path(process.phfCoincFilter4Th3)
# process.pphfCoincFilter5Th3 = cms.Path(process.phfCoincFilter5Th3)
# process.pphfCoincFilter1Th4 = cms.Path(process.phfCoincFilter1Th4)
# process.pphfCoincFilter2Th4 = cms.Path(process.phfCoincFilter2Th4)
# process.pphfCoincFilter3Th4 = cms.Path(process.phfCoincFilter3Th4)
# process.pphfCoincFilter4Th4 = cms.Path(process.phfCoincFilter4Th4)
# process.pphfCoincFilter5Th4 = cms.Path(process.phfCoincFilter5Th4)
# process.pphfCoincFilter1Th5 = cms.Path(process.phfCoincFilter1Th5)
# process.pphfCoincFilter2Th5 = cms.Path(process.phfCoincFilter2Th5)
# process.pphfCoincFilter3Th5 = cms.Path(process.phfCoincFilter3Th5)
# process.pphfCoincFilter4Th5 = cms.Path(process.phfCoincFilter4Th5)
# process.pphfCoincFilter5Th5 = cms.Path(process.phfCoincFilter5Th5)
process.pAna = cms.EndPath(process.skimanalysis)

from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel
process.hltfilter = hltHighLevel.clone(
   HLTPaths = [
       #"HLT_HIZeroBias_v4",
       # "HLT_MinimumBiasHF_OR_BptxAND_v*"
       "HLT_MinimumBiasHF_OR_BptxAND_v*",
       # "HLT_OxyL1SingleJet*"
       # "HLT_HIMinimumBias_v*",
   ]
)
process.filterSequence = cms.Sequence(
   process.hltfilter
)

process.superFilterPath = cms.Path(process.filterSequence)
process.skimanalysis.superFilters = cms.vstring("superFilterPath")
#
for path in process.paths:
   getattr(process, path)._seq = process.filterSequence * getattr(process,path)._seq

process.MessageLogger.cerr.FwkReport.reportEvery = 300

# Save the EEC input contract directly during forest production.
from HeavyIonsAnalysis.Configuration.eecSlimContent_cff import configureEECSlimForest
configureEECSlimForest(process, isMC=False, isOO=True)
