### HiForest Configuration
# Collisions: pp
# Type: Data
# Input: AOD

import FWCore.ParameterSet.Config as cms
process = cms.Process('HiForest')
process.options = cms.untracked.PSet()

#####################################################################################
# HiForest labelling info
#####################################################################################

process.load("HeavyIonsAnalysis.JetAnalysis.HiForest_cff")
process.HiForest.inputLines = cms.vstring("HiForest V3",)
import subprocess
version = subprocess.Popen(["(cd $CMSSW_BASE/src && git describe --tags)"], stdout=subprocess.PIPE, shell=True).stdout.read()
if version == '':
    version = 'no git info'
process.HiForest.HiForestVersion = cms.string(version)

#####################################################################################
# Input source
#####################################################################################

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                                "file:/afs/cern.ch/user/m/mverweij/work/soft/forest/cs3/test/data/failEvent/pickevents.root"
#                                "file:/afs/cern.ch/user/m/mverweij/work/soft/forest/test/samples/PbPb_DATA_AOD.root"
#                                "/store/group/phys_heavyions/velicanu/reco/HIPhysicsMinBiasUPC/v0/000/262/548/recoExpress_84.root"
                            )
)


# Number of events we want to process, -1 = all events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1))


#####################################################################################
# Load Global Tag, Geometry, etc.
#####################################################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

#process.MessageLogger = cms.Service(
#    "MessageLogger",
#    destinations = cms.untracked.vstring(
#        'detailedInfo',
#         'critical'
#         ),
#    detailedInfo = cms.untracked.PSet(
#        threshold  = cms.untracked.string('DEBUG') 
#         ),
#    debugModules = cms.untracked.vstring('*')
#    )


from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '75X_dataRun2_v12', '')  #for now track GT manually, since centrality tables updated ex post facto
process.HiForest.GlobalTagLabel = process.GlobalTag.globaltag

from HeavyIonsAnalysis.Configuration.CommonFunctions_cff import overrideJEC_PbPb5020
process = overrideJEC_PbPb5020(process)

process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")
process.centralityBin.Centrality = cms.InputTag("hiCentrality")
process.centralityBin.centralityVariable = cms.string("HFtowers")

#####################################################################################
# Define tree output
#####################################################################################

process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("HiForestAOD.root"))

#####################################################################################
# Additional Reconstruction and Analysis: Main Body
#####################################################################################

####################################################################################

#############################
# Jets
#############################
from Configuration.StandardSequences.ReconstructionHeavyIons_cff import voronoiBackgroundPF, voronoiBackgroundCalo

process.voronoiBackgroundPF = voronoiBackgroundPF
process.voronoiBackgroundCalo = voronoiBackgroundCalo
process.load('HeavyIonsAnalysis.JetAnalysis.jets.HiReRecoJets_HI_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu2CaloJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs2CaloJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs2PFJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu2PFJetSequence_PbPb_data_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3CaloJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3CaloJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3PFJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3PFJetSequence_PbPb_data_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4CaloJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4CaloJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4PFJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4PFJetSequence_PbPb_data_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5CaloJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs5CaloJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs5PFJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5PFJetSequence_PbPb_data_cff')


process.highPurityTracks = cms.EDFilter("TrackSelector",
                                        src = cms.InputTag("hiGeneralTracks"),
                                        cut = cms.string('quality("highPurity")'))

process.load("RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi")
process.offlinePrimaryVertices.TrackLabel = 'highPurityTracks'

process.jetSequences = cms.Sequence(
    voronoiBackgroundPF+
    voronoiBackgroundCalo+

    process.akPu2CaloJets +
    process.akPu2PFJets +
    process.akVs2CaloJets +
    process.akVs2PFJets +

    #process.akPu3CaloJets +
    #process.akPu3PFJets +
    process.akVs3CaloJets +
    process.akVs3PFJets +

    #process.akPu4CaloJets +
    #process.akPu4PFJets +
    process.akVs4CaloJets +
    process.akVs4PFJets +

    process.akPu5CaloJets +
    process.akPu5PFJets +
    process.akVs5CaloJets +
    process.akVs5PFJets +

    process.highPurityTracks +
    process.offlinePrimaryVertices +

    process.akPu2CaloJetSequence +
    process.akVs2CaloJetSequence +
    process.akVs2PFJetSequence +
    process.akPu2PFJetSequence +

    process.akPu3CaloJetSequence +
    process.akVs3CaloJetSequence +
    process.akVs3PFJetSequence +
    process.akPu3PFJetSequence +

    process.akPu4CaloJetSequence +
    process.akVs4CaloJetSequence +
    process.akVs4PFJetSequence +
    process.akPu4PFJetSequence +

    process.akPu5CaloJetSequence +
    process.akVs5CaloJetSequence +
    process.akVs5PFJetSequence +
    process.akPu5PFJetSequence
    )

#####################################################################################

############################
# Event Analysis
############################
process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_data_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.hltobject_PbPb_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cff')
from HeavyIonsAnalysis.EventAnalysis.dummybranches_cff import addHLTdummybranches
addHLTdummybranches(process)

process.load("HeavyIonsAnalysis.JetAnalysis.pfcandAnalyzer_cfi")
process.pfcandAnalyzer.skipCharged = False
process.pfcandAnalyzer.pfPtMin = 0
process.load("HeavyIonsAnalysis.JetAnalysis.hcalNoise_cff")

#####################################################################################

#########################
# Track Analyzer
#########################
process.load('HeavyIonsAnalysis.JetAnalysis.ExtraTrackReco_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.TrkAnalyzers_cff')
# process.load("HeavyIonsAnalysis.TrackAnalysis.METAnalyzer_cff")


####################################################################################

#####################
# Photons
#####################
process.load('HeavyIonsAnalysis.PhotonAnalysis.ggHiNtuplizer_cfi')
process.ggHiNtuplizer.doGenParticles = False
process.ggHiNtuplizerGED = process.ggHiNtuplizer.clone(recoPhotonSrc = cms.InputTag('gedPhotonsTmp'),
                                                       recoPhotonHiIsolationMap = cms.InputTag('photonIsolationHIProducerGED')
)


####################################################################################

#####################
# tupel and necessary PAT sequences
#####################

process.load("HeavyIonsAnalysis.VectorBosonAnalysis.tupelSequence_PbPb_cff")

#####################################################################################


#####################################################################################

## Rho and rhom producer
process.load('RecoJets.JetProducers.kt4PFJets_cfi')
process.load('RecoHI.HiJetAlgos.hiFJRhoProducer')
process.kt4PFJets.src = cms.InputTag('particleFlowTmp')
process.kt4PFJets.doAreaFastjet = True
process.kt4PFJets.jetPtMin      = cms.double(0.0)
process.kt4PFJets.GhostArea     = cms.double(0.005)

process.load('HeavyIonsAnalysis.JetAnalysis.akCS4PFJetSequence_Marta')
process.load('HeavyIonsAnalysis.JetAnalysis.akCs4PFJetSequence_PbPb_data_cff')

process.akCs4PFJets.verbosity = cms.int32(0)
process.akCs4PFJets.jetCollInstanceName = cms.string("pfParticlesCs")
#process.akCs4PFJets.writeJetsWithConst = cms.bool(False)
#process.akCs4PFJets.writeCompound = cms.bool(True)

process.load("HeavyIonsAnalysis.JetAnalysis.pfcandAnalyzerCS_cfi")
#process.load('pfcandAnalyzerCS_cfi')
process.pfcandAnalyzerCS.skipCharged = False
process.pfcandAnalyzerCS.pfPtMin = 0

process.load('HeavyIonsAnalysis.JetAnalysis.akCs4PFJetsSoftDrop_cfi')
process.load('HeavyIonsAnalysis.JetAnalysis.akCs4PFSoftDropJetSequence_PbPb_data_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.akCs4PFJetsFilter_cfi')
process.load('HeavyIonsAnalysis.JetAnalysis.akCs4PFFilterJetSequence_PbPb_data_cff')

#########################
# Main analysis list
#########################

process.ana_step = cms.Path(process.hltanalysis *
			    process.hltobject *
                            process.centralityBin *
                            process.hiEvtAnalyzer*
                            process.jetSequences +
                            process.ggHiNtuplizer +
                            process.ggHiNtuplizerGED +
                            process.pfcandAnalyzer +
                            process.HiForest +
                            #process.trackSequencesPbPb +
                            #process.hcalNoise +
                            process.kt4PFJets *
                            process.hiFJRhoProducer *
                            process.akCs4PFJets *
                            process.akCs4PFJetSequence_data *
                            process.pfcandAnalyzerCS +
                            process.akCs4PFJetsSoftDrop *
                            process.akCs4PFSoftDropJetSequence_data +
                            #process.ak4PFJets *
                            #process.ak4PFRawJetSequence_data +
                            process.akCs4PFJetsFilter *
                            process.akCs4PFFilterJetSequence_data
                            #process.tupelPatSequence
                            )

#####################################################################################

#########################
# Event Selection
#########################

process.load('HeavyIonsAnalysis.JetAnalysis.EventSelection_cff')
process.pcollisionEventSelection = cms.Path(process.collisionEventSelectionAOD)
process.pHBHENoiseFilterResultProducer = cms.Path( process.HBHENoiseFilterResultProducer )
process.HBHENoiseFilterResult = cms.Path(process.fHBHENoiseFilterResult)
process.HBHENoiseFilterResultRun1 = cms.Path(process.fHBHENoiseFilterResultRun1)
process.HBHENoiseFilterResultRun2Loose = cms.Path(process.fHBHENoiseFilterResultRun2Loose)
process.HBHENoiseFilterResultRun2Tight = cms.Path(process.fHBHENoiseFilterResultRun2Tight)
process.HBHEIsoNoiseFilterResult = cms.Path(process.fHBHEIsoNoiseFilterResult)
process.pprimaryVertexFilter = cms.Path(process.primaryVertexFilter )

process.load('HeavyIonsAnalysis.Configuration.hfCoincFilter_cff')
process.phfCoincFilter1 = cms.Path(process.hfCoincFilter)
process.phfCoincFilter2 = cms.Path(process.hfCoincFilter2)
process.phfCoincFilter3 = cms.Path(process.hfCoincFilter3)
process.phfCoincFilter4 = cms.Path(process.hfCoincFilter4)
process.phfCoincFilter5 = cms.Path(process.hfCoincFilter5)

process.pclusterCompatibilityFilter = cms.Path(process.clusterCompatibilityFilter)

process.pAna = cms.EndPath(process.skimanalysis)

# Customization
##########################################UE##########################################
from CondCore.DBCommon.CondDBSetup_cfi import *
process.uetable = cms.ESSource("PoolDBESSource",
      DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0)
        ),
      timetype = cms.string('runnumber'),
      toGet = cms.VPSet(
          cms.PSet(record = cms.string("JetCorrectionsRecord"),
                   tag = cms.string("UETableCompatibilityFormat_PF_v02_offline"),
                   label = cms.untracked.string("UETable_PF")
          ),
          cms.PSet(record = cms.string("JetCorrectionsRecord"),
                   tag = cms.string("UETableCompatibilityFormat_Calo_v02_offline"),
                   label = cms.untracked.string("UETable_Calo")
          )
      ), 
      connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
)
process.es_prefer_uetable = cms.ESPrefer('PoolDBESSource','uetable')
##########################################UE##########################################

import HLTrigger.HLTfilters.hltHighLevel_cfi
process.hltfilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
process.hltfilter.HLTPaths = ["HLT_HIPuAK4CaloJet1*_Eta5p1_v*","HLT_HIPuAK4CaloJet1*_Eta5p1_Cent*100_v*","HLT_HIPuAK4CaloJet*_Jet35_Eta*_v*","HLT_HISinglePhoton30_Eta1p5_v*","HLT_HISinglePhoton30_Eta1p5_Cent*_100_v*","HLT_HISinglePhoton30_Eta3p1_v*","HLT_HISinglePhoton30_Eta3p1_Cent*_100_v*","HLT_HISinglePhoton40_Eta1p5_v*","HLT_HISinglePhoton40_Eta1p5_Cent*_100_v*","HLT_HISinglePhoton40_Eta3p1_v*","HLT_HISinglePhoton40_Eta3p1_Cent*_100_v*"]
process.superFilterPath = cms.Path(process.hltfilter)
process.skimanalysis.superFilters = cms.vstring("superFilterPath")
##filter all path with the production filter sequence
for path in process.paths:
   getattr(process,path)._seq = process.hltfilter * getattr(process,path)._seq
