import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.ReconstructionHeavyIons_cff import *
from GeneratorInterface.HiGenCommon.HeavyIon_cff import *
from RecoHI.HiCentralityAlgos.CentralityBin_cfi import *
from HLTrigger.HLTanalyzers.HLTAnalyser_cfi import * 

# jets
hltanalysis.genjets           = cms.InputTag("iterativeCone5HiGenJets")
hltanalysis.recjets           = cms.InputTag("hltIterativeCone5PileupSubtractionCaloJets")

# photons
hltanalysis.BarrelPhoton      = cms.InputTag("hltCorrectedIslandBarrelSuperClustersHI")
hltanalysis.EndcapPhoton      = cms.InputTag("hltCorrectedIslandEndcapSuperClustersHI")

# muons
#hltanalysis.MuCandTag3       = cms.InputTag("hltL3GenMuonCandidates") ### if DoL3Muons is true

hltanalysis.Centrality    = cms.InputTag("hiCentrality")
hltanalysis.CentralityBin    = cms.InputTag("centralityBin")
hltanalysis.EvtPlane      = cms.InputTag("hiEvtPlane")
hltanalysis.mctruth       = cms.InputTag("hiGenParticles")
hltanalysis.HiMC          = cms.InputTag("heavyIon")    

hltanalysis.PrimaryVertices             = cms.InputTag("hiSelectedVertex")

hltanalysis.RunParameters = cms.PSet(
    Monte                = cms.bool(False),
    Debug                = cms.bool(True),
    UseTFileService      = cms.untracked.bool(True),
    
    ### added in 2010 ###
    DoHeavyIon           = cms.untracked.bool(True),
    DoMC           = cms.untracked.bool(False),
    DoAlCa           = cms.untracked.bool(False),
    DoTracks           = cms.untracked.bool(False),
    DoVertex           = cms.untracked.bool(False),
    DoJets           = cms.untracked.bool(False),

    ### MCTruth
    DoParticles          = cms.untracked.bool(False),
    DoRapidity           = cms.untracked.bool(False),
    DoVerticesByParticle = cms.untracked.bool(False),
    
    ### Egamma
    DoPhotons            = cms.untracked.bool(False),
    DoElectrons          = cms.untracked.bool(False),
    DoSuperClusters      = cms.untracked.bool(False),
    
    ### Muon
    DoMuons            = cms.untracked.bool(False),
    DoL1Muons            = cms.untracked.bool(False),
    DoL2Muons            = cms.untracked.bool(False),
    DoL3Muons            = cms.untracked.bool(False),
    DoOfflineMuons       = cms.untracked.bool(False),
    DoQuarkonia          = cms.untracked.bool(False)
    )


