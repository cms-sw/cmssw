import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTanalyzers.HLTAnalyser_cfi import * 

#hltanalysis.MuCandTag3       = cms.InputTag("hltL3GenMuonCandidates") ### if DoL3Muons is true

hltanalysis.Centrality    = cms.InputTag("hiCentrality")
hltanalysis.CentralityBin    = cms.InputTag("centralityBin")
hltanalysis.EvtPlane      = cms.InputTag("hiEvtPlane","recoLevel")
hltanalysis.mctruth       = cms.InputTag("hiGenParticles")
hltanalysis.HiMC          = cms.InputTag("heavyIon")    

hltanalysis.PrimaryVertices             = cms.InputTag("hiSelectedVertex")
    
hltanalysis.l1GtObjectMapRecord = cms.InputTag("hltL1GtObjectMap","","ANALYSIS")
hltanalysis.l1GtReadoutRecord   = cms.InputTag("hltGtDigis","","ANALYSIS")
hltanalysis.hltresults          = cms.InputTag("TriggerResults","","ANALYSIS")
hltanalysis.HLTProcessName      = cms.string("ANALYSIS")

hltanalysis.RunParameters = cms.PSet(
        HistogramFile        = cms.untracked.string('openhlt.root'),
        EtaMin               = cms.untracked.double(-5.2),
        EtaMax               = cms.untracked.double( 5.2),
        CalJetMin            = cms.double(0.0),
        GenJetMin            = cms.double(0.0),
        Monte                = cms.bool(True),
        Debug                = cms.bool(True),

        ### added in 2010 ###
        DoHeavyIon           = cms.untracked.bool(True),

        ### MCTruth
        DoParticles          = cms.untracked.bool(True),
        DoRapidity           = cms.untracked.bool(True),
        DoVerticesByParticle = cms.untracked.bool(True),

        ### Egamma
        DoPhotons            = cms.untracked.bool(False),
        DoElectrons          = cms.untracked.bool(False),
        DoSuperClusters      = cms.untracked.bool(True),

        ### Muon
        DoL1Muons            = cms.untracked.bool(True),
        DoL2Muons            = cms.untracked.bool(False),
        DoL3Muons            = cms.untracked.bool(False),
        DoOfflineMuons       = cms.untracked.bool(False),
        DoQuarkonia          = cms.untracked.bool(False)
)


