#! /usr/bin/env python
import ROOT
from PhysicsTools.Heppy.analyzers  import * 
from PhysicsTools.Heppy.physicsobjects import *
import PhysicsTools.HeppyCore.framework.config as cfg
from PhysicsTools.HeppyCore.framework.looper import * 
from PhysicsTools.HeppyCore.framework.eventsfwlite import Events

treeProducer= cfg.Analyzer(
	class_object=AutoFillTreeProducer, 
#	name="AutoFillTreeProducer", 
	verbose=False, 
	vectorTree = True,
	collections = {
		"slimmedMuons" : ( AutoHandle( ("slimmedMuons",), "std::vector<pat::Muon>" ),
                           NTupleCollection("mu", particleType, 4, help="patMuons, directly from MINIAOD") ),
                "slimmedElectron" : ( AutoHandle( ("slimmedElectrons",), "std::vector<pat::Electron>" ),
                           NTupleCollection("mu", particleType, 4, help="patElectron, directly from MINIAOD") ),
   	        "selectedLeptons" : NTupleCollection("LepGood", leptonType, 8, help="Leptons after the preselection"),

	}
	)
# Lepton Analyzer (generic)
LepAna = cfg.Analyzer(
    verbose=False,
    class_object=LeptonAnalyzer,
    # input collections
    muons='slimmedMuons',
    electrons='slimmedElectrons',
    rhoMuon= 'fixedGridRhoFastjetAll',
    rhoElectron = 'fixedGridRhoFastjetAll',
##    photons='slimmedPhotons',
    # energy scale corrections and ghost muon suppression (off by default)
    doMuScleFitCorrections=False, # "rereco"
    doRochesterCorrections=False,
    doElectronScaleCorrections=False, # "embedded" in 5.18 for regression
    doSegmentBasedMuonCleaning=False,
    # inclusive very loose muon selection
    inclusive_muon_id  = "POG_ID_Loose",
    inclusive_muon_pt  = 3,
    inclusive_muon_eta = 2.4,
    inclusive_muon_dxy = 0.5,
    inclusive_muon_dz  = 1.0,
    # loose muon selection
    loose_muon_id     = "POG_ID_Loose",
    loose_muon_pt     = 5,
    loose_muon_eta    = 2.4,
    loose_muon_dxy    = 0.05,
    loose_muon_dz     = 0.2,
    loose_muon_relIso = 0.4,
    # inclusive very loose electron selection
    inclusive_electron_id  = "",
    inclusive_electron_pt  = 5,
    inclusive_electron_eta = 2.5,
    inclusive_electron_dxy = 0.5,
    inclusive_electron_dz  = 1.0,
    inclusive_electron_lostHits = 1.0,
    # loose electron selection
    loose_electron_id     = "", #POG_MVA_ID_NonTrig_full5x5",
    loose_electron_pt     = 7,
    loose_electron_eta    = 2.4,
    loose_electron_dxy    = 0.05,
    loose_electron_dz     = 0.2,
    loose_electron_relIso = 0.4,
    loose_electron_lostHits = 1.0,
    # electron isolation correction method (can be "rhoArea" or "deltaBeta")
    ele_isoCorr = "rhoArea" ,
    ele_tightId = "MVA" ,
    # minimum deltaR between a loose electron and a loose muon (on overlaps, discard the electron)
    min_dr_electron_muon = 0.02
    )
VertexAna = cfg.Analyzer(
#   name='VertexAnalyzer',
    class_object=VertexAnalyzer,
    vertexWeight = None,
    fixedWeight = 1,
    verbose = False
    )
sequence = [VertexAna,LepAna,treeProducer]
sample = cfg.Component(files = "E21AD523-E548-E411-8DF6-00261894388F.root", name="ATEST", isMC=False,isEmbed=False)
looper = Looper( 'Loop', sample,sequence, Events, nPrint = 5)
looper.loop()
looper.write()
