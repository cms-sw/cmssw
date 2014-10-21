#! /usr/bin/env python
import ROOT
from PhysicsTools.Heppy.analyzers  import * 
from PhysicsTools.Heppy.physicsobjects import *
import PhysicsTools.HeppyCore.framework.config as cfg
from PhysicsTools.HeppyCore.framework.looper import * 
from PhysicsTools.HeppyCore.framework.eventsfwlite import Events

treeProducer= cfg.Analyzer(
	class_object=AutoFillTreeProducer, 
	verbose=False, 
	vectorTree = True,
	collections = {
		#copying input collection p4 information
		"slimmedMuons" : ( AutoHandle( ("slimmedMuons",), "std::vector<pat::Muon>" ),
                           NTupleCollection("mu", particleType, 4, help="patMuons, directly from MINIAOD") ),
                "slimmedElectron" : ( AutoHandle( ("slimmedElectrons",), "std::vector<pat::Electron>" ),
                           NTupleCollection("ele", particleType, 4, help="patElectron, directly from MINIAOD") ),

		#standard dumping of objects
   	        "selectedLeptons" : NTupleCollection("leptons", leptonType, 8, help="Leptons after the preselection"),
                "selectedTaus"    : NTupleCollection("TauGood", tauType, 3, help="Taus after the preselection"),
	        "cleanJets"       : NTupleCollection("Jet",     jetType, 8, sortDescendingBy = lambda jet : jet.btag('combinedSecondaryVertexBJetTags'),
					 help="Cental jets after full selection and cleaning, sorted by b-tag"),
		#dump of gen objects
                "gentopquarks"    : NTupleCollection("GenTop",     genParticleType, 2, help="Generated top quarks from hard scattering"),
                "genbquarks"      : NTupleCollection("GenBQuark",  genParticleType, 2, help="Generated bottom quarks from top quark decays"),
                "genwzquarks"     : NTupleCollection("GenQuark",   genParticleWithSourceType, 6, help="Generated quarks from W/Z decays"),
                "genleps"         : NTupleCollection("GenLep",     genParticleWithSourceType, 6, help="Generated leptons from W/Z decays"),
                "gentauleps"      : NTupleCollection("GenLepFromTau", genParticleWithSourceType, 6, help="Generated leptons from decays of taus from W/Z/h decays"),

	}
	)
# Lepton Analyzer, take its default config
LepAna = LeptonAnalyzer.defaultConfig
#replace one parameter
LepAna.loose_muon_pt = 10

VertexAna = VertexAnalyzer.defaultConfig
PhoAna = PhotonAnalyzer.defaultConfig
TauAna = TauAnalyzer.defaultConfig
JetAna = JetAnalyzer.defaultConfig

sequence = [VertexAna,LepAna,TauAna,PhoAna,JetAna,treeProducer]
sample = cfg.Component(files = "/scratch/arizzi/heppy/CMSSW_7_2_0_pre8/src/PhysicsTools/Heppy/test/E21AD523-E548-E411-8DF6-00261894388F.root", name="ATEST", isMC=False,isEmbed=False)
looper = Looper( 'Loop', sample,sequence, Events, nPrint = 5)
looper.loop()
looper.write()
