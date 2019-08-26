import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
from Configuration.Eras.Modifier_run2_nanoAOD_94X2016_cff import run2_nanoAOD_94X2016

from PhysicsTools.NanoAOD.common_cff import Var, P4Vars
from PhysicsTools.NanoAOD.jets_cff import jetTable

from PhysicsTools.PatAlgos.jetCollectionTools import JETVARS, GenJetAdder, RecoJetAdder

#
# By default, these collections are saved in NanoAODs:
# - ak4gen (GenJet in NanoAOD)
# - ak8gen (GenJetAK8 in NanoAOD)
#
# Below is a list of genjets that we can save in NanoAOD. Switch enabled to be 
# true if you want to store the jets
config_genjets = [
  { "jet" : "ak6gen",  "enabled" : False,  "name" : "GenJetAK6",   "doc" : "AK6 jets"  }, 
  { "jet" : "ak10gen", "enabled" : False,  "name" : "GenJetAK10",  "doc" : "AK10 jets" }, 
]
config_genjets = list(filter(lambda k: k['enabled'], config_genjets))
#
#
# By default, these collections  are saved in NanoAODs:
# - ak4pfchs   (Jet    in NanoAOD) 
# - ak8pfpuppi (FatJet in NanoAOD)
# By default, the ak4pfchs (Jet) and ak8pfpuppi (FatJet) collections 
# are saved in NanoAODs. 
# Below is a list of recojets that we can save in NanoAOD. Switch enabled to be 
# true if you want to store the jets
config_recojets = [
  { "jet" : "ak4pfpuppi", "enabled" : True,   "name" : "JetPUPPI",     "doc" : "AK4PFPUPPI jets", "inputCollection" : "slimmedJetsPuppi", "genJetsCollection": "slimmedGenJets"    }, #Available in MiniAOD
  { "jet" : "ak4calo",    "enabled" : True,   "name" : "JetCalo",      "doc" : "AK4Calo jets",    "inputCollection" : "slimmedCaloJets" , "genJetsCollection": "slimmedGenJets"    }, #Available in MiniAOD
  { "jet" : "ak4pf",      "enabled" : True,   "name" : "JetPF",        "doc" : "AK4PF jets",      "inputCollection" : "",                 "genJetsCollection": "slimmedGenJets"    }, 
  { "jet" : "ak8pf",      "enabled" : True,   "name" : "FatJetPF",     "doc" : "AK8PF jets",      "inputCollection" : "",                 "genJetsCollection": "slimmedGenJetsAK8" },
  { "jet" : "ak8pfchs",   "enabled" : True,   "name" : "FatJetCHS",    "doc" : "AK8PFCHS jets",   "inputCollection" : "",                 "genJetsCollection": "slimmedGenJetsAK8" },
  { "jet" : "ak6pf",      "enabled" : False,  "name" : "JetAK6PF",     "doc" : "AK6PF jets",      "inputCollection" : "",                 "genJetsCollection": "AK6GenJetsNoNu"    },
  { "jet" : "ak10pf",     "enabled" : False,  "name" : "FatJetAK10PF", "doc" : "AK10PF jets",     "inputCollection" : "",                 "genJetsCollection": "AK10GenJetsNoNu"   },
]
config_recojets = list(filter(lambda k: k['enabled'], config_recojets))

for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_94X2016:
  modifier.toModify(JETVARS,
    jetId = Var("userInt('tightId')*2+userInt('looseId')", int, doc = "Jet ID flags bit1 is loose, bit2 is tight")
  )

def PrepJMECustomNanoAOD(process):
  #
  # Additional variables to AK4GenJets 
  #
  process.genJetTable.variables.area = JETVARS.area
  #
  # additional variables to AK8GenJets
  #
  process.genJetAK8Table.variables.area = JETVARS.area
  #
  # Additional variables for AK4PFCHS 
  #
  process.jetTable.variables.HFHEF  = JETVARS.HFHEF
  process.jetTable.variables.HFEMEF = JETVARS.HFEMEF
  #
  # Additional variables to AK8PFPUPPI
  #
  # These variables are not stored for AK8PFCHS (slimmedJetsAK8)
  # in MiniAOD if their pt < 170 GeV. Hence the conditional fill.
  #
  process.fatJetTable.variables.chHEF  = Var("?isPFJet()?chargedHadronEnergyFraction():-1", float, doc="charged Hadron Energy Fraction",                  precision = 6)
  process.fatJetTable.variables.neHEF  = Var("?isPFJet()?neutralHadronEnergyFraction():-1", float, doc="neutral Hadron Energy Fraction",                  precision = 6)
  process.fatJetTable.variables.chEmEF = Var("?isPFJet()?chargedEmEnergyFraction():-1",     float, doc="charged Electromagnetic Energy Fraction",         precision = 6)
  process.fatJetTable.variables.neEmEF = Var("?isPFJet()?neutralEmEnergyFraction():-1",     float, doc="neutral Electromagnetic Energy Fraction",         precision = 6)
  process.fatJetTable.variables.muEF   = Var("?isPFJet()?muonEnergyFraction():-1",          float, doc="muon Energy Fraction",                            precision = 6)
  process.fatJetTable.variables.HFHEF  = Var("?isPFJet()?HFHadronEnergyFraction():-1",      float, doc="energy fraction in forward hadronic calorimeter", precision = 6)
  process.fatJetTable.variables.HFEMEF = Var("?isPFJet()?HFEMEnergyFraction():-1",          float, doc="energy fraction in forward EM calorimeter",       precision = 6)
  #
  #
  #
  process.jercVarsFatJet = process.jercVars.clone(
    srcJet = cms.InputTag("updatedJetsAK8"),
    maxDR = cms.double(0.8),
  )
  process.jetSequence.insert(process.jetSequence.index(process.updatedJetsAK8WithUserData), process.jercVarsFatJet)
  
  process.updatedJetsAK8WithUserData.userFloats.jercCHPUF = cms.InputTag(
    "%s:chargedHadronPUEnergyFraction"  % process.jercVarsFatJet.label()
  )
  process.updatedJetsAK8WithUserData.userFloats.jercCHF = cms.InputTag(
    "%s:chargedHadronCHSEnergyFraction" % process.jercVarsFatJet.label()
  )
  process.fatJetTable.variables.jercCHPUF = JETVARS.jercCHPUF
  process.fatJetTable.variables.jercCHF   = JETVARS.jercCHF

  #
  # Remove any pT cuts.
  #
  process.finalJets.cut             = cms.string("") # 15 -> 10
  process.finalJetsAK8.cut          = cms.string("") # 170 -> 170
  process.genJetTable.cut           = cms.string("") # 10 -> 8
  process.genJetFlavourTable.cut    = cms.string("") # 10 -> 8
  process.genJetAK8Table.cut        = cms.string("") # 100 -> 80
  process.genJetAK8FlavourTable.cut = cms.string("") # 100 -> 80

  ######################################################################################################################

  genJA = GenJetAdder()
  for jetConfig in config_genjets:
    cfg = { k : v for k, v in jetConfig.items() if k != "enabled" }
    genJA.addGenJetCollection(process, **cfg)
  process.nanoSequenceMC += genJA.getSequence(process)

  recoJA = RecoJetAdder()
  for jetConfig in config_recojets:
    cfg = { k : v for k, v in jetConfig.items() if k != "enabled" }
    recoJA.addRecoJetCollection(process, **cfg)
  process.nanoSequenceMC += recoJA.getSequence(process)

