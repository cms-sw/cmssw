import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
from Configuration.Eras.Modifier_run2_nanoAOD_94X2016_cff import run2_nanoAOD_94X2016

from PhysicsTools.NanoAOD.common_cff import Var, P4Vars
from PhysicsTools.NanoAOD.jets_cff import jetTable

from PhysicsTools.PatAlgos.tools.jetCollectionTools import GenJetAdder, RecoJetAdder

import copy

#
# By default, these collections are saved in NanoAODs:
# - ak4gen (GenJet in NanoAOD)
# - ak8gen (GenJetAK8 in NanoAOD)
# Below is a list of genjets that we can save in NanoAOD. Set 
# "enabled" to true if you want to store the jet collection
config_genjets = [
  { 
    "jet"     : "ak5gen",    
    "enabled" : False, 
  }, 
  { 
    "jet"     : "ak6gen",    
    "enabled" : False, 
  }, 
  { 
    "jet"     : "ak7gen",    
    "enabled" : False, 
  },
  { 
    "jet"     : "ak9gen",    
    "enabled" : False, 
  },  
  { 
    "jet"     : "ak10gen", 
    "enabled" : False,   
  }, 
]
config_genjets = list(filter(lambda k: k['enabled'], config_genjets))
#
# GenJets info in NanoAOD
#
nanoInfo_genjets = {
  "ak5gen"  : {
    "name" : "GenJetAK5",
    "doc"  : "AK5 jets",
  },
  "ak6gen"  : {
    "name" : "GenJetAK6",
    "doc"  : "AK6 jets",
  },
  "ak7gen"  : {
    "name" : "GenJetAK7",
    "doc"  : "AK9 jets",
  },
  "ak9gen"  : {
    "name" : "GenJetAK9",
    "doc"  : "AK9 jets",
  },
  "ak10gen"  : {
    "name" : "GenJetAK10",
    "doc"  : "AK10 jets",
  },
}
#
# By default, these collections  are saved in NanoAODs:
# - ak4pfchs   (Jet    in NanoAOD) 
# - ak8pfpuppi (FatJet in NanoAOD)
# By default, the ak4pfchs (Jet) and ak8pfpuppi (FatJet) collections 
# are saved in NanoAODs. 
# Below is a list of recojets that we can save in NanoAOD. Set "enabled" 
# to true if you want to store the recojet collection.
#
config_recojets = [
  { 
    "jet"     : "ak4pfpuppi", 
    "enabled" : True,   
    "inputCollection"  : "slimmedJetsPuppi", #Exist in MiniAOD
    "genJetsCollection": "slimmedGenJets",   
  }, 
  { 
    "jet" : "ak4calo",    
    "enabled" : True,     
    "inputCollection"  : "slimmedCaloJets", #Exist in MiniAOD
    "genJetsCollection": "slimmedGenJets",  
  }, 
  { 
    "jet" : "ak4pf",  
    "enabled" : True,        
    "inputCollection" : "",                 
    "genJetsCollection": "slimmedGenJets",  
  }, 
  { 
    "jet" : "ak8pf",  
    "enabled" : True,   
    "inputCollection" : "",                 
    "genJetsCollection": "slimmedGenJetsAK8", 
  },
  { 
    "jet" : "ak8pfchs",   
    "enabled" : True,   
    "inputCollection" : "",                 
    "genJetsCollection": "slimmedGenJetsAK8",
  },
  { 
    "jet" : "ak6pf",  
    "enabled" : False,  
    "inputCollection" : "",                 
    "genJetsCollection": "AK6GenJetsNoNu",    
  },
  { 
    "jet" : "ak10pf", 
    "enabled" : False,  
    "inputCollection" : "",                 
    "genJetsCollection": "AK10GenJetsNoNu",
  },
]
config_recojets = list(filter(lambda k: k['enabled'], config_recojets))
#
# RecoJets info in NanoAOD
#
nanoInfo_recojets = {
  "ak4pfpuppi" : {
    "name" : "JetPUPPI",
    "doc"  : "AK4PFPUPPI jets", 
  },
  "ak4calo" : {
    "name": "JetCalo",
    "doc" : "AK4Calo jets",   
  },
  "ak4pf" : {
    "name": "JetPF",
    "doc" : "AK4PF jets",     
  },
  "ak8pf" : {
    "name": "FatJetPF",
    "doc" : "AK8PF jets", 
  },
  "ak8pfchs" : {
    "name" : "FatJetCHS",
    "doc"  : "AK8PFCHS jets",   
  },
  "ak6pf" : {
    "name": "JetAK6PF",
    "doc" : "AK6PF jets",
  },
  "ak10pf" : {
    "name" : "FatJetAK10PF",
    "doc"  : "AK10PF jets", 
  },
}

#
# The reco jet names already exists 
# in NanoAOD.
#
recojetNameInNano = [ "Jet", "FatJet" ]
#
# The gen jet names already exists 
# in NanoAOD.
#
genjetNameInNano = [ "GenJet", "GenJetAK8" ]

JETVARS = cms.PSet(P4Vars,
  HFHEF     = Var("HFHadronEnergyFraction()", float, doc = "energy fraction in forward hadronic calorimeter", precision = 6),
  HFEMEF    = Var("HFEMEnergyFraction()",     float, doc = "energy fraction in forward EM calorimeter",       precision = 6),
  area      = jetTable.variables.area,
  chHEF     = jetTable.variables.chHEF,
  neHEF     = jetTable.variables.neHEF,
  chEmEF    = jetTable.variables.chEmEF,
  neEmEF    = jetTable.variables.neEmEF,
  muEF      = jetTable.variables.muEF,
  rawFactor = jetTable.variables.rawFactor,
  jetId     = jetTable.variables.jetId,
  jercCHPUF = jetTable.variables.jercCHPUF,
  jercCHF   = jetTable.variables.jercCHF,
)

for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_94X2016:
  modifier.toModify(JETVARS,
    jetId = Var("userInt('tightId')*2+userInt('looseId')", int, doc = "Jet ID flags bit1 is loose, bit2 is tight")
  )

#============================================
#
# TableGenJetAdder
#
#============================================
class TableGenJetAdder(object):
  """
  Tool to store gen jet variables in NanoAOD for customized
  gen jet collections.
  """
  def __init__(self):
    self.main = []

  def getSequence(self, proc):
    """
    Tool to add 
    """
    tasks = self.main

    resultSequence = cms.Sequence()
    for idx, task in enumerate(tasks):
      if idx == 0:
        resultSequence = cms.Sequence(getattr(proc, task))
      else:
        resultSequence.insert(idx, getattr(proc, task))
    return resultSequence

  def addTable(self, proc, genJetInfo):
    currentTasks = []
    
    print("custom_jme_cff::TableGenJetAdder::addTable: Adding Table for GenJet Collection: {}".format(genJetInfo.jet))
    
    name = nanoInfo_genjets[genJetInfo.jet]["name"]
    doc  = nanoInfo_genjets[genJetInfo.jet]["doc"]
    
    if name in genjetNameInNano:
      raise RuntimeError('GenJet collection name (%s) taken in NanoAOD for %s' %(name, genJetInfo.jet))

    #
    # GenJet Table
    #
    table = "{}Table".format(genJetInfo.jetTagName)
    genJetsCollection = "{}{}{}".format(genJetInfo.jetAlgo.upper(), genJetInfo.jetSize, 'GenJetsNoNu')
    setattr(proc, table, cms.EDProducer("SimpleCandidateFlatTableProducer",
        src       = cms.InputTag(genJetsCollection),
        cut       = cms.string(""),
        name      = cms.string(name),
        doc       = cms.string('{} (generator level)'.format(doc)),
        singleton = cms.bool(False),
        extension = cms.bool(False),
        variables = cms.PSet(P4Vars,
        area      = jetTable.variables.area,
        ),
      )
    )    
    currentTasks.append(table)

    #
    # GenJet Flavour Table
    #
    genFlavour = "{}Flavour".format(genJetInfo.jetTagName)
    genFlavourTable = "{}Table".format(genFlavour)
    if genFlavourTable in self.main:
      raise ValueError("Step '%s' already implemented" % genFlavourTable)
    setattr(proc, genFlavourTable, cms.EDProducer("GenJetFlavourTableProducer",
        name            = cms.string(name),
        src             = cms.InputTag(genJetsCollection),
        cut             = cms.string(""),
        deltaR          = cms.double(0.1),
        jetFlavourInfos = cms.InputTag(genFlavour),
      )
    )
    currentTasks.append(genFlavourTable)
    self.main.extend(currentTasks)

#============================================
#
# TableRecoJetAdder
#
#============================================
class TableRecoJetAdder(object):
  """
  Tool to store reco jet variables in NanoAOD for customized
  reco jet collections.
  """
  def __init__(self):
    self.main = []

  def getSequence(self, proc):
    tasks = self.main

    resultSequence = cms.Sequence()
    for idx, task in enumerate(tasks):
      if idx == 0:
        resultSequence = cms.Sequence(getattr(proc, task))
      else:
        resultSequence.insert(idx, getattr(proc, task))
    return resultSequence

  def addTable(self, proc, recoJetInfo):

    currentTasks = []

    print("custom_jme_cff::TableRecoJetAdder::addTable: Adding Table for Reco Jet Collection: {}".format(recoJetInfo.jet))

    name = nanoInfo_recojets[recoJetInfo.jet]["name"]
    doc  = nanoInfo_recojets[recoJetInfo.jet]["doc"]

    if name in recojetNameInNano:
      raise RuntimeError('RecoJet collection name (%s) taken in NanoAOD for %s' %(name, recoJetInfo.jet))

    table = "{}Table".format(recoJetInfo.jetTagName)
    if recoJetInfo.skipUserData:
      if recoJetInfo.doCalo:
        tableContents = cms.PSet(
          P4Vars,
          area      = jetTable.variables.area,
          rawFactor = jetTable.variables.rawFactor,
          emf       = Var("emEnergyFraction()", float, doc = "electromagnetic energy fraction", precision = 10),
        )
      else:
        tableContents = cms.PSet(
          P4Vars,
          area      = jetTable.variables.area,
          rawFactor = jetTable.variables.rawFactor,
        )
    else:
      tableContents = JETVARS.clone()
    
    updatedJets = "updatedJets{}".format(recoJetInfo.jetTagName)
    setattr(proc, table, cms.EDProducer("SimpleCandidateFlatTableProducer",
        src       = cms.InputTag(updatedJets),
        cut       = cms.string(""),
        name      = cms.string(name),
        doc       = cms.string(doc),
        singleton = cms.bool(False),
        extension = cms.bool(False),
        variables = tableContents,
      )
    )
    currentTasks.append(table)

    tightJetIdLepVeto = "tightJetIdLepVeto{}".format(recoJetInfo.jetTagName)
    if not recoJetInfo.skipUserData:
      altTasks = copy.deepcopy(currentTasks)
      for idx, task in enumerate(altTasks):
        if task == tightJetIdLepVeto:
          altTasks[idx] = looseJetId
      for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_94X2016:
        modifier.toReplaceWith(currentTasks, altTasks)
    self.main.extend(currentTasks)

def AddPileUpJetIDVars(proc):

  print("custom_jme_cff::AddPileUpJetIDVars: Recalculate pile-up jet ID variables and save them")

  #
  # Recalculate PUJet ID variables
  #
  from RecoJets.JetProducers.PileupJetID_cfi import pileupJetIdCalculator
  proc.pileupJetIdCalculatorAK4PFCHS = pileupJetIdCalculator.clone(
    jets = "updatedJets",
    vertexes  = "offlineSlimmedPrimaryVertices",
    inputIsCorrected = True,
    applyJec  = False 
  )
  proc.jetSequence.insert(proc.jetSequence.index(proc.updatedJets)+1, proc.pileupJetIdCalculatorAK4PFCHS)

  #
  # Get the variables
  #
  proc.puJetVarAK4PFCHS = cms.EDProducer("PileupJetIDVarProducer",
    srcJet = cms.InputTag("updatedJets"),    
    srcPileupJetId = cms.InputTag("pileupJetIdCalculatorAK4PFCHS")
  )
  proc.jetSequence.insert(proc.jetSequence.index(proc.jercVars)+1, proc.puJetVarAK4PFCHS)

  #
  # Save variables as userFloats and userInts in each jet
  # 
  proc.updatedJetsWithUserData.userFloats.dR2Mean  = cms.InputTag("puJetVarAK4PFCHS:dR2Mean")
  proc.updatedJetsWithUserData.userFloats.majW     = cms.InputTag("puJetVarAK4PFCHS:majW")
  proc.updatedJetsWithUserData.userFloats.minW     = cms.InputTag("puJetVarAK4PFCHS:minW")
  proc.updatedJetsWithUserData.userFloats.frac01   = cms.InputTag("puJetVarAK4PFCHS:frac01")
  proc.updatedJetsWithUserData.userFloats.frac02   = cms.InputTag("puJetVarAK4PFCHS:frac02")
  proc.updatedJetsWithUserData.userFloats.frac03   = cms.InputTag("puJetVarAK4PFCHS:frac03")
  proc.updatedJetsWithUserData.userFloats.frac04   = cms.InputTag("puJetVarAK4PFCHS:frac04")
  proc.updatedJetsWithUserData.userFloats.ptD      = cms.InputTag("puJetVarAK4PFCHS:ptD")
  proc.updatedJetsWithUserData.userFloats.beta     = cms.InputTag("puJetVarAK4PFCHS:beta")
  proc.updatedJetsWithUserData.userFloats.pull     = cms.InputTag("puJetVarAK4PFCHS:pull")
  proc.updatedJetsWithUserData.userFloats.jetR     = cms.InputTag("puJetVarAK4PFCHS:jetR")
  proc.updatedJetsWithUserData.userFloats.jetRchg  = cms.InputTag("puJetVarAK4PFCHS:jetRchg")
  proc.updatedJetsWithUserData.userInts.nCharged   = cms.InputTag("puJetVarAK4PFCHS:nCharged")

  #
  # Specfiy variables in the jetTable to save in NanoAOD
  #
  proc.jetTable.variables.dR2Mean  = Var("userFloat('dR2Mean')", float, doc="pT^2-weighted average square distance of jet constituents from the jet axis", precision= 6)  
  proc.jetTable.variables.majW     = Var("userFloat('majW')",    float, doc="major axis of jet ellipsoid in eta-phi plane", precision= 6)  
  proc.jetTable.variables.minW     = Var("userFloat('minW')",    float, doc="minor axis of jet ellipsoid in eta-phi plane", precision= 6)  
  proc.jetTable.variables.frac01   = Var("userFloat('frac01')",  float, doc="frac of constituents' pT contained within dR<0.1", precision= 6)  
  proc.jetTable.variables.frac02   = Var("userFloat('frac02')",  float, doc="frac of constituents' pT contained within 0.1<dR<0.2", precision= 6) 
  proc.jetTable.variables.frac03   = Var("userFloat('frac03')",  float, doc="frac of constituents' pT contained within 0.2<dR<0.3", precision= 6) 
  proc.jetTable.variables.frac04   = Var("userFloat('frac04')",  float, doc="frac of constituents' pT contained within 0.3<dR<0.4", precision= 6) 
  proc.jetTable.variables.ptD      = Var("userFloat('ptD')",     float, doc="pT-weighted average pT of constituents", precision= 6) 
  proc.jetTable.variables.beta     = Var("userFloat('beta')",    float, doc="fraction of pT of charged constituents associated to PV", precision= 6) 
  proc.jetTable.variables.pull     = Var("userFloat('pull')",    float, doc="magnitude of pull vector", precision= 6) 
  proc.jetTable.variables.jetR     = Var("userFloat('jetR')",    float, doc="fraction of jet pT carried by the leading constituent", precision= 6) 
  proc.jetTable.variables.jetRchg  = Var("userFloat('jetRchg')", float, doc="fraction of jet pT carried by the leading charged constituent", precision= 6) 
  proc.jetTable.variables.nCharged = Var("userInt('nCharged')",  float, doc="number of charged constituents", precision= 6) 

def PrepJMECustomNanoAOD(process):
  #
  # Additional variables to AK4GenJets 
  #
  process.genJetTable.variables.area = JETVARS.area
  #
  # Additional variables to AK8GenJets
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
    srcJet = "updatedJetsAK8",
    maxDR = 0.8,
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
  process.finalJets.cut             = "" # 15 -> 10
  process.finalJetsAK8.cut          = "" # 170 -> 170
  process.genJetTable.cut           = "" # 10 -> 8
  process.genJetFlavourTable.cut    = "" # 10 -> 8
  process.genJetAK8Table.cut        = "" # 100 -> 80
  process.genJetAK8FlavourTable.cut = "" # 100 -> 80
  #
  # Add variables for pileup jet ID studies.
  #
  AddPileUpJetIDVars(process)

  ######################################################################################################################

  #
  # Add GenJets to NanoAOD
  #
  genJA = GenJetAdder()
  tableGenJA = TableGenJetAdder()

  for jetConfig in config_genjets:
    cfg = { k : v for k, v in jetConfig.items() if k != "enabled" }
    genJetInfo = genJA.addGenJetCollection(process, **cfg)
    tableGenJA.addTable(process, genJetInfo)

  process.nanoSequenceMC += genJA.getSequence(process)
  process.nanoSequenceMC += tableGenJA.getSequence(process)

  #
  # Add RecoJets to NanoAOD
  #
  recoJA = RecoJetAdder()
  tableRecoJA = TableRecoJetAdder()

  for jetConfig in config_recojets:
    cfg = { k : v for k, v in jetConfig.items() if k != "enabled" }
    recoJetInfo = recoJA.addRecoJetCollection(process, **cfg)
    tableRecoJA.addTable(process, recoJetInfo)

  process.nanoSequenceMC += recoJA.getSequence(process)
  process.nanoSequenceMC += tableRecoJA.getSequence(process)

