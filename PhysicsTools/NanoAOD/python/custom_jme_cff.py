import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_run2_jme_2016_cff import run2_jme_2016
from Configuration.Eras.Modifier_run2_jme_2017_cff import run2_jme_2017

from CommonTools.PileupAlgos.Puppi_cff import puppi

from RecoJets.JetProducers.PileupJetID_cfi import pileupJetIdCalculator, pileupJetId
from RecoJets.JetProducers.PileupJetID_cfi import _chsalgos_81x, _chsalgos_94x, _chsalgos_102x

from PhysicsTools.NanoAOD.common_cff import Var, P4Vars
from PhysicsTools.NanoAOD.jets_cff   import jetTable, jetCorrFactorsNano, updatedJets, finalJets, qgtagger, hfJetShowerShapeforNanoAOD
from PhysicsTools.NanoAOD.jets_cff   import genJetTable, genJetFlavourAssociation, genJetFlavourTable

from PhysicsTools.PatAlgos.tools.jetCollectionTools import GenJetAdder, RecoJetAdder
from PhysicsTools.PatAlgos.tools.jetTools import supportedJetAlgos
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection

import copy

bTagCSVV2    = ['pfDeepCSVJetTags:probb','pfDeepCSVJetTags:probbb','pfDeepCSVJetTags:probc']
bTagDeepCSV  = ['pfCombinedInclusiveSecondaryVertexV2BJetTags']
bTagDeepJet  = [
  'pfDeepFlavourJetTags:probb','pfDeepFlavourJetTags:probbb','pfDeepFlavourJetTags:problepb',
  'pfDeepFlavourJetTags:probc','pfDeepFlavourJetTags:probuds','pfDeepFlavourJetTags:probg'
]
bTagDiscriminatorsForAK4 = bTagCSVV2+bTagDeepCSV+bTagDeepJet

#
# By default, these collections are saved in NanoAODs:
# - ak4gen (GenJet in NanoAOD), slimmedGenJets in MiniAOD 
# - ak8gen (GenJetAK8 in NanoAOD), slimmedGenJetsAK8 in MiniAOD 
# Below is a list of genjets that we can save in NanoAOD. Set 
# "enabled" to true if you want to store the jet collection
config_genjets = [
  { 
    "jet"     : "ak8gen",    
    "enabled" : False, 
  },  
]
config_genjets = list(filter(lambda k: k['enabled'], config_genjets))
#
# GenJets info in NanoAOD
#
nanoInfo_genjets = {
  "ak8gen"  : {
    "name" : "GenJetAK8",
    "doc"  : "AK8 Gen jets",
  },
}
#
# By default, these collections are saved in the main NanoAODs:
# - ak4pfchs   (Jet    in NanoAOD), slimmedJets in MiniAOD  
# - ak8pfpuppi (FatJet in NanoAOD), slimmedJetsAK8 in MiniAOD 
# Below is a list of recojets that we can save in NanoAOD. Set 
# "enabled" to true if you want to store the recojet collection.
#
config_recojets = [
  { 
    "jet" : "ak4calo",    
    "enabled" : True,     
    "inputCollection"  : "slimmedCaloJets", #Exist in MiniAOD
    "genJetsCollection": "AK4GenJetsNoNu",  
  }, 
  { 
    "jet" : "ak4pf",  
    "enabled" : True,        
    "inputCollection" : "",
    "genJetsCollection": "AK4GenJetsNoNu",  
  }, 
  { 
    "jet" : "ak4pfpuppi",  
    "enabled" : True,        
    "inputCollection" : "",                 
    "genJetsCollection": "AK4GenJetsNoNu",  
    "bTagDiscriminators": bTagDiscriminatorsForAK4
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
]
config_recojets = list(filter(lambda k: k['enabled'], config_recojets))
#
# RecoJets info in NanoAOD
#
nanoInfo_recojets = {
  "ak4pfpuppi" : {
    "name"  : "JetPuppi",
    "doc"   : "AK4 PF Puppi jets with JECs applied, after basic selection (pt > 2)",  
    "ptcut" : "pt > 2",      
    "doQGL" : True,
    "doPUIDVar": True,
    "doBTag": True,
  },
  "ak4pf" : {
    "name"  : "JetPF",
    "doc"   : "AK4 PF jets with JECs applied, after basic selection (pt > 2)",
    "ptcut" : "pt > 2",   
  },
  "ak4calo" : {
    "name": "JetCalo",
    "doc" : "AK4 Calo jets with JECs applied",
  },
  "ak8pfchs" : {
    "name"  : "FatJetCHS",
    "doc"   : "AK8 PF CHS jets with JECs applied, after basic selection (pt > 100)", 
    "ptcut" : "pt > 100"    
  },
  "ak8pf" : {
    "name"  : "FatJetPF",
    "doc"   : "AK8 PF jets with JECs applied, after basic selection (pt > 100)", 
    "ptcut" : "pt > 100", 
  },
}



GENJETVARS = cms.PSet(P4Vars,
  nConstituents   = jetTable.variables.nConstituents,
)
PFJETVARS = cms.PSet(P4Vars,
  rawFactor       = jetTable.variables.rawFactor,
  area            = jetTable.variables.area,
  chHEF           = jetTable.variables.chHEF,
  neHEF           = jetTable.variables.neHEF,
  chEmEF          = jetTable.variables.chEmEF,
  neEmEF          = jetTable.variables.neEmEF,
  muEF            = jetTable.variables.muEF,
  hfHEF           = Var("HFHadronEnergyFraction()",float,doc = "hadronic energy fraction in HF",precision = 6),
  hfEmEF          = Var("HFEMEnergyFraction()",float,doc = "electromagnetic energy fraction in HF",precision = 6),
  nMuons          = jetTable.variables.nMuons,
  nElectrons      = jetTable.variables.nElectrons,
  nConstituents   = jetTable.variables.nConstituents,
  nConstChHads    = Var("chargedHadronMultiplicity()",int,doc="number of charged hadrons in the jet"),
  nConstNeuHads   = Var("neutralHadronMultiplicity()",int,doc="number of neutral hadrons in the jet"),
  nConstHFHads    = Var("HFHadronMultiplicity()", int,doc="number of HF hadrons in the jet"),
  nConstHFEMs     = Var("HFEMMultiplicity()",int,doc="number of HF EMs in the jet"),
  nConstMuons     = Var("muonMultiplicity()",int,doc="number of muons in the jet"),
  nConstElecs     = Var("electronMultiplicity()",int,doc="number of electrons in the jet"),
  nConstPhotons   = Var("photonMultiplicity()",int,doc="number of photons in the jet"),
)
PUIDVARS = cms.PSet(
  puId_dR2Mean    = Var("userFloat('puId_dR2Mean')",float,doc="pT^2-weighted average square distance of jet constituents from the jet axis (PileUp ID BDT input variable)", precision= 6),
  puId_majW       = Var("userFloat('puId_majW')",float,doc="major axis of jet ellipsoid in eta-phi plane (PileUp ID BDT input variable)", precision= 6)  ,
  puId_minW       = Var("userFloat('puId_minW')",float,doc="minor axis of jet ellipsoid in eta-phi plane (PileUp ID BDT input variable)", precision= 6)  ,
  puId_frac01     = Var("userFloat('puId_frac01')",float,doc="fraction of constituents' pT contained within dR <0.1 (PileUp ID BDT input variable)", precision= 6)  ,
  puId_frac02     = Var("userFloat('puId_frac02')",float,doc="fraction of constituents' pT contained within 0.1< dR <0.2 (PileUp ID BDT input variable)", precision= 6) ,
  puId_frac03     = Var("userFloat('puId_frac03')",float,doc="fraction of constituents' pT contained within 0.2< dR <0.3 (PileUp ID BDT input variable)", precision= 6) ,
  puId_frac04     = Var("userFloat('puId_frac04')",float,doc="fraction of constituents' pT contained within 0.3< dR <0.4 (PileUp ID BDT input variable)", precision= 6) ,
  puId_ptD        = Var("userFloat('puId_ptD')",float,doc="pT-weighted average pT of constituents (PileUp ID BDT input variable)", precision= 6) ,
  puId_beta       = Var("userFloat('puId_beta')",float,doc="fraction of pT of charged constituents associated to PV (PileUp ID BDT input variable)", precision= 6) ,
  puId_pull       = Var("userFloat('puId_pull')",float,doc="magnitude of pull vector (PileUp ID BDT input variable)", precision= 6) ,
  puId_jetR       = Var("userFloat('puId_jetR')",float,doc="fraction of jet pT carried by the leading constituent (PileUp ID BDT input variable)", precision= 6) ,
  puId_jetRchg    = Var("userFloat('puId_jetRchg')",float,doc="fraction of jet pT carried by the leading charged constituent (PileUp ID BDT input variable)", precision= 6) ,
  puId_nCharged   = Var("userInt('puId_nCharged')",int,doc="number of charged constituents (PileUp ID BDT input variable)"),
)
QGLVARS = cms.PSet(
  qgl_axis2       =  Var("userFloat('qgl_axis2')",float,doc="ellipse minor jet axis (Quark vs Gluon likelihood input variable)", precision= 6),
  qgl_ptD         =  Var("userFloat('qgl_ptD')",float,doc="pT-weighted average pT of constituents (Quark vs Gluon likelihood input variable)", precision= 6),
  qgl_mult        =  Var("userInt('qgl_mult')", int,doc="PF candidates multiplicity (Quark vs Gluon likelihood input variable)"),
)
BTAGVARS = cms.PSet(
  btagDeepB = jetTable.variables.btagDeepB,
  btagCSVV2 = jetTable.variables.btagCSVV2,
  btagDeepCvL = jetTable.variables.btagDeepCvL,
)
DEEPJETVARS = cms.PSet(
  btagDeepFlavB   = jetTable.variables.btagDeepFlavB,
  btagDeepFlavC   = Var("bDiscriminator('pfDeepFlavourJetTags:probc')",float,doc="DeepFlavour charm tag raw score",precision=10),
  btagDeepFlavG   = Var("bDiscriminator('pfDeepFlavourJetTags:probg')",float,doc="DeepFlavour gluon tag raw score",precision=10),
  btagDeepFlavUDS = Var("bDiscriminator('pfDeepFlavourJetTags:probuds')",float,doc="DeepFlavour uds tag raw score",precision=10)
)
CALOJETVARS = cms.PSet(P4Vars,
  area      = jetTable.variables.area,
  rawFactor = jetTable.variables.rawFactor,
  emf       = Var("emEnergyFraction()", float, doc = "electromagnetic energy fraction", precision = 10),
)


#******************************************
#
#
# Reco Jets related functions
# 
#
#******************************************
def AddJetID(proc, jetName="", jetSrc="", jetTableName="", jetSequenceName=""):
  """
  Setup modules to calculate PF jet ID 
  """

  isPUPPIJet = True if "Puppi" in jetName else False
  
  looseJetId = "looseJetId{}".format(jetName)
  setattr(proc, looseJetId, proc.looseJetId.clone(
      src = jetSrc,
      filterParams=proc.looseJetId.filterParams.clone(
        version ="WINTER16"
      ),
    )
  )

  tightJetId = "tightJetId{}".format(jetName)
  setattr(proc, tightJetId, proc.tightJetId.clone(
      src = jetSrc,
      filterParams=proc.tightJetId.filterParams.clone(
        version = "SUMMER18{}".format("PUPPI" if isPUPPIJet else "")
      ),
    )
  )
  
  tightJetIdLepVeto = "tightJetIdLepVeto{}".format(jetName)
  setattr(proc, tightJetIdLepVeto, proc.tightJetIdLepVeto.clone(
      src = jetSrc,
      filterParams=proc.tightJetIdLepVeto.filterParams.clone(
        version = "SUMMER18{}".format("PUPPI" if isPUPPIJet else "")
      ),
    )
  )
  run2_jme_2016.toModify(getattr(proc, tightJetId) .filterParams,        version = "WINTER16" )
  run2_jme_2016.toModify(getattr(proc, tightJetIdLepVeto) .filterParams, version = "WINTER16" )
  run2_jme_2017.toModify(getattr(proc, tightJetId) .filterParams,        version = "WINTER17{}".format("PUPPI" if isPUPPIJet else ""))
  run2_jme_2017.toModify(getattr(proc, tightJetIdLepVeto) .filterParams, version = "WINTER17{}".format("PUPPI" if isPUPPIJet else ""))
  
  #
  # Save variables as userInts in each jet
  # 
  patJetWithUserData = "{}WithUserData".format(jetSrc)
  getattr(proc, patJetWithUserData).userInts.tightId = cms.InputTag(tightJetId)
  getattr(proc, patJetWithUserData).userInts.tightIdLepVeto = cms.InputTag(tightJetIdLepVeto)
  run2_jme_2016.toModify(getattr(proc, patJetWithUserData).userInts, looseId = cms.InputTag(looseJetId))

  #
  # Specfiy variables in the jetTable to save in NanoAOD
  #
  getattr(proc, jetTableName).variables.jetId = Var("userInt('tightId')*2+4*userInt('tightIdLepVeto')",int,doc="Jet ID flags bit1 is loose (always false in 2017 since it does not exist), bit2 is tight, bit3 is tightLepVeto")
  run2_jme_2016.toModify(getattr(proc, jetTableName).variables, jetId = Var("userInt('tightIdLepVeto')*4+userInt('tightId')*2+userInt('looseId')",int, doc="Jet ID flags bit1 is loose, bit2 is tight, bit3 is tightLepVeto"))

  getattr(proc,jetSequenceName).insert(getattr(proc,jetSequenceName).index(getattr(proc, jetSrc))+1, getattr(proc, tightJetId))
  getattr(proc,jetSequenceName).insert(getattr(proc,jetSequenceName).index(getattr(proc, tightJetId))+1, getattr(proc, tightJetIdLepVeto))
  
  setattr(proc,"_"+jetSequenceName+"_2016", getattr(proc,jetSequenceName).copy())
  getattr(proc,"_"+jetSequenceName+"_2016").insert(getattr(proc, "_"+jetSequenceName+"_2016").index(getattr(proc, tightJetId)), getattr(proc, looseJetId))
  run2_jme_2016.toReplaceWith(getattr(proc,jetSequenceName), getattr(proc, "_"+jetSequenceName+"_2016"))

  return proc

def AddPileUpJetIDVars(proc, jetName="", jetSrc="", jetTableName="", jetSequenceName=""):
  """
  Setup modules to calculate pileup jet ID input variables for PF jet
  """

  #
  # Calculate pileup jet ID variables
  #
  puJetIdVarsCalculator = "puJetIdCalculator{}".format(jetName) 
  setattr(proc, puJetIdVarsCalculator, pileupJetIdCalculator.clone(
      jets = jetSrc,
      vertexes  = "offlineSlimmedPrimaryVertices",
      inputIsCorrected = True,
      applyJec  = False,
      usePuppi = True if "Puppi" in jetName else False
    )
  )
  getattr(proc,jetSequenceName).insert(getattr(proc,jetSequenceName).index(getattr(proc, jetSrc))+1, getattr(proc, puJetIdVarsCalculator))

  #
  # Get the variables
  #
  puJetIDVar = "puJetIDVar{}".format(jetName)
  setattr(proc, puJetIDVar, cms.EDProducer("PileupJetIDVarProducer",
      srcJet = cms.InputTag(jetSrc),    
      srcPileupJetId = cms.InputTag(puJetIdVarsCalculator)
    )
  )
  getattr(proc,jetSequenceName).insert(getattr(proc,jetSequenceName).index(getattr(proc, puJetIdVarsCalculator))+1, getattr(proc, puJetIDVar))

  #
  # Save variables as userFloats and userInts for each jet
  # 
  patJetWithUserData = "{}WithUserData".format(jetSrc)
  getattr(proc,patJetWithUserData).userFloats.puId_dR2Mean  = cms.InputTag("{}:dR2Mean".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_majW     = cms.InputTag("{}:majW".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_minW     = cms.InputTag("{}:minW".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_frac01   = cms.InputTag("{}:frac01".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_frac02   = cms.InputTag("{}:frac02".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_frac03   = cms.InputTag("{}:frac03".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_frac04   = cms.InputTag("{}:frac04".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_ptD      = cms.InputTag("{}:ptD".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_beta     = cms.InputTag("{}:beta".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_pull     = cms.InputTag("{}:pull".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_jetR     = cms.InputTag("{}:jetR".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_jetRchg  = cms.InputTag("{}:jetRchg".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userInts.puId_nCharged   = cms.InputTag("{}:nCharged".format(puJetIDVar))

  #
  # Specfiy variables in the jet table to save in NanoAOD
  #
  getattr(proc,jetTableName).variables.puId_dR2Mean  = PUIDVARS.puId_dR2Mean 
  getattr(proc,jetTableName).variables.puId_majW     = PUIDVARS.puId_majW    
  getattr(proc,jetTableName).variables.puId_minW     = PUIDVARS.puId_minW    
  getattr(proc,jetTableName).variables.puId_frac01   = PUIDVARS.puId_frac01  
  getattr(proc,jetTableName).variables.puId_frac02   = PUIDVARS.puId_frac02  
  getattr(proc,jetTableName).variables.puId_frac03   = PUIDVARS.puId_frac03  
  getattr(proc,jetTableName).variables.puId_frac04   = PUIDVARS.puId_frac04  
  getattr(proc,jetTableName).variables.puId_ptD      = PUIDVARS.puId_ptD      
  getattr(proc,jetTableName).variables.puId_beta     = PUIDVARS.puId_beta    
  getattr(proc,jetTableName).variables.puId_pull     = PUIDVARS.puId_pull     
  getattr(proc,jetTableName).variables.puId_jetR     = PUIDVARS.puId_jetR    
  getattr(proc,jetTableName).variables.puId_jetRchg  = PUIDVARS.puId_jetRchg 
  getattr(proc,jetTableName).variables.puId_nCharged = PUIDVARS.puId_nCharged 

  return proc

def AddQGLTaggerVars(proc, jetName="", jetSrc="", jetTableName="", jetSequenceName="", calculateQGLVars=False):
  """
  Schedule the QGTagger module to calculate input variables to the QG likelihood 
  """

  QGLTagger="qgtagger{}".format(jetName)
  patJetWithUserData="{}WithUserData".format(jetSrc)

  if calculateQGLVars:
    setattr(proc, QGLTagger, qgtagger.clone(
        srcJets=jetSrc
      )
    )

  #
  # Save variables as userFloats and userInts for each jet
  # 
  getattr(proc,patJetWithUserData).userFloats.qgl_axis2 = cms.InputTag(QGLTagger+":axis2")
  getattr(proc,patJetWithUserData).userFloats.qgl_ptD   = cms.InputTag(QGLTagger+":ptD")
  getattr(proc,patJetWithUserData).userInts.qgl_mult    = cms.InputTag(QGLTagger+":mult")

  #
  # Specfiy variables in the jet table to save in NanoAOD
  #
  getattr(proc,jetTableName).variables.qgl_axis2 =  QGLVARS.qgl_axis2
  getattr(proc,jetTableName).variables.qgl_ptD   =  QGLVARS.qgl_ptD
  getattr(proc,jetTableName).variables.qgl_mult  =  QGLVARS.qgl_mult

  if calculateQGLVars:
    getattr(proc,jetSequenceName).insert(getattr(proc,jetSequenceName).index(getattr(proc, jetSrc))+1, getattr(proc, QGLTagger))

  return proc

def AddBTaggingScores(proc, jetTableName=""):
  """
  Store b-tagging scores from various algortihm
  """

  getattr(proc, jetTableName).variables.btagDeepB     = jetTable.variables.btagDeepB
  getattr(proc, jetTableName).variables.btagCSVV2     = jetTable.variables.btagCSVV2
  getattr(proc, jetTableName).variables.btagDeepCvL     = jetTable.variables.btagDeepCvL
  getattr(proc, jetTableName).variables.btagDeepFlavB = jetTable.variables.btagDeepFlavB
  getattr(proc, jetTableName).variables.btagDeepFlavCvL = jetTable.variables.btagDeepFlavCvL

  return proc

def AddDeepJetGluonLQuarkScores(proc, jetTableName=""):
  """
  Store DeepJet raw score in jetTable for gluon and light quark
  """

  getattr(proc, jetTableName).variables.btagDeepFlavG   = DEEPJETVARS.btagDeepFlavG  
  getattr(proc, jetTableName).variables.btagDeepFlavUDS = DEEPJETVARS.btagDeepFlavUDS

  return proc

def AddNewPatJets(proc, recoJetInfo, runOnMC):
  """
  Add patJet into custom nanoAOD
  """

  jetName = recoJetInfo.jetUpper
  payload = recoJetInfo.jetCorrPayload 
  doPF    = recoJetInfo.doPF
  doCalo  = recoJetInfo.doCalo

  if recoJetInfo.inputCollection != "":
    patJetFinalColl = recoJetInfo.inputCollection
  else: 
    patJetFinalColl = "selectedUpdatedPatJets{}Final".format(jetName)

  if doCalo:
    patJetFinalColl = "selectedPatJets{}".format(jetName)

  nanoInfoForJet = nanoInfo_recojets[recoJetInfo.jet]
  jetTablePrefix = nanoInfoForJet["name"]
  jetTableDoc    = nanoInfoForJet["doc"]
  ptcut          = nanoInfoForJet["ptcut"] if "ptcut" in nanoInfoForJet else ""
  doPUIDVar      = nanoInfoForJet["doPUIDVar"] if "doPUIDVar" in nanoInfoForJet else False
  doQGL          = nanoInfoForJet["doQGL"] if "doQGL" in nanoInfoForJet else False
  doBTag         = nanoInfoForJet["doBTag"] if "doBTag" in nanoInfoForJet else False

  SavePatJets(proc, 
    jetName, payload, patJetFinalColl, jetTablePrefix, jetTableDoc, doPF, doCalo, 
    ptcut=ptcut, doPUIDVar=doPUIDVar, doQGL=doQGL, doBTag=doBTag, runOnMC=runOnMC
  )

  return proc

def SavePatJets(proc, jetName, payload, patJetFinalColl, jetTablePrefix, jetTableDoc, 
                doPF, doCalo, ptcut="", doPUIDVar=False, doQGL=False, doBTag=False, runOnMC=False):
  """
  Schedule modules for a given patJet collection and save its variables into custom NanoAOD
  """

  #
  # Setup jet correction factors
  #
  jetCorrFactors = "jetCorrFactorsNano{}".format(jetName)
  setattr(proc, jetCorrFactors, jetCorrFactorsNano.clone(
      src = patJetFinalColl,
      payload = payload,
    )
  )

  #
  # Update jets
  #
  srcJets = "updatedJets{}".format(jetName)
  setattr(proc, srcJets, updatedJets.clone(
      jetSource = patJetFinalColl,
      jetCorrFactorsSource=[jetCorrFactors],
    )
  )
   
  #
  # Setup UserDataEmbedder
  #
  srcJetsWithUserData = "updatedJets{}WithUserData".format(jetName)
  setattr(proc, srcJetsWithUserData, cms.EDProducer("PATJetUserDataEmbedder",
      src = cms.InputTag(srcJets),
      userFloats = cms.PSet(),    
      userInts = cms.PSet(),
    )
  )
  
  #
  # Filter jets with pt cut
  #
  finalJetsForTable = "finalJets{}".format(jetName)
  setattr(proc, finalJetsForTable, finalJets.clone(
      src = srcJetsWithUserData,
      cut = ptcut
    )
  )

  #
  # Save jets in table
  #
  tableContent = PFJETVARS
  if doCalo:
    tableContent =  CALOJETVARS

  jetTable = "jet{}Table".format(jetName)
  setattr(proc,jetTable, cms.EDProducer("SimpleCandidateFlatTableProducer",
      src = cms.InputTag(finalJetsForTable),
      cut = cms.string(""), # Don't specify cuts here
      name = cms.string(jetTablePrefix),
      doc  = cms.string(jetTableDoc),
      singleton = cms.bool(False), # the number of entries is variable
      extension = cms.bool(False), # this is the main table for the jets
      variables = cms.PSet(tableContent)
    )
  )
  getattr(proc,jetTable).variables.pt.precision=10

  #
  # Save MC-only jet variables in table
  #
  jetMCTable = "jet{}MCTable".format(jetName)
  setattr(proc, jetMCTable, cms.EDProducer("SimpleCandidateFlatTableProducer",
      src = cms.InputTag(finalJetsForTable),
      cut = getattr(proc,jetTable).cut,
      name = cms.string(jetTablePrefix),
      singleton = cms.bool(False),
      extension = cms.bool(True), # this is an extension table
      variables = cms.PSet(
        partonFlavour = Var("partonFlavour()", int, doc="flavour from parton matching"),
        hadronFlavour = Var("hadronFlavour()", int, doc="flavour from hadron ghost clustering"),
        genJetIdx = Var("?genJetFwdRef().backRef().isNonnull()?genJetFwdRef().backRef().key():-1", int, doc="index of matched gen jet"),
      )
    )
  )

  #
  # Define the jet modules sequence first
  #
  jetSequenceName = "jet{}Sequence".format(jetName)
  setattr(proc, jetSequenceName, cms.Sequence(
      getattr(proc,jetCorrFactors)+
      getattr(proc,srcJets)+
      getattr(proc,srcJetsWithUserData)+
      getattr(proc,finalJetsForTable)
    )
  )

  #
  # Define the jet table sequences 
  #
  jetTableSequenceName = "jet{}TablesSequence".format(jetName)
  setattr(proc, jetTableSequenceName, cms.Sequence(getattr(proc,jetTable)))

  jetTableSequenceMCName = "jet{}MCTablesSequence".format(jetName)
  setattr(proc, jetTableSequenceMCName, cms.Sequence(getattr(proc,jetMCTable)))
  
  if runOnMC:
    proc.nanoSequenceMC += getattr(proc,jetSequenceName)
    proc.nanoSequenceMC += getattr(proc,jetTableSequenceName)
    proc.nanoSequenceMC += getattr(proc,jetTableSequenceMCName)
  else:
    proc.nanoSequence += getattr(proc,jetSequenceName)
    proc.nanoSequence += getattr(proc,jetTableSequenceName)

  #
  # Schedule plugins to calculate Jet ID, PileUp Jet ID input variables, and Quark-Gluon Likehood input variables.
  #
  if doPF:
    proc = AddJetID(proc, jetName=jetName, jetSrc=srcJets, jetTableName=jetTable, jetSequenceName=jetSequenceName)
    if doPUIDVar:
      proc = AddPileUpJetIDVars(proc, jetName=jetName, jetSrc=srcJets, jetTableName=jetTable, jetSequenceName=jetSequenceName)
    if doQGL:
      proc = AddQGLTaggerVars(proc,jetName=jetName, jetSrc=srcJets, jetTableName=jetTable, jetSequenceName=jetSequenceName, calculateQGLVars=True)
  
  #
  # Save b-tagging algorithm scores. Should only be done for jet collection with b-tagging 
  # calculated when reclustered or collection saved with b-tagging info in MiniAOD
  # 
  if doBTag:
    AddBTaggingScores(proc,jetTableName=jetTable)
    AddDeepJetGluonLQuarkScores(proc,jetTableName=jetTable)

  return proc

def ReclusterAK4CHSJets(proc, recoJA, runOnMC):
  """
  Recluster AK4 CHS jets and replace slimmedJets
  that is used as default to save AK4 CHS jets 
  in NanoAODs.  
  """
  print("custom_jme_cff::ReclusterAK4CHSJets: Recluster AK4 PF CHS jets")

  #
  # Recluster AK4 CHS jets
  #
  cfg = { 
    "jet" : "ak4pfchs",   
    "inputCollection" : "",                 
    "genJetsCollection": "AK4GenJetsNoNu",
    "bTagDiscriminators": bTagDiscriminatorsForAK4
  }
  recoJetInfo = recoJA.addRecoJetCollection(proc, **cfg) 

  jetName = recoJetInfo.jetUpper
  patJetFinalColl = "selectedUpdatedPatJets{}Final".format(jetName)
  
  #
  # Change the input jet source for jetCorrFactorsNano 
  # and updatedJets
  # 
  proc.jetCorrFactorsNano.src=patJetFinalColl
  proc.updatedJets.jetSource=patJetFinalColl

  #
  # Change pt cut
  #
  proc.finalJets.cut = "pt > 2"
  proc.simpleCleanerTable.jetSel = "pt > 10" # Change this from 15 -> 10 

  #
  # Add variables 
  #
  proc.jetTable.variables.hfHEF         = PFJETVARS.hfHEF
  proc.jetTable.variables.hfEmEF        = PFJETVARS.hfEmEF
  proc.jetTable.variables.nConstChHads  = PFJETVARS.nConstChHads
  proc.jetTable.variables.nConstNeuHads = PFJETVARS.nConstNeuHads
  proc.jetTable.variables.nConstHFHads  = PFJETVARS.nConstHFHads
  proc.jetTable.variables.nConstHFEMs   = PFJETVARS.nConstHFEMs
  proc.jetTable.variables.nConstMuons   = PFJETVARS.nConstMuons
  proc.jetTable.variables.nConstElecs   = PFJETVARS.nConstElecs
  proc.jetTable.variables.nConstPhotons = PFJETVARS.nConstPhotons

  proc.jetTable.doc = cms.string("AK4 PF CHS Jets with JECs applied, after basic selection (pt > 2)")

  #
  # Setup pileup jet ID with 80X training.
  # 
  pileupJetId80X = "pileupJetId80X"
  setattr(proc, pileupJetId80X, pileupJetId.clone(
      jets="updatedJets",
      algos=cms.VPSet(_chsalgos_81x),
      inputIsCorrected=True,
      applyJec=False,
      vertexes="offlineSlimmedPrimaryVertices"
    )
  )
  proc.jetSequence.insert(proc.jetSequence.index(proc.pileupJetId94X), getattr(proc, pileupJetId80X)) 

  proc.updatedJetsWithUserData.userInts.puId80XfullId = cms.InputTag('pileupJetId80X:fullId')
  run2_jme_2016.toModify(proc.updatedJetsWithUserData.userFloats, puId80XDisc = cms.InputTag("pileupJetId80X:fullDiscriminant"))

  proc.jetTable.variables.puId = Var("userInt('puId80XfullId')", int, doc="Pilup ID flags with 80X (2016) training")
  run2_jme_2016.toModify(proc.jetTable.variables, puIdDisc = Var("userFloat('puId80XDisc')",float,doc="Pilup ID discriminant with 80X (2016) training",precision=10))

  #
  # Add variables for pileup jet ID studies.
  #
  proc = AddPileUpJetIDVars(proc, 
    jetName="", 
    jetSrc="updatedJets", 
    jetTableName="jetTable",
    jetSequenceName="jetSequence"
  )
  #
  # Add variables for quark guon likelihood tagger studies.
  # Save variables as userFloats and userInts in each jet
  #
  proc.updatedJetsWithUserData.userFloats.qgl_axis2 = cms.InputTag("qgtagger:axis2")
  proc.updatedJetsWithUserData.userFloats.qgl_ptD   = cms.InputTag("qgtagger:ptD")
  proc.updatedJetsWithUserData.userInts.qgl_mult    = cms.InputTag("qgtagger:mult")
  #
  # Specfiy variables in the jetTable to save in NanoAOD
  #
  proc.jetTable.variables.qgl_axis2 =  QGLVARS.qgl_axis2
  proc.jetTable.variables.qgl_ptD   =  QGLVARS.qgl_ptD
  proc.jetTable.variables.qgl_mult  =  QGLVARS.qgl_mult
  #
  # Save DeepJet raw score for gluon and light quarks
  #
  proc.jetTable.variables.btagDeepFlavG   = DEEPJETVARS.btagDeepFlavG  
  proc.jetTable.variables.btagDeepFlavUDS = DEEPJETVARS.btagDeepFlavUDS

  #Adding hf shower shape producer to the jet sequence. By default this producer is not automatically rerun at the NANOAOD step
  #The following lines make sure it is.
  hfJetShowerShapeforCustomNanoAOD = "hfJetShowerShapeforCustomNanoAOD"
  setattr(proc, hfJetShowerShapeforCustomNanoAOD, hfJetShowerShapeforNanoAOD.clone(jets="updatedJets",vertices="offlineSlimmedPrimaryVertices") )
  proc.jetSequence.insert(proc.jetSequence.index(proc.updatedJetsWithUserData), getattr(proc, hfJetShowerShapeforCustomNanoAOD))
  proc.updatedJetsWithUserData.userFloats.hfsigmaEtaEta = cms.InputTag('hfJetShowerShapeforCustomNanoAOD:sigmaEtaEta')
  proc.updatedJetsWithUserData.userFloats.hfsigmaPhiPhi = cms.InputTag('hfJetShowerShapeforCustomNanoAOD:sigmaPhiPhi')
  proc.updatedJetsWithUserData.userInts.hfcentralEtaStripSize = cms.InputTag('hfJetShowerShapeforCustomNanoAOD:centralEtaStripSize')
  proc.updatedJetsWithUserData.userInts.hfadjacentEtaStripsSize = cms.InputTag('hfJetShowerShapeforCustomNanoAOD:adjacentEtaStripsSize')
  proc.jetTable.variables.hfsigmaEtaEta = Var("userFloat('hfsigmaEtaEta')",float,doc="sigmaEtaEta for HF jets (noise discriminating variable)",precision=10)
  proc.jetTable.variables.hfsigmaPhiPhi = Var("userFloat('hfsigmaPhiPhi')",float,doc="sigmaPhiPhi for HF jets (noise discriminating variable)",precision=10)
  proc.jetTable.variables.hfcentralEtaStripSize = Var("userInt('hfcentralEtaStripSize')", int, doc="eta size of the central tower strip in HF (noise discriminating variable) ")
  proc.jetTable.variables.hfadjacentEtaStripsSize = Var("userInt('hfadjacentEtaStripsSize')", int, doc="eta size of the strips next to the central tower strip in HF (noise discriminating variable) ")

  return proc

def AddVariablesForAK8PuppiJets(proc):
  """
  Add more variables for AK8 PFPUPPI jets
  """

  #
  #  These variables are not stored for AK8PFCHS (slimmedJetsAK8)
  #  in MiniAOD if their pt < 170 GeV. Hence the conditional fill.
  #
  proc.fatJetTable.variables.chHEF  = Var("?isPFJet()?chargedHadronEnergyFraction():-1", float, doc="charged Hadron Energy Fraction", precision = 6)
  proc.fatJetTable.variables.neHEF  = Var("?isPFJet()?neutralHadronEnergyFraction():-1", float, doc="neutral Hadron Energy Fraction", precision = 6)
  proc.fatJetTable.variables.chEmEF = Var("?isPFJet()?chargedEmEnergyFraction():-1", float, doc="charged Electromagnetic Energy Fraction", precision = 6)
  proc.fatJetTable.variables.neEmEF = Var("?isPFJet()?neutralEmEnergyFraction():-1", float, doc="neutral Electromagnetic Energy Fraction", precision = 6)
  proc.fatJetTable.variables.muEF   = Var("?isPFJet()?muonEnergyFraction():-1", float, doc="muon Energy Fraction", precision = 6)
  proc.fatJetTable.variables.hfHEF  = Var("?isPFJet()?HFHadronEnergyFraction():-1", float, doc="energy fraction in forward hadronic calorimeter", precision = 6)
  proc.fatJetTable.variables.hfEmEF = Var("?isPFJet()?HFEMEnergyFraction():-1", float, doc="energy fraction in forward EM calorimeter", precision = 6)
  proc.fatJetTable.variables.nConstChHads   = Var("?isPFJet()?chargedHadronMultiplicity():-1",int, doc="number of charged hadrons in the jet")
  proc.fatJetTable.variables.nConstNeuHads  = Var("?isPFJet()?neutralHadronMultiplicity():-1",int, doc="number of neutral hadrons in the jet")
  proc.fatJetTable.variables.nConstHFHads   = Var("?isPFJet()?HFHadronMultiplicity():-1", int, doc="number of HF Hadrons in the jet")
  proc.fatJetTable.variables.nConstHFEMs    = Var("?isPFJet()?HFEMMultiplicity():-1", int, doc="number of HF EMs in the jet")
  proc.fatJetTable.variables.nConstMuons    = Var("?isPFJet()?muonMultiplicity():-1", int, doc="number of muons in the jet")
  proc.fatJetTable.variables.nConstElecs    = Var("?isPFJet()?electronMultiplicity():-1", int, doc="number of electrons in the jet")
  proc.fatJetTable.variables.nConstPhotons  = Var("?isPFJet()?photonMultiplicity():-1", int, doc="number of photons in the jet")

  return proc
#******************************************
#
#
# Gen Jets related functions
# 
#
#******************************************
def AddNewGenJets(proc, genJetInfo):
  """
  Add genJet into custom nanoAOD
  """

  genJetName         = genJetInfo.jetUpper
  genJetAlgo         = genJetInfo.jetAlgo
  genJetSize         = genJetInfo.jetSize
  genJetSizeNr       = genJetInfo.jetSizeNr
  genJetFinalColl    = "{}{}{}".format(genJetAlgo.upper(), genJetSize, "GenJetsNoNu")
  genJetTablePrefix  = nanoInfo_genjets[genJetInfo.jet]["name"]
  genJetTableDoc     = nanoInfo_genjets[genJetInfo.jet]["doc"]

  SaveGenJets(proc, genJetName, genJetAlgo, genJetSizeNr, genJetFinalColl, genJetTablePrefix, genJetTableDoc, runOnMC=False)

  return proc

def SaveGenJets(proc, genJetName, genJetAlgo, genJetSizeNr, genJetFinalColl, genJetTablePrefix, genJetTableDoc, runOnMC=False):
  """
  Schedule modules for a given genJet collection and save its variables into custom NanoAOD
  """

  genJetTableThisJet = "jet{}Table".format(genJetName)
  setattr(proc, genJetTableThisJet, genJetTable.clone(
      src       = genJetFinalColl,
      cut       = "pt > 1",
      name      = genJetTablePrefix,
      doc       = genJetTableDoc,
      variables = GENJETVARS
    )
  )

  genJetFlavourAssociationThisJet = "genJet{}FlavourAssociation".format(genJetName)
  setattr(proc, genJetFlavourAssociationThisJet, genJetFlavourAssociation.clone(
      jets           = getattr(proc,genJetTableThisJet).src,
      jetAlgorithm   = supportedJetAlgos[genJetAlgo],
      rParam         = genJetSizeNr,
    )
  )

  genJetFlavourTableThisJet = "genJet{}FlavourTable".format(genJetName)
  setattr(proc, genJetFlavourTableThisJet, genJetFlavourTable.clone(
      name            = getattr(proc,genJetTableThisJet).name,
      src             = getattr(proc,genJetTableThisJet).src,
      cut             = getattr(proc,genJetTableThisJet).cut,
      jetFlavourInfos = genJetFlavourAssociationThisJet,
    )
  )

  genJetSequenceName = "genJet{}Sequence".format(genJetName)
  setattr(proc, genJetSequenceName, cms.Sequence(
      getattr(proc,genJetTableThisJet)+
      getattr(proc,genJetFlavourAssociationThisJet)+
      getattr(proc,genJetFlavourTableThisJet)
    )
  )
  proc.nanoSequenceMC.insert(proc.nanoSequenceMC.index(proc.jetMC)+1, getattr(proc,genJetSequenceName)) 

  return proc

def ReclusterAK4GenJets(proc, genJA):
  """
  Recluster AK4 Gen jets and replace 
  slimmedGenJets that is used as default 
  to save AK4 Gen jets in NanoAODs. 
  """
  print("custom_jme_cff::ReclusterAK4GenJets: Recluster AK4 Gen jets")

  #
  # Recluster AK4 Gen jet
  #
  cfg = { 
    "jet" : "ak4gen",   
  }
  genJetInfo = genJA.addGenJetCollection(proc, **cfg)

  genJetName            = genJetInfo.jetUpper
  genJetAlgo            = genJetInfo.jetAlgo
  genJetSize            = genJetInfo.jetSize
  genJetSizeNr          = genJetInfo.jetSizeNr
  selectedPatGenJets    = "{}{}{}".format(genJetAlgo.upper(), genJetSize, "GenJetsNoNu")
  
  #
  # Change jet source to the newly clustered jet collection. Set very low pt cut for jets 
  # to be stored in the GenJet Table
  #
  proc.genJetTable.src = selectedPatGenJets
  proc.genJetTable.cut = "pt > 1"
  proc.genJetTable.doc  ="AK4 Gen jets (made with visible genparticles)"

  genJetFlavourAssociationThisJet = "genJet{}FlavourAssociation".format(genJetName)
  setattr(proc, genJetFlavourAssociationThisJet, genJetFlavourAssociation.clone(
      jets           = proc.genJetTable.src,
      jetAlgorithm   = supportedJetAlgos[genJetAlgo],
      rParam         = genJetSizeNr,
    )
  )
  proc.jetMC.insert(proc.jetMC.index(proc.genJetFlavourTable), getattr(proc, genJetFlavourAssociationThisJet)) 
  return proc

def AddVariablesForAK4GenJets(proc):
  proc.genJetTable.variables.nConstituents = GENJETVARS.nConstituents
  return proc

def AddVariablesForAK8GenJets(proc):
  proc.genJetAK8Table.variables.nConstituents = GENJETVARS.nConstituents
  return proc

#===========================================================================
#
# Misc. functions
#
#===========================================================================
def RemoveAllJetPtCuts(proc):
  """
  Remove default pt cuts for all jets set in jets_cff.py 
  """

  proc.finalJets.cut             = "" # 15 -> 10
  proc.finalJetsAK8.cut          = "" # 170 -> 170
  proc.genJetTable.cut           = "" # 10 -> 8
  proc.genJetFlavourTable.cut    = "" # 10 -> 8
  proc.genJetAK8Table.cut        = "" # 100 -> 80
  proc.genJetAK8FlavourTable.cut = "" # 100 -> 80

  return proc

#===========================================================================
#
# CUSTOMIZATION function
#
#===========================================================================
def PrepJMECustomNanoAOD(process,runOnMC):
  
  ############################################################################
  # Remove all default jet pt cuts from jets_cff.py
  ############################################################################
  process = RemoveAllJetPtCuts(process)

  ###########################################################################
  #
  # Gen-level jets related functions. Only for MC.
  #
  ###########################################################################
  genJA = GenJetAdder()
  if runOnMC:
    ############################################################################
    # Save additional variables for AK8 GEN jets
    ############################################################################
    process = AddVariablesForAK8GenJets(process)
    ###########################################################################
    # Recluster AK4 GEN jets
    ###########################################################################
    process = ReclusterAK4GenJets(process, genJA)
    process = AddVariablesForAK4GenJets(process)
    ###########################################################################
    # Add additional GEN jets to NanoAOD
    ###########################################################################
    for jetConfig in config_genjets:
      cfg = { k : v for k, v in jetConfig.items() if k != "enabled"}
      genJetInfo = genJA.addGenJetCollection(process, **cfg)
      AddNewGenJets(process, genJetInfo)

  ###########################################################################
  #
  # Reco-level jets related functions. For both MC and data.
  #
  ###########################################################################
  recoJA = RecoJetAdder(runOnMC=runOnMC)
  ###########################################################################
  # Save additional variables for AK8Puppi jets
  ###########################################################################
  process = AddVariablesForAK8PuppiJets(process)
  ###########################################################################
  # Recluster AK4 CHS jets and replace "slimmedJets"
  ###########################################################################
  process = ReclusterAK4CHSJets(process, recoJA, runOnMC)
  ###########################################################################
  # Add additional Reco jets to NanoAOD
  ###########################################################################
  for jetConfig in config_recojets:
    cfg = { k : v for k, v in jetConfig.items() if k != "enabled"}
    recoJetInfo = recoJA.addRecoJetCollection(process, **cfg)
    AddNewPatJets(process, recoJetInfo, runOnMC)

  ###########################################################################
  # Save Maximum of Pt Hat Max
  ###########################################################################
  if runOnMC:
    process.puTable.savePtHatMax = True
  
  return process

def PrepJMECustomNanoAOD_MC(process):
  PrepJMECustomNanoAOD(process,runOnMC=True)
  return process

def PrepJMECustomNanoAOD_Data(process):
  PrepJMECustomNanoAOD(process,runOnMC=False)
  return process
