import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import Var, P4Vars
from PhysicsTools.NanoAOD.jets_cff import jetTable

from RecoJets.JetProducers.PFJetParameters_cfi import PFJetParameters
from RecoJets.JetProducers.GenJetParameters_cfi import GenJetParameters
from RecoJets.JetProducers.AnomalousCellParameters_cfi import AnomalousCellParameters

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
from Configuration.Eras.Modifier_run2_nanoAOD_94X2016_cff import run2_nanoAOD_94X2016

from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection, supportedJetAlgos
from PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cfi import updatedPatJets
from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import patJetCorrFactors

from CommonTools.PileupAlgos.Puppi_cff import puppi
from CommonTools.PileupAlgos.softKiller_cfi import softKiller

import copy
import re


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
# JetInfo
#
#============================================
class GenJetInfo(object):
  def __init__(self, jet, inputCollection):
    self.jet = jet
    self.inputCollection = inputCollection
    algoKey     = 'algo'
    sizeKey     = 'size'
    recoKey     = 'reco'
    jetRegex = re.compile(
      r'(?P<{algo}>({algoList}))(?P<{size}>[0-9]+)gen'.format(
        algo     = algoKey,
        algoList = '|'.join(supportedJetAlgos.keys()),
        size     = sizeKey,
      )
    )

    jetMatch = jetRegex.match(jet.lower())
    if not jetMatch:
      raise RuntimeError('Invalid jet collection: %s' % jet)
    self.jetAlgo     = jetMatch.group(algoKey)
    self.jetSize     = jetMatch.group(sizeKey)
    self.jetSizeNr = float(self.jetSize) / 10.

#============================================
#
# GenJetAdder
#
#============================================
class GenJetAdder(object):
  def __init__(self):
    self.prerequisites = []
    self.main = []
    self.gpLabel = "prunedGenParticles"

  def getSequence(self, proc):
    tasks = self.prerequisites + self.main

    resultSequence = cms.Sequence()
    for idx, task in enumerate(tasks):
      if idx == 0:
        resultSequence = cms.Sequence(getattr(proc, task))
      else:
        resultSequence.insert(idx, getattr(proc, task))
    return resultSequence
  
  def addGenJetCollection(self,
      proc,
      jet,
      name,
      doc,
      inputCollection    = "",
      genName            = "",
      minPt              = 5.,
    ):
    print("custom_jme_cff::GenJetAdder::addGenJetCollection: Adding Gen Jet Collection: {}".format(jet))
    currentTasks = []
    #
    # Decide which jet collection we're dealing with
    #
    jetLower = jet.lower()
    jetUpper = jet.upper()
    tagName  = jetUpper
    genJetInfo = GenJetInfo(jet,inputCollection)

    #
    # Skip AK4GenJets and AK8GenJets.
    # They're already available in the default NanoAOD
    #
    if genJetInfo.jetSize  == "4" and genJetInfo.jetAlgo == "ak":
      pass
    elif genJetInfo.jetSize  == "8" and genJetInfo.jetAlgo == "ak":
      pass
    #=======================================================
    #
    # If jet collection in MiniAOD is  not 
    # specified, build the jet collection.
    #
    #========================================================
    if not inputCollection:
      print("custom_jme_cff::GenJetAdder::addGenJetCollection: inputCollection in NanoAOD not specified. Building genjet collection now")
      #
      # Setup Gen Particles
      #
      packedGenPartNoNu = "packedGenParticlesForJetsNoNu"
      if packedGenPartNoNu not in self.prerequisites:
        setattr(proc, packedGenPartNoNu, cms.EDFilter("CandPtrSelector",
            src = cms.InputTag("packedGenParticles"),
            cut = cms.string("abs(pdgId) != 12 && abs(pdgId) != 14 && abs(pdgId) != 16"),
          )
        )
        self.prerequisites.append(packedGenPartNoNu)    
      #
      # Create the GenJet collection
      #
      genJetsCollection = "{}{}{}".format(genJetInfo.jetAlgo.upper(), genJetInfo.jetSize, 'GenJetsNoNu')
      setattr(proc, genJetsCollection,
        cms.EDProducer("FastjetJetProducer",
          GenJetParameters.clone(
            src          = packedGenPartNoNu,
            doAreaFastjet = cms.bool(True),
          ),
          AnomalousCellParameters,
          jetAlgorithm = cms.string(supportedJetAlgos[genJetInfo.jetAlgo]),
          rParam       = cms.double(genJetInfo.jetSizeNr),
        )
      )
      self.prerequisites.append(genJetsCollection)

    ############################
    #
    # Tables
    #
    ############################
    genTable = "{}Table".format(tagName)
    if genTable in self.main:
      raise ValueError("Step '%s' already implemented" % genTable)

    setattr(proc, genTable, cms.EDProducer("SimpleCandidateFlatTableProducer",
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
    currentTasks.append(genTable)
    #
    # GenJet Flavour Labelling
    #
    genFlavour = "{}Flavour".format(tagName)
    if genFlavour in self.main:
      raise ValueError("Step '%s' already implemented" % genFlavour)
    setattr(proc, genFlavour, cms.EDProducer("JetFlavourClustering",
        jets                     = cms.InputTag(genJetsCollection),
        bHadrons                 = cms.InputTag("patJetPartons", "bHadrons"),
        cHadrons                 = cms.InputTag("patJetPartons", "cHadrons"),
        partons                  = cms.InputTag("patJetPartons", "physicsPartons"),
        leptons                  = cms.InputTag("patJetPartons", "leptons"),
        jetAlgorithm             = cms.string(supportedJetAlgos[genJetInfo.jetAlgo]),
        rParam                   = cms.double(genJetInfo.jetSizeNr),
        ghostRescaling           = cms.double(1e-18),
        hadronFlavourHasPriority = cms.bool(False),
      )
    )
    currentTasks.append(genFlavour)
    #
    # GenJet Flavour Table
    #
    genFlavourTable = "{}FlavourTable".format(tagName)
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
# RecoJetInfo
#
#============================================
class RecoJetInfo(object):
  def __init__(self, jet, inputCollection):
    self.jet = jet
    self.inputCollection = inputCollection
    algoKey     = 'algo'
    sizeKey     = 'size'
    recoKey     = 'reco'
    puMethodKey = 'puMethod'
    jetRegex = re.compile(
      r'(?P<{algo}>({algoList}))(?P<{size}>[0-9]+)(?P<{reco}>(pf|calo))(?P<{puMethod}>(chs|puppi|sk|cs|))'.format(
        algo     = algoKey,
        algoList = '|'.join(supportedJetAlgos.keys()),
        size     = sizeKey,
        reco     = recoKey,
        puMethod = puMethodKey,
      )
    )
    jetMatch = jetRegex.match(jet.lower())
    if not jetMatch:
      raise RuntimeError('Invalid jet collection: %s' % jet)

    self.jetAlgo     = jetMatch.group(algoKey)
    self.jetSize     = jetMatch.group(sizeKey)
    self.jetReco     = jetMatch.group(recoKey)
    self.jetPUMethod = jetMatch.group(puMethodKey)

    self.jetSizeNr = float(self.jetSize) / 10.

    self.doCalo = self.jetReco == "calo"
    self.doCS   = self.jetPUMethod == "cs"
    self.skipUserData = self.doCalo or (self.jetPUMethod in [ "puppi", "sk" ] and inputCollection == "")
    
    self.jetCorrPayload = "{}{}{}".format(
      self.jetAlgo.upper(), self.jetSize, "Calo" if self.doCalo else self.jetReco.upper()
    )
    if self.jetPUMethod == "puppi":
      self.jetCorrPayload += "Puppi"
    elif self.jetPUMethod in [ "cs", "sk" ]:
      self.jetCorrPayload += "chs"
    else:
      self.jetCorrPayload += self.jetPUMethod.lower()
      
#============================================
#
# RecoJetAdder
#
#============================================
class RecoJetAdder(object):
  def __init__(self):
    self.prerequisites = []
    self.main = []
    self.bTagDiscriminators = [
      'pfTrackCountingHighEffBJetTags',
      'pfTrackCountingHighPurBJetTags',
      'pfJetProbabilityBJetTags',
      'pfJetBProbabilityBJetTags',
      'pfSimpleSecondaryVertexHighEffBJetTags',
      'pfSimpleSecondaryVertexHighPurBJetTags',
      'pfCombinedSecondaryVertexV2BJetTags',
      'pfCombinedInclusiveSecondaryVertexV2BJetTags',
      'pfCombinedMVAV2BJetTags',
      'pfDeepCSVJetTags:probb',
      'pfDeepCSVJetTags:probbb',
      'pfBoostedDoubleSecondaryVertexAK8BJetTags',
    ]
    self.JETCorrLevels = [ "L1FastJet", "L2Relative", "L3Absolute" ]
    self.pfLabel = "packedPFCandidates"
    self.pvLabel = "offlineSlimmedPrimaryVertices"
    self.svLabel = "slimmedSecondaryVertices"
    self.muLabel = "slimmedMuons"
    self.elLabel = "slimmedElectrons"
    self.gpLabel = "prunedGenParticles"

  def getSequence(self, proc):
    tasks = self.prerequisites + self.main

    resultSequence = cms.Sequence()
    for idx, task in enumerate(tasks):
      if idx == 0:
        resultSequence = cms.Sequence(getattr(proc, task))
      else:
        resultSequence.insert(idx, getattr(proc, task))
    return resultSequence
  
  def addRecoJetCollection(self,
      proc,
      jet,
      name,
      doc,
      inputCollection    = "",
      genJetsCollection  = "",
      minPt              = 5.,
      bTagDiscriminators = None,
      JETCorrLevels      = None,
    ):
    print("custom_jme_cff::RecoJetAdder::addRecoJetCollection: Adding Reco Jet Collection: {}".format(jet))

    currentTasks = []

    #
    # Check if name already exists in NanoAOD.
    # Hard-coded at the moment.
    #
    if name in [ "Jet", "FatJet" ]:
      raise RuntimeError("Name already taken: %s" % name)

    if inputCollection and inputCollection not in [
        "slimmedJets", "slimmedJetsAK8", "slimmedJetsPuppi", "slimmedCaloJets",
      ]:
      raise RuntimeError("Invalid input collection: %s" % inputCollection)

    if bTagDiscriminators is None:
      bTagDiscriminators = self.bTagDiscriminators

    if JETCorrLevels is None:
      JETCorrLevels = self.JETCorrLevels
    
    #
    # Decide which jet collection we're dealing with
    #
    jetLower = jet.lower()
    jetUpper = jet.upper()
    tagName = jetUpper
    recoJetInfo = RecoJetInfo(jet, inputCollection)

    if inputCollection == "slimmedJets":
      assert(jetLower == "ak4pfchs")
    elif inputCollection == "slimmedJetsAK8":
      assert(jetLower == "ak8pfpuppi")
    elif inputCollection == "slimmedJetsPuppi":
      assert(jetLower == "ak4pfpuppi")
    elif inputCollection == "slimmedCaloJets":
      assert(jetLower == "ak4calo")
    
    #=======================================================
    #
    # If jet collection in MiniAOD is  not 
    # specified, build the jet collection.
    #
    #========================================================
    if not inputCollection or recoJetInfo.doCalo:
      print("custom_jme_cff::RecoJetAdder::addRecoJetCollection: inputCollection in NanoAOD not specified. Building recojet collection now")

      #=======================================================
      #
      # Prepare the inputs to jet clustering
      #
      #========================================================
      #
      # Set up PF candidates
      #
      pfCand = self.pfLabel
      #
      # Setup PU method for PF candidates
      # 
      if recoJetInfo.jetPUMethod not in [ "", "cs" ]:
        pfCand += recoJetInfo.jetPUMethod
      #
      #
      #
      if pfCand not in self.prerequisites:
        #
        # Skip if no PU Method or CS specified
        #
        if recoJetInfo.jetPUMethod in [ "", "cs" ]:
          pass
        #
        # CHS
        #
        elif recoJetInfo.jetPUMethod == "chs":
          setattr(proc, pfCand,
            cms.EDFilter("CandPtrSelector",
              src = cms.InputTag(self.pfLabel),
              cut = cms.string("fromPV"),
            )
          )
          self.prerequisites.append(pfCand)
        #
        # PUPPI
        #
        elif recoJetInfo.jetPUMethod == "puppi":
          setattr(proc, pfCand,
            puppi.clone(
              candName   = cms.InputTag(self.pfLabel),
              vertexName = cms.InputTag(self.pvLabel),
            )
          )
          self.prerequisites.append(pfCand)
        #
        # Softkiller
        #
        elif recoJetInfo.jetPUMethod == "sk":
          setattr(proc, pfCand,
            softKiller.clone(
              PFCandidates = cms.InputTag(self.pfLabel),
              rParam       = cms.double(recoJetInfo.jetSizeNr),
            )
          )
          self.prerequisites.append(pfCand)
        else:
          raise RuntimeError("Currently unsupported PU method: '%s'" % recoJetInfo.jetPUMethod)
      #============================================
      #
      # Create the recojet collection
      #
      #============================================
      if not recoJetInfo.doCalo:
        jetCollection = '{}Collection'.format(tagName)

        if jetCollection in self.main:
          raise ValueError("Step '%s' already implemented" % jetCollection)

        setattr(proc, jetCollection,
          cms.EDProducer("FastjetJetProducer",
            PFJetParameters.clone(
              src           = cms.InputTag(pfCand),
              doAreaFastjet = cms.bool(True),
              jetPtMin      = cms.double(minPt),
            ),
            AnomalousCellParameters,
            jetAlgorithm              = cms.string(supportedJetAlgos[recoJetInfo.jetAlgo]),
            rParam                    = cms.double(recoJetInfo.jetSizeNr),
            useConstituentSubtraction = cms.bool(recoJetInfo.doCS),
            csRParam                  = cms.double(0.4 if recoJetInfo.doCS else -1.),
            csRho_EtaMax              = PFJetParameters.Rho_EtaMax if recoJetInfo.doCS else cms.double(-1.),
            useExplicitGhosts         = cms.bool(recoJetInfo.doCS or recoJetInfo.jetPUMethod == "sk"),
          )
        )
        currentTasks.append(jetCollection)
      else:
        jetCollection = inputCollection
      
      #
      # PATify
      #
      if recoJetInfo.jetPUMethod == "puppi":
        jetCorrLabel = "Puppi"
      elif recoJetInfo.jetPUMethod in [ "cs", "sk" ]:
        jetCorrLabel = "chs"
      else:
        jetCorrLabel = recoJetInfo.jetPUMethod
      
      #
      # Jet correction
      #
      jetCorrections = (
        "{}{}{}{}".format(
          recoJetInfo.jetAlgo.upper(),
          recoJetInfo.jetSize,
          "Calo" if recoJetInfo.doCalo else recoJetInfo.jetReco.upper(),
          jetCorrLabel
        ),
        JETCorrLevels,
        "None",
      )
      
      addJetCollection(
        proc,
        labelName          = tagName,
        jetSource          = cms.InputTag(jetCollection),
        algo               = recoJetInfo.jetAlgo,
        rParam             = recoJetInfo.jetSizeNr,
        pvSource           = cms.InputTag(self.pvLabel),
        pfCandidates       = cms.InputTag(self.pfLabel),
        svSource           = cms.InputTag(self.svLabel),
        muSource           = cms.InputTag(self.muLabel),
        elSource           = cms.InputTag(self.elLabel),
        btagDiscriminators = bTagDiscriminators if not recoJetInfo.doCalo else [ "None" ],
        jetCorrections     = jetCorrections,
        genJetCollection   = cms.InputTag(genJetsCollection),
        genParticles       = cms.InputTag(self.gpLabel),
      )
      
      getJetMCFlavour = not recoJetInfo.doCalo and recoJetInfo.jetPUMethod != "cs"
      
      setattr(getattr(proc, "patJets{}".format(tagName)),           "getJetMCFlavour", cms.bool(getJetMCFlavour))
      setattr(getattr(proc, "patJetCorrFactors{}".format(tagName)), "payload",         cms.string(recoJetInfo.jetCorrPayload))
      selJet = "selectedPatJets{}".format(tagName)
    else:
      selJet = inputCollection

    if not recoJetInfo.skipUserData:
      #
      jercVar = "jercVars{}".format(tagName)
      if jercVar in self.main:
        raise ValueError("Step '%s' already implemented" % jercVar)
      setattr(proc, jercVar, proc.jercVars.clone(srcJet = cms.InputTag(selJet)))
      currentTasks.append(jercVar)
      #
      looseJetId = "looseJetId{}".format(tagName)
      if looseJetId in self.main:
        raise ValueError("Step '%s' already implemented" % looseJetId)
      setattr(proc, looseJetId, proc.looseJetId.clone(src = cms.InputTag(selJet)))
      #
      tightJetId = "tightJetId{}".format(tagName)
      if tightJetId in self.main:
        raise ValueError("Step '%s' already implemented" % tightJetId)
      setattr(proc, tightJetId, proc.tightJetId.clone(src = cms.InputTag(selJet)))
      currentTasks.append(tightJetId)
      #
      tightJetIdLepVeto = "tightJetIdLepVeto{}".format(tagName)
      if tightJetIdLepVeto in self.main:
        raise ValueError("Step '%s' already implemented" % tightJetIdLepVeto)
      setattr(proc, tightJetIdLepVeto, proc.tightJetIdLepVeto.clone(src = cms.InputTag(selJet)))
      currentTasks.append(tightJetIdLepVeto)
      #
      selectedPatJetsWithUserData = "{}WithUserData".format(selJet)
      if selectedPatJetsWithUserData in self.main:
        raise ValueError("Step '%s' already implemented" % selectedPatJetsWithUserData)
      setattr(proc, selectedPatJetsWithUserData,
        cms.EDProducer("PATJetUserDataEmbedder",
          src = cms.InputTag(selJet),
          userFloats = cms.PSet(
            jercCHPUF = cms.InputTag("{}:chargedHadronPUEnergyFraction".format(jercVar)),
            jercCHF   = cms.InputTag("{}:chargedHadronCHSEnergyFraction".format(jercVar)),
          ),
          userInts = cms.PSet(
            tightId        = cms.InputTag(tightJetId),
            tightIdLepVeto = cms.InputTag(tightJetIdLepVeto),
          ),
        )
      )
      for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_94X2016:
        selectedPatJetsWithUserDataObj = getattr(proc, selectedPatJetsWithUserData)
        modifier.toModify(selectedPatJetsWithUserDataObj.userInts,
          looseId        = cms.InputTag(looseJetId),
          tightIdLepVeto = None,
        )
      currentTasks.append(selectedPatJetsWithUserData)
    else:
      selectedPatJetsWithUserData = "selectedPatJets{}".format(tagName)

    # Not sure why we can't re-use patJetCorrFactors* created by addJetCollection() (even cloning doesn't work)
    # Let's just create our own
    jetCorrFactors = "jetCorrFactors{}".format(tagName)
    if jetCorrFactors in self.main:
      raise ValueError("Step '%s' already implemented" % jetCorrFactors)
    setattr(proc, jetCorrFactors, patJetCorrFactors.clone(
        src             = selectedPatJetsWithUserData,
        levels          = cms.vstring(JETCorrLevels),
        primaryVertices = cms.InputTag(self.pvLabel),
        payload         = cms.string(recoJetInfo.jetCorrPayload),
        rho             = "fixedGridRhoFastjetAll{}".format("Calo" if recoJetInfo.doCalo else ""),
      )
    )
    currentTasks.append(jetCorrFactors)

    updatedJets = "updatedJets{}".format(tagName)
    if updatedJets in self.main:
      raise ValueError("Step '%s' already implemented" % updatedJets)
    
    setattr(proc, updatedJets, updatedPatJets.clone(
        addBTagInfo          = False,
        jetSource            = selectedPatJetsWithUserData,
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag(jetCorrFactors)),
      )
    )
    currentTasks.append(updatedJets)
    
    ############################
    #
    # Tables
    #
    ############################
    table = "{}Table".format(tagName)
    if table in self.main:
      raise ValueError("Step '%s' already implemented" % table)
    
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

    if not recoJetInfo.skipUserData:
      altTasks = copy.deepcopy(currentTasks)
      for idx, task in enumerate(altTasks):
        if task == tightJetIdLepVeto:
          altTasks[idx] = looseJetId
      for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_94X2016:
        modifier.toReplaceWith(currentTasks, altTasks)

    self.main.extend(currentTasks)

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

