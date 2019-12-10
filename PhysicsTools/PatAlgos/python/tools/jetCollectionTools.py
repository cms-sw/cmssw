import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.ConfigToolBase import *

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
from Configuration.Eras.Modifier_run2_nanoAOD_94X2016_cff import run2_nanoAOD_94X2016

from RecoJets.JetProducers.PFJetParameters_cfi import PFJetParameters
from RecoJets.JetProducers.GenJetParameters_cfi import GenJetParameters
from RecoJets.JetProducers.AnomalousCellParameters_cfi import AnomalousCellParameters
from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets
from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJetsCS

from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection, supportedJetAlgos
from PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cfi import updatedPatJets
from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import patJetCorrFactors

from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import patJetFlavourAssociation

from CommonTools.PileupAlgos.Puppi_cff import puppi
from CommonTools.PileupAlgos.softKiller_cfi import softKiller

import re

#============================================
#
# GenJetInfo
#
#============================================
class GenJetInfo(object):
  """
  Class to hold information of a genjet collection
  """
  def __init__(self, jet, inputCollection):
    self.jet = jet
    self.jetLower = jet.lower()
    self.jetUpper = jet.upper()
    self.jetTagName = self.jetUpper
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
  """
  Tool to schedule modules for building a genjet collection with input MiniAODs
  """
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
      inputCollection    = "",
      genName            = "",
      minPt              = 5.,
    ):
    print("jetCollectionTools::GenJetAdder::addGenJetCollection: Adding Gen Jet Collection: {}".format(jet))
    currentTasks = []

    #
    # Decide which jet collection we're dealing with
    #
    jetLower = jet.lower()
    jetUpper = jet.upper()
    tagName  = jetUpper
    genJetInfo = GenJetInfo(jet,inputCollection)

    #=======================================================
    #
    # If gen jet collection in MiniAOD is not 
    # specified, build the genjet collection.
    #
    #========================================================
    if not inputCollection:
      print("jetCollectionTools::GenJetAdder::addGenJetCollection: inputCollection not specified. Building genjet collection now")
      #
      # Setup GenParticles
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
      setattr(proc, genJetsCollection, ak4GenJets.clone(
          src           = packedGenPartNoNu,
          jetAlgorithm  = cms.string(supportedJetAlgos[genJetInfo.jetAlgo]),
          rParam        = cms.double(genJetInfo.jetSizeNr),
        )
      )
      self.prerequisites.append(genJetsCollection)
    #
    # GenJet Flavour Labelling
    #
    genFlavour = "{}Flavour".format(genJetInfo.jetTagName)
    setattr(proc, genFlavour, patJetFlavourAssociation.clone(
        jets         = cms.InputTag(genJetsCollection),
        jetAlgorithm = cms.string(supportedJetAlgos[genJetInfo.jetAlgo]),
        rParam       = cms.double(genJetInfo.jetSizeNr),
      )
    )

    currentTasks.append(genFlavour)
    self.main.extend(currentTasks)

    return genJetInfo

#============================================
#
# RecoJetInfo
#
#============================================
class RecoJetInfo(object):
  """
  Class to hold information of a recojet collection
  """
  def __init__(self, jet, inputCollection):
    self.jet = jet
    self.jetLower   = jet.lower()
    self.jetUpper   = jet.upper()
    self.jetTagName = self.jetUpper
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
  """
  Tool to schedule modules for building a recojet collection with input MiniAODs
  """
  def __init__(self):    
    self.prerequisites = []
    self.main = []
    self.bTagDiscriminators = ["None"] # No b-tagging by default
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
    inputCollection    = "",
    genJetsCollection  = "",
    minPt              = 5.,
    bTagDiscriminators = None,
    JETCorrLevels      = None,
    ):
    print("jetCollectionTools::RecoJetAdder::addRecoJetCollection: Adding Reco Jet Collection: {}".format(jet))

    currentTasks = []

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
    recoJetInfo = RecoJetInfo(jet, inputCollection)
    jetLower = recoJetInfo.jetLower
    jetUpper = recoJetInfo.jetUpper
    tagName  = recoJetInfo.jetTagName

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
    # If jet collection in MiniAOD is not 
    # specified, build the jet collection.
    #
    #========================================================
    if not inputCollection or recoJetInfo.doCalo:
      print("jetCollectionTools::RecoJetAdder::addRecoJetCollection: inputCollection not specified. Building recojet collection now")

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
              candName   = self.pfLabel,
              vertexName = self.pvLabel,
            )
          )
          self.prerequisites.append(pfCand)
        #
        # Softkiller
        #
        elif recoJetInfo.jetPUMethod == "sk":
          setattr(proc, pfCand,
            softKiller.clone(
              PFCandidates = self.pfLabel,
              rParam       = recoJetInfo.jetSizeNr,
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

        setattr(proc, jetCollection, ak4PFJetsCS.clone(
            src                       = pfCand,
            doAreaFastjet             = True,
            jetPtMin                  = minPt,
            jetAlgorithm              = supportedJetAlgos[recoJetInfo.jetAlgo],
            rParam                    = recoJetInfo.jetSizeNr,
            useConstituentSubtraction = recoJetInfo.doCS,
            csRParam                  = 0.4 if recoJetInfo.doCS else -1.,
            csRho_EtaMax              = PFJetParameters.Rho_EtaMax if recoJetInfo.doCS else -1.,
            useExplicitGhosts         = recoJetInfo.doCS or recoJetInfo.jetPUMethod == "sk",
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
      # 
      #
      jercVar = "jercVars{}".format(tagName)
      if jercVar in self.main:
        raise ValueError("Step '%s' already implemented" % jercVar)
      setattr(proc, jercVar, proc.jercVars.clone(srcJet = selJet))
      currentTasks.append(jercVar)
      #
      # 
      #
      looseJetId = "looseJetId{}".format(tagName)
      if looseJetId in self.main:
        raise ValueError("Step '%s' already implemented" % looseJetId)
      setattr(proc, looseJetId, proc.looseJetId.clone(src = selJet))
      #
      # 
      #
      tightJetId = "tightJetId{}".format(tagName)
      if tightJetId in self.main:
        raise ValueError("Step '%s' already implemented" % tightJetId)
      setattr(proc, tightJetId, proc.tightJetId.clone(src = selJet))
      currentTasks.append(tightJetId)
      #
      # 
      #
      tightJetIdLepVeto = "tightJetIdLepVeto{}".format(tagName)
      if tightJetIdLepVeto in self.main:
        raise ValueError("Step '%s' already implemented" % tightJetIdLepVeto)
      setattr(proc, tightJetIdLepVeto, proc.tightJetIdLepVeto.clone(src = selJet))
      currentTasks.append(tightJetIdLepVeto)
      #
      # 
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
          looseId        = looseJetId,
          tightIdLepVeto = None,
        )
      currentTasks.append(selectedPatJetsWithUserData)
    else:
      selectedPatJetsWithUserData = "selectedPatJets{}".format(tagName)
    
    #
    # Not sure why we can't re-use patJetCorrFactors* created by addJetCollection() 
    # (even cloning doesn't work) Let's just create our own
    #
    jetCorrFactors = "jetCorrFactors{}".format(tagName)
    if jetCorrFactors in self.main:
      raise ValueError("Step '%s' already implemented" % jetCorrFactors)

    setattr(proc, jetCorrFactors, patJetCorrFactors.clone(
        src             = selectedPatJetsWithUserData,
        levels          = JETCorrLevels,
        primaryVertices = self.pvLabel,
        payload         = recoJetInfo.jetCorrPayload,
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
        jetCorrFactorsSource = [jetCorrFactors],
      )
    )

    currentTasks.append(updatedJets)
    self.main.extend(currentTasks)

    return recoJetInfo
