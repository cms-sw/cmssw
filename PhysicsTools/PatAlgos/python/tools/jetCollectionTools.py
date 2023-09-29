import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.ConfigToolBase import *

from CommonTools.ParticleFlow.pfCHS_cff import pfCHS

from CommonTools.PileupAlgos.softKiller_cfi import softKiller

from RecoJets.JetProducers.PFJetParameters_cfi         import PFJetParameters
from RecoJets.JetProducers.GenJetParameters_cfi        import GenJetParameters
from RecoJets.JetProducers.GenJetParameters_cfi        import GenJetParameters
from RecoJets.JetProducers.AnomalousCellParameters_cfi import AnomalousCellParameters

from RecoJets.JetProducers.ak4GenJets_cfi  import ak4GenJets
from RecoJets.JetProducers.ak4PFJets_cfi   import ak4PFJets, ak4PFJetsCHS, ak4PFJetsPuppi, ak4PFJetsSK, ak4PFJetsCS
from RecoJets.JetProducers.ak4CaloJets_cfi import ak4CaloJets

from PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cfi import updatedPatJets
from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi  import patJetCorrFactors
from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import patJetFlavourAssociation

from PhysicsTools.PatAlgos.tools.jetTools import supportedJetAlgos, addJetCollection, updateJetCollection
from PhysicsTools.PatAlgos.tools.jetTools import setupPuppiForPackedPF
from PhysicsTools.PatAlgos.tools.helpers  import getPatAlgosToolsTask, addToProcessAndTask

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
    self.jetAlgo    = jetMatch.group(algoKey)
    self.jetSize    = jetMatch.group(sizeKey)
    self.jetSizeNr  = float(self.jetSize) / 10.

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

  def addProcessAndTask(self, proc, label, module):
    task = getPatAlgosToolsTask(proc)
    addToProcessAndTask(label, module, proc, task)

  def addGenJetCollection(self,
      proc,
      jet,
      inputCollection    = "",
      minPtFastjet       = None,
    ):
    print("jetCollectionTools::GenJetAdder::addGenJetCollection: Adding Gen Jet Collection: {}".format(jet))

    #
    # Decide which genJet collection we are dealing with
    #
    genJetInfo = GenJetInfo(jet,inputCollection)
    jetLower = genJetInfo.jetLower
    jetUpper = genJetInfo.jetUpper

    #=======================================================
    #
    # If genJet collection in MiniAOD is not
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
        self.addProcessAndTask(proc, packedGenPartNoNu, cms.EDFilter("CandPtrSelector",
            src = cms.InputTag("packedGenParticles"),
            cut = cms.string("abs(pdgId) != 12 && abs(pdgId) != 14 && abs(pdgId) != 16"),
          )
        )
        self.prerequisites.append(packedGenPartNoNu)
      #
      # Create the GenJet collection
      #
      genJetsCollection = "{}{}{}".format(genJetInfo.jetAlgo.upper(), genJetInfo.jetSize, 'GenJetsNoNu')
      self.addProcessAndTask(proc, genJetsCollection, ak4GenJets.clone(
          src           = packedGenPartNoNu,
          jetAlgorithm  = cms.string(supportedJetAlgos[genJetInfo.jetAlgo]),
          rParam        = cms.double(genJetInfo.jetSizeNr),
        )
      )
      #
      # Set minimum pt threshold of gen jets to be saved after fastjet clustering
      #
      if minPtFastjet != None:
        getattr(proc, genJetsCollection).jetPtMin = minPtFastjet
      self.prerequisites.append(genJetsCollection)

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
    self.doPF   = self.jetReco == "pf"

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

    self.patJetFinalCollection = ""

#============================================
#
# RecoJetAdder
#
#============================================
class RecoJetAdder(object):
  """
  Tool to schedule modules for building a patJet collection from MiniAODs
  """
  def __init__(self,runOnMC=True):
    self.prerequisites = []
    self.main = []
    self.pfLabel = "packedPFCandidates"
    self.pvLabel = "offlineSlimmedPrimaryVertices"
    self.svLabel = "slimmedSecondaryVertices"
    self.muLabel = "slimmedMuons"
    self.elLabel = "slimmedElectrons"
    self.gpLabel = "prunedGenParticles"
    self.runOnMC = runOnMC
    self.patJetsInMiniAOD = ["slimmedJets", "slimmedJetsAK8", "slimmedJetsPuppi", "slimmedCaloJets"]

  def addProcessAndTask(self, proc, label, module):
    task = getPatAlgosToolsTask(proc)
    addToProcessAndTask(label, module, proc, task)

  def addRecoJetCollection(self,
    proc,
    jet,
    inputCollection    = "",
    minPtFastjet       = None,
    genJetsCollection  = "",
    bTagDiscriminators = ["None"],
    JETCorrLevels      = ["L1FastJet", "L2Relative", "L3Absolute", "L2L3Residual"],
    ):
    print("jetCollectionTools::RecoJetAdder::addRecoJetCollection: Adding Reco Jet Collection: {}".format(jet))

    currentTasks = []

    if inputCollection and inputCollection not in self.patJetsInMiniAOD:
      raise RuntimeError("Invalid input collection: %s" % inputCollection)

    #=======================================================
    #
    # Figure out which jet collection we're dealing with
    #
    #=======================================================
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
    # If the patJet collection in MiniAOD is not specified,
    # we have to build the patJet collection from scratch.
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
      # Specify PF candidates
      #
      pfCand = self.pfLabel
      #
      # Setup PU method for PF candidates
      #
      if recoJetInfo.jetPUMethod not in [ "", "cs" ]:
        pfCand += recoJetInfo.jetPUMethod


      #
      # Setup modules to perform PU mitigation for
      # PF candidates
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
          self.addProcessAndTask(proc, pfCand, pfCHS.clone(
              src = self.pfLabel
             )
           )
          self.prerequisites.append(pfCand)
        #
        # PUPPI
        #
        elif recoJetInfo.jetPUMethod == "puppi":
          pfCandWeight = setupPuppiForPackedPF(proc)[0]
          self.prerequisites.append(pfCandWeight)
        #
        # Softkiller
        #
        elif recoJetInfo.jetPUMethod == "sk":
          self.addProcessAndTask(proc, pfCand, softKiller.clone(
              PFCandidates = self.pfLabel,
              rParam = recoJetInfo.jetSizeNr,
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
        jetCollection = '{}Collection'.format(jetUpper)

        if jetCollection in self.main:
          raise ValueError("Step '%s' already implemented" % jetCollection)
        #
        # Cluster new jet
        #
        if recoJetInfo.jetPUMethod == "chs":
          self.addProcessAndTask(proc, jetCollection, ak4PFJetsCHS.clone(
              src = pfCand,
            )
          )
        elif recoJetInfo.jetPUMethod == "puppi":
          self.addProcessAndTask(proc, jetCollection, ak4PFJetsPuppi.clone(
              src = self.pfLabel,
              srcWeights = pfCandWeight
            )
          )
        elif recoJetInfo.jetPUMethod == "sk":
          self.addProcessAndTask(proc, pfCand, ak4PFJetsSK.clone(
              src = pfCand,
            )
          )
        elif recoJetInfo.jetPUMethod == "cs":
          self.addProcessAndTask(proc, jetCollection, ak4PFJetsCS.clone(
            src = pfCand,
          )
        )
        else:
          self.addProcessAndTask(proc, jetCollection, ak4PFJets.clone(
            src = pfCand,
          )
        )
        getattr(proc, jetCollection).jetAlgorithm = supportedJetAlgos[recoJetInfo.jetAlgo]
        getattr(proc, jetCollection).rParam = recoJetInfo.jetSizeNr
        #
        # Set minimum pt threshold of reco jets to be saved after fastjet clustering
        #
        if minPtFastjet != None:
          getattr(proc, jetCollection).jetPtMin = minPtFastjet
        currentTasks.append(jetCollection)
      else:
        jetCollection = inputCollection


      #=============================================
      #
      # Make patJet collection
      #
      #=============================================
      #
      # Jet correction
      #
      if recoJetInfo.jetPUMethod == "puppi":
        jetCorrLabel = "Puppi"
      elif recoJetInfo.jetPUMethod in [ "cs", "sk" ]:
        jetCorrLabel = "chs"
      else:
        jetCorrLabel = recoJetInfo.jetPUMethod

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

      postfix = "Recluster" if inputCollection == "" else ""
      addJetCollection(
        proc,
        labelName          = jetUpper,
        postfix            = postfix,
        jetSource          = cms.InputTag(jetCollection),
        algo               = recoJetInfo.jetAlgo,
        rParam             = recoJetInfo.jetSizeNr,
        pvSource           = cms.InputTag(self.pvLabel),
        pfCandidates       = cms.InputTag(self.pfLabel),
        svSource           = cms.InputTag(self.svLabel),
        muSource           = cms.InputTag(self.muLabel),
        elSource           = cms.InputTag(self.elLabel),
        genJetCollection   = cms.InputTag(genJetsCollection),
        genParticles       = cms.InputTag(self.gpLabel),
        jetCorrections     = jetCorrections,
      )

      #
      # Need to set this explicitly for PUPPI jets
      #
      if recoJetInfo.jetPUMethod == "puppi":
        getattr(proc, "patJetFlavourAssociation{}{}".format(jetUpper,postfix)).weights = cms.InputTag(pfCandWeight)

      getJetMCFlavour = not recoJetInfo.doCalo and recoJetInfo.jetPUMethod != "cs"
      if not self.runOnMC: #Remove modules for Gen-level object matching
        delattr(proc, 'patJetGenJetMatch{}{}'.format(jetUpper,postfix))
        delattr(proc, 'patJetPartonMatch{}{}'.format(jetUpper,postfix))
        getJetMCFlavour = False
      setattr(getattr(proc, "patJets{}{}".format(jetUpper,postfix)), "getJetMCFlavour", cms.bool(getJetMCFlavour))

      selectedPatJets = "selectedPatJets{}{}".format(jetUpper,postfix)
      #=============================================
      #
      # Update the patJet collection.
      # This is where we setup
      # -  JEC
      # -  b-tagging discriminators
      #
      #=============================================
      updateJetCollection(
        proc,
        labelName          = jetUpper,
        postfix            = "Final",
        jetSource          = cms.InputTag(selectedPatJets),
        jetCorrections     = jetCorrections,
        btagDiscriminators = bTagDiscriminators,
      )

      recoJetInfo.patJetFinalCollection = "selectedUpdatedPatJets{}{}".format(jetUpper,"Final")
    else:
      recoJetInfo.patJetFinalCollection = inputCollection

    self.main.extend(currentTasks)

    return recoJetInfo