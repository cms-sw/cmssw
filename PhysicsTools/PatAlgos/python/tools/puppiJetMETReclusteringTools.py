import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.coreTools import runOnData
from PhysicsTools.PatAlgos.tools.jetTools import supportedJetAlgos, addJetCollection, updateJetCollection
from PhysicsTools.PatAlgos.tools.jetTools import setupPuppiForPackedPF
from PhysicsTools.PatAlgos.tools.helpers  import getPatAlgosToolsTask, addToProcessAndTask

def puppiAK4METReclusterFromMiniAOD(process, runOnMC, useExistingWeights, btagDiscriminatorsAK4=None):

  task = getPatAlgosToolsTask(process)

  pfLabel = "packedPFCandidates"
  pvLabel = "offlineSlimmedPrimaryVertices"
  svLabel = "slimmedSecondaryVertices"
  muLabel = "slimmedMuons"
  elLabel = "slimmedElectrons"
  gpLabel = "prunedGenParticles"

  genJetsCollection = "slimmedGenJets"

  #########################
  #
  # Setup puppi weights
  # Two instances of PuppiProducer:
  # 1) puppi (for jet reclustering)
  # 2) puppiNoLep (for MET reclustering)
  #
  ########################
  puppiLabel, puppiNoLepLabel = setupPuppiForPackedPF(process, useExistingWeights)

  #########################
  #
  # AK4 Puppi jets
  #
  ########################
  #
  # Recluster jets
  #
  process.load("RecoJets.JetProducers.ak4PFJets_cfi")
  task.add(process.ak4PFJetsPuppi)
  process.ak4PFJetsPuppi.src = pfLabel
  process.ak4PFJetsPuppi.srcWeights = puppiLabel

  from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import j2tParametersVX
  process.ak4PFJetsPuppiTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("ak4PFJetsPuppi")
  )
  task.add(process.ak4PFJetsPuppiTracksAssociatorAtVertex)
  process.patJetPuppiCharge = cms.EDProducer("JetChargeProducer",
    src = cms.InputTag("ak4PFJetsPuppiTracksAssociatorAtVertex"),
    var = cms.string("Pt"),
    exp = cms.double(1.0)
  )

  #
  # PATify jets
  #
  addJetCollection(
    process,
    postfix            = "",
    labelName          = "Puppi",
    jetSource          = cms.InputTag("ak4PFJetsPuppi"),
    algo               = "ak",
    rParam             = 0.4,
    pfCandidates       = cms.InputTag(pfLabel),
    pvSource           = cms.InputTag(pvLabel),
    svSource           = cms.InputTag(svLabel),
    muSource           = cms.InputTag(muLabel),
    elSource           = cms.InputTag(elLabel),
    genJetCollection   = cms.InputTag(genJetsCollection),
    genParticles       = cms.InputTag(gpLabel),
    jetCorrections     = ('AK4PFPuppi', cms.vstring(["L2Relative", "L3Absolute"]), ''),
    getJetMCFlavour    = runOnMC
  )
  process.patJetsPuppi.jetChargeSource = cms.InputTag("patJetPuppiCharge")
  process.selectedPatJetsPuppi.cut = cms.string("pt > 10")
  if hasattr(process,"patJetFlavourAssociationPuppi"):
    process.patJetFlavourAssociationPuppi.weights = cms.InputTag(puppiLabel)

  process.load("RecoJets.JetProducers.PileupJetID_cfi")
  task.add(process.pileUpJetIDPuppiTask)
  process.pileupJetIdPuppi.srcConstituentWeights = puppiLabel
  process.pileupJetIdPuppi.vertexes = pvLabel
  process.patJetsPuppi.userData.userFloats.src += [cms.InputTag("pileupJetIdPuppi:fullDiscriminant")]
  process.patJetsPuppi.userData.userInts.src += [cms.InputTag("pileupJetIdPuppi:fullId")]

  #=============================================
  #
  # Update the selectedPatJet collection.
  # This is where we setup
  # -  JEC
  # -  b-tagging discriminators
  #
  #=============================================
  # update slimmedJetsPuppi to include taggers
  from PhysicsTools.PatAlgos.slimming.slimmedJets_cfi import slimmedJets
  addToProcessAndTask('slimmedJetsPuppiNoDeepTags', slimmedJets.clone(
      src = "selectedPatJetsPuppi",
      packedPFCandidates = pfLabel,
      dropDaughters = "0",
      rekeyDaughters = "0",
    ),
    process, task
  )

  updateJetCollection(
    process,
    jetSource = cms.InputTag("slimmedJetsPuppiNoDeepTags"),
    # updateJetCollection defaults to MiniAOD inputs but
    # here it is made explicit (as in training or MINIAOD redoing)
    pfCandidates = cms.InputTag(pfLabel),
    pvSource = cms.InputTag(pvLabel),
    svSource = cms.InputTag(svLabel),
    muSource = cms.InputTag(muLabel),
    elSource = cms.InputTag(elLabel),
    jetCorrections = ("AK4PFPuppi", cms.vstring(["L2Relative", "L3Absolute"]), "None"),
    btagDiscriminators = btagDiscriminatorsAK4.names.value() if btagDiscriminatorsAK4 is not None else ['None'],
    postfix = 'SlimmedDeepFlavour',
    printWarning = False
  )

  addToProcessAndTask("slimmedJetsPuppi", process.selectedUpdatedPatJetsSlimmedDeepFlavour.clone(), process, task)
  del process.selectedUpdatedPatJetsSlimmedDeepFlavour

  ########################
  #
  # Recluster PuppiMET
  #
  ########################
  from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD
  runMetCorAndUncFromMiniAOD(process,
    isData=not(runOnMC),
    jetCollUnskimmed="slimmedJetsPuppi",
    metType="Puppi",
    postfix="Puppi",
    jetFlavor="AK4PFPuppi",
    puppiProducerLabel=puppiLabel,
    puppiProducerForMETLabel=puppiNoLepLabel,
    recoMetFromPFCs=True
  )

  ########################
  #
  # Modify JECs when processing real Data
  # Disable any MC-only features.
  #
  ########################
  if not(runOnMC):
    runOnData(process, names=["Jets","METs"], outputModules = [])

  return process


def puppiAK8ReclusterFromMiniAOD(process, runOnMC, useExistingWeights, btagDiscriminatorsAK8=None, btagDiscriminatorsAK8Subjets=None):

  task = getPatAlgosToolsTask(process)

  pfLabel = "packedPFCandidates"
  pvLabel = "offlineSlimmedPrimaryVertices"
  svLabel = "slimmedSecondaryVertices"
  muLabel = "slimmedMuons"
  elLabel = "slimmedElectrons"
  gpLabel = "prunedGenParticles"

  genJetsAK8Collection = "slimmedGenJetsAK8"
  genSubJetsForAK8Collection = "slimmedGenJetsAK8SoftDropSubJets"

  #########################
  #
  # Setup puppi weights
  # Two instances of PuppiProducer:
  # 1) puppi (for jet reclustering)
  # 2) puppiNoLep (for MET reclustering)
  #
  ########################
  puppiLabel, puppiNoLepLabel = setupPuppiForPackedPF(process, useExistingWeights)

  ########################
  #
  # AK8 Puppi jets
  #
  ########################
  #
  # Recluster jets and do soft-drop grooming
  #
  process.load("RecoJets.JetProducers.ak8PFJets_cfi")
  task.add(process.ak8PFJetsPuppi)
  task.add(process.ak8PFJetsPuppiSoftDrop)

  # AK8 jet constituents for softdrop
  process.ak8PFJetsPuppi.src = pfLabel
  process.ak8PFJetsPuppi.srcWeights = puppiLabel

  # AK8 jet constituents for softdrop
  from CommonTools.RecoAlgos.miniAODJetConstituentSelector_cfi import miniAODJetConstituentSelector
  addToProcessAndTask("ak8PFJetsPuppiConstituents", miniAODJetConstituentSelector.clone(
      src = cms.InputTag("ak8PFJetsPuppi"),
      cut = cms.string("pt > 100.0 && abs(rapidity()) < 2.4")
    ),
    process, task
  )

  # Soft-drop grooming
  process.ak8PFJetsPuppiSoftDrop.src = "ak8PFJetsPuppiConstituents:constituents"
  process.ak8PFJetsPuppiSoftDrop.srcWeights = puppiLabel

  # Soft-drop mass
  process.load("RecoJets.JetProducers.ak8PFJetsPuppi_groomingValueMaps_cfi")
  task.add(process.ak8PFJetsPuppiSoftDropMass)
  process.ak8PFJetsPuppiSoftDropMass.src = "ak8PFJetsPuppi"
  process.ak8PFJetsPuppiSoftDropMass.matched = "ak8PFJetsPuppiSoftDrop"
  #=============================================
  #
  # PATify
  #
  #=============================================
  #
  # AK8 jets
  #
  addJetCollection(
    process,
    labelName          = "AK8Puppi",
    jetSource          = cms.InputTag("ak8PFJetsPuppi"),
    algo               = "ak",
    rParam             = 0.8,
    pfCandidates       = cms.InputTag(pfLabel),
    pvSource           = cms.InputTag(pvLabel),
    svSource           = cms.InputTag(svLabel),
    muSource           = cms.InputTag(muLabel),
    elSource           = cms.InputTag(elLabel),
    genJetCollection   = cms.InputTag(genJetsAK8Collection),
    genParticles       = cms.InputTag(gpLabel),
    jetCorrections     = ("AK8PFPuppi", cms.vstring(["L2Relative", "L3Absolute"]), "None"),
    getJetMCFlavour    = runOnMC,
  )
  if hasattr(process,"patJetFlavourAssociationAK8Puppi"):
    process.patJetFlavourAssociationAK8Puppi.weights = cms.InputTag(puppiLabel)

  process.patJetsAK8Puppi.userData.userFloats.src = [] # start with empty list of user floats
  process.patJetsAK8Puppi.userData.userFloats.src += ["ak8PFJetsPuppiSoftDropMass"]
  process.patJetsAK8Puppi.addTagInfos = cms.bool(False)

  process.selectedPatJetsAK8Puppi.cut = cms.string("pt > 100")
  process.selectedPatJetsAK8Puppi.cutLoose = cms.string("pt > 30")
  process.selectedPatJetsAK8Puppi.nLoose = cms.uint32(3)

  #
  # Add AK8 Njetiness
  #
  from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness
  addToProcessAndTask("NjettinessAK8Puppi", Njettiness.clone(
      src = "ak8PFJetsPuppi",
      srcWeights = puppiLabel
    ),
    process, task
  )
  process.patJetsAK8Puppi.userData.userFloats.src += [
    "NjettinessAK8Puppi:tau1",
    "NjettinessAK8Puppi:tau2",
    "NjettinessAK8Puppi:tau3",
    "NjettinessAK8Puppi:tau4"
  ]

  #
  # AK8 soft-drop jets
  #
  addJetCollection(
    process,
    labelName = "AK8PFPuppiSoftDrop",
    jetSource = cms.InputTag("ak8PFJetsPuppiSoftDrop"),
    btagDiscriminators = ["None"],
    pfCandidates = cms.InputTag(pfLabel),
    pvSource = cms.InputTag(pvLabel),
    svSource = cms.InputTag(svLabel),
    muSource = cms.InputTag(muLabel),
    elSource = cms.InputTag(elLabel),
    genJetCollection = cms.InputTag(genJetsAK8Collection),
    genParticles = cms.InputTag(gpLabel),
    jetCorrections = ("AK8PFPuppi", cms.vstring(["L2Relative", "L3Absolute"]), "None"),
    getJetMCFlavour = False # jet flavor disabled regardless if running on MC or data
  )

  #
  # Soft-drop subjets
  #
  addJetCollection(
    process,
    labelName = "AK8PFPuppiSoftDropSubjets",
    jetSource = cms.InputTag("ak8PFJetsPuppiSoftDrop", "SubJets"),
    algo = "ak",  # needed for subjet flavor clustering
    rParam = 0.8, # needed for subjet flavor clustering
    explicitJTA = True,  # needed for subjet b tagging
    svClustering = True, # needed for subjet b tagging
    pfCandidates = cms.InputTag(pfLabel),
    pvSource = cms.InputTag(pvLabel),
    svSource = cms.InputTag(svLabel),
    muSource = cms.InputTag(muLabel),
    elSource = cms.InputTag(elLabel),
    genJetCollection = cms.InputTag(genSubJetsForAK8Collection),
    genParticles = cms.InputTag(gpLabel),
    fatJets = cms.InputTag("ak8PFJetsPuppi"),               # needed for subjet flavor clustering
    groomedFatJets = cms.InputTag("ak8PFJetsPuppiSoftDrop"), # needed for subjet flavor clustering
    jetCorrections = ("AK4PFPuppi", cms.vstring(["L2Relative", "L3Absolute"]), "None"),
  )
  if hasattr(process,"patJetFlavourAssociationAK8PFPuppiSoftDropSubjets"):
    process.patJetFlavourAssociationAK8PFPuppiSoftDropSubjets.weights = cms.InputTag(puppiLabel)

  #=============================================
  #
  #
  #
  #=============================================
  #
  # add groomed ECFs and N-subjettiness to soft dropped pat::Jets for fat jets and subjets
  #
  process.load('RecoJets.JetProducers.ECF_cff')

  addToProcessAndTask('nb1AK8PuppiSoftDrop', process.ecfNbeta1.clone(
      src = cms.InputTag("ak8PFJetsPuppiSoftDrop"),
      srcWeights = puppiLabel,
      cuts = cms.vstring('', '', 'pt > 250')
    ),
    process, task
  )
  process.patJetsAK8PFPuppiSoftDrop.userData.userFloats.src += [
    'nb1AK8PuppiSoftDrop:ecfN2',
    'nb1AK8PuppiSoftDrop:ecfN3',
  ]

  addToProcessAndTask('nb2AK8PuppiSoftDrop', process.ecfNbeta2.clone(
      src = cms.InputTag("ak8PFJetsPuppiSoftDrop"),
      srcWeights = puppiLabel,
      cuts = cms.vstring('', '', 'pt > 250')
    ),
    process, task
  )
  process.patJetsAK8PFPuppiSoftDrop.userData.userFloats.src += [
    'nb2AK8PuppiSoftDrop:ecfN2',
    'nb2AK8PuppiSoftDrop:ecfN3',
  ]

  #
  # add groomed ECFs and N-subjettiness to soft drop subjets
  #
  addToProcessAndTask("nb1AK8PuppiSoftDropSubjets", process.ecfNbeta1.clone(
      src = cms.InputTag("ak8PFJetsPuppiSoftDrop", "SubJets"),
      srcWeights = puppiLabel,
    ),
    process, task
  )

  process.patJetsAK8PFPuppiSoftDropSubjets.userData.userFloats.src += [
    'nb1AK8PuppiSoftDropSubjets:ecfN2',
    'nb1AK8PuppiSoftDropSubjets:ecfN3'
  ]

  addToProcessAndTask("nb2AK8PuppiSoftDropSubjets", process.ecfNbeta2.clone(
      src = cms.InputTag("ak8PFJetsPuppiSoftDrop", "SubJets"),
      srcWeights = puppiLabel,
    ),
    process, task
  )

  process.patJetsAK8PFPuppiSoftDropSubjets.userData.userFloats.src += [
    'nb2AK8PuppiSoftDropSubjets:ecfN2',
    'nb2AK8PuppiSoftDropSubjets:ecfN3'
  ]

  addToProcessAndTask("NjettinessAK8Subjets", Njettiness.clone(
      src = cms.InputTag("ak8PFJetsPuppiSoftDrop", "SubJets"),
      srcWeights = puppiLabel
    ),
    process, task
  )
  process.patJetsAK8PFPuppiSoftDropSubjets.userData.userFloats.src += [
    "NjettinessAK8Subjets:tau1",
    "NjettinessAK8Subjets:tau2",
    "NjettinessAK8Subjets:tau3",
    "NjettinessAK8Subjets:tau4",
  ]

  addToProcessAndTask("slimmedJetsAK8PFPuppiSoftDropSubjetsNoDeepTags", cms.EDProducer("PATJetSlimmer",
      src = cms.InputTag("selectedPatJetsAK8PFPuppiSoftDropSubjets"),
      packedPFCandidates = cms.InputTag(pfLabel),
      dropJetVars = cms.string("1"),
      dropDaughters = cms.string("0"),
      rekeyDaughters = cms.string("0"),
      dropTrackRefs = cms.string("1"),
      dropSpecific = cms.string("1"),
      dropTagInfos = cms.string("1"),
      modifyJets = cms.bool(True),
      mixedDaughters = cms.bool(False),
      modifierConfig = cms.PSet( modifications = cms.VPSet() )
    ),
    process, task
  )

  updateJetCollection(
    process,
    labelName = "AK8PFPuppiSoftDropSubjets",
    postfix = 'SlimmedDeepFlavour',
    jetSource = cms.InputTag("slimmedJetsAK8PFPuppiSoftDropSubjetsNoDeepTags"),
    # updateJetCollection defaults to MiniAOD inputs but
    # here it is made explicit (as in training or MINIAOD redoing)
    pfCandidates = cms.InputTag(pfLabel),
    pvSource = cms.InputTag(pvLabel),
    svSource = cms.InputTag(svLabel),
    muSource = cms.InputTag(muLabel),
    elSource = cms.InputTag(elLabel),
    jetCorrections = ("AK4PFPuppi", cms.vstring(["L2Relative", "L3Absolute"]), "None"),
    printWarning = False,
    btagDiscriminators = btagDiscriminatorsAK8Subjets.names.value() if btagDiscriminatorsAK8Subjets is not None else ['None'],
  )

  ## Establish references between PATified fat jets and subjets using the BoostedJetMerger
  addToProcessAndTask("slimmedJetsAK8PFPuppiSoftDropPacked", cms.EDProducer("BoostedJetMerger",
      jetSrc    = cms.InputTag("selectedPatJetsAK8PFPuppiSoftDrop"),
      subjetSrc = cms.InputTag("selectedUpdatedPatJetsAK8PFPuppiSoftDropSubjetsSlimmedDeepFlavour")
    ),
    process, task
  )

  addToProcessAndTask("packedPatJetsAK8", cms.EDProducer("JetSubstructurePacker",
      jetSrc = cms.InputTag("selectedPatJetsAK8Puppi"),
      distMax = cms.double(0.8),
      algoTags = cms.VInputTag(
        cms.InputTag("slimmedJetsAK8PFPuppiSoftDropPacked")
      ),
      algoLabels = cms.vstring(
        'SoftDropPuppi'
      ),
      fixDaughters = cms.bool(False),
      packedPFCandidates = cms.InputTag(pfLabel),
    ),
    process, task
  )

  #=============================================
  #
  # Update the selectedPatJet collection.
  # This is where we setup
  # -  JEC
  # -  b-tagging discriminators
  #
  #=============================================
  from PhysicsTools.PatAlgos.slimming.slimmedJets_cfi import slimmedJetsAK8
  addToProcessAndTask("slimmedJetsAK8NoDeepTags", slimmedJetsAK8.clone(rekeyDaughters = "0"), process, task)
  # Reconfigure the slimmedAK8 jet information to keep
  process.slimmedJetsAK8NoDeepTags.dropDaughters = cms.string("pt < 170")
  process.slimmedJetsAK8NoDeepTags.dropSpecific = cms.string("pt < 170")
  process.slimmedJetsAK8NoDeepTags.dropTagInfos = cms.string("pt < 170")

  updateJetCollection(
    process,
    jetSource = cms.InputTag("slimmedJetsAK8NoDeepTags"),
    # updateJetCollection defaults to MiniAOD inputs but
    # here it is made explicit (as in training or MINIAOD redoing)
    pfCandidates = cms.InputTag(pfLabel),
    pvSource = cms.InputTag(pvLabel),
    svSource = cms.InputTag(svLabel),
    muSource = cms.InputTag(muLabel),
    elSource = cms.InputTag(elLabel),
    rParam = 0.8,
    jetCorrections = ('AK8PFPuppi', cms.vstring(["L2Relative", "L3Absolute"]), 'None'),
    btagDiscriminators = btagDiscriminatorsAK8.names.value() if btagDiscriminatorsAK8 is not None else ['None'],
    postfix = "SlimmedAK8DeepTags",
    printWarning = False
  )

  addToProcessAndTask("slimmedJetsAK8", process.selectedUpdatedPatJetsSlimmedAK8DeepTags.clone(), process, task)
  del process.selectedUpdatedPatJetsSlimmedAK8DeepTags

  ########################
  #
  # Modify JECs when processing real Data
  # Disable any MC-only features.
  #
  ########################
  if not(runOnMC):
    runOnData(process, names=["Jets"], outputModules = [])

  return process

