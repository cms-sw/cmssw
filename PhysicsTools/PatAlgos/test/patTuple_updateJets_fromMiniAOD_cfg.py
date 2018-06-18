# This configuration is an example that recalibrates the slimmedJets from MiniAOD
# and adds a new userfloat "oldJetMass" and an additional b-tag discriminator to them

## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

#process.Tracer = cms.Service("Tracer")

## uncomment the following line to update different jet collections
## and add them to the event content
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection

## An example where the jet energy correction are updated to the current GlobalTag
## and a userFloat containing the previous mass of the jet and an additional
## b-tag discriminator are added
from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsCHSSoftDropMass
process.oldJetMass = ak8PFJetsCHSSoftDropMass.clone(
  src = cms.InputTag("slimmedJets"),
  matched = cms.InputTag("slimmedJets") )
patAlgosToolsTask.add(process.oldJetMass)

updateJetCollection(
   process,
   jetSource = cms.InputTag('slimmedJets'),
   jetCorrections = ('AK4PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
   btagDiscriminators = ['pfCombinedSecondaryVertexV2BJetTags', 'pfDeepCSVDiscriminatorsJetTags:BvsAll', 'pfDeepCSVDiscriminatorsJetTags:CvsB', 'pfDeepCSVDiscriminatorsJetTags:CvsL'], ## to add discriminators,
   btagPrefix = 'TEST',
)
process.updatedPatJets.userData.userFloats.src += ['oldJetMass']

## An example where the jet corrections are undone
updateJetCollection(
   process,
   labelName = 'UndoneJEC',
   jetSource = cms.InputTag('slimmedJets'),
   jetCorrections = ('AK4PFchs', cms.vstring([]), 'None')
)
process.updatedPatJetsUndoneJEC.userData.userFloats.src = []

## An example where the jet corrections are reapplied
updateJetCollection(
   process,
   labelName = 'ReappliedJEC',
   jetSource = cms.InputTag('selectedUpdatedPatJetsUndoneJEC'),
   jetCorrections = ('AK4PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None')
)
process.updatedPatJetsReappliedJEC.userData.userFloats.src = []

## An example where the jet energy corrections are updated to the current GlobalTag
## and specified b-tag discriminators are rerun and added to SoftDrop subjets
updateJetCollection(
   process,
   labelName = 'SoftDropSubjets',
   jetSource = cms.InputTag('slimmedJetsAK8PFPuppiSoftDropPacked:SubJets'),
   jetCorrections = ('AK4PFPuppi', cms.vstring(['L2Relative', 'L3Absolute']), 'None'),
   btagDiscriminators = ['pfCombinedSecondaryVertexV2BJetTags', 'pfCombinedInclusiveSecondaryVertexV2BJetTags'],
   explicitJTA = True,          # needed for subjet b tagging
   svClustering = False,        # needed for subjet b tagging (IMPORTANT: Needs to be set to False to disable ghost-association which does not work with slimmed jets)
   fatJets = cms.InputTag('slimmedJetsAK8'), # needed for subjet b tagging
   rParam = 0.8,                # needed for subjet b tagging
   algo = 'ak'                  # has to be defined but is not used with svClustering=False
)
process.updatedPatJetsSoftDropSubjets.userData.userFloats.src = []

## An example where puppi jet specifics are computed
from PhysicsTools.PatAlgos.patPuppiJetSpecificProducer_cfi import patPuppiJetSpecificProducer
process.patPuppiJetSpecificProducer = patPuppiJetSpecificProducer.clone(
  src=cms.InputTag("slimmedJetsPuppi"),
  )
patAlgosToolsTask.add(process.patPuppiJetSpecificProducer)

updateJetCollection(
   process,
   labelName = 'PuppiJetSpecific',
   jetSource = cms.InputTag('slimmedJetsPuppi'),
)
process.updatedPatJetsPuppiJetSpecific.userData.userFloats.src = ['patPuppiJetSpecificProducer:puppiMultiplicity', 'patPuppiJetSpecificProducer:neutralPuppiMultiplicity', 'patPuppiJetSpecificProducer:neutralHadronPuppiMultiplicity', 'patPuppiJetSpecificProducer:photonPuppiMultiplicity', 'patPuppiJetSpecificProducer:HFHadronPuppiMultiplicity', 'patPuppiJetSpecificProducer:HFEMPuppiMultiplicity' ]

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpMINIAODSIM
process.source.fileNames = filesRelValTTbarPileUpMINIAODSIM
#                                         ##
process.maxEvents.input = 100
#                                         ##
from Configuration.EventContent.EventContent_cff import MINIAODSIMEventContent
process.out.outputCommands = MINIAODSIMEventContent.outputCommands
process.out.outputCommands.append('keep *_selectedUpdatedPatJets*_*_*')
process.out.outputCommands.append('drop *_selectedUpdatedPatJets*_caloTowers_*')
process.out.outputCommands.append('drop *_selectedUpdatedPatJets*_genJets_*')
process.out.outputCommands.append('drop *_selectedUpdatedPatJets*_pfCandidates_*')
#                                         ##
process.out.fileName = 'patTuple_updateJets_fromMiniAOD.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
