# This configuration is an example that recalibrates the slimmedJets from MiniAOD
# and adds a new userfloat "oldJetMass" and an additional b-tag discriminator to them

## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *
## switch to uncheduled mode
process.options.allowUnscheduled = cms.untracked.bool(True)
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

updateJetCollection(
   process,
   jetSource = cms.InputTag('slimmedJets'),
   jetCorrections = ('AK4PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
   btagDiscriminators = ['pfCombinedSecondaryVertexV2BJetTags'] ## to add discriminators
)
process.updatedPatJets.userData.userFloats.src += ['oldJetMass']

## An example where the jet correction is undone
updateJetCollection(
   process,
   labelName = 'UndoneJEC',
   jetSource = cms.InputTag('slimmedJets'),
   jetCorrections = ('AK4PFchs', cms.vstring([]), 'None')
)
process.updatedPatJetsUndoneJEC.userData.userFloats.src = []

## An example where the jet correction are reapplied
updateJetCollection(
   process,
   labelName = 'ReappliedJEC',
   jetSource = cms.InputTag('selectedUpdatedPatJetsUndoneJEC'),
   jetCorrections = ('AK4PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None')
)
process.updatedPatJetsReappliedJEC.userData.userFloats.src = []

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
