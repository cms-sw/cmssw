# This configuration is an example that recalibrates the slimmedJets from MiniAOD
# and adds a new userfloat "oldJetMass" and an additional b-tag discriminator to them

## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *
#process.Tracer = cms.Service("Tracer")

## uncomment the following line to update different jet collections
## and add them to the event content
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection, addJetCollection

## An example where the jet energy correction are updated to the current GlobalTag
## and a userFloat containing the previous mass of the jet and an additional
## b-tag discriminator are added
from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsPuppiSoftDropMass
process.oldJetMass = ak8PFJetsPuppiSoftDropMass.clone(
  src = cms.InputTag("slimmedJets"),
  matched = cms.InputTag("slimmedJets") )
patAlgosToolsTask.add(process.oldJetMass)

updateJetCollection(
   process,
   jetSource = cms.InputTag('slimmedJets'),
   jetCorrections = ('AK4PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
   btagDiscriminators = [
      'pfCombinedSecondaryVertexV2BJetTags',

      'pfDeepCSVJetTags:probudsg', 
      'pfDeepCSVJetTags:probb', 
      'pfDeepCSVJetTags:probc', 
      'pfDeepCSVJetTags:probbb', 
      ## probcc was merged with probc
      ## 'pfDeepCSVJetTags:probcc',

      ## 'pfDeepCMVAJetTags:probudsg', 
      ## 'pfDeepCMVAJetTags:probb', 
      ## 'pfDeepCMVAJetTags:probc', 
      ## 'pfDeepCMVAJetTags:probbb', 
      ## 'pfDeepCMVAJetTags:probcc'
      ] ## to add discriminators
)

process.updatedPatJets.userData.userFloats.src += ['oldJetMass']

from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpMINIAODSIM
process.source.fileNames = filesRelValTTbarPileUpMINIAODSIM
#                                         ##
process.maxEvents.input = 100
#                                         ##
from Configuration.EventContent.EventContent_cff import MINIAODSIMEventContent
process.out.outputCommands = MINIAODSIMEventContent.outputCommands
process.out.outputCommands.append('keep *_selectedUpdatedPatJets*_*_*')
process.out.outputCommands.append('keep *_pfDeepCSVTagInfos*_*_*')
process.out.outputCommands.append('keep *_updatedPatJets*_*_*')
#                                         ##
process.out.fileName = 'testDeepCSV.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
# process.add_(cms.Service("InitRootHandlers", DebugLevel =cms.untracked.int32(3)))
