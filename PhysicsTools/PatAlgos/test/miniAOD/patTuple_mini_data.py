## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import cms, process
## switch to uncheduled mode
process.options.allowUnscheduled = cms.untracked.bool(True)
#process.Tracer = cms.Service("Tracer")

process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
process.source.fileNames = {'/store/relval/CMSSW_7_0_0/SingleMu/RECO/GR_R_70_V1_RelVal_zMu2012D-v2/00000/0259E46E-F698-E311-8CFD-003048FF9AC6.root'}

#                                         ##
process.maxEvents.input = 10000

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("PhysicsTools.PatAlgos.slimming.slimming_cff")

process.patMuons.isoDeposits = cms.PSet()
process.patElectrons.isoDeposits = cms.PSet()
process.patTaus.isoDeposits = cms.PSet()
process.patPhotons.isoDeposits = cms.PSet()

process.patMuons.embedTrack         = True  # used for IDs
process.patMuons.embedCombinedMuon  = True  # used for IDs
process.patMuons.embedMuonBestTrack = True  # used for IDs
process.patMuons.embedStandAloneMuon = True # maybe?
process.patMuons.embedPickyMuon = False   # no, use best track
process.patMuons.embedTpfmsMuon = False   # no, use best track
process.patMuons.embedDytMuon   = False   # no, use best track

process.patElectrons.embedPflowSuperCluster         = False
process.patElectrons.embedPflowBasicClusters        = False
process.patElectrons.embedPflowPreshowerClusters    = False

process.selectedPatJets.cut = cms.string("pt > 10")
process.selectedPatMuons.cut = cms.string("pt > 5 || isPFMuon || (pt > 3 && (isGlobalMuon || isStandAloneMuon || numberOfMatches > 0 || muonID('RPCMuLoose')))") 
process.selectedPatElectrons.cut = cms.string("") 
process.selectedPatTaus.cut = cms.string("pt > 20 && tauID('decayModeFinding')> 0.5")
process.selectedPatPhotons.cut = cms.string("pt > 15 && hadTowOverEm()<0.15 ")

process.slimmedJets.clearDaughters = False

from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection

addJetCollection(process, labelName = 'CA8', jetSource = cms.InputTag('ca8PFJetsCHS') )
process.selectedPatJetsCA8.cut = cms.string("pt > 100")

process.slimmedJetsCA8 = cms.EDProducer("PATJetSlimmer",
   src = cms.InputTag("selectedPatJetsCA8"),
   map = cms.InputTag("packedPFCandidates"),
   clearJetVars = cms.bool(True),
   clearDaughters = cms.bool(False),
   clearTrackRefs = cms.bool(True),
   dropSpecific = cms.bool(False),
)
process.slimmedJetsCA8.clearDaughters = False

## PU JetID
process.load("PhysicsTools.PatAlgos.slimming.pileupJetId_cfi")
process.patJets.userData.userFloats.src = [ cms.InputTag("pileupJetId:fullDiscriminant"), ]

from PhysicsTools.PatAlgos.tools.trigTools import switchOnTriggerStandAlone
switchOnTriggerStandAlone( process )
process.patTrigger.packTriggerPathNames = cms.bool(True)

#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
#                                         ##

# apply type I/type I + II PFMEt corrections to pat::MET object
# and estimate systematic uncertainties on MET
from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
from PhysicsTools.PatUtils.tools.metUncertaintyTools import runMEtUncertainties
addJetCollection(process, postfix   = "ForMetUnc", labelName = 'AK5PF', jetSource = cms.InputTag('ak5PFJets'), jetCorrections = ('AK5PF', ['L1FastJet', 'L2Relative', 'L3Absolute', 'L2L3Residual'], ''), btagDiscriminators = ['combinedSecondaryVertexBJetTags' ] )
runMEtUncertainties(process,jetCollection="selectedPatJetsAK5PFForMetUnc", outputModule=None)

#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_mini_singlemu.root'
process.out.outputCommands = process.MicroEventContent.outputCommands
process.out.dropMetaData = cms.untracked.string('ALL')
process.out.fastCloning= cms.untracked.bool(False)
process.out.overrideInputFileSplitLevels = cms.untracked.bool(True)
process.out.compressionAlgorithm = cms.untracked.string('LZMA')

from PhysicsTools.PatAlgos.tools.coreTools import runOnData
runOnData( process )
process.GlobalTag.globaltag = "GR_R_70_V1::All"
