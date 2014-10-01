import FWCore.ParameterSet.Config as cms

## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

from PhysicsTools.PatAlgos.tools.coreTools import *
removeMCMatching(process, ['All'])

removeSpecificPATObjects(process,
                         ['Photons'],  # 'Tau' has currently been taken out due to problems with tau discriminators
                         outputModules=[])

removeCleaning(process,
               outputModules=[])

process.patJetCorrFactors.payload = 'AK5Calo'
# For data:
#process.patJetCorrFactors.levels = ['L2Relative', 'L3Absolute', 'L2L3Residual', 'L5Flavor', 'L7Parton']
# For MC:
process.patJetCorrFactors.levels = ['L2Relative', 'L3Absolute']
#process.patJetCorrFactors.flavorType = "T"

process.patMuons.usePV = False

#-------------------------------------------------
# selection step 1: trigger
#-------------------------------------------------

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
process.step1 = hltHighLevel.clone(TriggerResultsTag = "TriggerResults::HLT", HLTPaths = ["HLT_Mu15_eta2p1_v5"])

#-------------------------------------------------
# selection step 2: vertex filter
#-------------------------------------------------

# vertex filter
process.step2 = cms.EDFilter("VertexSelector",
                             src = cms.InputTag("offlinePrimaryVertices"),
                             cut = cms.string("!isFake && ndof > 4 && abs(z) < 15 && position.Rho < 2"),
                             filter = cms.bool(True),
                             )

#-------------------------------------------------
# selection steps 3 and 4: muon selection
#-------------------------------------------------

from PhysicsTools.PatAlgos.cleaningLayer1.muonCleaner_cfi import *
process.isolatedMuons010 = cleanPatMuons.clone(preselection =
                                               'isGlobalMuon & isTrackerMuon &'
                                               'pt > 20. &'
                                               'abs(eta) < 2.1 &'
                                               '(trackIso+caloIso)/pt < 0.1 &'
                                               'innerTrack.numberOfValidHits > 10 &'
                                               'globalTrack.normalizedChi2 < 10.0 &'
                                               'globalTrack.hitPattern.numberOfValidMuonHits > 0 &'
                                               'abs(dB) < 0.02'
                                               )

process.isolatedMuons010.checkOverlaps = cms.PSet(
        jets = cms.PSet(src       = cms.InputTag("goodJets"),
                        algorithm = cms.string("byDeltaR"),
                        preselection        = cms.string(""),
                        deltaR              = cms.double(0.3),
                        checkRecoComponents = cms.bool(False),
                        pairCut             = cms.string(""),
                        requireNoOverlaps   = cms.bool(True),
                        )
            )
process.isolatedMuons005 = cleanPatMuons.clone(src = 'isolatedMuons010',
                                               preselection = '(trackIso+caloIso)/pt < 0.05'
                                               )

process.vetoMuons = cleanPatMuons.clone(preselection =
                                        'isGlobalMuon &'
                                        'pt > 10. &'
                                        'abs(eta) < 2.5 &'
                                        '(trackIso+caloIso)/pt < 0.2'
                                        )

from PhysicsTools.PatAlgos.selectionLayer1.muonCountFilter_cfi import *
process.step3a = countPatMuons.clone(src = 'isolatedMuons005', minNumber = 1, maxNumber = 1)
process.step3b = countPatMuons.clone(src = 'isolatedMuons010', minNumber = 1, maxNumber = 1)
process.step4  = countPatMuons.clone(src = 'vetoMuons', maxNumber = 1)

#-------------------------------------------------
# selection step 5: electron selection
#-------------------------------------------------

from PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi import *
process.vetoElectrons = selectedPatElectrons.clone(src = 'selectedPatElectrons',
                                                   cut =
                                                   'et > 15. &'
                                                   'abs(eta) < 2.5 &'
                                                   '(dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/et <  0.2'
                                                   )

from PhysicsTools.PatAlgos.selectionLayer1.electronCountFilter_cfi import *
process.step5  = countPatMuons.clone(src = 'vetoElectrons', maxNumber = 0)

#-------------------------------------------------
# selection steps 6 and 7: jet selection
#-------------------------------------------------

from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import *
process.goodJets = selectedPatJets.clone(src = 'patJets',
                                         cut =
                                         'pt > 30. &'
                                         'abs(eta) < 2.4 &'
                                         'emEnergyFraction > 0.01 &'
                                         'jetID.n90Hits > 1 &'
                                         'jetID.fHPD < 0.98'
                                         )

from PhysicsTools.PatAlgos.selectionLayer1.jetCountFilter_cfi import *
process.step6a = countPatJets.clone(src = 'goodJets', minNumber = 1)
process.step6b = countPatJets.clone(src = 'goodJets', minNumber = 2)
process.step6c = countPatJets.clone(src = 'goodJets', minNumber = 3)
process.step7  = countPatJets.clone(src = 'goodJets', minNumber = 4)

#-------------------------------------------------
# paths
#-------------------------------------------------

process.looseSequence = cms.Path(process.step1 *
                                 process.step2 *
                                 process.patDefaultSequence *
                                 process.goodJets *
                                 process.isolatedMuons010 *
                                 process.step3b *
                                 process.vetoMuons *
                                 process.step4 *
                                 process.vetoElectrons *
                                 process.step5 *
                                 process.step6a *
                                 process.step6b *
                                 process.step6c
                                 )

process.tightSequence = cms.Path(process.step1 *
                                 process.step2 *
                                 process.patDefaultSequence *
                                 process.goodJets *
                                 process.isolatedMuons010 *
                                 process.isolatedMuons005 *
                                 process.step3a *
                                 process.vetoMuons *
                                 process.step4 *
                                 process.vetoElectrons *
                                 process.step5 *
                                 process.step6a *
                                 process.step6b *
                                 process.step6c *
                                 process.step7
                                )


process.out.SelectEvents.SelectEvents = ['tightSequence',
                                         'looseSequence' ]

from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
process.out.outputCommands = cms.untracked.vstring('drop *', *patEventContentNoCleaning )

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarAODSIM
process.source.fileNames = filesRelValProdTTbarAODSIM
#                                         ##
process.maxEvents.input = 1000
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_topSelection.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
