import FWCore.ParameterSet.Config as cms

process = cms.Process('GETGBR')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('/store/data/Run2015D/DoubleEG/MINIAOD/PromptReco-v4/000/258/159/00000/027612B0-306C-E511-BD47-02163E014496.root'),
                            )

from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '74X_dataRun2_Prompt_v4', '')

process.selectedPhotons = cms.EDFilter("PATPhotonSelector",
                                       src = cms.InputTag("slimmedPhotons"),
                                       cut = cms.string("pt > 8 && chargedHadronIso()/pt < 0.3"),
                                       )

process.load('EgammaAnalysis.ElectronTools.calibratedPhotonsRun2_cfi')
process.calibratedPatPhotons.photons = "selectedPhotons"

process.zeeUncalib = cms.EDProducer("CandViewShallowCloneCombiner",
                                    decay = cms.string("selectedPhotons@+ selectedPhotons@-"),
                                    checkCharge = cms.bool(False),                                    
                                    cut   = cms.string("min(daughter(0).pt,daughter(1).pt) > 15 && mass > 50"),
                                    )

process.zeeCalib = process.zeeUncalib.clone(
    decay = cms.string("calibratedPatPhotons@+ calibratedPatPhotons@-"),
    )

process.zeeUncalibTree = cms.EDFilter("ProbeTreeProducer",
                                      src = cms.InputTag("zeeUncalib"),
                                      variables = cms.PSet(
        mass = cms.string("mass"),
        #        reco::Photon::P4type::regression2

        #0286     float getCorrectedEnergy( P4type type) const;
        #0287     float getCorrectedEnergyError( P4type type) const ;
        #massErr = cms.string("0.5 * mass * sqrt( pow( daughter(0).masterClone.getCorrectedEnergyError(3) / daughter(0).masterClone.p4().P(), 2 ) + "+
        #                     "pow( daughter(0).masterClone.getCorrectedEnergyError(3) / daughter(0).masterClone.p4().P(), 2 ) ) "),
        l1pt = cms.string("daughter(0).pt"),
        l2pt = cms.string("daughter(1).pt"),
        l1eta = cms.string("daughter(0).eta"),
        l2eta = cms.string("daughter(1).eta"),
        ),
                                      flags = cms.PSet(),
                                      )

process.zeeCalibTree = process.zeeUncalibTree.clone(
    src = cms.InputTag("zeeCalib"),
    )

process.path = cms.Path(process.selectedPhotons + process.calibratedPatPhotons +
                        process.zeeUncalib + process.zeeUncalibTree +
                        process.zeeCalib + process.zeeCalibTree
                        )

process.TFileService = cms.Service("TFileService", fileName = cms.string("plots_photons.root"))


#process.out = cms.OutputModule("PoolOutputModule",
#                               outputCommands = cms.untracked.vstring('keep *_*_*_GETGBR'),
#                               #    fileName = cms.untracked.string('CandidateZ_newEscale.root')
#                               fileName = cms.untracked.string('testMC.root')
#                               )
#
#process.end = cms.EndPath(process.out)
