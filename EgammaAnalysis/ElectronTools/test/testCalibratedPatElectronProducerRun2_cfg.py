import FWCore.ParameterSet.Config as cms

process = cms.Process('GETGBR')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
        calibratedPatElectrons = cms.PSet(
        initialSeed = cms.untracked.uint32(1),
        engineName = cms.untracked.string('TRandom3')
        ),
        calibratedElectrons = cms.PSet(
        initialSeed = cms.untracked.uint32(1),
        engineName = cms.untracked.string('TRandom3')
        ),
                                                   )
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        #'/store/data/Run2015D/DoubleEG/MINIAOD/PromptReco-v4/000/258/159/00000/027612B0-306C-E511-BD47-02163E014496.root',
        #'/store/data/Run2015D/DoubleEG/MINIAOD/PromptReco-v4/000/258/159/00000/06D4AACE-306C-E511-8E3B-02163E011FE7.root',
        #'/store/data/Run2015D/DoubleEG/MINIAOD/PromptReco-v4/000/258/159/00000/0CB85859-316C-E511-8139-02163E01299C.root',
        #'/store/data/Run2015D/DoubleEG/MINIAOD/PromptReco-v4/000/258/159/00000/34346FDB-306C-E511-9A4C-02163E013641.root',
        #'/store/data/Run2015D/DoubleEG/MINIAOD/PromptReco-v4/000/258/159/00000/366AFECD-306C-E511-B475-02163E01358D.root',
        #'/store/data/Run2015D/DoubleEG/MINIAOD/PromptReco-v4/000/258/159/00000/483AA3A6-306C-E511-A5F2-02163E01437B.root',
        #'/store/data/Run2015D/DoubleEG/MINIAOD/PromptReco-v4/000/258/159/00000/4E0ED931-316C-E511-9F15-02163E014186.root',
        '/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/FE69D710-7E6D-E511-A2F0-008CFA197FAC.root',
        '/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/FC529DE1-736D-E511-9386-008CFA1CB8A8.root',
        '/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/F8F8A7A4-726D-E511-A351-008CFA166008.root'
                            )
)

from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '74X_dataRun2_Prompt_v4', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '74X_mcRun2_asymptotic_v2', '')

process.selectedElectrons = cms.EDFilter("PATElectronSelector",
                                         src = cms.InputTag("slimmedElectrons"),
                                         cut = cms.string("pt > 8 && ((abs(eta)<1.479 && pfIsolationVariables().sumChargedHadronPt/pt < 0.1 && abs(deltaEtaSuperClusterTrackAtVtx)<0.0126 && abs(deltaPhiSuperClusterTrackAtVtx)<0.107 && hcalOverEcal<0.186 && passConversionVeto && pfIsolationVariables().sumNeutralHadronEt/pt < 0.1 && pfIsolationVariables().sumPhotonEt/pt < 0.1 && full5x5_sigmaIetaIeta<0.012) || (abs(eta)>1.479 && pfIsolationVariables().sumChargedHadronPt/pt < 0.1 && abs(deltaEtaSuperClusterTrackAtVtx)<0.0109 && abs(deltaPhiSuperClusterTrackAtVtx)<0.217 && hcalOverEcal<0.09 && passConversionVeto && pfIsolationVariables().sumNeutralHadronEt/pt < 0.1 && pfIsolationVariables().sumPhotonEt/pt < 0.1 && full5x5_sigmaIetaIeta<0.034))")
)

process.load('EgammaAnalysis.ElectronTools.calibratedElectronsRun2_cfi')
process.calibratedPatElectrons.electrons = "selectedElectrons"
process.calibratedPatElectrons.isMC = True

process.zeeUncalib = cms.EDProducer("CandViewShallowCloneCombiner",
                                    decay = cms.string("selectedElectrons@+ selectedElectrons@-"),
                                    cut   = cms.string("min(daughter(0).pt,daughter(1).pt) > 15 && mass > 50"),
                                    )

process.zeeCalib = process.zeeUncalib.clone(
    decay = cms.string("calibratedPatElectrons@+ calibratedPatElectrons@-"),
    )

process.zeeUncalibTree = cms.EDFilter("ProbeTreeProducer",
                                      src = cms.InputTag("zeeUncalib"),
                                      variables = cms.PSet(
        mass = cms.string("mass"),
        massErr = cms.string("0.5 * mass * sqrt( pow( daughter(0).masterClone.p4Error('P4_COMBINATION') / daughter(0).masterClone.p4('P4_COMBINATION').P(), 2 ) + "+
                             "pow( daughter(0).masterClone.p4Error('P4_COMBINATION') / daughter(0).masterClone.p4('P4_COMBINATION').P(), 2 ) ) "),
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

process.path = cms.Path(process.selectedElectrons + process.calibratedPatElectrons +
                        process.zeeUncalib + process.zeeUncalibTree +
                        process.zeeCalib + process.zeeCalibTree
                        )

process.TFileService = cms.Service("TFileService", fileName = cms.string("plots_mc.root"))
