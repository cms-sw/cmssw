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
        
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/3E38C5C7-87A6-E511-8620-002618943910.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/7EB36900-77A6-E511-ACCB-0CC47A4D764A.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/C67A894C-83A6-E511-A90D-0025905A6068.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/FED54E49-83A6-E511-AB65-0CC47A4C8ED8.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/3E65CB18-77A6-E511-A3DE-0025905A612C.root',  
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/7EF48202-89A6-E511-9089-0025905A6110.root',  
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/C68195C6-73A6-E511-A893-0CC47A4C8E34.root',  
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/3E9B66CD-87A6-E511-943C-0025905938A4.root',  
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/802ED605-79A6-E511-A555-0CC47A4D762A.root',  
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/C8E870D0-61A6-E511-8D4C-0CC47A74524E.root', 
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/ECA73F4E-82A6-E511-8843-0CC47A4D76A2.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/EE3246BE-6AA6-E511-A292-002590A2CCFE.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/EEC83EE3-75A6-E511-AE15-0025905A6136.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/EEE889E4-77A6-E511-A7D7-0025905B85D8.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/F066634D-83A6-E511-8874-0CC47A4C8E1E.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/F22451C8-87A6-E511-BEE7-0CC47A4D769A.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/F2515AE1-74A6-E511-9524-0CC47A74527A.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/F419DFC8-7AA6-E511-BCF7-0CC47A4D769E.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/F4D15347-66A6-E511-A73F-E41D2D08DE00.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/F4F4C3E9-76A6-E511-9572-0025905A6110.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/F60AEA20-66A6-E511-BDC4-0CC47A78A3EC.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/F67E523E-80A6-E511-836A-0025905964B6.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/F687B99A-81A6-E511-9BE4-0CC47A78A440.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/F6901A39-6AA6-E511-A634-0025901D4B20.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/F6E918C9-87A6-E511-B3D3-0CC47A4D76B2.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/F80C3E00-89A6-E511-8B04-0CC47A78A33E.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/F886D4C6-7AA6-E511-A3DF-00261894398B.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/F890A515-5BA6-E511-A1FD-0CC47A4C8E86.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/FA5697C8-7AA6-E511-A010-0CC47A78A4B8.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/FA78CBE0-75A6-E511-80EB-0025905AA9CC.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/FABF4750-7AA6-E511-BBBF-0CC47A4D762A.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/FAC03F87-6DA6-E511-90F5-0025901D4A0E.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/FC122FD8-74A6-E511-ADBA-0025905A6110.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/FC6E00D4-78A6-E511-BB82-003048FFD76E.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/FCAB2A25-57A6-E511-95F9-0CC47A4C8E16.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/FCC55B5B-5DA6-E511-9505-0CC47A78A33E.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/FE092324-64A6-E511-B90A-00266CFFA0B0.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/FE2483E5-77A6-E511-A182-0025905A609E.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/FEAD8C6B-64A6-E511-B1FC-002618943985.root',
        '/store/data/Run2015D/DoubleEG/MINIAOD/16Dec2015-v2/00000/FED54E49-83A6-E511-AB65-0CC47A4C8ED8.root'

        #'/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/FE69D710-7E6D-E511-A2F0-008CFA197FAC.root',
        #'/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/FC529DE1-736D-E511-9386-008CFA1CB8A8.root',
        #'/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/E66EB7F4-776D-E511-98A1-008CFA0A5A94.root',
        #'/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/E6C6E3D4-9F6D-E511-B817-008CFA197D2C.root',
        #'/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/E6D897DA-836D-E511-B02B-008CFA56D894.root',
        #'/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/EA0DB6DD-836D-E511-878E-008CFA14F9D4.root',
        #'/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/EA0FBE97-7E6D-E511-8172-008CFA197A5C.root',
        #'/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/EA90F1B8-5D6D-E511-9265-008CFA1C64B0.root',
        #'/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/F8F8A7A4-726D-E511-A351-008CFA166008.root'
                            )
)

from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '74X_dataRun2_Prompt_v4', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '76X_mcRun2_asymptotic_v12', '')

process.selectedElectrons = cms.EDFilter("PATElectronSelector",
                                         src = cms.InputTag("slimmedElectrons"),
                                         cut = cms.string("pt > 8 && ((abs(eta)<1.479 && pfIsolationVariables().sumChargedHadronPt/pt < 0.1 && abs(deltaEtaSuperClusterTrackAtVtx)<0.0126 && abs(deltaPhiSuperClusterTrackAtVtx)<0.107 && hcalOverEcal<0.186 && passConversionVeto && pfIsolationVariables().sumNeutralHadronEt/pt < 0.1 && pfIsolationVariables().sumPhotonEt/pt < 0.1 && full5x5_sigmaIetaIeta<0.012) || (abs(eta)>1.479 && pfIsolationVariables().sumChargedHadronPt/pt < 0.1 && abs(deltaEtaSuperClusterTrackAtVtx)<0.0109 && abs(deltaPhiSuperClusterTrackAtVtx)<0.217 && hcalOverEcal<0.09 && passConversionVeto && pfIsolationVariables().sumNeutralHadronEt/pt < 0.1 && pfIsolationVariables().sumPhotonEt/pt < 0.1 && full5x5_sigmaIetaIeta<0.034))")
)

process.load('EgammaAnalysis.ElectronTools.calibratedElectronsRun2_cfi')
process.calibratedPatElectrons.electrons = "selectedElectrons"
process.calibratedPatElectrons.isMC = False

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

process.TFileService = cms.Service("TFileService", fileName = cms.string("plots_data.root"))
