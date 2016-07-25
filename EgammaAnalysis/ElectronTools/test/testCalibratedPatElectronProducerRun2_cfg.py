import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

varOptions = VarParsing('analysis')
varOptions.register(
    "isMC",
    False,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.bool,
    "setup MC as in/out put"
    )
varOptions.parseArguments()

process = cms.Process('GETGBR')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 1000


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
    input = cms.untracked.int32(varOptions.maxEvents)
)

# Input source
if not varOptions.isMC :    
    process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(
            '/store/data/Run2016B/DoubleEG/MINIAOD/PromptReco-v2/000/275/073/00000/C281D9DB-5235-E611-8663-02163E0122D7.root'
            )
                                )

else:
    process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(
            '/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/FE69D710-7E6D-E511-A2F0-008CFA197FAC.root',
            '/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/FC529DE1-736D-E511-9386-008CFA1CB8A8.root',
            '/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/E66EB7F4-776D-E511-98A1-008CFA0A5A94.root',
            '/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/E6C6E3D4-9F6D-E511-B817-008CFA197D2C.root',
            '/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/E6D897DA-836D-E511-B02B-008CFA56D894.root',
            '/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/EA0DB6DD-836D-E511-878E-008CFA14F9D4.root',
            '/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/EA0FBE97-7E6D-E511-8172-008CFA197A5C.root',
            '/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/EA90F1B8-5D6D-E511-9265-008CFA1C64B0.root',
            '/store/mc/RunIISpring15MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/F8F8A7A4-726D-E511-A351-008CFA166008.root'
            )
                                )
    
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '74X_dataRun2_Prompt_v4', '')

if varOptions.isMC:
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc'  , '')
else:
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
    
process.selectedElectrons = cms.EDFilter("PATElectronSelector",
                                         src = cms.InputTag("slimmedElectrons"),
                                         cut = cms.string("pt > 8 && ((abs(eta)<1.479 && pfIsolationVariables().sumChargedHadronPt/pt < 0.1 && abs(deltaEtaSuperClusterTrackAtVtx)<0.0126 && abs(deltaPhiSuperClusterTrackAtVtx)<0.107 && hcalOverEcal<0.186 && passConversionVeto && pfIsolationVariables().sumNeutralHadronEt/pt < 0.1 && pfIsolationVariables().sumPhotonEt/pt < 0.1 && full5x5_sigmaIetaIeta<0.012) || (abs(eta)>1.479 && pfIsolationVariables().sumChargedHadronPt/pt < 0.1 && abs(deltaEtaSuperClusterTrackAtVtx)<0.0109 && abs(deltaPhiSuperClusterTrackAtVtx)<0.217 && hcalOverEcal<0.09 && passConversionVeto && pfIsolationVariables().sumNeutralHadronEt/pt < 0.1 && pfIsolationVariables().sumPhotonEt/pt < 0.1 && full5x5_sigmaIetaIeta<0.034))")
)

process.load('EgammaAnalysis.ElectronTools.calibratedElectronsRun2_cfi')
process.calibratedPatElectrons.electrons = "selectedElectrons"
process.calibratedPatElectrons.isMC = False
if varOptions.isMC:
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

outputfilename = 'plots_data.root'
if varOptions.isMC:
    outputfilename = 'plots_mc2.root'
process.TFileService = cms.Service("TFileService", 
                                   fileName = cms.string(outputfilename),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

