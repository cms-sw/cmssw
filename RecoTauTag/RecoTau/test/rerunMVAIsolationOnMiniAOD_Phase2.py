import FWCore.ParameterSet.Config as cms

process = cms.Process("rerunMVAIsolationOnMiniAODPhase2")

process.load("Configuration.StandardSequences.MagneticField_cff")

#process.Tracer = cms.Service("Tracer")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'/store/mc/PhaseIIMTDTDRAutumn18MiniAOD/QCD_Pt-15To7000_TuneCP5_Flat_14TeV-pythia8/MINIAODSIM/PU200_103X_upgrade2023_realistic_v2-v1/40000/FFE6B9AD-6109-FA47-9273-24C908EC90EE.root',
        '/store/mc/PhaseIIMTDTDRAutumn18MiniAOD/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack/MINIAODSIM/PU200_103X_upgrade2023_realistic_v2-v1/120000/2AD029AA-54C4-3D44-B934-24F19454A6FD.root',
    )
)

process.TFileService = cms.Service("TFileService",
    #fileName = cms.string('output_QCD_Phase2.root')
    fileName = cms.string('output_VBFHToTauTau_Phase2.root')
)

from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
### Load payloads via GlobalTag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')


from RecoTauTag.RecoTau.PATTauDiscriminationByMVAIsolationRun2_cff import *
process.rerunDiscriminationByIsolationMVADBnewDMwLTPhase2raw = patDiscriminationByIsolationMVArun2v1raw.clone(
    PATTauProducer = 'slimmedTaus',
    Prediscriminants = noPrediscriminants,
    loadMVAfromDB = True,
    #loadMVAfromDB = False,
    #inputFileName = 'gbrDiscriminationByIsolationMVAPhase2.root',
    mvaName = 'RecoTauTag_tauIdMVAIsoPhase2',
    mvaOpt = 'DBnewDMwLTwGJPhase2',
    verbosity = 0
)

process.rerunDiscriminationByIsolationMVADBnewDMwLTPhase2 = patDiscriminationByIsolationMVArun2v1.clone(
    PATTauProducer = 'slimmedTaus',
    Prediscriminants = noPrediscriminants,
    toMultiplex = 'rerunDiscriminationByIsolationMVADBnewDMwLTPhase2raw',
    loadMVAfromDB = True,
    #loadMVAfromDB = False,
    #inputFileName = 'wpDiscriminationByIsolationMVAPhase2_tauIdMVAIsoPhase2.root',
    mvaOutput_normalization = 'RecoTauTag_tauIdMVAIsoPhase2_mvaOutput_normalization',
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAIsoPhase2"),
            variable = cms.string("pt"),
        )
    ),
    workingPoints = cms.vstring(
        "_VVLoose",
        "_VLoose",
        "_Loose",
        "_Medium",
        "_Tight",
        "_VTight",
        "_VVTight"
    )
)

process.rerunMvaIsolation2Seq_Phase2 = cms.Sequence(
    process.rerunDiscriminationByIsolationMVADBnewDMwLTPhase2raw
    * process.rerunDiscriminationByIsolationMVADBnewDMwLTPhase2
)

# embed new id's into tau
def tauIDMVAinputs(module, wp):
    return cms.PSet(inputTag = cms.InputTag(module), workingPointIndex = cms.int32(-1 if wp=="raw" else -2 if wp=="category" else getattr(process, module).workingPoints.index(wp)))
embedID = cms.EDProducer("PATTauIDEmbedder",
    src = cms.InputTag('slimmedTaus'),
    tauIDSources = cms.PSet(
        byIsolationMVADBnewDMwLTPhase2raw = tauIDMVAinputs("rerunDiscriminationByIsolationMVADBnewDMwLTPhase2", "raw"),
        byVVLooseIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs("rerunDiscriminationByIsolationMVADBnewDMwLTPhase2", "_VVLoose"),
        byVLooseIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs("rerunDiscriminationByIsolationMVADBnewDMwLTPhase2", "_VLoose"),
        byLooseIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs("rerunDiscriminationByIsolationMVADBnewDMwLTPhase2", "_Loose"),
        byMediumIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs("rerunDiscriminationByIsolationMVADBnewDMwLTPhase2", "_Medium"),
        byTightIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs("rerunDiscriminationByIsolationMVADBnewDMwLTPhase2", "_Tight"),
        byVTightIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs("rerunDiscriminationByIsolationMVADBnewDMwLTPhase2", "_VTight"),
        byVVTightIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs("rerunDiscriminationByIsolationMVADBnewDMwLTPhase2", "_VVTight")
    ),
)
setattr(process, "newTauIDsEmbedded", embedID)

process.rerunMVAIsolationOnMiniAOD_Phase2 = cms.EDAnalyzer(
    'rerunMVAIsolationOnMiniAOD_Phase2',
    tauCollection = cms.InputTag("newTauIDsEmbedded"),
    genJetCollection = cms.InputTag("slimmedGenJets"), #comment out to run on data
)

process.p = cms.Path(
    process.rerunMvaIsolation2Seq_Phase2
    * process.newTauIDsEmbedded
    * process.rerunMVAIsolationOnMiniAOD_Phase2
)
