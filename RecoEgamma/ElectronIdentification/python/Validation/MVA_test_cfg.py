import FWCore.ParameterSet.Config as cms
# from RecoEgamma.ElectronIdentification.ElectronMVAValueMapProducer_cfi import electronMVAVariableHelper
from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
from Configuration.AlCa.GlobalTag import GlobalTag

process = cms.Process("ElectronMVANtuplizer")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# File with the ID variables to include in the Ntuplizer
mvaVariablesFile = "RecoEgamma/ElectronIdentification/data/ElectronIDVariables.txt"

outputFile = "electron_validation_ntuple_new.root"

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
# process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:/data_CMS/cms/rembser/store/mc/RunIISummer17MiniAOD/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/92X_upgrade2017_realistic_v10_ext1-v1/110000/02CF84A2-6086-E711-A3A1-0CC47A7C3458.root'
    )
)

useAOD = False

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
# turn on VID producer, indicate data format  to be
# DataFormat.AOD or DataFormat.MiniAOD, as appropriate
if useAOD == True :
    dataFormat = DataFormat.AOD
else :
    dataFormat = DataFormat.MiniAOD

switchOnVIDElectronIdProducer(process, dataFormat)

# define which IDs we want to produce
my_id_modules = [
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_25ns_nonTrig_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_25ns_Trig_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_50ns_Trig_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_HZZ_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V1_cff',

        # 'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_25ns_nonTrig_V0_cff',
        # 'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_25ns_nonTrig_V2_cff',
        # 'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_25ns_nonTrig_V2p1_cff',
        # 'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_50ns_nonTrig_V0_cff',
        # 'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_50ns_nonTrig_V1_cff',
        # 'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_50ns_nonTrig_V2_cff',
        # 'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_50ns_nonTrig_V2p1_cff',
        # 'RecoEgamma.PhotonIdentification.Identification.vaPhotonID_Spring16_nonTrig_V1_cff',
        # 'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1_cff',
        # 'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1p1_cff',
                 ]

#add them to the VID producer
for idmod in my_id_modules:
    setupAllVIDIdsInModule(process,idmod,setupVIDElectronSelection)

process.ntuplizer = cms.EDAnalyzer('ElectronMVANtuplizer',
        # AOD case
        src                  = cms.InputTag('gedGsfElectrons'),
        vertices             = cms.InputTag('offlinePrimaryVertices'),
        pileup               = cms.InputTag('addPileupInfo'),
        genParticles         = cms.InputTag('genParticles'),
        # MET                  = cms.InputTag('pfMet'),
        MET                  = cms.InputTag('slimmedMETs'),
        # miniAOD case
        srcMiniAOD           = cms.InputTag('slimmedElectrons'),
        verticesMiniAOD      = cms.InputTag('offlineSlimmedPrimaryVertices'),
        pileupMiniAOD        = cms.InputTag('slimmedAddPileupInfo'),
        genParticlesMiniAOD  = cms.InputTag('prunedGenParticles'),
        METMiniAOD           = cms.InputTag('slimmedMETs'),
        #
        eleMVAs             = cms.vstring(
                                          ),
        eleMVALabels        = cms.vstring(
                                          ),
        eleMVAValMaps        = cms.vstring(
                                           "electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring15NonTrig25nsV1Values",
                                           "electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring15Trig25nsV1Values",
                                           "electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring15Trig50nsV1Values",
                                           "electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring16GeneralPurposeV1Values",
                                           "electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring16HZZV1Values",
                                           "electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17IsoV1Values",
                                           "electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17NoIsoV1Values",
                                           ),
        eleMVAValMapLabels   = cms.vstring(
                                           "Spring15_25nsnonTrigVals",
                                           "Spring15_25nsTrigVals",
                                           "Spring15_50nsTrigVals",
                                           "Spring16GPVals",
                                           "Spring16HZZVals",
                                           "Fall17IsoV1Vals",
                                           "Fall17NoIsoV1Vals",
                                           ),
        eleMVACats           = cms.vstring(
                                           "electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17NoIsoV1Categories",
                                           ),
        eleMVACatLabels      = cms.vstring(
                                           "EleMVACats",
                                           ),
        #
        variableDefinition   = cms.string(mvaVariablesFile),
        isMC                 = cms.bool(False),
        deltaR               = cms.double(0.1),
        )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string( outputFile )
                                   )

# process.electronMVAVariableHelper = electronMVAVariableHelper
process.p = cms.Path(process.egmGsfElectronIDSequence * process.ntuplizer)
