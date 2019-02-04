import FWCore.ParameterSet.Config as cms
from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
from Configuration.AlCa.GlobalTag import GlobalTag

process = cms.Process("PhotonMVANtuplizer")

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")

process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# File with the ID variables form the text file to include in the Ntuplizer
mvaVariablesFile = "RecoEgamma/PhotonIdentification/data/PhotonMVAEstimatorRun2VariablesFall17V1p1.txt"

outputFile = "photon_ntuple.root"

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/mc/RunIIFall17MiniAODv2/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/00000/00AE0E2A-6F42-E811-8EA2-0025905B85AA.root'
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

switchOnVIDPhotonIdProducer(process, dataFormat)

# define which IDs we want to produce
my_id_modules = [
        'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring16_nonTrig_V1_cff',
        'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1_cff',
        'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1p1_cff',
        'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V2_cff',
                 ]

#add them to the VID producer
for idmod in my_id_modules:
    setupAllVIDIdsInModule(process,idmod,setupVIDPhotonSelection)

process.ntuplizer = cms.EDAnalyzer('PhotonMVANtuplizer',
        phoMVAs              = cms.untracked.vstring(
                                          ),
        phoMVALabels         = cms.untracked.vstring(
                                          ),
        phoMVAValMaps        = cms.untracked.vstring(
                                           "photonMVAValueMapProducer:PhotonMVAEstimatorRun2Spring16NonTrigV1Values",
                                           "photonMVAValueMapProducer:PhotonMVAEstimatorRunIIFall17v1Values",
                                           "photonMVAValueMapProducer:PhotonMVAEstimatorRunIIFall17v1p1Values",
                                           "photonMVAValueMapProducer:PhotonMVAEstimatorRunIIFall17v2Values",
                                           ),
        phoMVAValMapLabels   = cms.untracked.vstring(
                                           "Spring16NonTrigV1",
                                           "Fall17v1",
                                           "Fall17v1p1",
                                           "Fall17v2",
                                           ),
        phoMVACats           = cms.untracked.vstring(
                                           "photonMVAValueMapProducer:PhotonMVAEstimatorRunIIFall17v1Categories",
                                           ),
        phoMVACatLabels      = cms.untracked.vstring(
                                           "PhoMVACats",
                                           ),
        variableDefinition = cms.string(mvaVariablesFile),
        )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string( outputFile )
                                   )

process.p = cms.Path(process.egmPhotonIDSequence * process.ntuplizer)
