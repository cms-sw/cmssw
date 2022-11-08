import FWCore.ParameterSet.Config as cms
from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
from Configuration.AlCa.GlobalTag import GlobalTag

process = cms.Process("PhotonMVANtuplizer")

#process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
#process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")


#process.GlobalTag = GlobalTag(process.GlobalTag, '122X_mcRun3_2021_realistic_v9', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

# File with the ID variables form the text file to include in the Ntuplizer0
mvaVariablesFile = "RecoEgamma/PhotonIdentification/data/PhotonMVAEstimatorRun3VariablesWinter22V1.txt"


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )


inputFilesAOD = cms.untracked.vstring(
        #'/store/mc/Run3Winter22DR/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_13p6TeV_pythia8/AODSIM/FlatPU0to70_122X_mcRun3_2021_realistic_v9-v2/2430000/00acca83-e889-4d35-8339-0a0ba956e74e.root',
        '/store/mc/Run3Winter22DR/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_13p6TeV_pythia8/AODSIM/FlatPU0to70_122X_mcRun3_2021_realistic_v9-v2/2430000/023a9365-aa97-4a10-8a5a-178eaedd94d4.root',
        '/store/mc/Run3Winter22DR/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_13p6TeV_pythia8/AODSIM/FlatPU0to70_122X_mcRun3_2021_realistic_v9-v2/2430000/0725adeb-68f1-4023-a2e8-373c52e85ff4.root',
        '/store/mc/Run3Winter22DR/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_13p6TeV_pythia8/AODSIM/FlatPU0to70_122X_mcRun3_2021_realistic_v9-v2/2430000/0817e2b0-81e3-49f4-90c1-19cfa46ab46a.root')

inputFilesMiniAOD = cms.untracked.vstring(
        'file:/eos/cms/store/group/phys_egamma/ec/prrout/Run3IDchecks/MiniAOD_Hgg.root'
        #'/store/mc/Run3Winter22MiniAOD/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_13p6TeV_pythia8/MINIAODSIM/FlatPU0to70_122X_mcRun3_2021_realistic_v9-v2/2430000/019c9ef2-86db-4258-a076-bdb5169dc3d0.root',
        #'/store/mc/Run3Winter22MiniAOD/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_13p6TeV_pythia8/MINIAODSIM/FlatPU0to70_122X_mcRun3_2021_realistic_v9-v2/2430000/0a80915b-ce51-4d0e-ae0d-a170a6736a19.root',
       #'/store/mc/Run3Winter22MiniAOD/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_13p6TeV_pythia8/MINIAODSIM/FlatPU0to70_122X_mcRun3_2021_realistic_v9-v2/2430000/1020066d-ab70-4718-8535-efbdfd3356cd.root',
       #'/store/mc/Run3Winter22MiniAOD/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_13p6TeV_pythia8/MINIAODSIM/FlatPU0to70_122X_mcRun3_2021_realistic_v9-v2/2430000/11fa80b0-204f-49ad-8682-5bf232b9f927.root'
)

useAOD = False

if useAOD == True :
    inputFiles = inputFilesAOD
    outputFile = "photon_ntuple_Run3_Winter22_IDtest_Hgg_xmlweights_AOD.root"
    print("AOD input files are used")
else :
    inputFiles = inputFilesMiniAOD
    outputFile = "photon_ntuple_Run3_Winter22_IDtest_Hgg_xmlweights_MiniAOD.root"
    print("MiniAOD input files are used")
process.source = cms.Source ("PoolSource", fileNames = inputFiles )

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
# turn on VID producer, indicate data format  to be
# DataFormat.AOD or DataFormat.MiniAOD, as appropriate
if useAOD == True :
    dataFormat = DataFormat.AOD
    input_tags = dict(
        src = cms.InputTag("gedPhotons"),
        vertices = cms.InputTag("offlinePrimaryVertices"),
        pileup = cms.InputTag("addPileupInfo"),
        genParticles = cms.InputTag("genParticles"),
        ebReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
        eeReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
    )
else :
    dataFormat = DataFormat.MiniAOD
    input_tags = dict()

switchOnVIDPhotonIdProducer(process, dataFormat)

# define which IDs we want to produce
my_id_modules = [
        'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Winter22_122X_V1_cff',
                 ]

#add them to the VID producer
for idmod in my_id_modules:
    setupAllVIDIdsInModule(process,idmod,setupVIDPhotonSelection)

process.ntuplizer = cms.EDAnalyzer('PhotonMVANtuplizer',
        phoMVAs              = cms.vstring(
                                          ),
        phoMVALabels         = cms.vstring(
                                          ),
        phoMVAValMaps        = cms.vstring(
                                           #"photonMVAValueMapProducer:PhotonMVAEstimatorRunIIFall17v1p1Values",
                                           #"photonMVAValueMapProducer:PhotonMVAEstimatorRunIIFall17v2Values",
                                           "photonMVAValueMapProducer:PhotonMVAEstimatorRunIIIWinter22v1Values",
                                           ),
        phoMVAValMapLabels   = cms.vstring(
                                           #"Fall17v1p1",
                                           #"Fall17v2",
                                           "Winter22v1",
                                           ),
        phoMVACats           = cms.vstring(
                                           #"photonMVAValueMapProducer:PhotonMVAEstimatorRunIIFall17v1Categories",
                                           "photonMVAValueMapProducer:PhotonMVAEstimatorRunIIIWinter22v1Categories",
                                           ),
        phoMVACatLabels      = cms.vstring(
                                           "PhoMVACats",
                                           ),
        variableDefinition = cms.string(mvaVariablesFile),
        #
        doEnergyMatrix = cms.bool(False), # disabled by default due to large size
        energyMatrixSize = cms.int32(2), # corresponding to 5x5
        #
        **input_tags
        )
"""
The energy matrix is the n x n of raw rec-hit energies around the seed
crystal.

The size of the energy matrix is controlled with the parameter
"energyMatrixSize", which controlls the extension of crystals in each
direction away from the seed, in other words n = 2 * energyMatrixSize + 1.

The energy matrix gets saved as a vector but you can easily unroll it
to a two dimensional numpy array later, for example like that:

>>> import uproot
>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> tree = uproot.open("photon_ntuple.root")["ntuplizer/tree"]
>>> n = 5

>>> for a in tree.array("ele_energyMatrix"):
>>>     a = a.reshape((n,n))
>>>     plt.imshow(np.log10(a))
>>>     plt.colorbar()
>>>     plt.show()
"""

process.TFileService = cms.Service("TFileService", fileName = cms.string(outputFile))

process.p = cms.Path(process.egmPhotonIDSequence * process.ntuplizer)
