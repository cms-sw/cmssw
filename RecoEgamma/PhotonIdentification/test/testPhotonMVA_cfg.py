import FWCore.ParameterSet.Config as cms
from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
from Configuration.AlCa.GlobalTag import GlobalTag

process = cms.Process("PhotonMVANtuplizer")

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

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
        'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring16_nonTrig_V1_cff',
        'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1_cff',
        'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1p1_cff',
        'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V2_cff',
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
                                           "photonMVAValueMapProducer:PhotonMVAEstimatorRun2Spring16NonTrigV1Values",
                                           "photonMVAValueMapProducer:PhotonMVAEstimatorRunIIFall17v1Values",
                                           "photonMVAValueMapProducer:PhotonMVAEstimatorRunIIFall17v1p1Values",
                                           "photonMVAValueMapProducer:PhotonMVAEstimatorRunIIFall17v2Values",
                                           ),
        phoMVAValMapLabels   = cms.vstring(
                                           "Spring16NonTrigV1",
                                           "Fall17v1",
                                           "Fall17v1p1",
                                           "Fall17v2",
                                           ),
        phoMVACats           = cms.vstring(
                                           "photonMVAValueMapProducer:PhotonMVAEstimatorRunIIFall17v1Categories",
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
