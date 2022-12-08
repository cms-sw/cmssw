import FWCore.ParameterSet.Config as cms
from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
from Configuration.AlCa.GlobalTag import GlobalTag

process = cms.Process("ElectronMVANtuplizer")

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag = GlobalTag(process.GlobalTag, '122X_mcRun3_2021_realistic_v9', '')

# File with the ID variables to include in the Ntuplizer
mvaVariablesFile = "RecoEgamma/ElectronIdentification/data/ElectronIDVariablesRun3.txt"

outputFile = "electron_ntuple.root"

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
         #'/store/mc/Run3Winter22MiniAOD/DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/MINIAODSIM/122X_mcRun3_2021_realistic_v9_ext1-v1/2830000/009dec29-88c9-4721-bb6f-135a7005e281.root',
        '/store/mc/Run3Winter22MiniAOD/DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/MINIAODSIM/122X_mcRun3_2021_realistic_v9_ext2-v2/40000/004dd361-f49c-43fe-8344-f18aa036e286.root'
    )
)

useAOD = False

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
# turn on VID producer, indicate data format  to be
# DataFormat.AOD or DataFormat.MiniAOD, as appropriate
if useAOD == True :
    dataFormat = DataFormat.AOD
    input_tags = dict(
        src = cms.InputTag("gedGsfElectrons"),
        vertices = cms.InputTag("offlinePrimaryVertices"),
        pileup = cms.InputTag("addPileupInfo"),
        genParticles = cms.InputTag("genParticles"),
        ebReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
        eeReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
    )
else :
    dataFormat = DataFormat.MiniAOD
    input_tags = dict()

switchOnVIDElectronIdProducer(process, dataFormat)

# define which IDs we want to produce
my_id_modules = [
        #'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff',
        #'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_HZZ_V1_cff',
        #'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V1_cff',
        #'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_iso_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_noIso_V1_cff',
                 ]

#add them to the VID producer
for idmod in my_id_modules:
    setupAllVIDIdsInModule(process,idmod,setupVIDElectronSelection)

process.ntuplizer = cms.EDAnalyzer('ElectronMVANtuplizer',
        #
        eleMVAs             = cms.vstring(
                                          #"egmGsfElectronIDs:mvaEleID-Spring16-GeneralPurpose-V1-wp80",
                                          #"egmGsfElectronIDs:mvaEleID-Spring16-GeneralPurpose-V1-wp90",
                                          #"egmGsfElectronIDs:mvaEleID-Spring16-HZZ-V1-wpLoose",
                                          #"egmGsfElectronIDs:mvaEleID-Fall17-noIso-V2-wp80",
                                          #"egmGsfElectronIDs:mvaEleID-Fall17-noIso-V2-wpLoose",
                                           "egmGsfElectronIDs:mvaEleID-RunIIIWinter22-iso-V1-wp80",
                                           "egmGsfElectronIDs:mvaEleID-RunIIIWinter22-iso-V1-wp90",
                                           "egmGsfElectronIDs:mvaEleID-RunIIIWinter22-noIso-V1-wp80",
                                           "egmGsfElectronIDs:mvaEleID-RunIIIWinter22-noIso-V1-wp90",
                                          #"egmGsfElectronIDs:mvaEleID-Fall17-noIso-V2-wp90",
                                          #"egmGsfElectronIDs:mvaEleID-Fall17-iso-V2-wpHZZ",
                                          # "egmGsfElectronIDs:mvaEleID-Fall17-iso-V2-wp80",
                                          #"egmGsfElectronIDs:mvaEleID-Fall17-iso-V2-wp80",
                                          #"egmGsfElectronIDs:mvaEleID-Fall17-iso-V2-wpLoose",
                                          #"egmGsfElectronIDs:mvaEleID-Fall17-iso-V2-wp90",
                                          #"egmGsfElectronIDs:mvaEleID-Fall17-noIso-V1-wp90",
                                          #"egmGsfElectronIDs:mvaEleID-Fall17-noIso-V1-wp80",
                                          #"egmGsfElectronIDs:mvaEleID-Fall17-noIso-V1-wpLoose",
                                          #"egmGsfElectronIDs:mvaEleID-Fall17-iso-V1-wp90",
                                          #"egmGsfElectronIDs:mvaEleID-Fall17-iso-V1-wp80",
                                          #"egmGsfElectronIDs:mvaEleID-Fall17-iso-V1-wpLoose",
                                          ),
        eleMVALabels        = cms.vstring(
                                          #"Spring16GPV1wp80",
                                          #"Spring16GPV1wp90",
                                          #"Spring16HZZV1wpLoose",
                                          #"Fall17noIsoV2wp80",
                                          #"Fall17noIsoV2wpLoose",
                                          #"Fall17noIsoV2wp90",
                                          #"Fall17isoV2wpHZZ",
                                          #"Fall17isoV2wp80",
                                          #"Fall17isoV2wpLoose",
                                          #"Fall17isoV2wp90",
                                          #"Fall17noIsoV1wp90",
                                          #"Fall17noIsoV1wp80",
                                          #"Fall17noIsoV1wpLoose",
                                          #"Fall17isoV1wp90",
                                          #"Fall17isoV1wp80",
                                          #"Fall17isoV1wpLoose",
                                           "RunIIIWinter22isoV1wp80",
                                           "RunIIIWinter22isoV1wp90", 
                                           "RunIIIWinter22noIsoV1wp80",
                                           "RunIIIWinter22noIsoV1wp90", 
                                          ),
        eleMVAValMaps        = cms.vstring(
                                           #"electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring16GeneralPurposeV1Values",
                                           #"electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring16GeneralPurposeV1RawValues",
                                           #"electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring16HZZV1Values",
                                           #"electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring16HZZV1RawValues",
                                           #"electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17NoIsoV2Values",
                                           #"electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17NoIsoV2RawValues",
                                           #"electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17IsoV2Values",
                                           #"electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17IsoV2RawValues",
                                           #"electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17IsoV1Values",
                                           #"electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17NoIsoV1Values",
                                           "electronMVAValueMapProducer:ElectronMVAEstimatorRun2RunIIIWinter22IsoV1Values",
                                           "electronMVAValueMapProducer:ElectronMVAEstimatorRun2RunIIIWinter22IsoV1RawValues",
                                           "electronMVAValueMapProducer:ElectronMVAEstimatorRun2RunIIIWinter22NoIsoV1Values",
                                           "electronMVAValueMapProducer:ElectronMVAEstimatorRun2RunIIIWinter22NoIsoV1RawValues",
                                           ),
        eleMVAValMapLabels   = cms.vstring(
                                           #"Spring16GPV1Vals",
                                           #"Spring16GPV1RawVals",
                                           #"Spring16HZZV1Vals",
                                           #"Spring16HZZV1RawVals",
                                           #"Fall17NoIsoV2Vals",
                                           #"Fall17NoIsoV2RawVals",
                                           #"Fall17IsoV2Vals",
                                           #"Fall17IsoV2RawVals",
                                           #"Fall17IsoV1Vals",
                                           "RunIIIWinter22NoIsoV1Vals",
                                           "RunIIIWinter22NoIsoV1RawVals",
                                           "RunIIIWinter22IsoV1Vals",
                                           "RunIIIWinter22IsoV1RawVals",
                                           ),
        eleMVACats           = cms.vstring(
                                           #"electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17NoIsoV1Categories",
                                           "electronMVAValueMapProducer:ElectronMVAEstimatorRun2RunIIIWinter22NoIsoV1Categories",
                                           ),
        eleMVACatLabels      = cms.vstring(
                                           "EleMVACats",
                                           ),
        #
        variableDefinition   = cms.string(mvaVariablesFile),
        ptThreshold = cms.double(5.0),
        #
        doEnergyMatrix = cms.bool(False), # disabled by default due to large size
        energyMatrixSize = cms.int32(2), # corresponding to 5x5
        #
        **input_tags
        )
"""
The energy matrix is for ecal driven electrons the n x n of raw
rec-hit energies around the seed crystal.

The size of the energy matrix is controlled with the parameter
"energyMatrixSize", which controlls the extension of crystals in each
direction away from the seed, in other words n = 2 * energyMatrixSize + 1.

The energy matrix gets saved as a vector but you can easily unroll it
to a two dimensional numpy array later, for example like that:

>>> import uproot
>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> tree = uproot.open("electron_ntuple.root")["ntuplizer/tree"]
>>> n = 5

>>> for a in tree.array("ele_energyMatrix"):
>>>     a = a.reshape((n,n))
>>>     plt.imshow(np.log10(a))
>>>     plt.colorbar()
>>>     plt.show()
"""

process.TFileService = cms.Service("TFileService", fileName = cms.string(outputFile))

process.p = cms.Path(process.egmGsfElectronIDSequence * process.ntuplizer)
