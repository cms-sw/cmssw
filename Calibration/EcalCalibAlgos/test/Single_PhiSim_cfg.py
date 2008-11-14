from FWCore.ParameterSet.Config import *

process = cms.Process("Calibration")

process.extend(include("FWCore/MessageLogger/data/MessageLogger.cfi"))
process.extend(include("RecoEcal/EgammaClusterProducers/data/geometryForClustering.cff"))

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
     '/store/relval/CMSSW_2_2_0_pre1/RelValZEE/ALCARECO/STARTUP_V7_StreamALCARECOEcalCalElectron_v1/0000/AEEDE438-11B0-DD11-8F7F-001617C3B5D8.root'
   ),
    secondaryFileNames = cms.untracked.vstring (
   )
)

process.CalibFilter = cms.EDFilter("IMASelector",
	src = cms.InputTag("electronFilter"),
	ESCOPinMin=cms.double(0.9),
	ESCOPinMax=cms.double(1.2),
	ESeedOPoutMin=cms.double(0.1),
	ESeedOPoutMax=cms.double(9),
	PinMPoutOPinMin=cms.double(-0.1),
	PinMPoutOPinMax=cms.double(0.5),
	EMPoutMin=cms.double(0.9),
	EMPoutMax=cms.double(1.2)
	)

process.looper = cms.Looper("InvRingCalib",
    ElectronLabel = cms.InputTag('CalibFilter'),
    barrelAlca = cms.InputTag("alCaIsolatedElectrons:alcaBarrelHits"),
    endcapAlca = cms.InputTag("alCaIsolatedElectrons:alcaEndcapHits"),
    recoWindowSidex = cms.int32 (5),
    recoWindowSidey = cms.int32 (5),
    minEnergyPerCrystal = cms.double (0.1),
    maxEnergyPerCrystal = cms.double (1200),
    etaStart = cms.int32 (3), #Intervals of the tipe [start,end)
    etaEnd = cms.int32 (83),
    etaWidth = cms.int32 (20),
    maxNumPerRing = cms.int32 (-1), # -1 means "get all"
    startRing = cms.int32 (2), #0-40, eta increasing with ring index
    endRing = cms.int32 (22),
    EEZone = cms.int32 (1), #1 for EE+, -1 for EE-, 0 for both 
    minCoeff = cms.double (0.2),
    maxCoeff = cms.double (1.9),
    usingBlockSolver = cms.int32 (0),
    loops = cms.int32 (1),
    algorithm = cms.string ("L3"),
    L3EventWeight = cms.untracked.int32 (2),
    EBcoeffs = cms.string(""),#pass the output of phi-sym in input 
    EEcoeffs = cms.string(""),
    FillType = cms.string ("Matrix")

)

process.p = cms.Path(process.CalibFilter)
