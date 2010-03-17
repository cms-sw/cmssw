from FWCore.ParameterSet.Config import *

process = cms.Process("Calibration")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#     '/store/relval/CMSSW_2_2_0_pre1/RelValZEE/ALCARECO/STARTUP_V7_StreamALCARECOEcalCalElectron_v1/0000/AEEDE438-11B0-DD11-8F7F-001617C3B5D8.root'
#     '/store/relval/CMSSW_3_5_0_pre3/RelValZEE/ALCARECO/STARTUP3X_V15_StreamEcalCalElectron-v2/0006/C8EBACB1-2F05-DF11-9B94-0030487A3C9A.root'
     '/store/relval/CMSSW_3_5_4/RelValZEE/ALCARECO/START3X_V24_EcalCalElectron-v1/0004/2AB6B40D-2D2C-DF11-814B-002618943948.root'
          ),
    secondaryFileNames = cms.untracked.vstring (
   )
)

process.CalibFilter = cms.EDFilter("IMASelector",
	src = cms.InputTag("gsfElectrons"),
	ESCOPinMin=cms.double(0.9),
	ESCOPinMax=cms.double(1.2),
	ESeedOPoutMin=cms.double(0.1),
	ESeedOPoutMax=cms.double(9),
	PinMPoutOPinMin=cms.double(-0.1),
	PinMPoutOPinMax=cms.double(0.5),
	EMPoutMin=cms.double(0.9),
	EMPoutMax=cms.double(1.2)
	)

process.looper = cms.Looper("EcalEleCalibLooper",
    electronLabel = cms.InputTag('CalibFilter'),
    alcaBarrelHitCollection = cms.InputTag("alCaIsolatedElectrons:alcaBarrelHits"),
    alcaEndcapHitCollection = cms.InputTag("alCaIsolatedElectrons:alcaEndcapHits"),
    recoWindowSidex = cms.int32 (5),
    recoWindowSidey = cms.int32 (5),
    minEnergyPerCrystal = cms.double (0.1),
    maxEnergyPerCrystal = cms.double (1200),
    etaStart = cms.int32 (3), #Intervals of the tipe [start,end)
    etaEnd = cms.int32 (83),
    etaWidth = cms.int32 (20),
    phiWidthEB = cms.int32 (20),
    phiStartEB = cms.int32 (0),
    phiEndEB = cms.int32 (40), 
    phiWidthEE = cms.int32 (20),
    phiStartEE = cms.int32 (0),
    phiEndEE = cms.int32 (40), 
    radStart = cms.int32 (10),
    radEnd = cms.int32 (40),
    radWidth = cms.int32(10),
    maxSelectedNumPerCrystal = cms.int32 (-1), # -1 means "get all"
#    EEZone = cms.int32 (1), #1 for EE+, -1 for EE-, 0 for both 
    minCoeff = cms.double (0.2),
    maxCoeff = cms.double (1.9),
    usingBlockSolver = cms.int32 (0),
    loops = cms.int32 (3),
    algorithm = cms.string ("L3"),
    L3EventWeight = cms.untracked.int32 (2),
    FillType = cms.string ("Matrix")

)

process.p = cms.Path(process.CalibFilter)


# # per dumpare un cfg esplicitando tutti i load e include
# # aggiungere al cfg (mio_cfg.py):
# temp = process.dumpPython()
# outputfile = file("recoT0DQM_EvContent.py",'w')
# # poi fai 
# # python tuo_cfg.py
