import FWCore.ParameterSet.Config as cms

# config db parameters
SiStripConfigDb = cms.Service("SiStripConfigDb",
    UsingDbCache = cms.untracked.bool(True),
    UsingDb = cms.untracked.bool(True),
    SharedMemory = cms.untracked.string('FEDSM00')
)

# use the config db rather than conditions db configuration parameters
SiStripCondObjBuilderFromDb = cms.Service("SiStripCondObjBuilderFromDb")
FedCablingFromConfigDb = cms.ESSource("SiStripFedCablingBuilderFromDb",
    CablingSource = cms.untracked.string('UNDEFINED')
)
myFedCablingPrefer = cms.ESPrefer("SiStripFedCablingBuilderFromDb", "FedCablingFromConfigDb")
PedestalsFromConfigDb = cms.ESSource("SiStripPedestalsBuilderFromDb")
myPedestalsPrefer = cms.ESPrefer("SiStripPedestalsBuilderFromDb", "PedestalsFromConfigDb")
NoiseFromConfigDb = cms.ESSource("SiStripNoiseBuilderFromDb")
myNoisePrefer = cms.ESPrefer("SiStripNoiseBuilderFromDb", "NoiseFromConfigDb")
# produce SiStripFecCabling and SiStripDetCabling out of SiStripFedCabling
sistripconn = cms.ESProducer("SiStripConnectivity")
SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")
