import FWCore.ParameterSet.Config as cms

# config db parameters
from OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi import *
SiStripConfigDb.UsingDb = True                    # should be true!
SiStripConfigDb.ConfDb = 'user/password@account'  # taken from $CONFDB

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
