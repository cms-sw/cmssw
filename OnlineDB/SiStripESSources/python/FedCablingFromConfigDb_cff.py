import FWCore.ParameterSet.Config as cms

from OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi import *
FedCablingFromConfigDb = cms.ESSource("SiStripFedCablingBuilderFromDb",
    CablingSource = cms.untracked.string('UNDEFINED')
)

SiStripConfigDb.UsingDb = True
SiStripConfigDb.ConfDb = ''
SiStripConfigDb.Partition = ''

