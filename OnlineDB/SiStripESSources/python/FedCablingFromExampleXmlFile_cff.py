import FWCore.ParameterSet.Config as cms

from OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi import *
FedCablingFromConfigDb = cms.ESSource("SiStripFedCablingBuilderFromDb",
    CablingSource = cms.untracked.string('UNDEFINED')
)

SiStripConfigDb.UsingDb = False
SiStripConfigDb.InputModuleXml = '/afs/cern.ch/cms/cmt/onlinedev/data/examples/module.xml'
SiStripConfigDb.InputDcuInfoXml = '/afs/cern.ch/cms/cmt/onlinedev/data/examples/dcuinfo.xml'

