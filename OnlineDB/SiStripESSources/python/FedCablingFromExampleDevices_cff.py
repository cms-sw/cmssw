import FWCore.ParameterSet.Config as cms

from OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi import *
FedCablingFromConfigDb = cms.ESSource("SiStripFedCablingBuilderFromDb",
    CablingSource = cms.untracked.string('DEVICES')
)

SiStripConfigDb.UsingDb = False
SiStripConfigDb.InputModuleXml = ''
SiStripConfigDb.InputDcuInfoXml = ''
SiStripConfigDb.InputFecXml = ['/afs/cern.ch/cms/cmt/onlinedev/data/telescope/20991/fec.xml']
SiStripConfigDb.InputFedXml = ['/afs/cern.ch/cms/cmt/onlinedev/data/telescope/20991/fed4.xml', '/afs/cern.ch/cms/cmt/onlinedev/data/telescope/20991/fed5.xml', '/afs/cern.ch/cms/cmt/onlinedev/data/telescope/20991/fed6.xml', '/afs/cern.ch/cms/cmt/onlinedev/data/telescope/20991/fed7.xml']

