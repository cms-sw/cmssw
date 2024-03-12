import FWCore.ParameterSet.Config as cms

from OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi import *

SiStripConfigDb.UsingDb = False
SiStripConfigDb.Partitions.PrimaryPartition.InputModuleXml = '/afs/cern.ch/cms/cmt/onlinedev/data/module.xml'
SiStripConfigDb.Partitions.PrimaryPartition.InputDcuInfoXml = ''
SiStripConfigDb.Partitions.PrimaryPartition.InputFedXml = ['/afs/cern.ch/cms/cmt/onlinedev/data/fed9.xml', '/afs/cern.ch/cms/cmt/onlinedev/data/fed15.xml']

PedestalsFromConfigDb = cms.ESSource("SiStripPedestalsBuilderFromDb")

# foo bar baz
# 0vmuCnQvULzQ2
# yT30KqBFojI34
