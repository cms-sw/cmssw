# The following comments couldn't be translated into the new config version:

#@@ A non-zero run number will override any versions specified
#@@ A null run number means use the versions specified
#@@ For all versions, "0" means "current state"  
#@@ If "ForceDcuDetIdsVersions" is true, versions not overriden by run number (useful for O2O)
#@@ If TNS_ADMIN is set, it will override the environmental variable!

import FWCore.ParameterSet.Config as cms

SiStripConfigDb = cms.Service("SiStripConfigDb",
    FedMinorVersion = cms.untracked.uint32(0),
    InputDcuInfoXml = cms.untracked.string('/afs/cern.ch/cms/cmt/onlinedev/data/examples/dcuinfo.xml'),
    OutputFedXml = cms.untracked.string('/tmp/fed.xml'),
    OutputDcuInfoXml = cms.untracked.string('/tmp/dcuinfo.xml'),
    CalibMajorVersion = cms.untracked.uint32(0),
    SharedMemory = cms.untracked.string(''),
    MinorVersion = cms.untracked.uint32(0), ##@@ cabling

    OutputFecXml = cms.untracked.string('/tmp/fec.xml'),
    FecMajorVersion = cms.untracked.uint32(0),
    CalibMinorVersion = cms.untracked.uint32(0),
    TNS_ADMIN = cms.untracked.string('/afs/cern.ch/project/oracle/admin'),
    UsingDb = cms.untracked.bool(False),
    DcuDetIdMajorVersion = cms.untracked.uint32(0),
    OutputModuleXml = cms.untracked.string('/tmp/module.xml'),
    FedMajorVersion = cms.untracked.uint32(0),
    ForceDcuDetIdsVersions = cms.untracked.bool(True),
    RunNumber = cms.untracked.uint32(0),
    Partition = cms.untracked.string(''),
    ConfDb = cms.untracked.string(''),
    UsingDbCache = cms.untracked.bool(False),
    MajorVersion = cms.untracked.uint32(0), ##@@ cabling

    InputFedXml = cms.untracked.vstring(''),
    InputFecXml = cms.untracked.vstring(''),
    InputModuleXml = cms.untracked.string('/afs/cern.ch/cms/cmt/onlinedev/data/examples/module.xml'),
    FecMinorVersion = cms.untracked.uint32(0),
    DcuDetIdMinorVersion = cms.untracked.uint32(0)
)


