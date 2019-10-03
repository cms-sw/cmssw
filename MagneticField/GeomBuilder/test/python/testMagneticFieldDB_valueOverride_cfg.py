#


import FWCore.ParameterSet.Config as cms

process = cms.Process("MAGNETICFIELDTEST")
process.maxEvents.input = 2

process.source = cms.Source("EmptySource",
                            firstLuminosityBlockForEachRun = cms.untracked.VLuminosityBlockID(
                                cms.LuminosityBlockID(10,1),
                                cms.LuminosityBlockID(20,2),
                            ),
                            numberEventsInLuminosityBlock =cms.untracked.uint32(1)
)

#the job should not use the data product made by this ESProducer
# it is here with different avg_current than the valueOverride
process.add_( cms.ESProducer("RunInfoTestESProducer",
                             runInfos = cms.VPSet(cms.PSet(run = cms.int32(10), avg_current = cms.double(0.)),
                                              cms.PSet(run = cms.int32(20), avg_current = cms.double(9000.)),
 ) ) )

process.riSource = cms.ESSource("EmptyESSource", recordName = cms.string("RunInfoRcd"),
                                iovIsRunNotTime = cms.bool(True),
                                firstValid = cms.vuint32(10,20))

process.load("MagneticField.Engine.volumeBasedMagneticFieldFromDB_cfi")
process.VolumeBasedMagneticFieldESProducer.valueOverride = 18000

process.load("Configuration/StandardSequences/CondDBESSource_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')


#configuration of GlobalTag comes from MagneticField/Engine/test/testMagneticFieldDB.py
process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("MFGeometryFileRcd"),
             tag = cms.string("MFGeometry_160812"),
             connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
             label = cms.untracked.string("160812")
             ),
    # Configurations
     cms.PSet(record = cms.string("MagFieldConfigRcd"),
              tag = cms.string("MFConfig_71212_0T"),
              connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
              label = cms.untracked.string("0T")
              ),
     cms.PSet(record = cms.string("MagFieldConfigRcd"),
              tag = cms.string("MFConfig_71212_2T"),
              connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
              label = cms.untracked.string("2T")
              ),
     cms.PSet(record = cms.string("MagFieldConfigRcd"), # version 160812, 3T (good also for Run1)
              tag = cms.string("MFConfig_160812_Run2_3T"),
              connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
              label = cms.untracked.string("3T")
              ),
     cms.PSet(record = cms.string("MagFieldConfigRcd"), # version 160812, 3.5T (good also for Run1)
              tag = cms.string("MFConfig_160812_Run2_3_5T"),
              connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
              label = cms.untracked.string("3.5T")
              ),
     cms.PSet(record = cms.string("MagFieldConfigRcd"), # version 160812, 3.8T
              tag = cms.string("MFConfig_160812_Run2_3_8T"), #Run 2, version 160812, 3.8T (single IOV, for MC)
              connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
              label = cms.untracked.string("3.8T")
              ),
     cms.PSet(record = cms.string("MagFieldConfigRcd"),
              tag = cms.string("MFConfig_71212_4T"),
              connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
              label = cms.untracked.string("4T")
              ),
)

process.MessageLogger = cms.Service("MessageLogger",
    categories   = cms.untracked.vstring("MagneticField"),
    destinations = cms.untracked.vstring("cout"),
    cout = cms.untracked.PSet(  
    noLineBreaks = cms.untracked.bool(True),
    threshold = cms.untracked.string("WARNING"),
    WARNING = cms.untracked.PSet(
      limit = cms.untracked.int32(0)
    ),
    MagneticField = cms.untracked.PSet(
     limit = cms.untracked.int32(10000000)
    )
  )
)


process.testMagneticField = cms.EDAnalyzer("testMagneticField"
)

process.p1 = cms.Path(process.testMagneticField)

