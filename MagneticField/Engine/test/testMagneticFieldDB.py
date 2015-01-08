#

import FWCore.ParameterSet.Config as cms

process = cms.Process("MAGNETICFIELDTEST")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("MagneticField.Engine.volumeBasedMagneticFieldFromDB_cfi")

process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')



process.GlobalTag.toGet = cms.VPSet(
    # Geometries
    cms.PSet(record = cms.string("MFGeometryFileRcd"),
#             tag = cms.string("MagneticFieldGeometry_90322"),
#             connect = cms.untracked.string("sqlite_file:DB_Geom/mfGeometry_90322.db"),
             tag = cms.string("MFGeometry_90322"),
             connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_GEOMETRY"),
             label = cms.untracked.string("90322")
             ),
    cms.PSet(record = cms.string("MFGeometryFileRcd"),
#             tag = cms.string("MagneticFieldGeometry_120812"),
#             connect = cms.untracked.string("sqlite_file:DB_Geom/mfGeometry_120812.db"),
             tag = cms.string("MFGeometry_120812"),
             connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_GEOMETRY"),
             label = cms.untracked.string("120812")
             ),
    cms.PSet(record = cms.string("MFGeometryFileRcd"),
#             tag = cms.string("MagneticFieldGeometry_130503"),
#             connect = cms.untracked.string("sqlite_file:DB_Geom/mfGeometry_130503.db"),
             tag = cms.string("MFGeometry_130503"),
             connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_GEOMETRY"),
             label = cms.untracked.string("130503")
             ),

    # Configurations
    cms.PSet(record = cms.string("MagFieldConfigRcd"),
             tag = cms.string("MagFieldConfig"),
             connect = cms.untracked.string("sqlite_file:DB_Conf/MFConfig_71212_0T.db"),
             label = cms.untracked.string("0T")
             ),
    cms.PSet(record = cms.string("MagFieldConfigRcd"),
             tag = cms.string("MagFieldConfig"),
             connect = cms.untracked.string("sqlite_file:DB_Conf/MFConfig_71212_2T.db"),
             label = cms.untracked.string("2T")
             ),
    cms.PSet(record = cms.string("MagFieldConfigRcd"),
             tag = cms.string("MagFieldConfig"),
             connect = cms.untracked.string("sqlite_file:DB_Conf/MFConfig_71212_3T.db"),
             label = cms.untracked.string("3T")
             ),
    cms.PSet(record = cms.string("MagFieldConfigRcd"),
             tag = cms.string("MagFieldConfig"),
             connect = cms.untracked.string("sqlite_file:DB_Conf/MFConfig_71212_3_5T.db"),
             label = cms.untracked.string("3.5T")
             ),
#     cms.PSet(record = cms.string("MagFieldConfigRcd"), # Run 2, new version (130503)
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.untracked.string("sqlite_file:DB_Conf/MFConfig_130503_Run2_3_5T.db"),
#              label = cms.untracked.string("3.5T")
#              ),    
    cms.PSet(record = cms.string("MagFieldConfigRcd"), #Run 1 default
             tag = cms.string("MagFieldConfig"),
             connect = cms.untracked.string("sqlite_file:DB_Conf/MFConfig_90322_2pi_scaled_3_8T.db"),
             label = cms.untracked.string("3.8T")
             ),
#     cms.PSet(record = cms.string("MagFieldConfigRcd"), #Run 2 default
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.untracked.string("sqlite_file:DB_Conf/MFConfig_120812_Run2_3_8T.db"),
#              label = cms.untracked.string("3.8T")
#              ),
#     cms.PSet(record = cms.string("MagFieldConfigRcd"), #Run 1, version 120812
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.untracked.string("sqlite_file:DB_Conf/MFConfig_120812_Run1_3_8T.db"),
#              label = cms.untracked.string("3.8T")
#              ),
#     cms.PSet(record = cms.string("MagFieldConfigRcd"), #Run 1, version 130503
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.untracked.string("sqlite_file:DB_Conf/MFConfig_130503_Run1_3_8T.db"),
#              label = cms.untracked.string("3.8T")
#              ),
#     cms.PSet(record = cms.string("MagFieldConfigRcd"), #Run 2, new version (130503)
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.untracked.string("sqlite_file:DB_Conf/MFConfig_130503_Run2_3_8T.db"),
#              label = cms.untracked.string("3.8T")
#              ),
    cms.PSet(record = cms.string("MagFieldConfigRcd"),
             tag = cms.string("MagFieldConfig"),
             connect = cms.untracked.string("sqlite_file:DB_Conf/MFConfig_71212_4T.db"),
             label = cms.untracked.string("4T")
             ),


)

# process.VolumeBasedMagneticFieldESProducer.valueOverride =  17000 

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

process.testField  = cms.EDAnalyzer("testMagneticField")
process.p1 = cms.Path(process.testField)

