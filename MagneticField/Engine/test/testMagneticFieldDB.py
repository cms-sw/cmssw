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
#     cms.PSet(record = cms.string("MFGeometryFileRcd"),
# #             tag = cms.string("MagneticFieldGeometry_90322"),
# #             connect = cms.untracked.string("sqlite_file:DB_Geom/mfGeometry_90322.db"),
#              tag = cms.string("MFGeometry_90322"),
#              connect = cms.string("frontier://FrontierPrep/CMS_COND_GEOMETRY"),
#              label = cms.untracked.string("90322")
#              ),
#     cms.PSet(record = cms.string("MFGeometryFileRcd"),
# #             tag = cms.string("MagneticFieldGeometry_120812"),
# #             connect = cms.string("sqlite_file:DB_Geom/mfGeometry_120812.db"),
#              tag = cms.string("MFGeometry_120812"),
#              connect = cms.string("frontier://FrontierPrep/CMS_COND_GEOMETRY"),
#              label = cms.untracked.string("120812")
#              ),
#     cms.PSet(record = cms.string("MFGeometryFileRcd"),
# #             tag = cms.string("MagneticFieldGeometry_130503"),
# #             connect = cms.string("sqlite_file:DB_Geom/mfGeometry_130503.db"),
#              tag = cms.string("MFGeometry_130503"),
#              connect = cms.string("frontier://FrontierPrep/CMS_COND_GEOMETRY"),
#              label = cms.untracked.string("130503")
#              ),
    cms.PSet(record = cms.string("MFGeometryFileRcd"),
#              tag = cms.string("MagneticFieldGeometry_160812"),
#              connect = cms.string("sqlite_file:DB_Geom/mfGeometry_160812.db"),
             tag = cms.string("MFGeometry_160812"),
#             connect = cms.string("frontier://FrontierPrep/CMS_CONDITIONS"),
             connect = cms.string("frontier://PromptProd/CMS_CONDITIONS"),
             label = cms.untracked.string("160812")
             ),




    # Configurations
#     cms.PSet(record = cms.string("MagFieldConfigRcd"),
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_71212_0T.db"),
#              label = cms.untracked.string("0T")
#              ),
#     cms.PSet(record = cms.string("MagFieldConfigRcd"),
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_71212_2T.db"),
#              label = cms.untracked.string("2T")
#              ),
#     cms.PSet(record = cms.string("MagFieldConfigRcd"),
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_71212_3T.db"),
#              label = cms.untracked.string("3T")
#              ),
#     cms.PSet(record = cms.string("MagFieldConfigRcd"),
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_71212_3_5T.db"),
#              label = cms.untracked.string("3.5T")
#              ),
#     cms.PSet(record = cms.string("MagFieldConfigRcd"), # Run 2, new version (130503)
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_130503_Run2_3_5T.db"),
#              label = cms.untracked.string("3.5T")
#              ),    
#     cms.PSet(record = cms.string("MagFieldConfigRcd"), #Run 1 default
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_90322_2pi_scaled_3_8T.db"),
#              label = cms.untracked.string("3.8T")
#              ),
#     cms.PSet(record = cms.string("MagFieldConfigRcd"), #Run 2 default
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_120812_Run2_3_8T.db"),
#              label = cms.untracked.string("3.8T")
#              ),
#     cms.PSet(record = cms.string("MagFieldConfigRcd"), #Run 1, version 120812
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_120812_Run1_3_8T.db"),
#              label = cms.untracked.string("3.8T")
#              ),
#     cms.PSet(record = cms.string("MagFieldConfigRcd"), #Run 1, version 130503
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_130503_Run1_3_8T.db"),
#              label = cms.untracked.string("3.8T")
#              ),
#     cms.PSet(record = cms.string("MagFieldConfigRcd"), #Run 2, version 130503
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_130503_Run2_3_8T.db"),
#              label = cms.untracked.string("3.8T")
#              ),
     cms.PSet(record = cms.string("MagFieldConfigRcd"), #Run 2, version 160812
              tag = cms.string("MagFieldConfig"),
              connect = cms.string("sqlite_file:DB_Conf/MFConfig_160812_Run2_3_8T.db"),
              label = cms.untracked.string("3.8T")
              ),

#     cms.PSet(record = cms.string("MagFieldConfigRcd"),
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_71212_4T.db"),
#              label = cms.untracked.string("4T")
#              ),


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


process.testMagneticField = cms.EDAnalyzer("testMagneticField",

## Use the specified reference file 
	inputTable = cms.untracked.string("/afs/cern.ch/cms/OO/mag_field/CMSSW/regression/referenceField_160812_3_8t.txt"),
#        inputTable = cms.untracked.string("newtable.txt"),
                                           

## Valid input file types: "xyz_cm", "rpz_m", "xyz_m", "TOSCA" 
	inputTableType = cms.untracked.string("xyz_cm"),

## Resolution used for validation, number of points
	resolution     = cms.untracked.double(0.0001),
	numberOfPoints = cms.untracked.int32(1000000),

## Size of testing volume (cm):
	InnerRadius = cms.untracked.double(0),    
	OuterRadius = cms.untracked.double(900),  
        HalfLength  = cms.untracked.double(2400)

)

process.p1 = cms.Path(process.testMagneticField)


# process.testField  = cms.EDAnalyzer("testMagneticField")

# process.p1 = cms.Path(process.testField)

