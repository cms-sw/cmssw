#


import FWCore.ParameterSet.Config as cms

process = cms.Process("MAGNETICFIELDTEST")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#process.source.firstRun = cms.untracked.uint32(220642) # To get Run 2 IOVs

process.load("MagneticField.Engine.volumeBasedMagneticFieldFromDB_cfi")

process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')


process.GlobalTag.toGet = cms.VPSet(
    # Geometries
#     cms.PSet(record = cms.string("MFGeometryFileRcd"),
#              tag = cms.string("MFGeometry_90322"),
#              connect = cms.string("frontier://FrontierProd/CMS_COND_GEOMETRY"),
#              label = cms.untracked.string("90322")
#              ),
#     cms.PSet(record = cms.string("MFGeometryFileRcd"),
#              tag = cms.string("MFGeometry_120812"),
#              connect = cms.string("frontier://FrontierPrep/CMS_COND_GEOMETRY"),
#              label = cms.untracked.string("120812")
#              ),
#     cms.PSet(record = cms.string("MFGeometryFileRcd"),
#              tag = cms.string("MFGeometry_130503"),
#              connect = cms.string("frontier://FrontierPrep/CMS_COND_GEOMETRY"),
#              label = cms.untracked.string("130503")
#              ),
    cms.PSet(record = cms.string("MFGeometryFileRcd"),
#              tag = cms.string("MagneticFieldGeometry_160812"),
#              connect = cms.string("sqlite_file:DB_Geom/mfGeometry_160812.db"),
             tag = cms.string("MFGeometry_160812"),
#             connect = cms.string("frontier://FrontierPrep/CMS_CONDITIONS"),
#             connect = cms.string("frontier://PromptProd/CMS_CONDITIONS"),
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
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_160812_Run2_3T.db"),
              tag = cms.string("MFConfig_160812_Run2_3T"),
              connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
              label = cms.untracked.string("3T")
              ),

     cms.PSet(record = cms.string("MagFieldConfigRcd"), # version 160812, 3.5T (good also for Run1)
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_160812_Run2_3_5T.db"),
              tag = cms.string("MFConfig_160812_Run2_3_5T"),
              connect = cms.string("frontier://PromptProd/CMS_CONDITIONS"),
              label = cms.untracked.string("3.5T")
              ),

     cms.PSet(record = cms.string("MagFieldConfigRcd"), # version 160812, 3.8T
#              tag = cms.string("MagFieldConfig"),
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_160812_Run1_3_8T.db"), #Run 1, version 160812, 3.8T (single IOV, for MC)
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_160812_Run2_3_8T.db"), #Run 2, version 160812, 3.8T (single IOV, for MC)
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_RI_RII_160812_3_8T.db"),#Run 1+2, version 160812, 3.8T (two IOVs, for data)
#              connect = cms.string("sqlite_file:DB_Conf/MFConfig_RI90322_RII160812_3_8T.db"),#Run 1+2, version 160812, 3.8T (two IOVs, for data)              
#              tag = cms.string("MFConfig_90322_2pi_scaled_3_8T"), #Run 1, version 90322, 3.8T (single IOV, for MC)
#              tag = cms.string("MFConfig_160812_Run1_3_8T"), #Run 1, version 160812, 3.8T (single IOV, for MC)
              tag = cms.string("MFConfig_160812_Run2_3_8T"), #Run 2, version 160812, 3.8T (single IOV, for MC)
              connect = cms.string("frontier://PromptProd/CMS_CONDITIONS"),
#              connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
              label = cms.untracked.string("3.8T")
              ),

     cms.PSet(record = cms.string("MagFieldConfigRcd"),
              tag = cms.string("MFConfig_71212_4T"),
              connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
              label = cms.untracked.string("4T")
              ),

)

# Set the nominal current 
# process.VolumeBasedMagneticFieldESProducer.valueOverride =   9500 #2T
# process.VolumeBasedMagneticFieldESProducer.valueOverride =  14340 #3T
# process.VolumeBasedMagneticFieldESProducer.valueOverride =  16730 #3.5T
# process.VolumeBasedMagneticFieldESProducer.valueOverride =  18164 #3.8T (default for MC)
# process.VolumeBasedMagneticFieldESProducer.valueOverride =  19140 #4T

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        MagneticField = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING')
    )
)


process.testMagneticField = cms.EDAnalyzer("testMagneticField",

## Use the specified reference file 
#        inputTable = cms.untracked.string("/afs/cern.ch/cms/OO/mag_field/CMSSW/regression/referenceField_90322_2pi_scaled_3.8T_750.txt"),
#	inputTable = cms.untracked.string("/afs/cern.ch/cms/OO/mag_field/CMSSW/regression/referenceField_160812_Run1_3_8t.txt"),
#	inputTable = cms.untracked.string("/afs/cern.ch/cms/OO/mag_field/CMSSW/regression/referenceField_160812_3t.txt"),
#	inputTable = cms.untracked.string("/afs/cern.ch/cms/OO/mag_field/CMSSW/regression/referenceField_160812_3_5t.txt"),
	inputTable = cms.untracked.string("/afs/cern.ch/cms/OO/mag_field/CMSSW/regression/referenceField_160812_3_8t.txt"),
                                           

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

