
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()

options.register('producerType',
                 'static_DDD', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "MF producer to use. Valid values: 'static_DDD', 'static_DD4Hep', 'fromDB', 'fromDB_DD4Hep'")

options.register('era',
                 'RunII', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "'RunI'or 'RunII'")

options.register('current',
                 18000, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "Magnet current (nominal values: 18164=3.8T; 16730=3.5T; 14340=3T; 9500=2T")

options.parseArguments()


process = cms.Process("MAGNETICFIELDTEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

REFERENCEFILE = 'none'
if options.current > 18765 or options.current <= 4779 :
    print '\nERROR: invalid current value: ',  options.current,'\n'
    exit()
elif options.current > 17543 :
    if options.era == 'RunII' :
        REFERENCEFILE = 'MagneticField/Engine/data/Regression/referenceField_160812_RII_3_8T.bin'
    elif options.era == 'RunI' :
        REFERENCEFILE = 'MagneticField/Engine/data/Regression/referenceField_160812_RI_3_8T.bin'
    else: 
        print '\nERROR: Invalid era: ',  options.era,'\n'
        exit()
elif options.current > 15617 :
    REFERENCEFILE = 'MagneticField/Engine/data/Regression/referenceField_160812_3_5T.bin'
elif options.current > 11987 :
    REFERENCEFILE = 'MagneticField/Engine/data/Regression/referenceField_160812_3T.bin'
elif options.current > 4779 :
    REFERENCEFILE = 'MagneticField/Engine/data/Regression/referenceField_71212_2T.bin'


if options.producerType == 'static_DDD':
    if options.current > 17543 :
        process.load("MagneticField.Engine.volumeBasedMagneticField_160812_cfi") # 3.8T, RII
        if options.era == 'RunI' :
            process.VolumeBasedMagneticFieldESProducer.version = cms.string('grid_160812_3_8t_Run1') # 3.8T, RII
    elif options.current > 15617 :
        process.load("MagneticField.Engine.volumeBasedMagneticField_160812_cfi")
        process.VolumeBasedMagneticFieldESProducer.version = cms.string('grid_160812_3_5t') # 3.5T
        process.ParametrizedMagneticFieldProducer.parameters.BValue = cms.string('3_5T')
    elif options.current > 11987 :
        process.load("MagneticField.Engine.volumeBasedMagneticField_160812_cfi")
        process.VolumeBasedMagneticFieldESProducer.version = cms.string('grid_160812_3t')
        process.ParametrizedMagneticFieldProducer.parameters.BValue = cms.string('3_0T')
    elif options.current > 4779 :
        process.load("MagneticField.Engine.volumeBasedMagneticField_71212_cfi") #2.0T


elif options.producerType == 'static_DD4Hep' :
    process.load("MagneticField.Engine.volumeBasedMagneticField_dd4hep_160812_cfi") 
    if options.current > 17543 :
        if options.era == 'RunI' :
            process.VolumeBasedMagneticFieldESProducer.version = cms.string('grid_160812_3_8t_Run1') # 3.8T, RII
    elif options.current > 15617 :
        process.VolumeBasedMagneticFieldESProducer.version = cms.string('grid_160812_3_5t') # 3.5T
        process.ParametrizedMagneticFieldProducer.parameters.BValue = cms.string('3_5T')
    elif options.current > 11987 :
        process.VolumeBasedMagneticFieldESProducer.version = cms.string('grid_160812_3t')
        process.ParametrizedMagneticFieldProducer.parameters.BValue = cms.string('3_0T')
    elif options.current > 4779 :
        print '\nERROR: Unsupported current for static_DD4Hep: ',  options.current,'\n'
        exit()


elif options.producerType == 'fromDB' or options.producerType == 'fromDB_DD4Hep':
    if options.producerType == 'fromDB':
        process.load("Configuration.StandardSequences.MagneticField_cff")
    elif options.producerType == 'fromDB_DD4Hep':
        process.load("MagneticField.Engine.volumeBasedMagneticFieldFromDB_dd4hep_cfi")

    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
    from Configuration.AlCa.GlobalTag import GlobalTag
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '') # Note: testing the proper set-up of data and MC iovs is beyond the scope of this script.

    if options.era == 'RunII' :
        process.source.firstRun = cms.untracked.uint32(300000)
    elif options.era == 'RunI' :
        process.source.firstRun = cms.untracked.uint32(100)

    # Note that this tests the map and the valueOverride method, but not the mechanism using the actual RunInfo, one mor set of tests could be added for this purpose.
    if options.current > 17543:
        process.VolumeBasedMagneticFieldESProducer.valueOverride = -1 #3.8T is the default
    elif options.current > 15617 :
        process.VolumeBasedMagneticFieldESProducer.valueOverride = 17000 #3.5 T
    elif options.current > 11987 :
        process.VolumeBasedMagneticFieldESProducer.valueOverride = 14000 #3 T
    elif options.current > 4779 :
        process.VolumeBasedMagneticFieldESProducer.valueOverride = 10000 #2 T



else :
    print '\nERROR: invalid producerType', producerType,'\n'


print '\nRegression for MF built with', options.producerType, 'era:', options.era, 'current:', options.current,'\n'


process.testMagneticField = cms.EDAnalyzer("testMagneticField",

## Uncomment to write down the reference file
#	outputTable = cms.untracked.string("newtable.bin"),

## Use the specified reference file to compare with
	inputTable = cms.untracked.string(REFERENCEFILE),

## Valid input file types: "xyz", "rpz_m", "xyz_m", "TOSCA" 
	inputTableType = cms.untracked.string("xyz"),

## Resolution used for validation, number of points
	resolution     = cms.untracked.double(0.0001),
        numberOfPoints = cms.untracked.int32(1000000),

## Size of testing volume (cm):
	InnerRadius = cms.untracked.double(0),    #  default: 0 
	OuterRadius = cms.untracked.double(900),  #  default: 900 
        HalfLength  = cms.untracked.double(2400)  #  default: 2400 

)

process.p1 = cms.Path(process.testMagneticField)


