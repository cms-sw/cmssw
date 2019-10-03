#


import FWCore.ParameterSet.Config as cms

process = cms.Process("MAGNETICFIELDTEST")
process.maxEvents.input = 5

#NOTE: parameterized magnetic field does not allow 0T so skip lumi 1
process.source = cms.Source("EmptySource",
                            firstLuminosityBlockForEachRun = cms.untracked.VLuminosityBlockID(
                                #cms.LuminosityBlockID(10,1),
                                cms.LuminosityBlockID(20,2),
                                cms.LuminosityBlockID(30,3),
                                cms.LuminosityBlockID(40,4),
                                cms.LuminosityBlockID(50,5),
                                cms.LuminosityBlockID(60,6)
                            ),
                            firstLuminosityBlock = cms.untracked.uint32(2),
                            numberEventsInLuminosityBlock =cms.untracked.uint32(1)
)

process.add_( cms.ESProducer("RunInfoTestESProducer",
                             runInfos = cms.VPSet(cms.PSet(run = cms.int32(10), avg_current = cms.double(0.)), #0T
                                              cms.PSet(run = cms.int32(20), avg_current = cms.double(9000.)),  #2T
                                              cms.PSet(run = cms.int32(30), avg_current = cms.double(14000.)), #3T
                                              cms.PSet(run = cms.int32(40), avg_current = cms.double(17000.)), #3.5T
                                              cms.PSet(run = cms.int32(50), avg_current = cms.double(18000.)), #3.8T
                                              cms.PSet(run = cms.int32(60), avg_current = cms.double(19000.)), #4T
 ) ) )

process.riSource = cms.ESSource("EmptyESSource", 
                                recordName = cms.string("RunInfoRcd"),
                                iovIsRunNotTime = cms.bool(True),
                                firstValid = cms.vuint32(10,20,30,40,50,60))

#the values used for the MagFieldConfig come from
# CondFormats/MFObjects/test/writeMagFieldConfigDB.py
process.magFieldConfig0= cms.ESProducer("MagFieldConfigTestESProducer",
                                       configs = cms.VPSet(cms.PSet(run = cms.uint32(10),
                                                                    config = cms.PSet(
                                                                        scalingVolumes = cms.vint32(),
                                                                        scalingFactors = cms.vdouble(),
                                                                        version = cms.string('parametrizedMagneticField'),
                                                                        geometryVersion = cms.int32(90322),
                                                                        paramLabel = cms.string('OAE_1103l_071212'),
                                                                        paramData = cms.vdouble(0),
                                                                        gridFiles = cms.VPSet()
                                                                     )
                                                                 )
                                                        ),
                                        appendToDataLabel = cms.string('0T')
                                    )
process.magFieldConfig2= process.magFieldConfig0.clone(appendToDataLabel = '2T',
                                                       configs = {0: dict(config = dict(paramData = [2.0]) ) } )
process.magFieldConfig3= process.magFieldConfig0.clone(appendToDataLabel = '3T',
                                                       configs = {0: dict(config = dict(paramData = [3.0]) ) } )
process.magFieldConfig35= process.magFieldConfig0.clone(appendToDataLabel = '3.5T',
                                                        configs = {0: dict(config = dict(paramData = [3.5]) ) } )
process.magFieldConfig38= process.magFieldConfig0.clone(appendToDataLabel = '3.8T',
                                                        configs = {0: dict(config = dict(paramData = [3.8],
                                                                                         geometryVersion = 130503) ) } )
process.magFieldConfig4= process.magFieldConfig0.clone(appendToDataLabel = '4T',
                                                       configs = {0: dict(config = dict(paramData = [4.0]) ) } )


process.mfcSource = cms.ESSource("EmptyESSource", 
                                 recordName = cms.string("MagFieldConfigRcd"),
                                 iovIsRunNotTime = cms.bool(True),
                                 firstValid = cms.vuint32(10) )


process.load("MagneticField.Engine.volumeBasedMagneticFieldFromDB_cfi")


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

