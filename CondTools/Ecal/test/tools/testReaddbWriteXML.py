import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.maxEvents = cms.untracked.PSet(
     input = cms.untracked.int32(1)
)


process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        treshold = cms.untracked.string('INFO')
    )
)

##process.load("CondCore.DBCommon.CondDBSetup_cfi")
from CondCore.DBCommon.CondDBSetup_cfi import *

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(3),
    lastValue  = cms.uint64(4),
    interval = cms.uint64(1)
)


records = ["EcalADCToGeVConstant","EcalChannelStatus","EcalGainRatios","EcalTBWeights","EcalWeightXtalGroups","EcalIntercalibConstants","EcalIntercalibErrors","EcalIntercalibConstantsMC"]

source = []

for i in range (len(records)) :

        recordname = records[i] + 'Rcd'
        filename   = 'sqlite_file:test' + records[i] + '.db'

        source.append(cms.ESSource("PoolDBESSource",
                                   CondDBSetup,
                                   toGet = cms.VPSet(cms.PSet(
                                   record = cms.string(recordname),
                                   tag = cms.string('mytest'))), 
                                   connect=cms.string(filename)
                                   )
                      )

process.source0=source[0]
process.source1=source[1]
process.source2=source[2]
process.source3=source[3]
process.source4=source[4]
process.source5=source[5]
process.source6=source[6]

process.mytest = cms.EDAnalyzer("EcalTestConditionAnalyzer")                            


process.p = cms.Path(process.mytest)
