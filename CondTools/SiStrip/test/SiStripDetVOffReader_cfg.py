import FWCore.ParameterSet.Config as cms

process = cms.Process("DetVOffReader")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring(''),
    files = cms.untracked.PSet(
        SiStripDetVOffReader = cms.untracked.PSet(

        )
    ),
    threshold = cms.untracked.string('INFO')
)

process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string('timestamp'),
                            firstValue = cms.uint64(6318323863869830144),
                            lastValue = cms.uint64(6318587115701303296),
                            interval = cms.uint64(1)
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

from CondCore.CondDB.CondDB_cfi import *
CondDBDetVOff = CondDB.clone(connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'))
process.dbInput = cms.ESSource("PoolDBESSource",
                               CondDBDetVOff,
                               toGet = cms.VPSet(cms.PSet(record = cms.string('SiStripDetVOffRcd'),
                                                          tag = cms.string('SiStripDetVOff_1hourDelay_v1_Validation') #choose your own favourite
                                                          )
                                                 )
                               )

process.fedcablingreader = cms.EDAnalyzer("SiStripDetVOffReader")

process.p1 = cms.Path(process.fedcablingreader)


