import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = cms.string('sqlite_file:testExample.db')
#process.CondDBCommon.connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_30X_HCAL')
process.CondDBCommon.connect = cms.string('oracle://cms_orcon_prod/CMS_COND_31X_HCAL')
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/nfshome0/popcondev/conddb')

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.es_ascii = cms.ESSource("CastorTextCalibrations",
    input = cms.VPSet(cms.PSet(
        object = cms.string('GainWidths'),
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_gain_widths.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_gainwidths_Run124009-132957.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_gainwidths_Run132958-138559.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_gainwidths_Run138560-141955.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_gainwidths_Run141956-150430.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_gainwidths_Run150431-150589.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_gainwidths_Run150590-158416.txt')
        file = cms.FileInPath('CondFormats/CastorObjects/data/castor_gainwidths_Run158417.txt')
    ))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
#    logconnect= cms.untracked.string('sqlite_file:log.db'),
    logconnect= cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('CastorGainWidthsRcd'),
        #tag = cms.string('CastorGainWidths_v2.1_hlt')
        tag = cms.string('castor_gainwidths_v1.0_hlt')
         ))
)

process.mytest = cms.EDAnalyzer("CastorGainWidthsPopConAnalyzer",
    record = cms.string('CastorGainWidthsRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
#    firstSince=cms.untracked.double(300) 
    #IOVRun=cms.untracked.uint32(1)
    #IOVRun=cms.untracked.uint32(124009)
    #IOVRun=cms.untracked.uint32(132958)
    #IOVRun=cms.untracked.uint32(138560)
    #IOVRun=cms.untracked.uint32(141956)
    #IOVRun=cms.untracked.uint32(150431)
    #IOVRun=cms.untracked.uint32(150590)
    #IOVRun=cms.untracked.uint32(158417)
    IOVRun=cms.untracked.uint32(164799)
    )
)

process.p = cms.Path(process.mytest)
