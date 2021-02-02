from __future__ import print_function
import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
#    lastRun = cms.untracked.uint32(1),
#    timetype = cms.string('runnumber'),
#    interval = cms.uint32(1),
    firstRun = cms.untracked.uint32(1)
)


#process.TFileService = cms.Service("TFileService",
#                                   fileName = cms.string("siPixelDynamicInefficiency_histo.root")
#                                   )


process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING')
    )
)

process.Timing = cms.Service("Timing")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run2_design']
#In case you of conditions missing, or if you want to test a specific GT
#process.GlobalTag.globaltag = 'PRE_DES72_V6'
print(process.GlobalTag.globaltag)

process.load("Configuration.StandardSequences.GeometryDB_cff")

process.QualityReader = cms.ESSource("PoolDBESSource",
#    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('')
    ),
    toGet = cms.VPSet(
		cms.PSet(
			record = cms.string("SiPixelDynamicInefficiencyRcd"),
			tag = cms.string("SiPixelDynamicInefficiency_v1")
		),
	),
    connect = cms.string('sqlite_file:siPixelDynamicInefficiency.db')
)

process.es_prefer_QualityReader = cms.ESPrefer("PoolDBESSource","QualityReader")

process.DynamicInefficiencyReader = cms.EDAnalyzer("SiPixelDynamicInefficiencyReader",
    printDebug = cms.untracked.bool(False),
    #Dynamic Inefficiency factors for 13TeV 25ns case
    thePixelColEfficiency_BPix1 = cms.double(1.0),
    thePixelColEfficiency_BPix2 = cms.double(1.0),
    thePixelColEfficiency_BPix3 = cms.double(1.0),
    thePixelColEfficiency_FPix1 = cms.double(0.999),
    thePixelColEfficiency_FPix2 = cms.double(0.999),
    thePixelEfficiency_BPix1 = cms.double(1.0),
    thePixelEfficiency_BPix2 = cms.double(1.0),
    thePixelEfficiency_BPix3 = cms.double(1.0),
    thePixelEfficiency_FPix1 = cms.double(0.999),
    thePixelEfficiency_FPix2 = cms.double(0.999),
    thePixelChipEfficiency_BPix1 = cms.double(1.0),
    thePixelChipEfficiency_BPix2 = cms.double(1.0),
    thePixelChipEfficiency_BPix3 = cms.double(1.0),
    thePixelChipEfficiency_FPix1 = cms.double(0.999),
    thePixelChipEfficiency_FPix2 = cms.double(0.999),
    theInstLumiScaleFactor = cms.double(364),
    theLadderEfficiency_BPix1 = cms.vdouble( [1]*20 ),
    theLadderEfficiency_BPix2 = cms.vdouble( [1]*32 ),
    theLadderEfficiency_BPix3 = cms.vdouble( [1]*44 ),
    theModuleEfficiency_BPix1 = cms.vdouble( 1, 1, 1, 1, ),
    theModuleEfficiency_BPix2 = cms.vdouble( 1, 1, 1, 1, ),
    theModuleEfficiency_BPix3 = cms.vdouble( 1, 1, 1, 1 ),
    thePUEfficiency_BPix1 = cms.vdouble( 1.00023, -3.18350e-06, 5.08503e-10, -6.79785e-14 ),
    thePUEfficiency_BPix2 = cms.vdouble( 9.99974e-01, -8.91313e-07, 5.29196e-12, -2.28725e-15 ),
    thePUEfficiency_BPix3 = cms.vdouble( 1.00005, -6.59249e-07, 2.75277e-11, -1.62683e-15 ),
    theInnerEfficiency_FPix1 = cms.double(1.0),
    theInnerEfficiency_FPix2 = cms.double(1.0),
    theOuterEfficiency_FPix1 = cms.double(1.0),
    theOuterEfficiency_FPix2 = cms.double(1.0),
    thePUEfficiency_FPix_Inner = cms.vdouble(
        1.0
        ),
    thePUEfficiency_FPix_Outer = cms.vdouble(
        1.0
        ),
  )
process.p = cms.Path(process.DynamicInefficiencyReader)
