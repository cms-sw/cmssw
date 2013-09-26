import FWCore.ParameterSet.Config as cms
import os

user = os.environ['USER']

process = cms.Process("TEST")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(threshold = cms.untracked.string( "INFO" ),
                              FwkReport = cms.untracked.PSet(reportEvery = cms.untracked.int32(3000)),
                              ),
    destinations = cms.untracked.vstring( 'cout' ),
    categories = cms.untracked.vstring('FwkReport') 
    )

process.source = cms.Source("FedRawDataInputSource",
rootDirectory = cms.untracked.string("/data/appliance/"+user+"/BU-000")
    )

process.a = cms.EDAnalyzer("ExceptionGenerator",
    defaultAction = cms.untracked.int32(0),
    defaultQualifier = cms.untracked.int32(10)
    )

process.psa = cms.EDFilter("HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
    )

process.aa = cms.Path(process.psa*process.a)

process.streamA = cms.OutputModule("Stream",
    baseDir = cms.untracked.string("/data/appliance/"+user+"/sm"),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'aa' ) )
    )

process.ep = cms.EndPath(process.streamA)

