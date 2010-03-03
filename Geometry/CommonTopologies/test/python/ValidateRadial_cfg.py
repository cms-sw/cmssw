import FWCore.ParameterSet.Config as cms   

process = cms.Process('TEST')
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')

process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_3XY_V24::All'

process.test = cms.EDAnalyzer('ValidateRadial',
                              Epsilon = cms.double(3e-1),
                              FileName = cms.string("failureLimits.root"),
                              PrintOut = cms.bool(False))


process.p1 = cms.Path(process.test)

process.source = cms.Source( "EmptySource" )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
