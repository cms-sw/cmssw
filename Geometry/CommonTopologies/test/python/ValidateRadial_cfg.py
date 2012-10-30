import FWCore.ParameterSet.Config as cms   

process = cms.Process('TEST')
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')


process.test = cms.EDAnalyzer('ValidateRadial',
                              Epsilon = cms.double(3e-1),
                              FileName = cms.string("failureLimits.root"),
                              PrintOut = cms.bool(True),
                              PosOnly  = cms.bool(True)
                              )


process.p1 = cms.Path(process.test)

process.source = cms.Source( "EmptySource" )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
