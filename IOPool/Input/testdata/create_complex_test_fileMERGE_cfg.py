import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
      'file:testComplex0.root',
      'file:testComplex1.root',
      'file:testComplex2e.root',
      'file:testComplex3e.root',
      'file:testComplex4.root',
      'file:testComplex5.root',
      'file:testComplex6RL.root',
      'file:testComplex7R.root',
      'file:testComplex8RL.root',
      'file:testComplex9R.root',
      'file:testComplex10.root',
      'file:testComplex11RL.root'
   ),
   inputCommands = cms.untracked.vstring(
      'keep *', 
      'drop *_*_*_EXTRA'
   )
)

process.i = cms.EDProducer("IntProducer", ivalue = cms.int32(4))

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('complex_old_format_CMSSW_x_y_z.root')
)

process.p1 = cms.Path(process.i)

process.e = cms.EndPath(process.out)
