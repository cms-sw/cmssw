# $Id: qcdLowPtDQM_cfg.py,v 1.6 2009/11/29 10:21:39 loizides Exp $

import FWCore.ParameterSet.Config as cms

process = cms.Process("QcdLowPtDQM")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('DQM/Physics/qcdLowPtDQM_cfi')

process.GlobalTag.globaltag = 'STARTUP3X_V8D::All'

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring("ProductNotFound")
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.dump = cms.EDAnalyzer('EventContentAnalyzer')

process.source = cms.Source("PoolSource",
   duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
   fileNames = cms.untracked.vstring(
        'file:/putfilehere.root'
    )
)

##uncomment if you run on MC raw
#process.p1 = cms.Path(
#    process.myRecoSeq1
#)
#process.siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")

process.p2 = cms.Path(
    process.myRecoSeq2  *
    process.QcdLowPtDQM +
    process.dqmSaver
)

process.dqmSaver.workflow = cms.untracked.string('/Physics/QCDPhysics/LowPt')
