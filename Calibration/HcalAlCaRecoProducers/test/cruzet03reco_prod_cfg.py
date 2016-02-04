# The following comments couldn't be translated into the new config version:

#  include "FWCore/MessageLogger/data/MessageLogger.cfi"

import FWCore.ParameterSet.Config as cms

process = cms.Process("HOCalibAnalyser")
process.load("MagneticField.Engine.uniformMagneticField_cfi")

process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHO_cff")

process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHO_Output_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    dropMetaData = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring('/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0005/BE657CCB-DE26-DD11-BBB8-000423D99614.root', 
        '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0005/BEE6FCEF-E026-DD11-8B35-001D09F29849.root')
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet( ## kill all messages in the log

            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet( ## but FwkJob category - those unlimitted

            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('FwkJob'),
    destinations = cms.untracked.vstring('cout')
)

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    ignoreTotal = cms.untracked.int32(1) ## default is one

)

process.o1 = cms.OutputModule("PoolOutputModule",
    process.OutALCARECOHcalCalHO,
    fileName = cms.untracked.string('hocalibevent.root')
)

process.pathALCARECOHcalCalHO = cms.Path(process.hoCalibProducer)
process.e = cms.EndPath(process.o1)
process.UniformMagneticFieldESProducer.ZFieldInTesla = 0.001
process.hoCalibProducer.muons = 'cosmicMuons'
process.hoCalibProducer.digiInput = False
process.hoCalibProducer.hbinfo = True
process.hoCalibProducer.hotime = False
process.hoCalibProducer.firstTS = 4
process.hoCalibProducer.lastTS = 7
process.hoCalibProducer.m_scale = 4.0
process.hoCalibProducer.sigma = 0.15
process.hoCalibProducer.hoInput = 'horeco'
process.hoCalibProducer.hbheInput = 'hbhereco'
process.hoCalibProducer.hltInput = 'TriggerResults::FU'
process.hoCalibProducer.l1Input = 'gtDigis'
process.hoCalibProducer.towerInput = 'towerMaker'


