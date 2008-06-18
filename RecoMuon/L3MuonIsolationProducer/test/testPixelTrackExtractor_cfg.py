# The following comments couldn't be translated into the new config version:

# Raw data

#------------ Message logger ------------------------------------

import FWCore.ParameterSet.Config as cms

process = cms.Process("MUONHLT")
#------------ HLT Muon Paths -------------------------------------
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

import RecoMuon.L3MuonIsolationProducer.L3MuonIsolationProducerPixTE_cfi
process.hltL3MuonIsolations = RecoMuon.L3MuonIsolationProducer.L3MuonIsolationProducerPixTE_cfi.L3MuonIsolationProducerPixTE.clone()
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:RelVal_Pure_Raw.root')
)

process.MessageLogger = cms.Service("MessageLogger")

process.Timing = cms.Service("Timing")

process.OUTPUT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('*')
    ),
    fileName = cms.untracked.string('test.root')
)

process.p1 = cms.Path(process.hltL3MuonIsolations)
process.outpath = cms.EndPath(process.OUTPUT)
process.PoolSource.fileNames = ['/store/relval/2008/6/6/RelVal-RelValWM-1212543891-STARTUP-2nd-02/0000/0C965E23-E733-DD11-9730-000423D94524.root', '/store/relval/2008/6/6/RelVal-RelValWM-1212543891-STARTUP-2nd-02/0000/18AB1548-E533-DD11-8103-000423D9863C.root', '/store/relval/2008/6/6/RelVal-RelValWM-1212543891-STARTUP-2nd-02/0000/22230C75-E533-DD11-B855-001617E30F4C.root']

