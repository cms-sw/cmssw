# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TMuonEmulation")
import optparse

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1)
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False))

process.source = cms.Source('PoolSource',
 fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/g/gkaratha/private/bmtf/gen_samples/Singlemu_oneoverpt_100k.root')
	                    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2000))

# PostLS1 geometry used
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2015_cff')
############################
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')


##ES Producer

####Event Setup Producer
process.load('L1Trigger.L1TTwinMux.fakeTwinMuxParams_cff')
process.esProd = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(
      cms.PSet(record = cms.string('L1TTwinMuxParamsRcd'),
               data = cms.vstring('L1TTwinMuxParams'))
                   ),
   verbose = cms.untracked.bool(True)
)

# process.fakeTwinMuxParams.verbose = cms.bool(True)
# process.fakeTwinMuxParams.useOnlyRPC = cms.bool(False)


###TwinMux Emulator
process.load('L1Trigger.L1TTwinMux.simTwinMuxDigis_cfi')


process.L1TMuonSeq = cms.Sequence(  
        process.esProd
       +process.simTwinMuxDigis
)

process.L1TMuonPath = cms.Path(process.L1TMuonSeq)

process.out = cms.OutputModule("PoolOutputModule", 

    outputCommands = cms.untracked.vstring(
        'drop *',
        #'keep *CSC*_*_*_*',
        'keep *RPC*_*_*_*',
        'keep *DT*_*_*_*',
        'keep *L1Mu*_*_*_*',
        'keep *_*Muon*_*_*',
        'keep *_*gen*_*_*',
        'keep *_*TwinMux*_*_*',
        'keep *_*Bmtf*_*_*',
        'keep GenEventInfoProduct_generator_*_*'),  
   fileName = cms.untracked.string("l1twinmux.root")
)

process.output_step = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.L1TMuonPath)
process.schedule.extend([process.output_step])
