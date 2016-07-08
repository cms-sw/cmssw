# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TDQMOffline")
import os
import sys
import commands

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("DQMServices.Core.DQM_cfg")			#****** 20160707

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(50)
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False))
process.source = cms.Source('PoolSource',
# fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/g/gflouris/public/SingleMuPt6180_noanti_10k_eta1.root')
	fileNames = cms.untracked.vstring(
		'/store/data/Run2016B/SingleMuon/AOD/PromptReco-v2/000/273/730/00000/F617BDE8-5621-E611-97D7-02163E01426A.root',
	)
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10))
# PostLS1 geometry used
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2015_cff')

process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')		#****** 20160708 1



process.load('TrackingTools.GeomPropagators.SmartPropagatorAnyOpposite_cfi')		#****** 20160708 2d a
process.load('TrackingTools.GeomPropagators.SmartPropagatorAny_cfi')				#****** 20160708 2d b

#AXRHSTO
#process.TrackingComponentsRecord = cms.ESSource("EmptyESSource",					
#        recordName = cms.string(''),
#        iovIsRunNotTime = cms.bool(True),
#        firstValid = cms.vuint32(1)
#)	

####Event Setup Producer

#process.esProd = cms.EDAnalyzer("EventSetupRecordDataGetter",
#   toGet = cms.VPSet(
#      cms.PSet(record = cms.string('TrackingComponentsRecord'),
#			label = cms.untracked.string('PropagatorWithMaterial'),
#			data = cms.vstring('Propagator'))
#                   ),
#   verbose = cms.untracked.bool(True)
#)
																					#****** 20160708 2d c

############################
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag

#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')					#****** 20160708 2a

process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')					#****** 20160708 2b

#process.GlobalTag = GlobalTag(process.GlobalTag, '80X_dataRun2_Express_v10', '')		#****** 20160708 2c

####BMTF Emulator
print "c point "
process.load('DQMOffline.L1Trigger.L1TEfficiencyMuonsOffline_cff')
print "c point "
process.L1TMuonSeq = cms.Sequence(
#process.esProd+
process.l1tEfficiencyMuons_offline          
)

print "c point "
process.L1TMuonPath = cms.Path(
	process.L1TMuonSeq
)

print "c point "
process.out = cms.OutputModule("PoolOutputModule", 
   fileName = cms.untracked.string("dqm_l1tmuon.root")
)
print "c point "
process.output_step = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.L1TMuonPath)
process.schedule.extend([process.output_step])
