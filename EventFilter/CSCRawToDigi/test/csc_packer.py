# Test CSCValidation running on raw relval file - Tim Cox - 09.09.2013 
# For raw relval in 700, with useDigis ON, and request only 100 events.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Configuration/StandardSequences/Geometry_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')


"""Customise digi/reco geometry to use unganged ME1/a channels"""
process.CSCGeometryESModule.useGangedStripsInME1a = False
process.idealForDigiCSCGeometry.useGangedStripsInME1a = False

"""Settings for the upgrade raw vs offline condition channel translation"""
process.CSCIndexerESProducer.AlgoName=cms.string("CSCIndexerPostls1")
process.CSCChannelMapperESProducer.AlgoName=cms.string("CSCChannelMapperPostls1")

process.csc2DRecHits.readBadChannels = cms.bool(False)
process.csc2DRecHits.CSCUseGasGainCorrection = cms.bool(False)


""" Pick up upgrade condions data directly from DB tags using ESPrefer's.
Might be useful when dealing with a global tag that doesn't include
'unganged' CSC conditions.
"""
myconds = [
        ('CSCDBGainsRcd', 'CSCDBGains_ungangedME11A_mc'),
        ('CSCDBNoiseMatrixRcd', 'CSCDBNoiseMatrix_ungangedME11A_mc'),
        ('CSCDBCrosstalkRcd', 'CSCDBCrosstalk_ungangedME11A_mc'),
        ('CSCDBPedestalsRcd', 'CSCDBPedestals_ungangedME11A_mc'),
        ('CSCDBGasGainCorrectionRcd', 'CSCDBGasGainCorrection_ungangedME11A_mc'),
        ('CSCDBChipSpeedCorrectionRcd', 'CSCDBChipSpeedCorrection_ungangedME11A_mc')
]

from CalibMuon.Configuration.getCSCConditions_frontier_cff import cscConditions
for (classname, tag) in myconds:
      print classname, tag
      sourcename = 'unganged_' + classname
      process.__setattr__(sourcename, cscConditions.clone())
      process.__getattribute__(sourcename).toGet = cms.VPSet( cms.PSet( record = cms.string(classname), tag = cms.string(tag)) )
      process.__getattribute__(sourcename).connect = cms.string('frontier://FrontierProd/CMS_COND_CSC_000')
      process.__setattr__('esp_' + classname, cms.ESPrefer("PoolDBESSource", sourcename) )
    
del cscConditions



# As of 09.09.2013 only a temp gloabl tag exists for 620/700
#process.GlobalTag.globaltag = 'PRE_62_V8::All'
process.GlobalTag.globaltag = 'GR_E_V37::All'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.options = cms.untracked.PSet( SkipEvent = cms.untracked.vstring('ProductNotFound') )
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#	'file:/tmp/barvic/csc_00221766_Cosmic.root'
	'file:/tmp/barvic/digi_test.root'
#	'rfio:/castor/cern.ch/cms/store/data/Commissioning2014/Cosmics/RAW/v1/000/220/744/00000/0C7ECA47-C4BE-E311-BDAB-02163E00E734.root'
#                  '/store/relval/CMSSW_7_0_0_pre3/SingleMu/RAW/PRE_P62_V8_RelVal_mu2011B-v1/00000/127160CD-7215-E311-91F3-003048D15E14.root'
)
)

process.cscValidation = cms.EDAnalyzer("CSCValidation",
    rootFileName = cms.untracked.string('cscv_RAW.root'),
    isSimulation = cms.untracked.bool(False),
    writeTreeToFile = cms.untracked.bool(True),
    useDigis = cms.untracked.bool(True),
    detailedAnalysis = cms.untracked.bool(False),
    useTriggerFilter = cms.untracked.bool(False),
    useQualityFilter = cms.untracked.bool(False),
    makeStandalonePlots = cms.untracked.bool(False),
    makeTimeMonitorPlots = cms.untracked.bool(True),
    rawDataTag = cms.InputTag("rawDataCollector"),
    alctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCALCTDigi"),
    clctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCLCTDigi"),
    corrlctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCorrelatedLCTDigi"),
    stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    compDigiTag = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi"),
    cscRecHitTag = cms.InputTag("csc2DRecHits"),
    cscSegTag = cms.InputTag("cscSegments"),
    saMuonTag = cms.InputTag("standAloneMuons"),
    l1aTag = cms.InputTag("gtDigis"),
    hltTag = cms.InputTag("TriggerResults::HLT"),
    makeHLTPlots = cms.untracked.bool(True),
    simHitTag = cms.InputTag("g4SimHits", "MuonCSCHits")
)

process.cscpacker = cms.EDProducer("CSCDigiToRawModule",
    wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    comparatorDigiTag = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi"),
    alctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCALCTDigi"),
    clctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCLCTDigi"),
    preTriggerTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    correlatedLCTDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCorrelatedLCTDigi"),
    alctWindowMin = cms.int32(-999),
    alctWindowMax = cms.int32(3),
    clctWindowMin = cms.int32(-999),
    clctWindowMax = cms.int32(3),
    preTriggerWindowMin = cms.int32(-3),
    preTriggerWindowMax = cms.int32(1),
 
)

process.cscpacker.usePreTriggers = cms.bool(False)
process.cscpacker.useFormatVersion = cms.uint32(2013)

process.load("EventFilter.CSCRawToDigi.veiwDigi_cfi")

#process.dumpCSCdigi.WiresDigiDump = cms.untracked.bool(True)
#process.dumpCSCdigi.StripDigiDump = cms.untracked.bool(True)
#process.dumpCSCdigi.ComparatorDigiDump = cms.untracked.bool(True)
#process.dumpCSCdigi.RpcDigiDump = cms.untracked.bool(False)
#process.dumpCSCdigi.AlctDigiDump = cms.untracked.bool(True)
#process.dumpCSCdigi.ClctDigiDump = cms.untracked.bool(True)
#process.dumpCSCdigi.CorrClctDigiDump = cms.untracked.bool(True)

process.dumpCSCdigi.WiresDigiDump = cms.untracked.bool(False)
process.dumpCSCdigi.StripDigiDump = cms.untracked.bool(False)
process.dumpCSCdigi.ComparatorDigiDump = cms.untracked.bool(False)
process.dumpCSCdigi.RpcDigiDump = cms.untracked.bool(False)
process.dumpCSCdigi.AlctDigiDump = cms.untracked.bool(False)
process.dumpCSCdigi.ClctDigiDump = cms.untracked.bool(False)
process.dumpCSCdigi.CorrClctDigiDump = cms.untracked.bool(False)

process.dumpCSCdigi.StatusCFEBDump = cms.untracked.bool(False)
process.dumpCSCdigi.StatusDigiDump = cms.untracked.bool(False)
process.dumpCSCdigi.DDUStatus = cms.untracked.bool(False)
process.dumpCSCdigi.DCCStatus = cms.untracked.bool(False)

process.out = cms.OutputModule("PoolOutputModule",
                      dataset = cms.untracked.PSet(dataTier = cms.untracked.string('DIGI')),
		outputCommands = cms.untracked.vstring('keep *','drop FEDRawDataCollection_rawDataCollector_*_LHC'),
                               fileName = cms.untracked.string('/tmp/barvic/digi_packer_test.root'),
                               )

# From RECO
# process.p = cms.Path(process.cscValidation)
# From RAW
# process.p = cms.Path(process.muonCSCDigis * process.csc2DRecHits * process.cscSegments * process.cscValidation)
#process.p = cms.Path(process.muonCSCDigis * process.csc2DRecHits * process.cscSegments)
#process.p = cms.Path( process.cscpacker * process.muonCSCDigis * process.dumpCSCdigi)
process.p = cms.Path( process.cscpacker )

process.pend = cms.EndPath (process.out)

# Path and EndPath definitions
##process.raw2digi_step = cms.Path(process.RawToDigi)
##process.reconstruction_step = cms.Path(process.reconstruction)
##process.cscvalidation_step = cms.Path(process.cscValidation)
#process.endjob_step = cms.EndPath(process.out * process.endOfProcess)

# Schedule definition
##process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.cscvalidation_step,process.endjob_step)
#process.schedule = cms.Schedule(process.p,process.endjob_step)

