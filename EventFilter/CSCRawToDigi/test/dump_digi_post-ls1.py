import FWCore.ParameterSet.Config as cms

process = cms.Process("ViewDigi")

# Dump of different types of digis produced 
# by CSC RawToDigi chane

process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")



"""Customise digi/reco geometry to use unganged ME1/a channels"""
process.CSCGeometryESModule.useGangedStripsInME1a = False
#process.idealForDigiCSCGeometry.useGangedStripsInME1a = False

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



# initialize MessageLogger and output report
process.MessageLogger = cms.Service("MessageLogger",
    #destinations = cms.untracked.vstring('detailedInfo'),
    destinations = cms.untracked.vstring('cout'),
    #destinations = cms.untracked.vstring('DDUStatusDump'),
    debugModules = cms.untracked.vstring('muonCSCDigis'),
    categories = cms.untracked.vstring("CSCDDUEventData|CSCRawToDigi",
    #'CSCDCCUnpacker|CSCRawToDigi', 'StatusDigis', 'StatusDigi', 'CSCRawToDigi', 'CSCDCCUnpacker', 'EventInfo',
    'badData'),
    #detailedInfo = cms.untracked.PSet(
    cout = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        #TRACE = cms.untracked.PSet(limit = cms.untracked.int32(0) ),
        noTimeStamps = cms.untracked.bool(False),
        #FwkReport = cms.untracked.PSet(
        #    reportEvery = cms.untracked.int32(1),
        #    limit = cms.untracked.int32(10000000)
        #),
        #CSCRawToDigi = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
        #StatusDigi = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
        #EventInfo = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),

        default = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
        #Root_NoDictionary = cms.untracked.PSet(limit = cms.untracked.int32(0)),
        DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
        #FwkJob = cms.untracked.PSet(limit = cms.untracked.int32(0)),
        #FwkSummary = cms.untracked.PSet(reportEvery = cms.untracked.int32(1), limit = cms.untracked.int32(10000000) ),
        threshold = cms.untracked.string('DEBUG')
    )
)

#process.GlobalTag.globaltag = 'GR_R_39X_V5::All' GR_R_42_V18
#process.GlobalTag.globaltag = 'GR_R_42_V2::All'
# process.GlobalTag.globaltag = 'GR_R_42_V18::All'
# process.GlobalTag.globaltag = 'GR_R_44_V10::All'

#process.GlobalTag.globaltag = 'GR_R_50_V11::All'
process.GlobalTag.globaltag = 'GR_E_V37::All'



#----------------------------
# Event Source
#-----------------------------
 
process.source = cms.Source("PoolSource",
    #debugFlag = cms.untracked.bool(True),
    #debugVebosity = cms.untracked.uint32(10),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
    fileNames = cms.untracked.vstring(
# == Run  ==
#	'rfio:/castor/cern.ch/cms/store/data/Commissioning2014/Cosmics/RAW/v1/000/220/744/00000/0C7ECA47-C4BE-E311-BDAB-02163E00E734.root'
#	'file:/tmp/barvic/0C7ECA47-C4BE-E311-BDAB-02163E00E734.root'
#	'file:/tmp/barvic/csc_ME11_Sim_DDU_FED.root'
	'file:/tmp/barvic/csc_00221766_Cosmic.root'
#	'file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/Cosmics__RAW__v1__147647__0030E63A-1CD4-DF11-A1C2-0030487CD6B4.root'
#	'rfio:/castor/cern.ch/cms/store/data/Commissioning12/TestEnablesEcalHcalDT/RAW/v1/000/188/685/28DE3EDE-2F74-E111-A078-003048F118AA.root'

#        'rfio:/castor/cern.ch/cms/store/data/Commissioning12/Cosmics/RAW/v1/000/186/207/82A46FF0-0662-E111-9494-BCAEC518FF7A.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning12/Cosmics/RAW/v1/000/186/207/8EA3A657-0862-E111-9AF5-BCAEC53296F5.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning12/Cosmics/RAW/v1/000/186/207/C2E57EBC-0962-E111-9498-5404A63886C6.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning12/Cosmics/RAW/v1/000/186/207/F8B29BDB-0B62-E111-B88A-BCAEC532972E.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning12/Cosmics/RAW/v1/000/186/207/BC417E4A-0D62-E111-81BB-BCAEC5329702.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning12/Cosmics/RAW/v1/000/186/207/8889F9B4-0E62-E111-A98A-BCAEC532971E.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning12/Cosmics/RAW/v1/000/186/207/A696DA15-1062-E111-8AAC-5404A63886A9.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning12/Cosmics/RAW/v1/000/186/207/9E72967F-1162-E111-8273-E0CB4E55365D.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning12/Cosmics/RAW/v1/000/186/207/7EFB8AEC-1962-E111-9592-BCAEC5364C4C.root'
# ==
    )
) 


process.muonCSCDigis.SuppressZeroLCT=cms.untracked.bool(True)
# Accounts the right product name 
# for the unpacker in MC
#process.muonCSCDigis.InputObjects = cms.InputTag("rawDataCollector") 
#process.muonCSCDigis.VisualFEDInspect = cms.untracked.bool(True)
#process.muonCSCDigis.VisualFEDShort = cms.untracked.bool(True)
process.muonCSCDigis.FormatedEventDump = cms.untracked.bool(False)

# Unpack status digi
process.muonCSCDigis.UnpackStatusDigis = cms.bool(False)


process.load("EventFilter.CSCRawToDigi.veiwDigi_cfi")

#process.dumpCSCdigi.WiresDigiDump = cms.untracked.bool(True)
#process.dumpCSCdigi.StripDigiDump = cms.untracked.bool(True)
process.dumpCSCdigi.ComparatorDigiDump = cms.untracked.bool(True)
#process.dumpCSCdigi.RpcDigiDump = cms.untracked.bool(False)
#process.dumpCSCdigi.AlctDigiDump = cms.untracked.bool(True)
process.dumpCSCdigi.ClctDigiDump = cms.untracked.bool(True)
#process.dumpCSCdigi.CorrClctDigiDump = cms.untracked.bool(True)


process.dumpCSCdigi.WiresDigiDump = cms.untracked.bool(False)
process.dumpCSCdigi.StripDigiDump = cms.untracked.bool(False)
#process.dumpCSCdigi.ComparatorDigiDump = cms.untracked.bool(False)
process.dumpCSCdigi.RpcDigiDump = cms.untracked.bool(False)
process.dumpCSCdigi.AlctDigiDump = cms.untracked.bool(False)
#process.dumpCSCdigi.ClctDigiDump = cms.untracked.bool(False)
process.dumpCSCdigi.CorrClctDigiDump = cms.untracked.bool(False)

process.dumpCSCdigi.StatusCFEBDump = cms.untracked.bool(False)
process.dumpCSCdigi.StatusDigiDump = cms.untracked.bool(False)
process.dumpCSCdigi.DDUStatus = cms.untracked.bool(False)
process.dumpCSCdigi.DCCStatus = cms.untracked.bool(False)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.out = cms.OutputModule("PoolOutputModule",
        dataset = cms.untracked.PSet(dataTier = cms.untracked.string('DIGI')),
   	#outputCommands = cms.untracked.vstring('keep *','drop FEDRawDataCollection_rawDataCollector_*_RAW'),
        fileName = cms.untracked.string('/tmp/barvic/digi_test.root'),
                               )

process.EventContent=cms.EDAnalyzer('EventContentAnalyzer')

process.p1 = cms.Path(process.muonCSCDigis)
#process.p2 = cms.Path(process.EventContent)
#process.p2 = cms.Path(process.csc2DRecHits)
#process.ps = cms.Path(process.cscSegments)
process.p3 = cms.Path(process.dumpCSCdigi)

process.p4 = cms.EndPath(process.out)
