import FWCore.ParameterSet.Config as cms

process = cms.Process("GEMLocalRECO")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))

#process.Timing = cms.Service("Timing")
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

####################################
##### Choose your GEM Geometry #####
####################################
#  6 eta partitions :: command line option :: --geometry Geometry/GEMGeometry/cmsExtendedGeometryPostLS1plusGEMXML_cfi
#  8 eta partitions :: command line option :: --geometry Geometry/GEMGeometry/cmsExtendedGeometryPostLS1plusGEMr08v01XML_cfi.py
# 10 eta partitions :: command line option :: --geometry Geometry/GEMGeometry/cmsExtendedGeometryPostLS1plusGEMr10v01XML_cfi.py
###### This results in following lines
###  6 eta partitions
#process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMXML_cfi')
###  8 eta partitions
#process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMr08v01XML_cfi')
### 10 eta partitions
# process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMr10v01XML_cfi')          

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDev_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('RecoLocalMuon.GEMRecHit.gemRecHits_cfi')
#process.load('RecoLocalMuon.GEMRecHit.me0RecHits_cfi')
process.load('RecoLocalMuon.GEMRecHit.me0LocalReco_cff')

### Try to do RecoLocalMuon on all muon detectors ###
#####################################################
from RecoLocalMuon.Configuration.RecoLocalMuon_cff import *
process.localreco = cms.Sequence(muonlocalreco)

#????
#process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi')
#process.load('Geometry.CommonTopologies.globalTrackingGeometry_cfi')
#process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')
#process.load('Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometryDB_cff')

#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = 'auto:upgrade2019'
#process.GlobalTag.globaltag = 'DES19_62_V7::All'
#process.GlobalTag.globaltag = 'POSTLS161_V12::All'
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')
from Configuration.AlCa.GlobalTag import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Fix DT and CSC Alignment #
############################
from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixDTAlignmentConditions
process = fixDTAlignmentConditions(process)
from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixCSCAlignmentConditions
process = fixCSCAlignmentConditions(process)

# Skip Digi2Raw and Raw2Digi steps for Al Muon detectors #
##########################################################
process.gemRecHits.gemDigiLabel = cms.InputTag("simMuonGEMDigis","","GEMDIGI")
process.rpcRecHits.rpcDigiLabel = cms.InputTag('simMuonRPCDigis')
process.csc2DRecHits.wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
process.csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")
process.dt1DRecHits.dtDigiLabel = cms.InputTag("simMuonDTDigis")
process.dt1DCosmicRecHits.dtDigiLabel = cms.InputTag("simMuonDTDigis")

# Explicit configuration of CSC for postls1 = run2 #
####################################################
process.load("CalibMuon.CSCCalibration.CSCChannelMapper_cfi")
process.load("CalibMuon.CSCCalibration.CSCIndexer_cfi")
process.CSCIndexerESProducer.AlgoName = cms.string("CSCIndexerPostls1")
process.CSCChannelMapperESProducer.AlgoName = cms.string("CSCChannelMapperPostls1")
process.CSCGeometryESModule.useGangedStripsInME1a = False
process.csc2DRecHits.readBadChannels = cms.bool(False)
process.csc2DRecHits.CSCUseGasGainCorrections = cms.bool(False)
# process.csc2DRecHits.wireDigiTag  = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
# process.csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")

process.gemRecHits = cms.EDProducer("GEMRecHitProducer",
    recAlgoConfig = cms.PSet(),
    recAlgo = cms.string('GEMRecHitStandardAlgo'),
    gemDigiLabel = cms.InputTag("simMuonGEMDigis"),
    # maskSource = cms.string('File'),
    # maskvecfile = cms.FileInPath('RecoLocalMuon/GEMRecHit/data/GEMMaskVec.dat'),
    # deadSource = cms.string('File'),
    # deadvecfile = cms.FileInPath('RecoLocalMuon/GEMRecHit/data/GEMDeadVec.dat')
)

### Input and Output Files
##########################
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:out_digi.root'
        # 'file:out_digi_100GeV_1000evts.root'
        # 'file:out_digi_1To100GeV_1000evts.root'
    )
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( 
        'file:out_local_reco_test.root'
        # 'file:out_local_reco_100GeV_1000evts.root'
        # 'file:out_local_reco_1To100GeV_1000evts.root'
    ),
    outputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('rechit_step')
    )
)

### Paths and Schedules
#######################
process.rechit_step  = cms.Path(process.localreco+process.gemRecHits+process.me0LocalReco)
#process.rechit_step  = cms.Path(process.localreco+process.gemRecHits+process.me0RecHits)
process.endjob_step  = cms.Path(process.endOfProcess)
process.out_step     = cms.EndPath(process.output)


process.schedule = cms.Schedule(
    process.rechit_step,
    process.endjob_step,
    process.out_step
)

