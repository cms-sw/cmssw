import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.run2_HECollapse_2018_cff import run2_HECollapse_2018
process = cms.Process("DQM", run2_HECollapse_2018)
# for live online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")
# used in the old input source
#process.DQMEventStreamHttpReader.SelectHLTOutput = cms.untracked.string('hltOutputHLTDQM')

# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")

#process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(100)
#)

process.load("DQM.Integration.config.environment_cfi")
process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/hlt_reference.root"

process.dqmEnv.subSystemFolder = 'HLT'
process.dqmSaver.tag = 'HLT'

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.GlobalTrackingGeometryESProducer = cms.ESProducer( "GlobalTrackingGeometryESProducer" ) # for muon hlt dqm
process.HLTSiStripClusterChargeCutNone = cms.PSet(  value = cms.double( -1.0 ) )
process.ClusterShapeHitFilterESProducer = cms.ESProducer( "ClusterShapeHitFilterESProducer",
    ComponentName = cms.string( "ClusterShapeHitFilter" ),
    PixelShapeFileL1 = cms.string( "RecoPixelVertexing/PixelLowPtUtilities/data/pixelShapePhase1_loose.par" ),
    clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
    PixelShapeFile = cms.string( "RecoPixelVertexing/PixelLowPtUtilities/data/pixelShapePhase1_noL1.par" )
)
#SiStrip Local Reco
process.load("CalibTracker.SiStripCommon.TkDetMap_cff")

#---- for P5 (online) DB access
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Condition for lxplus: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')

process.hltESSHcalSeverityLevel = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "HcalSeverityLevelComputerRcd" ),
    firstValid = cms.vuint32( 1 )
)
process.hcalRecAlgos = cms.ESProducer( "HcalRecAlgoESProducer",
  RecoveredRecHitBits = cms.vstring( 'TimingAddedBit',
    'TimingSubtractedBit' ),
  SeverityLevels = cms.VPSet(
    cms.PSet(  ChannelStatus = cms.vstring(  ),
      RecHitFlags = cms.vstring(  ),
      Level = cms.int32( 0 )
    ),
    cms.PSet(  ChannelStatus = cms.vstring( 'HcalCellCaloTowerProb' ),
      RecHitFlags = cms.vstring(  ),
      Level = cms.int32( 1 )
    ),
    cms.PSet(  ChannelStatus = cms.vstring( 'HcalCellExcludeFromHBHENoiseSummary' ),
      RecHitFlags = cms.vstring( 'HSCP_R1R2',
        'HSCP_FracLeader',
        'HSCP_OuterEnergy',
        'HSCP_ExpFit',
        'ADCSaturationBit',
        'HBHEIsolatedNoise',
        'AddedSimHcalNoise' ),
      Level = cms.int32( 5 )
    ),
    cms.PSet(  ChannelStatus = cms.vstring(  ),
      RecHitFlags = cms.vstring( 'HBHEHpdHitMultiplicity',
        'HBHEPulseShape',
        'HOBit',
        'HFInTimeWindow',
        'ZDCBit',
        'CalibrationBit',
        'TimingErrorBit',
        'HBHETriangleNoise',
        'HBHETS4TS5Noise' ),
      Level = cms.int32( 8 )
    ),
    cms.PSet(  ChannelStatus = cms.vstring(  ),
      RecHitFlags = cms.vstring( 'HFLongShort',
        'HFPET',
        'HFS8S1Ratio',
        'HFDigiTime' ),
      Level = cms.int32( 11 )
    ),
    cms.PSet(  ChannelStatus = cms.vstring( 'HcalCellCaloTowerMask' ),
      RecHitFlags = cms.vstring( 'HBHEFlatNoise',
        'HBHESpikeNoise' ),
      Level = cms.int32( 12 )
    ),
    cms.PSet(  ChannelStatus = cms.vstring( 'HcalCellHot' ),
      RecHitFlags = cms.vstring(  ),
      Level = cms.int32( 15 )
    ),
    cms.PSet(  ChannelStatus = cms.vstring( 'HcalCellOff',
  'HcalCellDead' ),
      RecHitFlags = cms.vstring(  ),
      Level = cms.int32( 20 )
    )
  ),
  DropChannelStatusBits = cms.vstring( 'HcalCellMask',
    'HcalCellOff',
    'HcalCellDead' )
)

### for pp collisions
process.load("DQM.HLTEvF.HLTObjectMonitor_cff")

### for Proton-Lead collisions only (2016 Proton-Lead Era)
#process.load("DQM.HLTEvF.HLTObjectMonitorProtonLead_cff")

# added for hlt scalars
process.load("DQM.TrigXMonitor.HLTSeedL1LogicScalers_cfi")
# added for hlt scalars
process.hltSeedL1Logic.l1GtData = cms.InputTag("l1GtUnpack","","DQM")
process.hltSeedL1Logic.dqmFolder =    cms.untracked.string("HLT/HLTSeedL1LogicScalers_SM")

process.load("DQM.HLTEvF.HLTObjectMonitor_Client_cff")

#process.p = cms.EndPath(process.hlts+process.hltsClient)
process.p = cms.EndPath(process.hltSeedL1Logic)

process.pp = cms.Path(process.dqmEnv+process.dqmSaver)
#process.hltResults.plotAll = True


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
