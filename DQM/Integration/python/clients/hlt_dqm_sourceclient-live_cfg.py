import FWCore.ParameterSet.Config as cms

import sys
from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("DQM", Run3)

unitTest = False
if 'unitTest=True' in sys.argv:
	unitTest=True

if unitTest:
  process.load("DQM.Integration.config.unittestinputsource_cfi")
  from DQM.Integration.config.unittestinputsource_cfi import options
else:
  # for live online DQM in P5
  process.load("DQM.Integration.config.inputsource_cfi")
  from DQM.Integration.config.inputsource_cfi import options
# used in the old input source
#process.DQMEventStreamHttpReader.SelectHLTOutput = cms.untracked.string('hltOutputHLTDQM')

# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")
#from DQM.Integration.config.fileinputsource_cfi import options

#process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(100)
#)

process.load("DQM.Integration.config.environment_cfi")

process.dqmEnv.subSystemFolder = 'HLT'
process.dqmSaver.tag = 'HLT'
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'HLT'
process.dqmSaverPB.runNumber = options.runNumber

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.GlobalTrackingGeometryESProducer = cms.ESProducer( "GlobalTrackingGeometryESProducer" ) # for muon hlt dqm
process.HLTSiStripClusterChargeCutNone = cms.PSet(  value = cms.double( -1.0 ) )
process.ClusterShapeHitFilterESProducer = cms.ESProducer( "ClusterShapeHitFilterESProducer",
    ComponentName = cms.string( "ClusterShapeHitFilter" ),
    PixelShapeFileL1 = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_loose.par" ),
    clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
    PixelShapeFile = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_noL1.par" )
)
#SiStrip Local Reco
process.load("CalibTracker.SiStripCommon.TkDetMapESProducer_cfi")
#Track refitters
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

#---- for P5 (online) DB access
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Condition for lxplus: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run3_data', '')

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

process.load("DQM.HLTEvF.HLTObjectMonitor_Client_cff")

#process.p = cms.EndPath(process.hlts+process.hltsClient)

process.pp = cms.Path(process.dqmEnv+process.dqmSaver+process.dqmSaverPB)
#process.hltResults.plotAll = True


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
print("Final Source settings:", process.source)
