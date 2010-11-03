import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.BeamMonitor.BeamMonitor_cff")
process.load("DQM.BeamMonitor.BeamMonitorBx_cff")
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")

process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load("Configuration.StandardSequences.RawToDigi_Data_cff") ## For Data
#process.load("Configuration.StandardSequences.RawToDigi_cff") ## For MC
process.load("Configuration.StandardSequences.ReconstructionHeavyIons_cff") ## HI sequences
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['dqmBeamMonitor']
process.MessageLogger.categories = ['BeamMonitor']
process.MessageLogger.cerr.threshold = "INFO"

#--------------------------
# Input files (heavy-ion MC)
#--------------------------

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
        # Virgin raw
        '/store/data/Run2010B/HeavyIonTest/RAW/v1/000/146/421/FAD966AE-64C6-DF11-9649-0019B9F72BFF.root',
        '/store/data/Run2010B/HeavyIonTest/RAW/v1/000/146/421/F8C1BB4A-65C6-DF11-BFC4-001617C3B6FE.root',
        '/store/data/Run2010B/HeavyIonTest/RAW/v1/000/146/421/F6C87AAF-69C6-DF11-A9D0-001D09F23944.root',
        '/store/data/Run2010B/HeavyIonTest/RAW/v1/000/146/421/F473DF55-65C6-DF11-B04C-001D09F29538.root',
        '/store/data/Run2010B/HeavyIonTest/RAW/v1/000/146/421/F243463A-6AC6-DF11-A714-0030487C778E.root',
        '/store/data/Run2010B/HeavyIonTest/RAW/v1/000/146/421/EA5F2D9C-67C6-DF11-9FE8-001D09F241F0.root',
        '/store/data/Run2010B/HeavyIonTest/RAW/v1/000/146/421/E68F7D2E-68C6-DF11-A4A6-0030487D05B0.root',
        '/store/data/Run2010B/HeavyIonTest/RAW/v1/000/146/421/E4804C63-66C6-DF11-8F0A-003048D2BC42.root',
        '/store/data/Run2010B/HeavyIonTest/RAW/v1/000/146/421/E44578CC-66C6-DF11-B041-0030487CD704.root',
        '/store/data/Run2010B/HeavyIonTest/RAW/v1/000/146/421/E2062097-62C6-DF11-B596-003048D2C020.root',
        '/store/data/Run2010B/HeavyIonTest/RAW/v1/000/146/421/DC97531E-68C6-DF11-9ACE-0030487C8CB8.root',
        '/store/data/Run2010B/HeavyIonTest/RAW/v1/000/146/421/DA0796A8-69C6-DF11-88F3-003048D2BF1C.root',
    ## 36X heavy-ion MC sample with 'unrealistic' centered vertex smearing
    #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0005/54A93027-3A71-DF11-8B43-00E081300BDA.root',
    #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0005/409BE308-BD71-DF11-8494-00188B7ABC14.root',
    #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0005/0662E9D1-D070-DF11-8329-001E68865F6D.root',
    #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/FC6349FD-7F70-DF11-AE96-00188B7ACD5D.root',
    #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/FA3B59E5-7F70-DF11-BC81-001E68862AE3.root',
    #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/EACFBC1A-8170-DF11-996D-001E68865F71.root',
    #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/C2F8E48F-9770-DF11-97F0-00E0813000C2.root',
    #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/B69BA9EF-8170-DF11-9D35-001E68659F36.root',
    #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/B48E08CF-7F70-DF11-BDEE-00E081300BDA.root'

    ## 391 heavy-ion MC RelVal sample with 'realistic' vertex smearing
    #'/store/relval/CMSSW_3_9_1/RelValHydjetQ_MinBias_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0064/ECDC5F2E-88E4-DF11-AA34-002618943849.root',
    #'/store/relval/CMSSW_3_9_1/RelValHydjetQ_MinBias_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0064/8EB20D5F-69E4-DF11-8FF4-002618943907.root',
    #'/store/relval/CMSSW_3_9_1/RelValHydjetQ_MinBias_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0064/56925F2C-7AE4-DF11-A79F-00261894394D.root',
    #'/store/relval/CMSSW_3_9_1/RelValHydjetQ_MinBias_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0064/3AF6201E-70E4-DF11-B735-002354EF3BD0.root',
    #'/store/relval/CMSSW_3_9_1/RelValHydjetQ_MinBias_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0064/2EBABA5F-69E4-DF11-8022-003048679166.root',
    #'/store/relval/CMSSW_3_9_1/RelValHydjetQ_MinBias_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0062/FA36B575-42E4-DF11-913D-001A92971B3A.root',
    #'/store/relval/CMSSW_3_9_1/RelValHydjetQ_MinBias_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0062/D28E846B-3EE4-DF11-ABB8-0018F3D095EA.root',
    #'/store/relval/CMSSW_3_9_1/RelValHydjetQ_MinBias_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0062/A46F478D-4CE4-DF11-A331-002618943858.root',
    #'/store/relval/CMSSW_3_9_1/RelValHydjetQ_MinBias_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0062/8A8BBC03-4BE4-DF11-803B-0026189438A0.root',
    #'/store/relval/CMSSW_3_9_1/RelValHydjetQ_MinBias_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0062/7ADC78E4-3FE4-DF11-B51A-001A92971AD8.root',
    #'/store/relval/CMSSW_3_9_1/RelValHydjetQ_MinBias_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0062/5CC3D523-53E4-DF11-8525-003048679008.root',
    #'/store/relval/CMSSW_3_9_1/RelValHydjetQ_MinBias_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0062/483BB355-3AE4-DF11-AF87-0026189438EB.root',
    #'/store/relval/CMSSW_3_9_1/RelValHydjetQ_MinBias_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0062/18B7F1F5-50E4-DF11-B91C-002354EF3BE0.root'
    )
)

#--------------------------
# Filters
#--------------------------

# HLT Filter
process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter.SelectedTriggerType = 1

# L1 Trigger Bit Selection (bit 40 and 41 for BSC trigger)
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('40 OR 41')

# Select events based on the pixel cluster multiplicity
import  HLTrigger.special.hltPixelActivityFilter_cfi
process.multFilter = HLTrigger.special.hltPixelActivityFilter_cfi.hltPixelActivityFilter.clone(
    inputTag  = cms.InputTag('siPixelClusters'),
    minClusters = cms.uint32(150),
    maxClusters = cms.uint32(50000)
    )

process.phystrigger = cms.Sequence(
    process.hltTriggerTypeFilter*
    process.gtDigis*
    process.hltLevel1GTSeed)

process.filter_step = cms.Sequence(
    process.siPixelDigis*
    process.siPixelClusters*
    process.multFilter
)

#--------------------------
# Reconstruction
#--------------------------

# test with wrong offline beamspot
#process.load("RecoVertex.BeamSpotProducer.BeamSpot2000umOffset_cff")

# make pixel vertexing less sensitive to incorrect beamspot
process.hiPixel3ProtoTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.2
process.hiPixel3ProtoTracks.RegionFactoryPSet.RegionPSet.fixedError = 0.5
process.hiSelectedProtoTracks.maxD0Significance = 100
process.hiPixelAdaptiveVertex.TkFilterParameters.maxD0Significance = 100
process.hiPixelAdaptiveVertex.useBeamConstraint = False
process.hiPixelAdaptiveVertex.PVSelParameters.maxDistanceToBeam = 1.0

process.RecoForDQM = cms.Sequence(
    process.siPixelDigis*
    process.siPixelClusters*
    process.siPixelRecHits*
    process.offlineBeamSpot*
    process.hiPixelVertices*
    process.hiPixel3PrimTracks
 )

#--------------------------
# Beam Monitor
#--------------------------

# use HI pixel tracking and vertexing
process.dqmBeamMonitor.BeamFitter.TrackCollection = cms.untracked.InputTag('hiPixel3PrimTracks')
process.dqmBeamMonitor.primaryVertex = cms.untracked.InputTag('hiSelectedVertex')
process.dqmBeamMonitor.PVFitter.VertexCollection = cms.untracked.InputTag('hiSelectedVertex')

# Beamspot DQM options
process.dqmBeamMonitor.OnlineMode = False                  ## in MC the LS are not ordered??
#process.dqmBeamMonitor.OnlineMode = True                  ## in MC the LS are not ordered??
process.dqmBeamMonitor.BeamFitter.MinimumTotalLayers = 3   ## using pixel triplets
process.dqmBeamMonitor.resetEveryNLumi = 10                ## default is 20
process.dqmBeamMonitor.resetPVEveryNLumi = 5               ## default is 5
#process.dqmBeamMonitor.Debug = True
#process.dqmBeamMonitor.BeamFitter.Debug = True
#process.dqmBeamMonitor.BeamFitter.WriteAscii = True
#process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
#process.dqmBeamMonitor.BeamFitter.SaveFitResults = True
#process.dqmBeamMonitor.BeamFitter.SavePVVertices = True

process.load("FWCore.Modules.preScaler_cfi")
process.preScaler.prescaleFactor = 300

process.hi = cms.Path(
    process.preScaler*
    process.hltTriggerTypeFilter*
    process.RawToDigi*
    process.filter_step*
    process.RecoForDQM*
    process.dqmBeamMonitor+
    process.dqmEnv+
    process.dqmSaver)

#--------------------------
# DQM output
#--------------------------

# Setup DQM store parameters.
process.DQMStore.verbose = 0
#process.DQM.collectorHost = 'cmslpc17.fnal.gov'
process.DQM.collectorHost = 'localhost'
process.DQM.collectorPort = 9190
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_39Y_V3::All' ## 'realistic' offline beamspot, unlike 36x sample

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

process.schedule = cms.Schedule(process.hi)

