import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorSummary.OnDemandMonitoring_cfi import *
#  SiStripMonitorAnalyser ####
# for Online running
onlineAnalyser = cms.EDAnalyzer("SiStripAnalyser",
    StaticUpdateFrequency    = cms.untracked.int32(1),
    GlobalStatusFilling      = cms.untracked.int32(1),
    TkMapCreationFrequency   = cms.untracked.int32(1),
    SummaryCreationFrequency = cms.untracked.int32(1),
    ShiftReportFrequency     = cms.untracked.int32(1),
    SummaryConfigPath        = cms.untracked.string("DQM/SiStripMonitorClient/data/sistrip_monitorelement_config.xml"),
    PrintFaultyModuleList    = cms.untracked.bool(False),                                
    RawDataTag               = cms.untracked.InputTag("source"),                              
    TkmapParameters = cms.PSet(
        loadFedCabling = cms.untracked.bool(True),
        trackerdatPath = cms.untracked.string('CommonTools/TrackerMap/data/'),
        trackermaptxtPath = cms.untracked.string('DQM/SiStripMonitorClient/scripts/TkMap/')
    ),

# Parameters for On Demand Monitoring   
    MonitorSiStripPedestal      = OnDemandMonitoring.MonitorSiStripPedestal,
    MonitorSiStripNoise         = OnDemandMonitoring.MonitorSiStripNoise,
    MonitorSiStripQuality       = OnDemandMonitoring.MonitorSiStripQuality,
    MonitorSiStripApvGain       = OnDemandMonitoring.MonitorSiStripApvGain,
    MonitorSiStripLorentzAngle  = OnDemandMonitoring.MonitorSiStripLorentzAngle,

    MonitorSiStripCabling        = OnDemandMonitoring.MonitorSiStripCabling,
    MonitorSiStripLowThreshold   = OnDemandMonitoring.MonitorSiStripLowThreshold,
    MonitorSiStripHighThreshold  = OnDemandMonitoring.MonitorSiStripHighThreshold,

    FillConditions_PSet          = OnDemandMonitoring.FillConditions_PSet,     

    SiStripPedestalsDQM_PSet     = OnDemandMonitoring.SiStripPedestalsDQM_PSet,
    SiStripNoisesDQM_PSet        = OnDemandMonitoring.SiStripNoisesDQM_PSet,
    SiStripQualityDQM_PSet       = OnDemandMonitoring.SiStripQualityDQM_PSet,
    SiStripApvGainsDQM_PSet      = OnDemandMonitoring.SiStripApvGainsDQM_PSet,
    SiStripLorentzAngleDQM_PSet  = OnDemandMonitoring.SiStripLorentzAngleDQM_PSet,
    SiStripLowThresholdDQM_PSet  = OnDemandMonitoring.SiStripLowThresholdDQM_PSet,
    SiStripHighThresholdDQM_PSet = OnDemandMonitoring.SiStripHighThresholdDQM_PSet,
)

# for Offline running
offlineAnalyser = cms.EDAnalyzer("SiStripAnalyser",
    StaticUpdateFrequency    = cms.untracked.int32(-1),
    GlobalStatusFilling      = cms.untracked.int32(1),
    TkMapCreationFrequency   = cms.untracked.int32(-1),
    SummaryCreationFrequency = cms.untracked.int32(1),
    ShiftReportFrequency     = cms.untracked.int32(1),
    SummaryConfigPath        = cms.untracked.string("DQM/SiStripMonitorClient/data/sistrip_monitorelement_config.xml"),
    PrintFaultyModuleList    = cms.untracked.bool(False),
    RawDataTag               = cms.untracked.InputTag("source"),                               
    TkmapParameters = cms.PSet(
        loadFedCabling = cms.untracked.bool(True),
        trackerdatPath = cms.untracked.string('CommonTools/TrackerMap/data/'),
        trackermaptxtPath = cms.untracked.string('DQM/SiStripMonitorClient/scripts/TkMap/')
    ),
# Parameters for On Demand Monitoring                                  
    MonitorSiStripPedestal      = cms.untracked.bool(False),
    MonitorSiStripNoise         = cms.untracked.bool(False),
    MonitorSiStripQuality       = cms.untracked.bool(False),
    MonitorSiStripApvGain       = cms.untracked.bool(False),
    MonitorSiStripLorentzAngle  = cms.untracked.bool(False),

    MonitorSiStripCabling        = OnDemandMonitoring.MonitorSiStripCabling,
    MonitorSiStripLowThreshold   = OnDemandMonitoring.MonitorSiStripLowThreshold,
    MonitorSiStripHighThreshold  = OnDemandMonitoring.MonitorSiStripHighThreshold,

    FillConditions_PSet          = OnDemandMonitoring.FillConditions_PSet,     

    SiStripPedestalsDQM_PSet     = OnDemandMonitoring.SiStripPedestalsDQM_PSet,
    SiStripNoisesDQM_PSet        = OnDemandMonitoring.SiStripNoisesDQM_PSet,
    SiStripQualityDQM_PSet       = OnDemandMonitoring.SiStripQualityDQM_PSet,
    SiStripApvGainsDQM_PSet      = OnDemandMonitoring.SiStripApvGainsDQM_PSet,
    SiStripLorentzAngleDQM_PSet  = OnDemandMonitoring.SiStripLorentzAngleDQM_PSet,
    SiStripLowThresholdDQM_PSet  = OnDemandMonitoring.SiStripLowThresholdDQM_PSet,
    SiStripHighThresholdDQM_PSet = OnDemandMonitoring.SiStripHighThresholdDQM_PSet
)

# Sequence
SiStripOnlineDQMClient = cms.Sequence(onlineAnalyser)
SiStripOfflineDQMClient = cms.Sequence(offlineAnalyser)

# Services needed for TkHistoMap
TkDetMap = cms.Service("TkDetMap")

