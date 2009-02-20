import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorSummary.OnDemandMonitoring_cfi import *
#  SiStripMonitorAnalyser ####
SiStripAnalyser = cms.EDFilter("SiStripAnalyser",
    StaticUpdateFrequency    = cms.untracked.int32(1),
    GlobalStatusFilling      = cms.untracked.int32(1),
    TkMapCreationFrequency   = cms.untracked.int32(1),
    SummaryCreationFrequency = cms.untracked.int32(1),
    ShiftReportFrequency     = cms.untracked.int32(1),
    RawDataTag               = cms.untracked.InputTag("source"),                              
    TkmapParameters = cms.PSet(
        loadFedCabling = cms.untracked.bool(True),
        trackerdatPath = cms.untracked.string('CommonTools/TrackerMap/data/'),
        trackermaptxtPath = cms.untracked.string('DQM/Integration/test/TkMap/')
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
# Track Efficiency Client

from DQM.TrackingMonitor.TrackEfficiencyClient_cfi import *
TrackEffClient.FolderName = 'SiStrip/Tracks/Efficiencies'
TrackEffClient.AlgoName   = 'CKFTk'


