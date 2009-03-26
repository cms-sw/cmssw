import FWCore.ParameterSet.Config as cms

import DQMOffline.CalibCalo.MonitorAlCaHcalPhisym_cfi
import DQMOffline.CalibCalo.MonitorHcalDiJetsAlCaReco_cfi
import DQMOffline.CalibCalo.MonitorHcalIsoTrackAlCaReco_cfi
import DQMOffline.CalibCalo.MonitorHOAlCaRecoStream_cfi


ALCARECOHcalCalPhisymDQM =  DQMOffline.CalibCalo.MonitorAlCaHcalPhisym_cfi.HcalPhiSymMon.clone()

ALCARECOHcalCalDiJetsDQM =  DQMOffline.CalibCalo.MonitorHcalDiJetsAlCaReco_cfi.MonitorHcalDiJetsAlCaReco.clone()

ALCARECOHcalCalIsoTrackDQM =  DQMOffline.CalibCalo.MonitorHcalIsoTrackAlCaReco_cfi.MonitorHcalIsoTrackAlCaReco.clone()

ALCARECOHcalCalHODQM =  DQMOffline.CalibCalo.MonitorHOAlCaRecoStream_cfi.MonitorHOAlCaRecoStream.clone()

