import FWCore.ParameterSet.Config as cms

import DQMOffline.CalibCalo.MonitorAlCaEcalPhisym_cfi
import DQMOffline.CalibCalo.MonitorAlCaEcalPi0_cfi

ALCARECOEcalCalPhisymDQM = DQMOffline.CalibCalo.MonitorAlCaEcalPhisym_cfi.EcalPhiSymMon.clone()

ALCARECOEcalCalPi0CalibDQM =  DQMOffline.CalibCalo.MonitorAlCaEcalPi0_cfi.EcalPi0Mon.clone()
