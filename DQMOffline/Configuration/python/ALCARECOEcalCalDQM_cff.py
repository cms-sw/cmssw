import FWCore.ParameterSet.Config as cms

#import DQMOffline.CalibCalo.MonitorAlCaEcalPhisym_cfi
import DQMOffline.CalibCalo.MonitorAlCaEcalPi0_cfi
import DQMOffline.CalibCalo.MonitorAlCaEcalEleCalib_cfi

#ALCARECOEcalCalPhisymDQM = DQMOffline.CalibCalo.MonitorAlCaEcalPhisym_cfi.EcalPhiSymMonDQM.clone()

ALCARECOEcalCalPi0CalibDQM =  DQMOffline.CalibCalo.MonitorAlCaEcalPi0_cfi.EcalPi0MonDQM.clone()
ALCARECOEcalCalEtaCalibDQM =  DQMOffline.CalibCalo.MonitorAlCaEcalPi0_cfi.EcalPi0MonDQM.clone()
ALCARECOEcalCalElectronCalibDQM =  DQMOffline.CalibCalo.MonitorAlCaEcalEleCalib_cfi.EcalEleCalibMon.clone()

