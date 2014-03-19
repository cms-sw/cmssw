import FWCore.ParameterSet.Config as cms

siStripCertificationInfo = cms.EDAnalyzer("SiStripCertificationInfo")
from DQM.SiPixelCommon.SiPixelOfflineDQM_client_cff import *
from DQM.EcalBarrelMonitorClient.EcalCertification_cfi import *
from DQM.HcalMonitorClient.HcalDataCertification_cfi import *
from DQM.DTMonitorClient.dtDQMOfflineCertification_cff import *
from DQM.RPCMonitorClient.RPCDataCertification_cfi import *
from DQM.CSCMonitorModule.csc_certification_info_cfi import *
from DQM.EcalPreshowerMonitorModule.ESDataCertificationTask_cfi import *

from DQM.TrackingMonitorClient.TrackingCertification_cfi import *
from DQMOffline.JetMET.dataCertificationJetMET_cff import *
from DQMOffline.EGamma.egammaDataCertification_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_Cert_cff import *

crt_dqmoffline = cms.Sequence( siStripCertificationInfo *
                               sipixelCertification *
                               ecalCertification *
                               hcalDataCertification *
                               dtCertificationSummary *
                               rpcDataCertification *
                               cscCertificationInfo *
                               ecalPreshowerDataCertificationTask *
                               trackingCertificationInfo *
                               dataCertificationJetMETSequence *
                               egammaDataCertificationTask *
                               dqmOfflineTriggerCert )

