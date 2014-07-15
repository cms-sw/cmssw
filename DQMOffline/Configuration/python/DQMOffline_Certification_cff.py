import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMDaqInfo_cfi import *

from DQMOffline.Configuration.DQMOffline_DAQ_cff import *
from DQMOffline.Configuration.DQMOffline_DCS_cff import *
from DQMOffline.Configuration.DQMOffline_CRT_cff import *

DQMOffline_Certification = cms.Sequence(daq_dqmoffline*dcs_dqmoffline*crt_dqmoffline)

DQMCertCommon = cms.Sequence(siStripDaqInfo * sipixelDaqInfo * 
                             siStripDcsInfo * sipixelDcsInfo *
                             siStripCertificationInfo * sipixelCertification *
                             egammaDataCertificationTask *
                             dqmOfflineTriggerCert)

DQMCertMuon = cms.Sequence(dtDAQInfo * rpcDaqInfo * cscDaqInfo *
                           dtDCSByLumiSummary * rpcDCSSummary * cscDcsInfo *
                           dtCertificationSummary * rpcDataCertification * cscCertificationInfo)

DQMCertEcal = cms.Sequence(ecalDaqInfoTask * ecalPreshowerDaqInfoTask *
                           ecalDcsInfoTask * ecalPreshowerDcsInfoTask *
                           ecalCertification * ecalPreshowerDataCertificationTask)

DQMCertHcal = cms.Sequence(hcalDAQInfo *
                           hcalDCSInfo *
                           hcalDataCertification)                           

DQMCertJetMET = cms.Sequence(dataCertificationJetMETSequence)
