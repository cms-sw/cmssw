import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMDaqInfo_cfi import *

from DQMOffline.Configuration.DQMOffline_DAQ_cff import *
from DQMOffline.Configuration.DQMOffline_DCS_cff import *
from DQMOffline.Configuration.DQMOffline_CRT_cff import *

DQMOffline_Certification = cms.Sequence(daq_dqmoffline*dcs_dqmoffline*crt_dqmoffline)

DQMCertTrackerStrip = cms.Sequence(siStripDaqInfo * 
				   siStripDcsInfo * 
				   siStripCertificationInfo)

DQMCertTrackerPixel = cms.Sequence(sipixelDaqInfo *
				   sipixelDcsInfo *
				   sipixelCertification)

DQMCertTracking = cms.Sequence(trackingCertificationInfo) 

DQMCertEGamma = cms.Sequence(egammaDataCertificationTask)

DQMCertTrigger = cms.Sequence(dqmOfflineTriggerCert)

DQMCertMuon = cms.Sequence(dtDAQInfo * rpcDaqInfo * cscDaqInfo *
                           dtDCSByLumiSummary * rpcDCSSummary * cscDcsInfo *
                           dtCertificationSummary * rpcDataCertification * cscCertificationInfo)

DQMCertEcal = cms.Sequence(ecalDaqInfoTask * ecalPreshowerDaqInfoTask *
                           ecalDcsInfoTask * ecalPreshowerDcsInfoTask *
                           ecalCertification * ecalPreshowerDataCertificationTask)

DQMCertJetMET = cms.Sequence(dataCertificationJetMETSequence)

DQMCertCommon = cms.Sequence( DQMCertTrackerStrip *
			      DQMCertTrackerPixel *
			      DQMCertTracking *
			      DQMCertEGamma *
			      DQMCertTrigger)

from DQM.SiPixelPhase1Config.SiPixelPhase1OfflineDQM_harvesting_cff import *
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel

phase1Pixel.toReplaceWith(DQMCertCommon, DQMCertCommon.copyAndExclude([ # FIXME
    sipixelCertification # segfaults when included
]))


DQMCertCommonFakeHLT = cms.Sequence( DQMCertCommon )
DQMCertCommonFakeHLT.remove( dqmOfflineTriggerCert )
