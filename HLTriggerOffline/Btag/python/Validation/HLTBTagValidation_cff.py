import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Btag.Validation.HLTBTagPerformanceAnalyzer_cff import *

ValidationMCBTagIP3DFastPV  = bTagValidation.clone()
#ValidationMCBTagIP3DFastPV.HLTPathName    = cms.string('HLT_DiJet40Eta2p6_BTagIP3DFastPV')
#ValidationMCBTagIP3DFastPV.HLTPathName    = cms.string('HLT_BTagMu_DiJet70_Mu5_v15')
#ValidationMCBTagIP3DFastPV.HLTPathName    = cms.string('HLT_DiCentralJet36_BTagIP3DLoose')




#ValidationMCBTagIP3DFastPV.IsData         = cms.bool(False)
#ValidationMCBTagIP3DFastPV.BTagAlgorithm  = cms.string('TCHE')
#ValidationMCBTagIP3DFastPV.L25IPTagInfo   = cms.InputTag('NULL')
#ValidationMCBTagIP3DFastPV.L25JetTag      = cms.InputTag('NULL')
ValidationMCBTagIP3DFastPV.TrackIPTagInfo  = cms.InputTag('NULL')
ValidationMCBTagIP3DFastPV.OfflineJetTag   = cms.InputTag('NULL')


HLTBTagValSeq = cms.Sequence(ValidationMCBTagIP3DFastPV)

