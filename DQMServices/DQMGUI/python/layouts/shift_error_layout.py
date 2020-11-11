from ..layouts.layout_manager import register_layout

register_layout(source='BeamMonitor/BeamSpotProblemMonitor/FitFromScalars/BeamSpotError', destination='00 Shift/Errors/BeamSpotError', name='06 - BeamSpot missing from online', overlay='')
register_layout(source='L1T/L1TStage2uGMT/zeroSuppression/AllEvts/mismatchRatio', destination='00 Shift/Errors/mismatchRatio', name='08 - L1T Zero Suppression Error', overlay='')
register_layout(source='L1T/L1TStage2uGT/uGMToutput_vs_uGTinput/mismatchRatio', destination='00 Shift/Errors/mismatchRatio', name='09 - L1T Data Transmission Error', overlay='')
register_layout(source='L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy1/mismatchRatio', destination='00 Shift/Errors/mismatchRatio', name='10 - L1T uGMT Output Integrity Error', overlay='')
register_layout(source='L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy2/mismatchRatio', destination='00 Shift/Errors/mismatchRatio', name='10 - L1T uGMT Output Integrity Error', overlay='')
register_layout(source='L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy3/mismatchRatio', destination='00 Shift/Errors/mismatchRatio', name='10 - L1T uGMT Output Integrity Error', overlay='')
register_layout(source='L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy4/mismatchRatio', destination='00 Shift/Errors/mismatchRatio', name='10 - L1T uGMT Output Integrity Error', overlay='')
register_layout(source='L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy5/mismatchRatio', destination='00 Shift/Errors/mismatchRatio', name='10 - L1T uGMT Output Integrity Error', overlay='')
register_layout(source='L1T/L1TStage2CaloLayer1/MismatchDetail/maxEvtMismatchByLumiHCAL', destination='00 Shift/Errors/maxEvtMismatchByLumiHCAL', name='11 - HCAL uHTR-L1T Layer1 Mismatch', overlay='')
