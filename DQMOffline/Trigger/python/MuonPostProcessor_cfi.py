import FWCore.ParameterSet.Config as cms

HLTMuonPostVal = cms.EDAnalyzer("DQMGenericClient",
    #subDirs        = cms.untracked.vstring('HLT/Muon/Distributions/HLT_Mu15/*',
	#									   'HLT/Muon/Distributions/HLT_L1Mu/*'),
								
								subDirs = cms.untracked.vstring( "HLT/Muon/Distributions/*"),

    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    outputFileName = cms.untracked.string(''),
    commands       = cms.vstring(),
    resolution     = cms.vstring(),                                    
    efficiency     = cms.vstring(
        "recEffEta_L3Filtered '#eta Efficiency for L3Filtered; #eta of Reconstructed Muon;# Rec #mu Matched to L3 / # All Rec muons' recPassEta_L3Filtered recPassEta_All", 
        "recEffPhi_L3Filtered '#phi Efficiency for L3Filtered; #phi of Reconstructed Muon;# Rec #mu Matched to L3 / # All rec muons' recPassPhi_L3Filtered recPassPhi_All", 
        "recEffPtMax_L3Filtered 'P_T Efficiency for L3Filtered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3 / # All rec muons' recPassMaxPt_L3Filtered recPassMaxPt_All",
		"recEffPt_L3Filtered 'P_T Efficiency for L3Filtered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3 / # All rec muons' recPassPt_L3Filtered recPassPt_All",
		#"recEffPhiVsEta_L3Filtered 'Efficiency for L3Filtered; Reconstructed Muon #eta (GeV); Rec #mu #phi  / # All rec muons' recPhiVsRecEta_L3Filtered recPhiVsRecEta_All",
        "recEffZ0_L3Filtered 'Z0 (from origin) Efficiency for L3Filtered; Reconstructed Muon Z0;# Rec #mu Matched to L3 / # All rec muons' recPassZ0_L3Filtered recPassZ0_All",
		"recEffD0Beam_L3Filtered 'd0 (from beamspot) Efficiency for L3Filtered; Reconstructed Muon d0 ;# Rec #mu Matched to L3 / # All rec muons' recPassD0Beam_L3Filtered recPassD0Beam_All",
		"recEffCharge_L3Filtered 'Charge Efficiency for L3Filtered; Reconstructed Muon Charge ;# Rec #mu Matched to L3 / # All rec muons' recPassCharge_L3Filtered  recPassCharge_All",
		"fakeEffPt_L3Filtered 'Efficiency for Fakes passing L3Filtered; AOD Muon Pt; #Fakes / #HLT Muons' fakeHltCandPt_L3Filtered allHltCandPt_L3Filtered",
		"fakeEffEta_L3Filtered 'Efficiency for Fakes passing L3Filtered; AOD Muon Eta; #Fakes / #HLT Muons' fakeHltCandEta_L3Filtered allHltCandEta_L3Filtered", 
		"fakeEffPhi_L3Filtered 'Efficiency for Fakes passing L3Filtered; AOD Muon Phi; #Fakes / #HLT Muons' fakeHltCandPhi_L3Filtered allHltCandPhi_L3Filtered",

        "recEffEta_L3PreFiltered '#eta Efficiency for L3PreFiltered; #eta of Reconstructed Muon;# Rec #mu Matched to L3Pre / # All rec muons' recPassEta_L3PreFiltered recPassEta_All", 
        "recEffPhi_L3PreFiltered '#phi Efficiency for L3PreFiltered; #phi of Reconstructed Muon;# Rec #mu Matched to L3Pre / # All rec muons' recPassPhi_L3PreFiltered recPassPhi_All", 
		"recEffPtMax_L3PreFiltered '#P_t Efficiency for L3PreFiltered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3Pre / # All rec muons' recPassMaxPt_L3PreFiltered recPassMaxPt_All",
		"recEffPt_L3PreFiltered '#p_t Efficiency for L3PreFiltered; Reconstructed Muon #p_T (GeV);# Rec #mu Matched to L3Pre / # All rec muons' recPassPt_L3PreFiltered recPassPt_All",
		#"recEffPhiVsEta_L3PreFiltered 'Efficiency for L3PreFiltered; Reconstructed Muon #eta (GeV); Rec #mu #phi  / # All rec muons' recPhiVsRecEta_L3PreFiltered recPhiVsRecEta_All",
		"recEffZ0_L3PreFiltered 'Z0 (from origin) Efficiency for L3PreFiltered; Reconstructed Muon Z0;# Rec #mu Matched to L3 / # All rec muons' recPassZ0_L3PreFiltered recPassZ0_All",
		"recEffD0Beam_L3PreFiltered 'd0 (from beamspot) Efficiency for L3PreFiltered; Reconstructed Muon d0 ;# Rec #mu Matched to L3 / # All rec muons' recPassD0Beam_L3PreFiltered recPassD0Beam_All",
		"recEffCharge_L3PreFiltered 'Charge Efficiency for L3PreFiltered; Reconstructed Muon Charge ;# Rec #mu Matched to L3 / # All rec muons' recPassCharge_L3PreFiltered  recPassCharge_All",
		"fakeEffPt_L3PreFiltered 'Efficiency for Fakes passing L3PreFiltered; AOD Muon Pt; #Fakes / #HLT Muons' fakeHltCandPt_L3PreFiltered allHltCandPt_L3PreFiltered",
		"fakeEffEta_L3PreFiltered 'Efficiency for Fakes passing L3PreFiltered; AOD Muon Eta; #Fakes / #HLT Muons' fakeHltCandEta_L3PreFiltered allHltCandEta_L3PreFiltered", 
		"fakeEffPhi_L3PreFiltered 'Efficiency for Fakes passing L3PreFiltered; AOD Muon Phi; #Fakes / #HLT Muons' fakeHltCandPhi_L3PreFiltered allHltCandPhi_L3PreFiltered",


        "recEffEta_L3IsoFiltered '#eta Efficiency for L3IsoFiltered; #eta of Reconstructed Muon;# Rec #mu Matched to L3Iso / # All rec muons' recPassEta_L3IsoFiltered recPassEta_All", 
        "recEffPhi_L3IsoFiltered '#phi Efficiency for L3IsoFiltered; #phi of Reconstructed Muon;# Rec #mu Matched to L3Iso / # All rec muons' recPassPhi_L3IsoFiltered recPassPhi_All", 
        "recEffPtMax_L3IsoFiltered '#P_t Efficiency for L3IsoFiltered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3Iso / # All rec muons' recPassMaxPt_L3IsoFiltered recPassMaxPt_All",
		"recEffPt_L3IsoFiltered '#P_t Efficiency for L3IsoFiltered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3Iso / # All rec muons' recPassPt_L3IsoFiltered recPassPt_All"
		"recEffZ0_L3IsoFiltered 'Z0 (from origin) Efficiency for L3IsoFiltered; Reconstructed Muon Z0;# Rec #mu Matched to L3 / # All rec muons' recPassZ0_L3IsoFiltered recPassZ0_All",
		"recEffD0Beam_L3IsoFiltered 'd0 (from beamspot) Efficiency for L3IsoFiltered; Reconstructed Muon d0 ;# Rec #mu Matched to L3 / # All rec muons' recPassD0Beam_L3IsoFiltered recPassD0Beam_All",
		"recEffCharge_L3IsoFiltered 'Charge Efficiency for L3IsoFiltered; Reconstructed Muon Charge ;# Rec #mu Matched to L3 / # All rec muons' recPassCharge_L3IsoFiltered  recPassCharge_All",
		"fakeEffPt_L3IsoFiltered 'Efficiency for Fakes passing L3IsoFiltered; AOD Muon Pt; #Fakes / #HLT Muons' fakeHltCandPt_L3IsoFiltered allHltCandPt_L3IsoFiltered",
		"fakeEffEta_L3IsoFiltered 'Efficiency for Fakes passing L3IsoFiltered; AOD Muon Eta; #Fakes / #HLT Muons' fakeHltCandEta_L3IsoFiltered allHltCandEta_L3IsoFiltered", 
		"fakeEffPhi_L3IsoFiltered 'Efficiency for Fakes passing L3IsoFiltered; AOD Muon Phi; #Fakes / #HLT Muons' fakeHltCandPhi_L3IsoFiltered allHltCandPhi_L3IsoFiltered"

		
		#"recEffPhiVsEta_L3PreFiltered 'Efficiency for L3PreFiltered; Reconstructed Muon #eta (GeV); Rec #mu #phi  / # All rec muons' recPhiVsRecEta_L3PreFiltered recPhiVsRecEta_All"        

#        "recEffEta_L1Filtered '#eta Efficiency for L1Filtered; #eta of Reconstructed Muon;# Rec #mu Matched to L1 / # Rec #mu' recPassEta_L1Filtered recPassEta_All",
#        "recEffPhi_L1Filtered '#phi Efficiency for L1Filtered; #phi of Reconstructed Muon;# Rec #mu Matched to L1 / # Rec #mu' recPassPhi_L1Filtered recPassPhi_All", 
#        "recEffPt_L1Filtered 'pt Efficiency for L1Filtered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L1 / # Rec #mu' recPassPt_L1Filtered   recPassPt_All", 

#        "recTurnOn_L1Filtered 'pt Turn On for L1Filtered;p_{T} of Leading Reconstructed Muon (GeV);# Rec #mu Matched to L1 / # Rec #mu' recPassMaxPt_L1Filtered recPassMaxPt_All",
#        "recTurnOn_L3Filtered 'pt Turn On for L3Filtered;p_{T} of Leading Reconstructed Muon (GeV);# Rec #mu Matched to L3 / # Rec #mu Matched to L1' recPassMaxPt_L3Filtered recPassMaxPt_L1Filtered",
#        "recTurnOn_L3PreFiltered 'pt Turn On for L3PreFiltered;p_{T} of Leading Reconstructed Muon (GeV);# Rec #mu Matched to L3Pre / # Rec #mu Matched to L1' recPassMaxPt_L3PreFiltered recPassMaxPt_L1Filtered", 
#		"recTurnOn_L3IsoFiltered 'pt Turn On for L3IsoFiltered;p_{T} of Leading Reconstructed Muon (GeV);# Rec #mu Matched to L3Iso / # Rec #mu Matched to L1' recPassMaxPt_L3IsoFiltered recPassMaxPt_L1Filtered", 
## ------ Old ratio plots below
## ------ when using TriggerSummaryAOD, you don't have any L2 information
## 	"recEffEta_L2Filtered '#eta Efficiency for L2Filtered; #eta of Reconstructed Muon;# Rec #mu Matched to L2 / # Rec #mu Matched to L1' recPassEta_L2Filtered recPassEta_L1Filtered", 
##         "recEffPhi_L2Filtered '#phi Efficiency for L2Filtered; #phi of Reconstructed Muon;# Rec #mu Matched to L2 / # Rec #mu Matched to L1' recPassPhi_L2Filtered recPassPhi_L1Filtered", 
##         "recEffPt_L2Filtered 'pt Efficiency for L2Filtered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L2 / # Rec #mu Matched to L1' recPassPt_L2Filtered recPassPt_L1Filtered",
##         "recTurnOn_L2PreFiltered 'pt Turn On for L2PreFiltered;p_{T} of Leading Reconstructed Muon (GeV);# Rec #mu Matched to L2Pre / # Rec #mu Matched to L1' recPassMaxPt_L2PreFiltered recPassMaxPt_L1Filtered", 
##         "recEffEta_L2PreFiltered '#eta Efficiency for L2PreFiltered; #eta of Reconstructed Muon;# Rec #mu Matched to L2Pre / # Rec #mu Matched to L1' recPassEta_L2PreFiltered recPassEta_L1Filtered", 
##         "recEffPhi_L2PreFiltered '#phi Efficiency for L2PreFiltered; #phi of Reconstructed Muon;# Rec #mu Matched to L2Pre / # Rec #mu Matched to L1' recPassPhi_L2PreFiltered recPassPhi_L1Filtered", 
##         "recEffPt_L2PreFiltered 'pt Efficiency for L2PreFiltered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L2Pre / # Rec #mu Matched to L1' recPassPt_L2PreFiltered recPassPt_L1Filtered",
##         "recTurnOn_L2IsoFiltered 'pt Turn On for L2IsoFiltered;p_{T} of Leading Reconstructed Muon (GeV);# Rec #mu Matched to L2Iso / # Rec #mu Matched to L1' recPassMaxPt_L2IsoFiltered recPassMaxPt_L1Filtered", 
##         "recEffEta_L2IsoFiltered '#eta Efficiency for L2IsoFiltered; #eta of Reconstructed Muon;# Rec #mu Matched to L2Iso / # Rec #mu Matched to L1' recPassEta_L2IsoFiltered recPassEta_L1Filtered", 
##         "recEffPhi_L2IsoFiltered '#phi Efficiency for L2IsoFiltered; #phi of Reconstructed Muon;# Rec #mu Matched to L2Iso / # Rec #mu Matched to L1' recPassPhi_L2IsoFiltered recPassPhi_L1Filtered", 
##         "recEffPt_L2IsoFiltered 'pt Efficiency for L2IsoFiltered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L2Iso / # Rec #mu Matched to L1' recPassPt_L2IsoFiltered recPassPt_L1Filtered",
## 	"recTurnOn_L2Filtered 'pt Turn On for L2Filtered;p_{T} of Leading Reconstructed Muon (GeV);# Rec #mu Matched to L2 / # Rec #mu Matched to L1' recPassMaxPt_L2Filtered recPassMaxPt_L1Filtered",         
## 		"genTurnOn_L1Filtered 'pt Turn On for L1Filtered;p_{T} of Leading Generated Muon (GeV);# Gen #mu Matched to L1 / # Gen #mu' genPassMaxPt_L1Filtered genPassMaxPt_All",
##         "genEffEta_L1Filtered '#eta Efficiency for L1Filtered; #eta of Generated Muon;# Gen #mu Matched to L1 / # Gen #mu' genPassEta_L1Filtered genPassEta_All",
##         "genEffPhi_L1Filtered '#phi Efficiency for L1Filtered; #phi of Generated Muon;# Gen #mu Matched to L1 / # Gen #mu' genPassPhi_L1Filtered genPassPhi_All", 
##         "genEffPt_L1Filtered 'pt Efficiency for L1Filtered; Generated Muon pt (GeV);# Gen #mu Matched to L1 / # Gen #mu' genPassPt_L1Filtered genPassPt_All", 
##         "genTurnOn_L2Filtered 'pt Turn On for L2Filtered;p_{T} of Leading Generated Muon (GeV);# Gen #mu Matched to L2 / # Gen #mu Matched to L1' genPassMaxPt_L2Filtered genPassMaxPt_L1Filtered", 
##         "genEffEta_L2Filtered '#eta Efficiency for L2Filtered; #eta of Generated Muon;# Gen #mu Matched to L2 / # Gen #mu Matched to L1' genPassEta_L2Filtered genPassEta_L1Filtered", 
##         "genEffPhi_L2Filtered '#phi Efficiency for L2Filtered; #phi of Generated Muon;# Gen #mu Matched to L2 / # Gen #mu Matched to L1' genPassPhi_L2Filtered genPassPhi_L1Filtered", 
##         "genEffPt_L2Filtered 'pt Efficiency for L2Filtered; Generated Muon pt (GeV);# Gen #mu Matched to L2 / # Gen #mu Matched to L1' genPassPt_L2Filtered genPassPt_L1Filtered",
##         "genTurnOn_L2PreFiltered 'pt Turn On for L2PreFiltered;p_{T} of Leading Generated Muon (GeV);# Gen #mu Matched to L2Pre / # Gen #mu Matched to L1' genPassMaxPt_L2PreFiltered genPassMaxPt_L1Filtered", 
##         "genEffEta_L2PreFiltered '#eta Efficiency for L2PreFiltered; #eta of Generated Muon;# Gen #mu Matched to L2Pre / # Gen #mu Matched to L1' genPassEta_L2PreFiltered genPassEta_L1Filtered", 
##         "genEffPhi_L2PreFiltered '#phi Efficiency for L2PreFiltered; #phi of Generated Muon;# Gen #mu Matched to L2Pre / # Gen #mu Matched to L1' genPassPhi_L2PreFiltered genPassPhi_L1Filtered", 
##         "genEffPt_L2PreFiltered 'pt Efficiency for L2PreFiltered; Generated Muon pt (GeV);# Gen #mu Matched to L2Pre / # Gen #mu Matched to L1' genPassPt_L2PreFiltered genPassPt_L1Filtered",
##         "genTurnOn_L2IsoFiltered 'pt Turn On for L2IsoFiltered;p_{T} of Leading Generated Muon (GeV);# Gen #mu Matched to L2Iso / # Gen #mu Matched to L1' genPassMaxPt_L2IsoFiltered genPassMaxPt_L1Filtered", 
##         "genEffEta_L2IsoFiltered '#eta Efficiency for L2IsoFiltered; #eta of Generated Muon;# Gen #mu Matched to L2Iso / # Gen #mu Matched to L1' genPassEta_L2IsoFiltered genPassEta_L1Filtered", 
##         "genEffPhi_L2IsoFiltered '#phi Efficiency for L2IsoFiltered; #phi of Generated Muon;# Gen #mu Matched to L2Iso / # Gen #mu Matched to L1' genPassPhi_L2IsoFiltered genPassPhi_L1Filtered", 
##         "genEffPt_L2IsoFiltered 'pt Efficiency for L2IsoFiltered; Generated Muon pt (GeV);# Gen #mu Matched to L2Iso / # Gen #mu Matched to L1' genPassPt_L2IsoFiltered genPassPt_L1Filtered",
##         "genTurnOn_L3Filtered 'pt Turn On for L3Filtered;p_{T} of Leading Generated Muon (GeV);# Gen #mu Matched to L3 / # Gen #mu Matched to L1' genPassMaxPt_L3Filtered genPassMaxPt_L1Filtered", 
##         "genEffEta_L3Filtered '#eta Efficiency for L3Filtered; #eta of Generated Muon;# Gen #mu Matched to L3 / # Gen #mu Matched to L1' genPassEta_L3Filtered genPassEta_L1Filtered", 
##         "genEffPhi_L3Filtered '#phi Efficiency for L3Filtered; #phi of Generated Muon;# Gen #mu Matched to L3 / # Gen #mu Matched to L1' genPassPhi_L3Filtered genPassPhi_L1Filtered", 
##         "genEffPt_L3Filtered 'pt Efficiency for L3Filtered; Generated Muon pt (GeV);# Gen #mu Matched to L3 / # Gen #mu Matched to L1' genPassPt_L3Filtered genPassPt_L1Filtered",
##         "genTurnOn_L3PreFiltered 'pt Turn On for L3PreFiltered;p_{T} of Leading Generated Muon (GeV);# Gen #mu Matched to L3Pre / # Gen #mu Matched to L1' genPassMaxPt_L3PreFiltered genPassMaxPt_L1Filtered", 
##         "genEffEta_L3PreFiltered '#eta Efficiency for L3PreFiltered; #eta of Generated Muon;# Gen #mu Matched to L3Pre / # Gen #mu Matched to L1' genPassEta_L3PreFiltered genPassEta_L1Filtered", 
##         "genEffPhi_L3PreFiltered '#phi Efficiency for L3PreFiltered; #phi of Generated Muon;# Gen #mu Matched to L3Pre / # Gen #mu Matched to L1' genPassPhi_L3PreFiltered genPassPhi_L1Filtered", 
##         "genEffPt_L3PreFiltered 'pt Efficiency for L3PreFiltered; Generated Muon pt (GeV);# Gen #mu Matched to L3Pre / # Gen #mu Matched to L1' genPassPt_L3PreFiltered genPassPt_L1Filtered",
##         "genTurnOn_L3IsoFiltered 'pt Turn On for L3IsoFiltered;p_{T} of Leading Generated Muon (GeV);# Gen #mu Matched to L3Iso / # Gen #mu Matched to L1' genPassMaxPt_L3IsoFiltered genPassMaxPt_L1Filtered", 
##         "genEffEta_L3IsoFiltered '#eta Efficiency for L3IsoFiltered; #eta of Generated Muon;# Gen #mu Matched to L3Iso / # Gen #mu Matched to L1' genPassEta_L3IsoFiltered genPassEta_L1Filtered", 
##         "genEffPhi_L3IsoFiltered '#phi Efficiency for L3IsoFiltered; #phi of Generated Muon;# Gen #mu Matched to L3Iso / # Gen #mu Matched to L1' genPassPhi_L3IsoFiltered genPassPhi_L1Filtered", 
##         "genEffPt_L3IsoFiltered 'pt Efficiency for L3IsoFiltered; Generated Muon pt (GeV);# Gen #mu Matched to L3Iso / # Gen #mu Matched to L1' genPassPt_L3IsoFiltered genPassPt_L1Filtered",

    )
)

