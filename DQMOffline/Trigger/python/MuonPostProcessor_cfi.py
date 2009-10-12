import FWCore.ParameterSet.Config as cms

hLTMuonPostVal = cms.EDAnalyzer("DQMGenericClient",

    #subDirs        = cms.untracked.vstring('HLT/Muon/Distributions/HLT_Mu15/*',
	#									   'HLT/Muon/Distributions/HLT_L1Mu/*'),
								
								subDirs = cms.untracked.vstring( "HLT/Muon/Distributions/*"),

    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    outputFileName = cms.untracked.string(''),
    commands       = cms.vstring(),
    resolution     = cms.vstring(),                                    
    efficiency     = cms.vstring(

	    #### Comma Checking --- check for trailing commas by searching
   	    ###  this file for the regexp "[^,rf]

 	    #######################    L3 Filtered   #########################
	
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

		#######################    L3 PreFiltered   #########################

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


		#######################    L3 IsoFiltered   #########################

        "recEffEta_L3IsoFiltered '#eta Efficiency for L3IsoFiltered; #eta of Reconstructed Muon;# Rec #mu Matched to L3Iso / # All rec muons' recPassEta_L3IsoFiltered recPassEta_All", 
        "recEffPhi_L3IsoFiltered '#phi Efficiency for L3IsoFiltered; #phi of Reconstructed Muon;# Rec #mu Matched to L3Iso / # All rec muons' recPassPhi_L3IsoFiltered recPassPhi_All", 
        "recEffPtMax_L3IsoFiltered '#P_t Efficiency for L3IsoFiltered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3Iso / # All rec muons' recPassMaxPt_L3IsoFiltered recPassMaxPt_All",
		"recEffPt_L3IsoFiltered '#P_t Efficiency for L3IsoFiltered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3Iso / # All rec muons' recPassPt_L3IsoFiltered recPassPt_All",
		"recEffZ0_L3IsoFiltered 'Z0 (from origin) Efficiency for L3IsoFiltered; Reconstructed Muon Z0;# Rec #mu Matched to L3 / # All rec muons' recPassZ0_L3IsoFiltered recPassZ0_All",
		"recEffD0Beam_L3IsoFiltered 'd0 (from beamspot) Efficiency for L3IsoFiltered; Reconstructed Muon d0 ;# Rec #mu Matched to L3 / # All rec muons' recPassD0Beam_L3IsoFiltered recPassD0Beam_All",
		"recEffCharge_L3IsoFiltered 'Charge Efficiency for L3IsoFiltered; Reconstructed Muon Charge ;# Rec #mu Matched to L3 / # All rec muons' recPassCharge_L3IsoFiltered  recPassCharge_All",
		"fakeEffPt_L3IsoFiltered 'Efficiency for Fakes passing L3IsoFiltered; AOD Muon Pt; #Fakes / #HLT Muons' fakeHltCandPt_L3IsoFiltered allHltCandPt_L3IsoFiltered",
		"fakeEffEta_L3IsoFiltered 'Efficiency for Fakes passing L3IsoFiltered; AOD Muon Eta; #Fakes / #HLT Muons' fakeHltCandEta_L3IsoFiltered allHltCandEta_L3IsoFiltered", 
		"fakeEffPhi_L3IsoFiltered 'Efficiency for Fakes passing L3IsoFiltered; AOD Muon Phi; #Fakes / #HLT Muons' fakeHltCandPhi_L3IsoFiltered allHltCandPhi_L3IsoFiltered",


		#######################    L1MuL1Filtered   #########################
		
        "recEffEta_L1MuL1Filtered '#eta Efficiency for L1MuL1Filtered; #eta of Reconstructed Muon;# Rec #mu Matched to L3Iso / # All rec muons' recPassEta_L1MuL1Filtered recPassEta_All", 
        "recEffPhi_L1MuL1Filtered '#phi Efficiency for L1MuL1Filtered; #phi of Reconstructed Muon;# Rec #mu Matched to L3Iso / # All rec muons' recPassPhi_L1MuL1Filtered recPassPhi_All", 
        "recEffPtMax_L1MuL1Filtered '#P_t Efficiency for L1MuL1Filtered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3Iso / # All rec muons' recPassMaxPt_L1MuL1Filtered recPassMaxPt_All",
		"recEffPt_L1MuL1Filtered '#P_t Efficiency for L1MuL1Filtered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3Iso / # All rec muons' recPassPt_L1MuL1Filtered recPassPt_All",
		"recEffZ0_L1MuL1Filtered 'Z0 (from origin) Efficiency for L1MuL1Filtered; Reconstructed Muon Z0;# Rec #mu Matched to L3 / # All rec muons' recPassZ0_L1MuL1Filtered recPassZ0_All",
		"recEffD0Beam_L1MuL1Filtered 'd0 (from beamspot) Efficiency for L1MuL1Filtered; Reconstructed Muon d0 ;# Rec #mu Matched to L3 / # All rec muons' recPassD0Beam_L1MuL1Filtered recPassD0Beam_All",
		"recEffCharge_L1MuL1Filtered 'Charge Efficiency for L1MuL1Filtered; Reconstructed Muon Charge ;# Rec #mu Matched to L3 / # All rec muons' recPassCharge_L1MuL1Filtered  recPassCharge_All",
		"fakeEffPt_L1MuL1Filtered 'Efficiency for Fakes passing L1MuL1Filtered; AOD Muon Pt; #Fakes / #HLT Muons' fakeHltCandPt_L1MuL1Filtered allHltCandPt_L1MuL1Filtered",
		"fakeEffEta_L1MuL1Filtered 'Efficiency for Fakes passing L1MuL1Filtered; AOD Muon Eta; #Fakes / #HLT Muons' fakeHltCandEta_L1MuL1Filtered allHltCandEta_L1MuL1Filtered", 
		"fakeEffPhi_L1MuL1Filtered 'Efficiency for Fakes passing L1MuL1Filtered; AOD Muon Phi; #Fakes / #HLT Muons' fakeHltCandPhi_L1MuL1Filtered allHltCandPhi_L1MuL1Filtered",

		#######################    L1MuOpenL1Filtered   #########################
		
        "recEffEta_L1MuOpenL1Filtered '#eta Efficiency for L1MuOpenL1Filtered; #eta of Reconstructed Muon;# Rec #mu Matched to L3Iso / # All rec muons' recPassEta_L1MuOpenL1Filtered recPassEta_All", 
        "recEffPhi_L1MuOpenL1Filtered '#phi Efficiency for L1MuOpenL1Filtered; #phi of Reconstructed Muon;# Rec #mu Matched to L3Iso / # All rec muons' recPassPhi_L1MuOpenL1Filtered recPassPhi_All", 
        "recEffPtMax_L1MuOpenL1Filtered '#P_t Efficiency for L1MuOpenL1Filtered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3Iso / # All rec muons' recPassMaxPt_L1MuOpenL1Filtered recPassMaxPt_All",
		"recEffPt_L1MuOpenL1Filtered '#P_t Efficiency for L1MuOpenL1Filtered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3Iso / # All rec muons' recPassPt_L1MuOpenL1Filtered recPassPt_All",
		"recEffZ0_L1MuOpenL1Filtered 'Z0 (from origin) Efficiency for L1MuOpenL1Filtered; Reconstructed Muon Z0;# Rec #mu Matched to L3 / # All rec muons' recPassZ0_L1MuOpenL1Filtered recPassZ0_All",
		"recEffD0Beam_L1MuOpenL1Filtered 'd0 (from beamspot) Efficiency for L1MuOpenL1Filtered; Reconstructed Muon d0 ;# Rec #mu Matched to L3 / # All rec muons' recPassD0Beam_L1MuOpenL1Filtered recPassD0Beam_All",
		"recEffCharge_L1MuOpenL1Filtered 'Charge Efficiency for L1MuOpenL1Filtered; Reconstructed Muon Charge ;# Rec #mu Matched to L3 / # All rec muons' recPassCharge_L1MuOpenL1Filtered  recPassCharge_All",
		"fakeEffPt_L1MuOpenL1Filtered 'Efficiency for Fakes passing L1MuOpenL1Filtered; AOD Muon Pt; #Fakes / #HLT Muons' fakeHltCandPt_L1MuOpenL1Filtered allHltCandPt_L1MuOpenL1Filtered",
		"fakeEffEta_L1MuOpenL1Filtered 'Efficiency for Fakes passing L1MuOpenL1Filtered; AOD Muon Eta; #Fakes / #HLT Muons' fakeHltCandEta_L1MuOpenL1Filtered allHltCandEta_L1MuOpenL1Filtered", 
		"fakeEffPhi_L1MuOpenL1Filtered 'Efficiency for Fakes passing L1MuOpenL1Filtered; AOD Muon Phi; #Fakes / #HLT Muons' fakeHltCandPhi_L1MuOpenL1Filtered allHltCandPhi_L1MuOpenL1Filtered",

		#######################    L1OpenFiltered   #########################
		
        "recEffEta_L1OpenFiltered '#eta Efficiency for L1OpenFiltered; #eta of Reconstructed Muon;# Rec #mu Matched to L3Iso / # All rec muons' recPassEta_L1OpenFiltered recPassEta_All", 
        "recEffPhi_L1OpenFiltered '#phi Efficiency for L1OpenFiltered; #phi of Reconstructed Muon;# Rec #mu Matched to L3Iso / # All rec muons' recPassPhi_L1OpenFiltered recPassPhi_All", 
        "recEffPtMax_L1OpenFiltered '#P_t Efficiency for L1OpenFiltered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3Iso / # All rec muons' recPassMaxPt_L1OpenFiltered recPassMaxPt_All",
		"recEffPt_L1OpenFiltered '#P_t Efficiency for L1OpenFiltered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3Iso / # All rec muons' recPassPt_L1OpenFiltered recPassPt_All",
		"recEffZ0_L1OpenFiltered 'Z0 (from origin) Efficiency for L1OpenFiltered; Reconstructed Muon Z0;# Rec #mu Matched to L3 / # All rec muons' recPassZ0_L1OpenFiltered recPassZ0_All",
		"recEffD0Beam_L1OpenFiltered 'd0 (from beamspot) Efficiency for L1OpenFiltered; Reconstructed Muon d0 ;# Rec #mu Matched to L3 / # All rec muons' recPassD0Beam_L1OpenFiltered recPassD0Beam_All",
		"recEffCharge_L1OpenFiltered 'Charge Efficiency for L1OpenFiltered; Reconstructed Muon Charge ;# Rec #mu Matched to L3 / # All rec muons' recPassCharge_L1OpenFiltered  recPassCharge_All",
		"fakeEffPt_L1OpenFiltered 'Efficiency for Fakes passing L1OpenFiltered; AOD Muon Pt; #Fakes / #HLT Muons' fakeHltCandPt_L1OpenFiltered allHltCandPt_L1OpenFiltered",
		"fakeEffEta_L1OpenFiltered 'Efficiency for Fakes passing L1OpenFiltered; AOD Muon Eta; #Fakes / #HLT Muons' fakeHltCandEta_L1OpenFiltered allHltCandEta_L1OpenFiltered", 
		"fakeEffPhi_L1OpenFiltered 'Efficiency for Fakes passing L1OpenFiltered; AOD Muon Phi; #Fakes / #HLT Muons' fakeHltCandPhi_L1OpenFiltered allHltCandPhi_L1OpenFiltered",


		#######################    L2Mu9L2Filtered   #########################
		
        "recEffEta_L2Mu9L2Filtered '#eta Efficiency for L2Mu9L2Filtered; #eta of Reconstructed Muon;# Rec #mu Matched to L3Iso / # All rec muons' recPassEta_L2Mu9L2Filtered recPassEta_All", 
        "recEffPhi_L2Mu9L2Filtered '#phi Efficiency for L2Mu9L2Filtered; #phi of Reconstructed Muon;# Rec #mu Matched to L3Iso / # All rec muons' recPassPhi_L2Mu9L2Filtered recPassPhi_All", 
        "recEffPtMax_L2Mu9L2Filtered '#P_t Efficiency for L2Mu9L2Filtered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3Iso / # All rec muons' recPassMaxPt_L2Mu9L2Filtered recPassMaxPt_All",
		"recEffPt_L2Mu9L2Filtered '#P_t Efficiency for L2Mu9L2Filtered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3Iso / # All rec muons' recPassPt_L2Mu9L2Filtered recPassPt_All",
		"recEffZ0_L2Mu9L2Filtered 'Z0 (from origin) Efficiency for L2Mu9L2Filtered; Reconstructed Muon Z0;# Rec #mu Matched to L3 / # All rec muons' recPassZ0_L2Mu9L2Filtered recPassZ0_All",
		"recEffD0Beam_L2Mu9L2Filtered 'd0 (from beamspot) Efficiency for L2Mu9L2Filtered; Reconstructed Muon d0 ;# Rec #mu Matched to L3 / # All rec muons' recPassD0Beam_L2Mu9L2Filtered recPassD0Beam_All",
		"recEffCharge_L2Mu9L2Filtered 'Charge Efficiency for L2Mu9L2Filtered; Reconstructed Muon Charge ;# Rec #mu Matched to L3 / # All rec muons' recPassCharge_L2Mu9L2Filtered  recPassCharge_All",
		"fakeEffPt_L2Mu9L2Filtered 'Efficiency for Fakes passing L2Mu9L2Filtered; AOD Muon Pt; #Fakes / #HLT Muons' fakeHltCandPt_L2Mu9L2Filtered allHltCandPt_L2Mu9L2Filtered",
		"fakeEffEta_L2Mu9L2Filtered 'Efficiency for Fakes passing L2Mu9L2Filtered; AOD Muon Eta; #Fakes / #HLT Muons' fakeHltCandEta_L2Mu9L2Filtered allHltCandEta_L2Mu9L2Filtered",
		"fakeEffPhi_L2Mu9L2Filtered 'Efficiency for Fakes passing L2Mu9L2Filtered; AOD Muon Phi; #Fakes / #HLT Muons' fakeHltCandPhi_L2Mu9L2Filtered allHltCandPhi_L2Mu9L2Filtered",

		

		######################      L2Mu11L2Filtered
		"recEffEta_L2Mu11L2Filtered '#eta Efficiency for L2Mu11L2Filtered; #eta of Reconstructed Muon;# Rec #mu Matched to L3Iso / # All rec muons' recPassEta_L2Mu11L2Filtered recPassEta_All", 
        "recEffPhi_L2Mu11L2Filtered '#phi Efficiency for L2Mu11L2Filtered; #phi of Reconstructed Muon;# Rec #mu Matched to L3Iso / # All rec muons' recPassPhi_L2Mu11L2Filtered recPassPhi_All", 
        "recEffPtMax_L2Mu11L2Filtered '#P_t Efficiency for L2Mu11L2Filtered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3Iso / # All rec muons' recPassMaxPt_L2Mu11L2Filtered recPassMaxPt_All",
		"recEffPt_L2Mu11L2Filtered '#P_t Efficiency for L2Mu11L2Filtered; Reconstructed Muon pt (GeV);# Rec #mu Matched to L3Iso / # All rec muons' recPassPt_L2Mu11L2Filtered recPassPt_All",
		"recEffZ0_L2Mu11L2Filtered 'Z0 (from origin) Efficiency for L2Mu11L2Filtered; Reconstructed Muon Z0;# Rec #mu Matched to L3 / # All rec muons' recPassZ0_L2Mu11L2Filtered recPassZ0_All",
		"recEffD0Beam_L2Mu11L2Filtered 'd0 (from beamspot) Efficiency for L2Mu11L2Filtered; Reconstructed Muon d0 ;# Rec #mu Matched to L3 / # All rec muons' recPassD0Beam_L2Mu11L2Filtered recPassD0Beam_All",
		"recEffCharge_L2Mu11L2Filtered 'Charge Efficiency for L2Mu11L2Filtered; Reconstructed Muon Charge ;# Rec #mu Matched to L3 / # All rec muons' recPassCharge_L2Mu11L2Filtered  recPassCharge_All",
		"fakeEffPt_L2Mu11L2Filtered 'Efficiency for Fakes passing L2Mu11L2Filtered; AOD Muon Pt; #Fakes / #HLT Muons' fakeHltCandPt_L2Mu11L2Filtered allHltCandPt_L2Mu11L2Filtered",
		"fakeEffEta_L2Mu11L2Filtered 'Efficiency for Fakes passing L2Mu11L2Filtered; AOD Muon Eta; #Fakes / #HLT Muons' fakeHltCandEta_L2Mu11L2Filtered allHltCandEta_L2Mu11L2Filtered",
		"fakeEffPhi_L2Mu11L2Filtered 'Efficiency for Fakes passing L2Mu11L2Filtered; AOD Muon Phi; #Fakes / #HLT Muons' fakeHltCandPhi_L2Mu11L2Filtered allHltCandPhi_L2Mu11L2Filtered",

		


		

    )
)

