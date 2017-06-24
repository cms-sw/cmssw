import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

metEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/MET/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_met          'MET turnON;            PF MET [GeV]; efficiency'     met_numerator          met_denominator",
        "effic_met_variable 'MET turnON;            PF MET [GeV]; efficiency'     met_variable_numerator met_variable_denominator",
        "effic_metPhi       'MET efficiency vs phi; PF MET phi [rad]; efficiency' metPhi_numerator       metPhi_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_met_vs_LS 'MET efficiency vs LS; LS; PF MET efficiency' metVsLS_numerator metVsLS_denominator"
    ),
  
)

DJHTEfficiency = DQMEDHarvester("DQMGenericClient",

     subDirs        = cms.untracked.vstring("HLT/DisplacedJet/*HT"),
     verbose        = cms.untracked.uint32(0),
     resolution    = cms.vstring(),
     efficiency     = cms.vstring(
         "effic_caloHT           'CaloHT turnOn;       Calo HT [GeV]; efficiency'   caloHT_numerator        caloHT_denominator",
         "effic_caloHT_variable  'CaloHT turnOn;       Calo HT [GeV]; efficiency'   caloHT_numerator        caloHT_denominatro",
     ),

     efficiencyProfile = cms.untracked.vstring(
         "effic_caloHT_vs_LS     'CaloHT efficiency vs LS; LS; Calo HT efficiency'  caloHTVsLS_numerator caloHTVsLS_denominator"
     ),
)


DJTrackEfficiency = DQMEDHarvester("DQMGenericClient",

     subDirs        = cms.untracked.vstring("HLT/DisplacedJet/*Track"),
     verbose        = cms.untracked.uint32(0),
     resolution    = cms.vstring(),
     efficiency     = cms.vstring(
         "effic_npropmttrksjet1    '#PromptTracks in Jet 1 turnOn; #Prompt Tracks in Jet 1; efficiency' npropmttrksjet1_numerator    npropmttrksjet1_denominator",
         "effic_npropmttrksjet2    '#PromptTracks in Jet 2 turnOn; #Prompt Tracks in Jet 2; efficiency' npropmttrksjet2_numerator    npropmttrksjet2_denominator",
         "effic_ndisplacedtrksjet1 '#DisplacedTracks in Jet 1 turnOn; #Displaced Tracks in Jet1; efficiency' ndisplacedtrksjet1_numerator    ndisplacedtrksjet1_denominator",
         "effic_ndisplacedtrksjet2 '#DisplacedTracks in Jet 2 turnOn; #Displaced Tracks in Jet2; efficiency' ndisplacedtrksjet2_numerator    ndisplacedtrksjet2_denominator",
     ),

     efficiencyProfile = cms.untracked.vstring(
         "effic_npropmttrksjet1_vs_LS    '#PromptTracks in Jet1 efficiency vs LS; LS; #Prompt Tracks in Jet1 efficiency' npropmttrksjet1VsLS_numerator    npropmttrksjet1VsLS_denominator",
         "effic_npropmttrksjet2_vs_LS    '#PromptTracks in Jet2 efficiency vs LS; LS; #Prompt Tracks in Jet2 efficiency' npropmttrksjet2VsLS_numerator    npropmttrksjet2VsLS_denominator",
         "effic_ndisplacedtrksjet1_vs_LS    '#DisplacedTracks in Jet1 efficiency vs LS; LS; #Displaced Tracks in Jet1 efficiency' ndisplacedtrksjet1VsLS_numerator    ndisplacedtrksjet1VsLS_denominator",
         "effic_ndisplacedtrksjet2_vs_LS    '#DisplacedTracks in Jet2 efficiency vs LS; LS; #Displaced Tracks in Jet2 efficiency' ndisplacedtrksjet2VsLS_numerator    ndisplacedtrksjet2VsLS_denominator",
     ),
)



exoticaClient = cms.Sequence(
    metEfficiency
   +DJHTEfficiency
   +DJTrackEfficiency
)
