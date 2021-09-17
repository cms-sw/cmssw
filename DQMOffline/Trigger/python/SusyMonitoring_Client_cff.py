import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQMOffline.Trigger.VBFSUSYMonitor_Client_cff import *
from DQMOffline.Trigger.LepHTMonitor_cff import *
from DQMOffline.Trigger.susyHLTEleCaloJetsClient_cfi import *
from DQMOffline.Trigger.RazorMonitor_Client_cff import *
from DQMOffline.Trigger.SoftMuHardJetMETSUSYMonitor_Client_cff import *

double_soft_muon_muonpt_efficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSY/SOS/Muon"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages 
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
      "effic_muPt_1       'efficiency vs muon pt; muon pt [GeV]; efficiency'  muPt_1_variableBinning_numerator  muPt_1_variableBinning_denominator",
      "effic_muEta_1      'efficiency vs muon eta; muon eta ; efficiency'    muEta_1_variableBinning_numerator muEta_1_variableBinning_denominator",
      "effic_muPhi_1      'efficiency vs muon phi; muon phi ; efficiency'    muPhi_1_numerator                 muPhi_1_denominator",

      "effic_muPt_2       'efficiency vs muon pt; muon pt [GeV]; efficiency'  muPt_2_variableBinning_numerator  muPt_2_variableBinning_denominator",
      "effic_muEta_2      'efficiency vs muon eta; muon eta ; efficiency'    muEta_2_variableBinning_numerator muEta_2_variableBinning_denominator",
      "effic_muPhi_2      'efficiency vs muon phi; muon phi ; efficiency'    muPhi_2_numerator                 muPhi_2_denominator",

      "effic_mu1mu2Pt     'efficiency vs mu1mu2 Pt; mu1 Pt [GeV]; mu2 Pt [GeV]' mu1Pt_mu2Pt_numerator   mu1Pt_mu2Pt_denominator",
      "effic_mu1mu2Eta    'efficiency vs mu1mu2 Eta; mu1 Eta ; mu2 Eta' mu1Eta_mu2Eta_numerator   mu1Eta_mu2Eta_denominator",
    ),
)

double_soft_muon_metpt_efficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSY/SOS/MET"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages 
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
      "effic_met       'efficiency vs met pt; met [GeV]; efficiency' met_numerator       met_denominator",
    ),
)

double_soft_muon_mll_efficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSY/SOS/Mll"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages 
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
      "effic_Mll       'efficiency vs inv.mass; Mll [GeV]; efficiency' invMass_numerator       invMass_denominator",
      "effic_Mll_variableBinning       'efficiency vs inv.mass; Mll [GeV]; efficiency' invMass_variable_numerator       invMass_variable_denominator",
    ),
)

double_soft_muon_mhtpt_efficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSY/SOS/MHT"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages 
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
      "effic_MHT       'efficiency vs MHT; met [GeV]; efficiency' eventMHT_numerator       eventMHT_denominator",
      "effic_MHT_variableBinning       'efficiency vs MHT; met [GeV]; efficiency' eventMHT_variable_numerator       eventMHT_variable_denominator",
    ),
)

# backup1
double_soft_muon_backup_70_metpt_efficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSY/SOS/backup70/MET"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages 
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
      "effic_met       'efficiency vs met pt; met [GeV]; efficiency' met_numerator       met_denominator",
    ),
)

double_soft_muon_backup_70_mhtpt_efficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSY/SOS/backup70/MHT"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages 
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
      "effic_MHT       'efficiency vs MHT; met [GeV]; efficiency' eventMHT_numerator       eventMHT_denominator",
      "effic_MHT_variableBinning       'efficiency vs MHT; met [GeV]; efficiency' eventMHT_variable_numerator       eventMHT_variable_denominator",
    ),
)

# backup2
double_soft_muon_backup_90_metpt_efficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSY/SOS/backup90/MET"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages 
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
      "effic_met       'efficiency vs met pt; met [GeV]; efficiency' met_numerator       met_denominator",
    ),
)

double_soft_muon_backup_90_mhtpt_efficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSY/SOS/backup90/MHT"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages 
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
      "effic_MHT       'efficiency vs MHT; met [GeV]; efficiency' eventMHT_numerator       eventMHT_denominator",
      "effic_MHT_variableBinning       'efficiency vs MHT; met [GeV]; efficiency' eventMHT_variable_numerator       eventMHT_variable_denominator",
    ),
)

triple_muon_mupt_efficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSY/SOS/TripleMu/Muon"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages 
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
      "effic_muPt_1       'efficiency vs muon pt; muon pt [GeV]; efficiency'  muPt_1_variableBinning_numerator  muPt_1_variableBinning_denominator",
      "effic_muEta_1      'efficiency vs muon eta; muon eta ; efficiency'    muEta_1_variableBinning_numerator muEta_1_variableBinning_denominator",
      "effic_muPhi_1      'efficiency vs muon phi; muon phi ; efficiency'    muPhi_1_numerator                 muPhi_1_denominator",

      "effic_muPt_2       'efficiency vs muon pt; muon pt [GeV]; efficiency'  muPt_2_variableBinning_numerator  muPt_2_variableBinning_denominator",
      "effic_muEta_2      'efficiency vs muon eta; muon eta ; efficiency'    muEta_2_variableBinning_numerator muEta_2_variableBinning_denominator",
      "effic_muPhi_2      'efficiency vs muon phi; muon phi ; efficiency'    muPhi_2_numerator                 muPhi_2_denominator",

      "effic_muPt_3       'efficiency vs muon pt; muon pt [GeV]; efficiency'  muPt_3_variableBinning_numerator  muPt_3_variableBinning_denominator",
      "effic_muEta_3      'efficiency vs muon eta; muon eta ; efficiency'    muEta_3_variableBinning_numerator muEta_3_variableBinning_denominator",
      "effic_muPhi_3      'efficiency vs muon phi; muon phi ; efficiency'    muPhi_3_numerator                 muPhi_3_denominator",
    ),
)

# triple dca
triple_muon_dca_mupt_efficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSY/SOS/TripleMu/DCA/Muon"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages 
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
      "effic_muPt_1       'efficiency vs muon pt; muon pt [GeV]; efficiency'  muPt_1_variableBinning_numerator  muPt_1_variableBinning_denominator",
      "effic_muEta_1      'efficiency vs muon eta; muon eta ; efficiency'    muEta_1_variableBinning_numerator muEta_1_variableBinning_denominator",
      "effic_muPhi_1      'efficiency vs muon phi; muon phi ; efficiency'    muPhi_1_numerator                 muPhi_1_denominator",

      "effic_muPt_2       'efficiency vs muon pt; muon pt [GeV]; efficiency'  muPt_2_variableBinning_numerator  muPt_2_variableBinning_denominator",
      "effic_muEta_2      'efficiency vs muon eta; muon eta ; efficiency'    muEta_2_variableBinning_numerator muEta_2_variableBinning_denominator",
      "effic_muPhi_2      'efficiency vs muon phi; muon phi ; efficiency'    muPhi_2_numerator                 muPhi_2_denominator",

      "effic_muPt_3       'efficiency vs muon pt; muon pt [GeV]; efficiency'  muPt_3_variableBinning_numerator  muPt_3_variableBinning_denominator",
      "effic_muEta_3      'efficiency vs muon eta; muon eta ; efficiency'    muEta_3_variableBinning_numerator muEta_3_variableBinning_denominator",
      "effic_muPhi_3      'efficiency vs muon phi; muon phi ; efficiency'    muPhi_3_numerator                 muPhi_3_denominator",
    ),
)

# dca double muon
double_soft_dca_muonpt_efficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSY/SOS/DCA/Muon"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages 
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
      "effic_muPt_1       'efficiency vs muon pt; muon pt [GeV]; efficiency'  muPt_1_variableBinning_numerator  muPt_1_variableBinning_denominator",
      "effic_muEta_1      'efficiency vs muon eta; muon eta ; efficiency'    muEta_1_variableBinning_numerator muEta_1_variableBinning_denominator",
      "effic_muPhi_1      'efficiency vs muon phi; muon phi ; efficiency'    muPhi_1_numerator                 muPhi_1_denominator",

      "effic_muPt_2       'efficiency vs muon pt; muon pt [GeV]; efficiency'  muPt_2_variableBinning_numerator  muPt_2_variableBinning_denominator",
      "effic_muEta_2      'efficiency vs muon eta; muon eta ; efficiency'    muEta_2_variableBinning_numerator muEta_2_variableBinning_denominator",
      "effic_muPhi_2      'efficiency vs muon phi; muon phi ; efficiency'    muPhi_2_numerator                 muPhi_2_denominator",

      "effic_mu1mu2Pt     'efficiency vs mu1mu2 Pt; mu1 Pt [GeV]; mu2 Pt [GeV]'   mu1Pt_mu2Pt_numerator   mu1Pt_mu2Pt_denominator",
      "effic_mu1mu2Eta    'efficiency vs mu1mu2 Eta; mu1 Eta ; mu2 Eta'         mu1Eta_mu2Eta_numerator mu1Eta_mu2Eta_denominator",
    ),
)

double_soft_dca_metpt_efficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSY/SOS/DCA/MET"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages 
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
      "effic_met       'efficiency vs met pt; met [GeV]; efficiency' met_numerator       met_denominator",
    ),
)

susyClient = cms.Sequence(
    vbfsusyClient
  + LepHTClient
  + susyHLTEleCaloJetsClient
  + double_soft_muon_muonpt_efficiency
  + double_soft_muon_metpt_efficiency
  + double_soft_muon_mll_efficiency
  + double_soft_muon_mhtpt_efficiency
  + double_soft_muon_backup_70_metpt_efficiency
  + double_soft_muon_backup_70_mhtpt_efficiency
  + double_soft_muon_backup_90_metpt_efficiency
  + double_soft_muon_backup_90_mhtpt_efficiency
  + double_soft_dca_muonpt_efficiency
  + double_soft_dca_metpt_efficiency
  + susyHLTRazorClient
  + triple_muon_mupt_efficiency
  + triple_muon_dca_mupt_efficiency
  + susyHLTSoftMuHardJetMETClient
)
