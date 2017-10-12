import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQMOffline.Trigger.VBFMETMonitor_Client_cff import *
from DQMOffline.Trigger.VBFTauMonitor_Client_cff import *
from DQMOffline.Trigger.MssmHbbBtagTriggerMonitor_Client_cfi import *
from DQMOffline.Trigger.MssmHbbMonitoring_Client_cfi import *
from DQMOffline.Trigger.PhotonMonitor_cff import *

metbtagEfficiency_met = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/Higgs/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_met          'MET turnON;            PF MET [GeV]; efficiency'     met_numerator          met_denominator",
y = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/Higgs/VBFHbb/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_jetPhi_1      'efficiency vs 1st jet phi; jet phi ; efficiency'    jetPhi_1_numerator      jetPhi_1_denominator",
        "effic_jetPhi_2      'efficiency vs 2nd jet phi; jet phi ; efficiency'    jetPhi_2_numerator      jetPhi_2_denominator",
        "effic_jetPhi_3      'efficiency vs 3rd jet phi; jet phi ; efficiency'    jetPhi_3_numerator      jetPhi_3_denominator",
        "effic_jetPhi_4      'efficiency vs 4th jet phi; jet phi ; efficiency'    jetPhi_4_numerator      jetPhi_4_denominator",
        #
        "effic_bjetPhi_1     'efficiency vs 1st b-jet phi; bjet phi ; efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetCSV_1     'efficiency vs 1st b-jet csv; bjet CSV; efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        #
        "effic_bjetPhi_2     'efficiency vs 2nd b-jet phi; bjet phi ; efficiency'  bjetPhi_2_numerator   bjetPhi_2_denominator",
        "effic_bjetCSV_2     'efficiency vs 2nd b-jet csv; bjet CSV; efficiency' bjetCSV_2_numerator  bjetCSV_2_denominator",
        #
        "effic_jetPt_1_variableBinning       'efficiency vs 1st jet pt; jet pt [GeV]; efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetPt_2_variableBinning       'efficiency vs 2nd jet pt; jet pt [GeV]; efficiency' jetPt_2_variableBinning_numerator       jetPt_2_variableBinning_denominator",
        "effic_jetPt_3_variableBinning       'efficiency vs 3rd jet pt; jet pt [GeV]; efficiency' jetPt_3_variableBinning_numerator       jetPt_3_variableBinning_denominator",
        "effic_jetPt_4_variableBinning       'efficiency vs 4th jet pt; jet pt [GeV]; efficiency' jetPt_4_variableBinning_numerator       jetPt_4_variableBinning_denominator",
        #
        "effic_jetEta_1_variableBinning       'efficiency vs 1st jet eta; jet eta ; efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        "effic_jetEta_2_variableBinning       'efficiency vs 2nd jet eta; jet eta ; efficiency' jetEta_2_variableBinning_numerator       jetEta_2_variableBinning_denominator",
        "effic_jetEta_3_variableBinning       'efficiency vs 3rd jet eta; jet eta ; efficiency' jetEta_3_variableBinning_numerator       jetEta_3_variableBinning_denominator",
        "effic_jetEta_4_variableBinning       'efficiency vs 4th jet eta; jet eta ; efficiency' jetEta_4_variableBinning_numerator       jetEta_4_variableBinning_denominator",
        #
        "effic_bjetPt_1_variableBinning   'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_variableBinning_numerator   bjetPt_1_variableBinning_denominator",
        "effic_bjetEta_1_variableBinning  'efficiency vs 1st b-jet eta; bjet eta ; efficiency' bjetEta_1_variableBinning_numerator     bjetEta_1_variableBinning_denominator",
        #
        "effic_bjetPt_2_variableBinning   'efficiency vs 2nd b-jet pt; bjet pt [GeV]; efficiency' bjetPt_2_variableBinning_numerator   bjetPt_2_variableBinning_denominator",
        "effic_bjetEta_2_variableBinning  'efficiency vs 2nd b-jet eta; bjet eta ; efficiency' bjetEta_2_variableBinning_numerator     bjetEta_2_variableBinning_denominator",
        #
        "effic_jetMulti       'efficiency vs jet multiplicity; jet multiplicity; efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity; bjet multiplicity; efficiency' bjetMulti_numerator   bjetMulti_denominator",
        #
        "effic_jetPtEta_1     'efficiency vs 1st jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetPtEta_2     'efficiency vs 2nd jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetPtEta_3     'efficiency vs 3rd jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_3_numerator       jetPtEta_3_denominator",
        "effic_jetPtEta_4     'efficiency vs 4th jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_4_numerator       jetPtEta_4_denominator",
        "effic_jetEtaPhi_1    'efficiency vs 1st jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetEtaPhi_2    'efficiency vs 2nd jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_jetEtaPhi_3    'efficiency vs 3rd jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_3_numerator       jetEtaPhi_3_denominator",
        "effic_jetEtaPhi_4    'efficiency vs 4th jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_4_numerator       jetEtaPhi_4_denominator",
        #
        "effic_bjetPtEta_1    'efficiency vs 1st b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        #
        "effic_bjetEtaPhi_1    'efficiency vs 1st b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",
        #
        "effic_bjetPtEta_2    'efficiency vs 2nd b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_2_numerator   bjetPtEta_2_denominator",
        #
        "effic_bjetEtaPhi_2    'efficiency vs 2nd b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_2_numerator  bjetEtaPhi_2_denominator",
        ),
)


higgsClient = cms.Sequence(
    diphotonEfficiency
  + VBFEfficiency
  + vbfmetClient
  + vbftauClient
  + ele23Ele12CaloIdLTrackIdLIsoVL_effdz
  + dimu9ele9caloIdLTrackIdLdz_effmu
  + dimu9ele9caloIdLTrackIdLdz_effele
  + dimu9ele9caloIdLTrackIdLdz_effdz
  + mu8diEle12CaloIdLTrackIdL_effmu
  + mu8diEle12CaloIdLTrackIdL_effele
  + mu8diEle12CaloIdLTrackIdL_effdz
  + ele16ele12ele8caloIdLTrackIdL
  + triplemu12mu10mu5
  + triplemu10mu5mu5DZ
  + mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_effele
  + mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_effmu
  + mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_effele
  + mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_effmu
  + metbtagEfficiency_met
  + metbtagEfficiency_btag
  + mssmHbbBtagTriggerEfficiency 
  + mssmHbbHLTEfficiency 
)
