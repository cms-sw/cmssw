import FWCore.ParameterSet.Config as cms

from DQMOffline.L1Trigger.L1TMuonDQMOffline_cfi import muonEfficiencyThresholds, muonEfficiencyThresholds_HI

plots = ["EffvsPt", "EffvsEta", "EffvsPhi",
        "EffvsPt_OPEN", "EffvsEta_OPEN", "EffvsPhi_OPEN",
        "EffvsPt_DOUBLE", "EffvsEta_DOUBLE", "EffvsPhi_DOUBLE",
        "EffvsPt_SINGLE", "EffvsEta_SINGLE", "EffvsPhi_SINGLE"]

allEfficiencyPlots = []
for plot in plots:
    for threshold in muonEfficiencyThresholds:
        plotName = '{0}_{1}'.format(plot, threshold)
        allEfficiencyPlots.append(plotName)

allEfficiencyPlots_HI = []
for plot in plots:
    for threshold in muonEfficiencyThresholds_HI:
        plotName = '{0}_{1}'.format(plot, threshold)
        allEfficiencyPlots_HI.append(plotName)

from DQMOffline.L1Trigger.L1TEfficiencyHarvesting_cfi import l1tEfficiencyHarvesting
l1tMuonDQMEfficiency = l1tEfficiencyHarvesting.clone(
    plotCfgs = cms.untracked.VPSet(
        cms.untracked.PSet(
            numeratorDir = cms.untracked.string("L1T/L1TMuon/numerators_and_denominators"),
            outputDir = cms.untracked.string("L1T/L1TMuon"),
            numeratorSuffix = cms.untracked.string("_Num"),
            denominatorSuffix = cms.untracked.string("_Den"),
            plots = cms.untracked.vstring(allEfficiencyPlots)
        )
    )
)

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

numDenDir = "numerators_and_denominators/"

l1tMuonDQMEfficiencyGeneric = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("L1T/L1TMuon/"),
    efficiency = cms.vstring(),
    efficiencyProfile = cms.untracked.vstring(
       "eff_eta_3_qualOpen    'L1 muon efficiency vs. #eta' "+numDenDir+"effNum_eta_3_qualOpen    "+numDenDir+"effDen_eta_3",
       "eff_eta_7_qualDouble  'L1 muon efficiency vs. #eta' "+numDenDir+"effNum_eta_7_qualDouble  "+numDenDir+"effDen_eta_7",
       "eff_eta_15_qualDouble 'L1 muon efficiency vs. #eta' "+numDenDir+"effNum_eta_15_qualDouble "+numDenDir+"effDen_eta_15",
       "eff_eta_25_qualSingle 'L1 muon efficiency vs. #eta' "+numDenDir+"effNum_eta_25_qualSingle "+numDenDir+"effDen_eta_25",

       "eff_phi_3_etaMin0_etaMax2p4_qualOpen 'L1 muon efficiency vs. #phi' "+numDenDir+"effNum_phi_3_etaMin0_etaMax2p4_qualOpen "+numDenDir+"effDen_phi_3_etaMin0_etaMax2p4",
       "eff_phi_3_etaMin0_etaMax0p83_qualOpen 'L1 muon efficiency vs. #phi' "+numDenDir+"effNum_phi_3_etaMin0_etaMax0p83_qualOpen "+numDenDir+"effDen_phi_3_etaMin0_etaMax0p83",
       "eff_phi_3_etaMin0p83_etaMax1p24_qualOpen 'L1 muon efficiency vs. #phi' "+numDenDir+"effNum_phi_3_etaMin0p83_etaMax1p24_qualOpen "+numDenDir+"effDen_phi_3_etaMin0p83_etaMax1p24",
       "eff_phi_3_etaMin1p24_etaMax2p4_qualOpen 'L1 muon efficiency vs. #phi' "+numDenDir+"effNum_phi_3_etaMin1p24_etaMax2p4_qualOpen "+numDenDir+"effDen_phi_3_etaMin1p24_etaMax2p4",
       "eff_phi_7_etaMin0_etaMax2p4_qualDouble 'L1 muon efficiency vs. #phi' "+numDenDir+"effNum_phi_7_etaMin0_etaMax2p4_qualDouble "+numDenDir+"effDen_phi_7_etaMin0_etaMax2p4",
       "eff_phi_7_etaMin0_etaMax0p83_qualDouble 'L1 muon efficiency vs. #phi' "+numDenDir+"effNum_phi_7_etaMin0_etaMax0p83_qualDouble "+numDenDir+"effDen_phi_7_etaMin0_etaMax0p83",
       "eff_phi_7_etaMin0p83_etaMax1p24_qualDouble 'L1 muon efficiency vs. #phi' "+numDenDir+"effNum_phi_7_etaMin0p83_etaMax1p24_qualDouble "+numDenDir+"effDen_phi_7_etaMin0p83_etaMax1p24",
       "eff_phi_7_etaMin1p24_etaMax2p4_qualDouble 'L1 muon efficiency vs. #phi' "+numDenDir+"effNum_phi_7_etaMin1p24_etaMax2p4_qualDouble "+numDenDir+"effDen_phi_7_etaMin1p24_etaMax2p4",
       "eff_phi_15_etaMin0_etaMax2p4_qualDouble 'L1 muon efficiency vs. #phi' "+numDenDir+"effNum_phi_15_etaMin0_etaMax2p4_qualDouble "+numDenDir+"effDen_phi_15_etaMin0_etaMax2p4",
       "eff_phi_15_etaMin0_etaMax0p83_qualDouble 'L1 muon efficiency vs. #phi' "+numDenDir+"effNum_phi_15_etaMin0_etaMax0p83_qualDouble "+numDenDir+"effDen_phi_15_etaMin0_etaMax0p83",
       "eff_phi_15_etaMin0p83_etaMax1p24_qualDouble 'L1 muon efficiency vs. #phi' "+numDenDir+"effNum_phi_15_etaMin0p83_etaMax1p24_qualDouble "+numDenDir+"effDen_phi_15_etaMin0p83_etaMax1p24",
       "eff_phi_15_etaMin1p24_etaMax2p4_qualDouble 'L1 muon efficiency vs. #phi' "+numDenDir+"effNum_phi_15_etaMin1p24_etaMax2p4_qualDouble "+numDenDir+"effDen_phi_15_etaMin1p24_etaMax2p4",
       "eff_phi_25_etaMin0_etaMax2p4_qualSingle 'L1 muon efficiency vs. #phi' "+numDenDir+"effNum_phi_25_etaMin0_etaMax2p4_qualSingle "+numDenDir+"effDen_phi_25_etaMin0_etaMax2p4",
       "eff_phi_25_etaMin0_etaMax0p83_qualSingle 'L1 muon efficiency vs. #phi' "+numDenDir+"effNum_phi_25_etaMin0_etaMax0p83_qualSingle "+numDenDir+"effDen_phi_25_etaMin0_etaMax0p83",
       "eff_phi_25_etaMin0p83_etaMax1p24_qualSingle 'L1 muon efficiency vs. #phi' "+numDenDir+"effNum_phi_25_etaMin0p83_etaMax1p24_qualSingle "+numDenDir+"effDen_phi_25_etaMin0p83_etaMax1p24",
       "eff_phi_25_etaMin1p24_etaMax2p4_qualSingle 'L1 muon efficiency vs. #phi' "+numDenDir+"effNum_phi_25_etaMin1p24_etaMax2p4_qualSingle "+numDenDir+"effDen_phi_25_etaMin1p24_etaMax2p4",

       "eff_pt_3_etaMin0_etaMax2p4_qualOpen 'L1 muon efficiency vs. p_{T}' "+numDenDir+"effNum_pt_3_etaMin0_etaMax2p4_qualOpen "+numDenDir+"effDen_pt_etaMin0_etaMax2p4",
       "eff_pt_3_etaMin0_etaMax0p83_qualOpen 'L1 muon efficiency vs. p_{T}' "+numDenDir+"effNum_pt_3_etaMin0_etaMax0p83_qualOpen "+numDenDir+"effDen_pt_etaMin0_etaMax0p83",
       "eff_pt_3_etaMin0p83_etaMax1p24_qualOpen 'L1 muon efficiency vs. p_{T}' "+numDenDir+"effNum_pt_3_etaMin0p83_etaMax1p24_qualOpen "+numDenDir+"effDen_pt_etaMin0p83_etaMax1p24",
       "eff_pt_3_etaMin1p24_etaMax2p4_qualOpen 'L1 muon efficiency vs. p_{T}' "+numDenDir+"effNum_pt_3_etaMin1p24_etaMax2p4_qualOpen "+numDenDir+"effDen_pt_etaMin1p24_etaMax2p4",
       "eff_pt_7_etaMin0_etaMax2p4_qualDouble 'L1 muon efficiency vs. p_{T}' "+numDenDir+"effNum_pt_7_etaMin0_etaMax2p4_qualDouble "+numDenDir+"effDen_pt_etaMin0_etaMax2p4",
       "eff_pt_7_etaMin0_etaMax0p83_qualDouble 'L1 muon efficiency vs. p_{T}' "+numDenDir+"effNum_pt_7_etaMin0_etaMax0p83_qualDouble "+numDenDir+"effDen_pt_etaMin0_etaMax0p83",
       "eff_pt_7_etaMin0p83_etaMax1p24_qualDouble 'L1 muon efficiency vs. p_{T}' "+numDenDir+"effNum_pt_7_etaMin0p83_etaMax1p24_qualDouble "+numDenDir+"effDen_pt_etaMin0p83_etaMax1p24",
       "eff_pt_7_etaMin1p24_etaMax2p4_qualDouble 'L1 muon efficiency vs. p_{T}' "+numDenDir+"effNum_pt_7_etaMin1p24_etaMax2p4_qualDouble "+numDenDir+"effDen_pt_etaMin1p24_etaMax2p4",
       "eff_pt_15_etaMin0_etaMax2p4_qualDouble 'L1 muon efficiency vs. p_{T}' "+numDenDir+"effNum_pt_15_etaMin0_etaMax2p4_qualDouble "+numDenDir+"effDen_pt_etaMin0_etaMax2p4",
       "eff_pt_15_etaMin0_etaMax0p83_qualDouble 'L1 muon efficiency vs. p_{T}' "+numDenDir+"effNum_pt_15_etaMin0_etaMax0p83_qualDouble "+numDenDir+"effDen_pt_etaMin0_etaMax0p83",
       "eff_pt_15_etaMin0p83_etaMax1p24_qualDouble 'L1 muon efficiency vs. p_{T}' "+numDenDir+"effNum_pt_15_etaMin0p83_etaMax1p24_qualDouble "+numDenDir+"effDen_pt_etaMin0p83_etaMax1p24",
       "eff_pt_15_etaMin1p24_etaMax2p4_qualDouble 'L1 muon efficiency vs. p_{T}' "+numDenDir+"effNum_pt_15_etaMin1p24_etaMax2p4_qualDouble "+numDenDir+"effDen_pt_etaMin1p24_etaMax2p4",
       "eff_pt_25_etaMin0_etaMax2p4_qualSingle 'L1 muon efficiency vs. p_{T}' "+numDenDir+"effNum_pt_25_etaMin0_etaMax2p4_qualSingle "+numDenDir+"effDen_pt_etaMin0_etaMax2p4",
       "eff_pt_25_etaMin0_etaMax0p83_qualSingle 'L1 muon efficiency vs. p_{T}' "+numDenDir+"effNum_pt_25_etaMin0_etaMax0p83_qualSingle "+numDenDir+"effDen_pt_etaMin0_etaMax0p83",
       "eff_pt_25_etaMin0p83_etaMax1p24_qualSingle 'L1 muon efficiency vs. p_{T}' "+numDenDir+"effNum_pt_25_etaMin0p83_etaMax1p24_qualSingle "+numDenDir+"effDen_pt_etaMin0p83_etaMax1p24",
       "eff_pt_25_etaMin1p24_etaMax2p4_qualSingle 'L1 muon efficiency vs. p_{T}' "+numDenDir+"effNum_pt_25_etaMin1p24_etaMax2p4_qualSingle "+numDenDir+"effDen_pt_etaMin1p24_etaMax2p4",

    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string(""),
    verbose = cms.untracked.uint32(0)
)

# modifications for the pp reference run
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
ppRef_2017.toModify(l1tMuonDQMEfficiency,
    plotCfgs = {0:dict(plots = allEfficiencyPlots_HI)}
)

