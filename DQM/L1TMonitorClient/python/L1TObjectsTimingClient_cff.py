import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

# directory path shortening
l1tobjectstimingDqmDir = 'L1T/L1TObjects/'

# L1TObjects Timing Ratio Plots
l1tObjectsRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir),
    efficiency = cms.vstring(
       "L1TMuon/timing/Ratio_L1TMuons_BX_minus2 'Ratio for L1TMuons for BX=-2' L1TMuon/timing/muons_eta_phi_bx_minus2 L1TMuon/timing/denominator_muons",
       "L1TMuon/timing/Ratio_L1TMuons_BX_minus1 'Ratio for L1TMuons for BX=-1' L1TMuon/timing/muons_eta_phi_bx_minus1 L1TMuon/timing/denominator_muons",
       "L1TMuon/timing/Ratio_L1TMuons_BX_0 'Ratio for L1TMuons for BX=0' L1TMuon/timing/muons_eta_phi_bx_0 L1TMuon/timing/denominator_muons",
       "L1TMuon/timing/Ratio_L1TMuons_BX_plus1 'Ratio for L1TMuons for BX=+1' L1TMuon/timing/muons_eta_phi_bx_plus1 L1TMuon/timing/denominator_muons",
       "L1TMuon/timing/Ratio_L1TMuons_BX_plus2 'Ratio for L1TMuons for BX=+2' L1TMuon/timing/muons_eta_phi_bx_plus2 L1TMuon/timing/denominator_muons",
       "L1TJet/timing/Ratio_L1TJet_BX_minus2 'Ratio for L1TJet for BX=-2' L1TJet/timing/jet_eta_phi_bx_minus2 L1TJet/timing/denominator_jet",
       "L1TJet/timing/Ratio_L1TJet_BX_minus1 'Ratio for L1TJet for BX=-1' L1TJet/timing/jet_eta_phi_bx_minus1 L1TJet/timing/denominator_jet",
       "L1TJet/timing/Ratio_L1TJet_BX_0 'Ratio for L1TJet for BX=0' L1TJet/timing/jet_eta_phi_bx_0 L1TJet/timing/denominator_jet",
       "L1TJet/timing/Ratio_L1TJet_BX_plus1 'Ratio for L1TJet for BX=+1' L1TJet/timing/jet_eta_phi_bx_plus1 L1TJet/timing/denominator_jet",
       "L1TJet/timing/Ratio_L1TJet_BX_plus2 'Ratio for L1TJet for BX=+2' L1TJet/timing/jet_eta_phi_bx_plus2 L1TJet/timing/denominator_jet",
       "L1TEGamma/timing/Ratio_L1TEGamma_BX_minus2 'Ratio for L1TEGamma for BX=-2' L1TEGamma/timing/egamma_eta_phi_bx_minus2 L1TEGamma/timing/denominator_egamma",
       "L1TEGamma/timing/Ratio_L1TEGamma_BX_minus1 'Ratio for L1TEGamma for BX=-1' L1TEGamma/timing/egamma_eta_phi_bx_minus1 L1TEGamma/timing/denominator_egamma",
       "L1TEGamma/timing/Ratio_L1TEGamma_BX_0 'Ratio for L1TEGamma for BX=0' L1TEGamma/timing/egamma_eta_phi_bx_0 L1TEGamma/timing/denominator_egamma",
       "L1TEGamma/timing/Ratio_L1TEGamma_BX_plus1 'Ratio for L1TEGamma for BX=+1' L1TEGamma/timing/egamma_eta_phi_bx_plus1 L1TEGamma/timing/denominator_egamma",
       "L1TEGamma/timing/Ratio_L1TEGamma_BX_plus2 'Ratio for L1TEGamma for BX=+2' L1TEGamma/timing/egamma_eta_phi_bx_plus2 L1TEGamma/timing/denominator_egamma",
       "L1TTau/timing/Ratio_L1TTau_BX_minus2 'Ratio for L1TTau for BX=-2' L1TTau/timing/tau_eta_phi_bx_minus2 L1TTau/timing/denominator_tau",
       "L1TTau/timing/Ratio_L1TTau_BX_minus1 'Ratio for L1TTau for BX=-1' L1TTau/timing/tau_eta_phi_bx_minus1 L1TTau/timing/denominator_tau",
       "L1TTau/timing/Ratio_L1TTau_BX_0 'Ratio for L1TTau for BX=0' L1TTau/timing/tau_eta_phi_bx_0 L1TTau/timing/denominator_tau",
       "L1TTau/timing/Ratio_L1TTau_BX_plus1 'Ratio for L1TTau for BX=+1' L1TTau/timing/tau_eta_phi_bx_plus1 L1TTau/timing/denominator_tau",
       "L1TTau/timing/Ratio_L1TTau_BX_plus2 'Ratio for L1TTau for BX=+2' L1TTau/timing/tau_eta_phi_bx_plus2 L1TTau/timing/denominator_tau",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_MET_minus2 'Ratio for L1TEtSum MET for BX=-2' L1TEtSum/timing/etsum_phi_bx_MET_minus2 L1TEtSum/timing/denominator_etsum_MET",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_MET_minus1 'Ratio for L1TEtSum MET for BX=-1' L1TEtSum/timing/etsum_phi_bx_MET_minus1 L1TEtSum/timing/denominator_etsum_MET",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_MET_0 'Ratio for L1TEtSum MET for BX=0' L1TEtSum/timing/etsum_phi_bx_MET_0 L1TEtSum/timing/denominator_etsum_MET",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_MET_plus1 'Ratio for L1TEtSum MET for BX=+1' L1TEtSum/timing/etsum_phi_bx_MET_plus1 L1TEtSum/timing/denominator_etsum_MET",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_MET_plus2 'Ratio for L1TEtSum MET for BX=+2' L1TEtSum/timing/etsum_phi_bx_MET_plus2 L1TEtSum/timing/denominator_etsum_MET",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_METHF_minus2 'Ratio for L1TEtSum METHF for BX=-2' L1TEtSum/timing/etsum_phi_bx_METHF_minus2 L1TEtSum/timing/denominator_etsum_METHF",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_METHF_minus1 'Ratio for L1TEtSum METHF for BX=-1' L1TEtSum/timing/etsum_phi_bx_METHF_minus1 L1TEtSum/timing/denominator_etsum_METHF",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_METHF_0 'Ratio for L1TEtSum METHF for BX=0' L1TEtSum/timing/etsum_phi_bx_METHF_0 L1TEtSum/timing/denominator_etsum_METHF",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_METHF_plus1 'Ratio for L1TEtSum METHF for BX=+1' L1TEtSum/timing/etsum_phi_bx_METHF_plus1 L1TEtSum/timing/denominator_etsum_METHF",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_METHF_plus2 'Ratio for L1TEtSum METHF for BX=+2' L1TEtSum/timing/etsum_phi_bx_METHF_plus2 L1TEtSum/timing/denominator_etsum_METHF",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_MHTHF_minus2 'Ratio for L1TEtSum MHTHF for BX=-2' L1TEtSum/timing/etsum_phi_bx_MHTHF_minus2 L1TEtSum/timing/denominator_etsum_MHTHF",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_MHTHF_minus1 'Ratio for L1TEtSum MHTHF for BX=-1' L1TEtSum/timing/etsum_phi_bx_MHTHF_minus1 L1TEtSum/timing/denominator_etsum_MHTHF",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_MHTHF_0 'Ratio for L1TEtSum MHTHF for BX=0' L1TEtSum/timing/etsum_phi_bx_MHTHF_0 L1TEtSum/timing/denominator_etsum_MHTHF",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_MHTHF_plus1 'Ratio for L1TEtSum MHTHF for BX=+1' L1TEtSum/timing/etsum_phi_bx_MHTHF_plus1 L1TEtSum/timing/denominator_etsum_MHTHF",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_MHTHF_plus2 'Ratio for L1TEtSum MHTHF for BX=+2' L1TEtSum/timing/etsum_phi_bx_MHTHF_plus2 L1TEtSum/timing/denominator_etsum_MHTHF",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_MHT_minus2 'Ratio for L1TEtSum MHT for BX=-2' L1TEtSum/timing/etsum_phi_bx_MHT_minus2 L1TEtSum/timing/denominator_etsum_MHT",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_MHT_minus1 'Ratio for L1TEtSum MHT for BX=-1' L1TEtSum/timing/etsum_phi_bx_MHT_minus1 L1TEtSum/timing/denominator_etsum_MHT",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_MHT_0 'Ratio for L1TEtSum MHT for BX=0' L1TEtSum/timing/etsum_phi_bx_MHT_0 L1TEtSum/timing/denominator_etsum_MHT",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_MHT_plus1 'Ratio for L1TEtSum MHT for BX=+1' L1TEtSum/timing/etsum_phi_bx_MHT_plus1 L1TEtSum/timing/denominator_etsum_MHT",
       "L1TEtSum/timing/Ratio_L1TEtSum_BX_MHT_plus2 'Ratio for L1TEtSum MHT for BX=+2' L1TEtSum/timing/etsum_phi_bx_MHT_plus2 L1TEtSum/timing/denominator_etsum_MHT",
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)

)

l1tMuonFirstBunchMinus2RatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TMuon/timing/First_bunch_minus_2/'),
    efficiency = cms.vstring(
       "Ratio_L1TMuons_BX_FirstBunchMinus2_minus2 'Ratio for L1TMuons two buckets before train for BX=-2' muons_eta_phi_bx_firstbunchminus2_minus2 denominator_muons_firstbunchminus2",
       "Ratio_L1TMuons_BX_FirstBunchMinus2_minus1 'Ratio for L1TMuons two buckets before train for BX=-1' muons_eta_phi_bx_firstbunchminus2_minus1 denominator_muons_firstbunchminus2",
       "Ratio_L1TMuons_BX_FirstBunchMinus2_0 'Ratio for L1TMuons two buckets before train for BX=0' muons_eta_phi_bx_firstbunchminus2_0 denominator_muons_firstbunchminus2",
       "Ratio_L1TMuons_BX_FirstBunchMinus2_plus1 'Ratio for L1TMuons two buckets before train for BX=+1' muons_eta_phi_bx_firstbunchminus2_plus1 denominator_muons_firstbunchminus2",
       "Ratio_L1TMuons_BX_FirstBunchMinus2_plus2 'Ratio for L1TMuons two buckets before train for BX=+2' muons_eta_phi_bx_firstbunchminus2_plus2 denominator_muons_firstbunchminus2",
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tMuonFirstBunchMinus1RatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TMuon/timing/First_bunch_minus_1/'),
    efficiency = cms.vstring(
       "Ratio_L1TMuons_BX_FirstBunchMinus1_minus2 'Ratio for L1TMuons one bucket before train for BX=-2' muons_eta_phi_bx_firstbunchminus1_minus2 denominator_muons_firstbunchminus1",
       "Ratio_L1TMuons_BX_FirstBunchMinus1_minus1 'Ratio for L1TMuons one bucket before train for BX=-1' muons_eta_phi_bx_firstbunchminus1_minus1 denominator_muons_firstbunchminus1",
       "Ratio_L1TMuons_BX_FirstBunchMinus1_0 'Ratio for L1TMuons one bucket before train for BX=0' muons_eta_phi_bx_firstbunchminus1_0 denominator_muons_firstbunchminus1",
       "Ratio_L1TMuons_BX_FirstBunchMinus1_plus1 'Ratio for L1TMuons one bucket before train for BX=+1' muons_eta_phi_bx_firstbunchminus1_plus1 denominator_muons_firstbunchminus1",
       "Ratio_L1TMuons_BX_FirstBunchMinus1_plus2 'Ratio for L1TMuons one bucket before train for BX=+2' muons_eta_phi_bx_firstbunchminus1_plus2 denominator_muons_firstbunchminus1",
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tMuonFirstBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TMuon/timing/First_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TMuons_BX_FirstBunch_minus2 'Ratio for L1TMuons first bunch for BX=-2' muons_eta_phi_bx_firstbunch_minus2 denominator_muons_firstbunch",
       "Ratio_L1TMuons_BX_FirstBunch_minus1 'Ratio for L1TMuons first bunch for BX=-1' muons_eta_phi_bx_firstbunch_minus1 denominator_muons_firstbunch",
       "Ratio_L1TMuons_BX_FirstBunch_0 'Ratio for L1TMuons first bunch for BX=0' muons_eta_phi_bx_firstbunch_0 denominator_muons_firstbunch"
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tMuonLastBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TMuon/timing/Last_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TMuons_BX_LastBunch_0 'Ratio for L1TMuons last bunch for BX=0' muons_eta_phi_bx_lastbunch_0 denominator_muons_lastbunch",
       "Ratio_L1TMuons_BX_LastBunch_plus1 'Ratio for L1TMuons last bunch for BX=+1' muons_eta_phi_bx_lastbunch_plus1 denominator_muons_lastbunch",
       "Ratio_L1TMuons_BX_LastBunch_plus2 'Ratio for L1TMuons last bunch for BX=+2' muons_eta_phi_bx_lastbunch_plus2 denominator_muons_lastbunch"
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tMuonIsolatedBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TMuon/timing/Isolated_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TMuons_BX_Isolated_minus2 'Ratio for L1TMuons isolated bunch for BX=-2' muons_eta_phi_bx_isolated_minus2 denominator_muons_isolated",
       "Ratio_L1TMuons_BX_Isolated_minus1 'Ratio for L1TMuons isolated bunch for BX=-1' muons_eta_phi_bx_isolated_minus1 denominator_muons_isolated",
       "Ratio_L1TMuons_BX_Isolated_0 'Ratio for L1TMuons isolated bunch for BX=0' muons_eta_phi_bx_isolated_0 denominator_muons_isolated",
       "Ratio_L1TMuons_BX_Isolated_plus1 'Ratio for L1TMuons isolated bunch for BX=+1' muons_eta_phi_bx_isolated_plus1 denominator_muons_isolated",
       "Ratio_L1TMuons_BX_Isolated_plus2 'Ratio for L1TMuons isolated bunch for BX=+2' muons_eta_phi_bx_isolated_plus2 denominator_muons_isolated"
       ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

 
l1tJetFirstBunchMinus2RatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TJet/timing/First_bunch_minus_2/'),
    efficiency = cms.vstring(
       "Ratio_L1TJets_BX_FirstBunchMinus2_minus2 'Ratio for L1TJets two buckets before train for BX=-2' jet_eta_phi_bx_firstbunchminus2_minus2 denominator_jet_firstbunchminus2",
       "Ratio_L1TJets_BX_FirstBunchMinus2_minus1 'Ratio for L1TJets two buckets before train for BX=-1' jet_eta_phi_bx_firstbunchminus2_minus1 denominator_jet_firstbunchminus2",
       "Ratio_L1TJets_BX_FirstBunchMinus2_0 'Ratio for L1TJets two buckets before train for BX=0' jet_eta_phi_bx_firstbunchminus2_0 denominator_jet_firstbunchminus2",
       "Ratio_L1TJets_BX_FirstBunchMinus2_plus1 'Ratio for L1TJets two buckets before train for BX=+1' jet_eta_phi_bx_firstbunchminus2_plus1 denominator_jet_firstbunchminus2",
       "Ratio_L1TJets_BX_FirstBunchMinus2_plus2 'Ratio for L1TJets two buckets before train for BX=+2' jet_eta_phi_bx_firstbunchminus2_plus2 denominator_jet_firstbunchminus2",
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tJetFirstBunchMinus1RatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TJet/timing/First_bunch_minus_1/'),
    efficiency = cms.vstring(
       "Ratio_L1TJets_BX_FirstBunchMinus1_minus2 'Ratio for L1TJets one bucket before train for BX=-2' jet_eta_phi_bx_firstbunchminus1_minus2 denominator_jet_firstbunchminus1",
       "Ratio_L1TJets_BX_FirstBunchMinus1_minus1 'Ratio for L1TJets one bucket before train for BX=-1' jet_eta_phi_bx_firstbunchminus1_minus1 denominator_jet_firstbunchminus1",
       "Ratio_L1TJets_BX_FirstBunchMinus1_0 'Ratio for L1TJets one bucket before train for BX=0' jet_eta_phi_bx_firstbunchminus1_0 denominator_jet_firstbunchminus1",
       "Ratio_L1TJets_BX_FirstBunchMinus1_plus1 'Ratio for L1TJets one bucket before train for BX=+1' jet_eta_phi_bx_firstbunchminus1_plus1 denominator_jet_firstbunchminus1",
       "Ratio_L1TJets_BX_FirstBunchMinus1_plus2 'Ratio for L1TJets one bucket before train for BX=+2' jet_eta_phi_bx_firstbunchminus1_plus2 denominator_jet_firstbunchminus1",
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tJetFirstBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TJet/timing/First_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TJet_BX_FirstBunch_minus2 'Ratio for L1TJet first bunch for BX=-2' jet_eta_phi_bx_firstbunch_minus2 denominator_jet_firstbunch",
       "Ratio_L1TJet_BX_FirstBunch_minus1 'Ratio for L1TJet first bunch for BX=-1' jet_eta_phi_bx_firstbunch_minus1 denominator_jet_firstbunch",
       "Ratio_L1TJet_BX_FirstBunch_0 'Ratio for L1TJet first bunch for BX=0' jet_eta_phi_bx_firstbunch_0 denominator_jet_firstbunch"
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tJetLastBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TJet/timing/Last_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TJet_BX_LastBunch_0 'Ratio for L1TJet last bunch for BX=0' jet_eta_phi_bx_lastbunch_0 denominator_jet_lastbunch",
       "Ratio_L1TJet_BX_LastBunch_plus1 'Ratio for L1TJet last bunch for BX=+1' jet_eta_phi_bx_lastbunch_plus1 denominator_jet_lastbunch",
       "Ratio_L1TJet_BX_LastBunch_plus2 'Ratio for L1TJet last bunch for BX=+2' jet_eta_phi_bx_lastbunch_plus2 denominator_jet_lastbunch"
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tJetIsolatedBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TJet/timing/Isolated_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TJet_BX_Isolated_minus2 'Ratio for L1TJet isolated bunch for BX=-2' jet_eta_phi_bx_isolated_minus2 denominator_jet_isolated",
       "Ratio_L1TJet_BX_Isolated_minus1 'Ratio for L1TJet isolated bunch for BX=-1' jet_eta_phi_bx_isolated_minus1 denominator_jet_isolated",
       "Ratio_L1TJet_BX_Isolated_0 'Ratio for L1TJet isolated bunch for BX=0' jet_eta_phi_bx_isolated_0 denominator_jet_isolated",
       "Ratio_L1TJet_BX_Isolated_plus1 'Ratio for L1TJet isolated bunch for BX=+1' jet_eta_phi_bx_isolated_plus1 denominator_jet_isolated",
       "Ratio_L1TJet_BX_Isolated_plus2 'Ratio for L1TJet isolated bunch for BX=+2' jet_eta_phi_bx_isolated_plus2 denominator_jet_isolated"
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)


l1tEGammaFirstBunchMinus2RatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEGamma/timing/First_bunch_minus_2/'),
    efficiency = cms.vstring(
       "Ratio_L1TEGammas_BX_FirstBunchMinus2_minus2 'Ratio for L1TEGamma two buckets before train for BX=-2' egamma_eta_phi_bx_firstbunchminus2_minus2 denominator_egamma_firstbunchminus2",
       "Ratio_L1TEGammas_BX_FirstBunchMinus2_minus1 'Ratio for L1TEGamma two buckets before train for BX=-1' egamma_eta_phi_bx_firstbunchminus2_minus1 denominator_egamma_firstbunchminus2",
       "Ratio_L1TEGammas_BX_FirstBunchMinus2_0 'Ratio for L1TEGamma two buckets before train for BX=0' egamma_eta_phi_bx_firstbunchminus2_0 denominator_egamma_firstbunchminus2",
       "Ratio_L1TEGammas_BX_FirstBunchMinus2_plus1 'Ratio for L1TEGamma two buckets before train for BX=+1' egamma_eta_phi_bx_firstbunchminus2_plus1 denominator_egamma_firstbunchminus2",
       "Ratio_L1TEGammas_BX_FirstBunchMinus2_plus2 'Ratio for L1TEGamma two buckets before train for BX=+2' egamma_eta_phi_bx_firstbunchminus2_plus2 denominator_egamma_firstbunchminus2",
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tEGammaFirstBunchMinus1RatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEGamma/timing/First_bunch_minus_1/'),
    efficiency = cms.vstring(
       "Ratio_L1TEGammas_BX_FirstBunchMinus1_minus2 'Ratio for L1TEGamma one bucket before train for BX=-2' egamma_eta_phi_bx_firstbunchminus1_minus2 denominator_egamma_firstbunchminus1",
       "Ratio_L1TEGammas_BX_FirstBunchMinus1_minus1 'Ratio for L1TEGamma one bucket before train for BX=-1' egamma_eta_phi_bx_firstbunchminus1_minus1 denominator_egamma_firstbunchminus1",
       "Ratio_L1TEGammas_BX_FirstBunchMinus1_0 'Ratio for L1TEGamma one bucket before train for BX=0' egamma_eta_phi_bx_firstbunchminus1_0 denominator_egamma_firstbunchminus1",
       "Ratio_L1TEGammas_BX_FirstBunchMinus1_plus1 'Ratio for L1TEGamma one bucket before train for BX=+1' egamma_eta_phi_bx_firstbunchminus1_plus1 denominator_egamma_firstbunchminus1",
       "Ratio_L1TEGammas_BX_FirstBunchMinus1_plus2 'Ratio for L1TEGamma one bucket before train for BX=+2' egamma_eta_phi_bx_firstbunchminus1_plus2 denominator_egamma_firstbunchminus1",
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tEGammaFirstBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEGamma/timing/First_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TEGamma_ptmin_10_gev_BX_FirstBunch_minus2 'Ratio for L1TEGamma first bunch for BX=-2' ptmin_10_gev/egamma_eta_phi_bx_firstbunch_minus2 ptmin_10_gev/denominator_egamma_firstbunch",
       "Ratio_L1TEGamma_ptmin_10_gev_BX_FirstBunch_minus1 'Ratio for L1TEGamma first bunch for BX=-1' ptmin_10_gev/egamma_eta_phi_bx_firstbunch_minus1 ptmin_10_gev/denominator_egamma_firstbunch",
       "Ratio_L1TEGamma_ptmin_10_gev_BX_FirstBunch_0 'Ratio for L1TEGamma first bunch for BX=0' ptmin_10_gev/egamma_eta_phi_bx_firstbunch_0 ptmin_10_gev/denominator_egamma_firstbunch",
       "Ratio_L1TEGamma_ptmin_20_gev_BX_FirstBunch_minus2 'Ratio for L1TEGamma first bunch for BX=-2' ptmin_20_gev/egamma_eta_phi_bx_firstbunch_minus2 ptmin_20_gev/denominator_egamma_firstbunch",
       "Ratio_L1TEGamma_ptmin_20_gev_BX_FirstBunch_minus1 'Ratio for L1TEGamma first bunch for BX=-1' ptmin_20_gev/egamma_eta_phi_bx_firstbunch_minus1 ptmin_20_gev/denominator_egamma_firstbunch",
       "Ratio_L1TEGamma_ptmin_20_gev_BX_FirstBunch_0 'Ratio for L1TEGamma first bunch for BX=0' ptmin_20_gev/egamma_eta_phi_bx_firstbunch_0 ptmin_20_gev/denominator_egamma_firstbunch",
       "Ratio_L1TEGamma_ptmin_30_gev_BX_FirstBunch_minus2 'Ratio for L1TEGamma first bunch for BX=-2' ptmin_30_gev/egamma_eta_phi_bx_firstbunch_minus2 ptmin_30_gev/denominator_egamma_firstbunch",
       "Ratio_L1TEGamma_ptmin_30_gev_BX_FirstBunch_minus1 'Ratio for L1TEGamma first bunch for BX=-1' ptmin_30_gev/egamma_eta_phi_bx_firstbunch_minus1 ptmin_30_gev/denominator_egamma_firstbunch",
       "Ratio_L1TEGamma_ptmin_30_gev_BX_FirstBunch_0 'Ratio for L1TEGamma first bunch for BX=0' ptmin_30_gev/egamma_eta_phi_bx_firstbunch_0 ptmin_30_gev/denominator_egamma_firstbunch",
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tEGammaLastBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEGamma/timing/Last_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TEGamma_ptmin_10_gev_BX_LastBunch_0 'Ratio for L1TEGamma last bunch for BX=0' ptmin_10_gev/egamma_eta_phi_bx_lastbunch_0 ptmin_10_gev/denominator_egamma_lastbunch",
       "Ratio_L1TEGamma_ptmin_10_gev_BX_LastBunch_plus1 'Ratio for L1TEGamma last bunch for BX=+1' ptmin_10_gev/egamma_eta_phi_bx_lastbunch_plus1 ptmin_10_gev/denominator_egamma_lastbunch",
       "Ratio_L1TEGamma_ptmin_10_gev_BX_LastBunch_plus2 'Ratio for L1TEGamma last bunch for BX=+2' ptmin_10_gev/egamma_eta_phi_bx_lastbunch_plus2 ptmin_10_gev/denominator_egamma_lastbunch",
       "Ratio_L1TEGamma_ptmin_20_gev_BX_LastBunch_0 'Ratio for L1TEGamma last bunch for BX=0' ptmin_20_gev/egamma_eta_phi_bx_lastbunch_0 ptmin_20_gev/denominator_egamma_lastbunch",
       "Ratio_L1TEGamma_ptmin_20_gev_BX_LastBunch_plus1 'Ratio for L1TEGamma last bunch for BX=+1' ptmin_20_gev/egamma_eta_phi_bx_lastbunch_plus1 ptmin_20_gev/denominator_egamma_lastbunch",
       "Ratio_L1TEGamma_ptmin_20_gev_BX_LastBunch_plus2 'Ratio for L1TEGamma last bunch for BX=+2' ptmin_20_gev/egamma_eta_phi_bx_lastbunch_plus2 ptmin_20_gev/denominator_egamma_lastbunch",
       "Ratio_L1TEGamma_ptmin_30_gev_BX_LastBunch_0 'Ratio for L1TEGamma last bunch for BX=0' ptmin_30_gev/egamma_eta_phi_bx_lastbunch_0 ptmin_30_gev/denominator_egamma_lastbunch",
       "Ratio_L1TEGamma_ptmin_30_gev_BX_LastBunch_plus1 'Ratio for L1TEGamma last bunch for BX=+1' ptmin_30_gev/egamma_eta_phi_bx_lastbunch_plus1 ptmin_30_gev/denominator_egamma_lastbunch",
       "Ratio_L1TEGamma_ptmin_30_gev_BX_LastBunch_plus2 'Ratio for L1TEGamma last bunch for BX=+2' ptmin_30_gev/egamma_eta_phi_bx_lastbunch_plus2 ptmin_30_gev/denominator_egamma_lastbunch"
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tEGammaIsolatedBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEGamma/timing/Isolated_bunch/'),
    efficiency = cms.vstring( 
       "Ratio_L1TEGamma_ptmin_10_gev_BX_Isolated_minus2 'Ratio for L1TEGamma isolated bunch for BX=-2' ptmin_10_gev/egamma_eta_phi_bx_isolated_minus2 ptmin_10_gev/denominator_egamma_isolated",
       "Ratio_L1TEGamma_ptmin_10_gev_BX_Isolated_minus1 'Ratio for L1TEGamma isolated bunch for BX=-1' ptmin_10_gev/egamma_eta_phi_bx_isolated_minus1 ptmin_10_gev/denominator_egamma_isolated",
       "Ratio_L1TEGamma_ptmin_10_gev_BX_Isolated_0 'Ratio for L1TEGamma isolated bunch for BX=0' ptmin_10_gev/egamma_eta_phi_bx_isolated_0 ptmin_10_gev/denominator_egamma_isolated",
       "Ratio_L1TEGamma_ptmin_10_gev_BX_Isolated_plus1 'Ratio for L1TEGamma isolated bunch for BX=+1' ptmin_10_gev/egamma_eta_phi_bx_isolated_plus1 ptmin_10_gev/denominator_egamma_isolated",
       "Ratio_L1TEGamma_ptmin_10_gev_BX_Isolated_plus2 'Ratio for L1TEGamma isolated bunch for BX=+2' ptmin_10_gev/egamma_eta_phi_bx_isolated_plus2 ptmin_10_gev/denominator_egamma_isolated",
       "Ratio_L1TEGamma_ptmin_20_gev_BX_Isolated_minus2 'Ratio for L1TEGamma isolated bunch for BX=-2' ptmin_20_gev/egamma_eta_phi_bx_isolated_minus2 ptmin_20_gev/denominator_egamma_isolated",
       "Ratio_L1TEGamma_ptmin_20_gev_BX_Isolated_minus1 'Ratio for L1TEGamma isolated bunch for BX=-1' ptmin_20_gev/egamma_eta_phi_bx_isolated_minus1 ptmin_20_gev/denominator_egamma_isolated",
       "Ratio_L1TEGamma_ptmin_20_gev_BX_Isolated_0 'Ratio for L1TEGamma isolated bunch for BX=0' ptmin_20_gev/egamma_eta_phi_bx_isolated_0 ptmin_20_gev/denominator_egamma_isolated",
       "Ratio_L1TEGamma_ptmin_20_gev_BX_Isolated_plus1 'Ratio for L1TEGamma isolated bunch for BX=+1' ptmin_20_gev/egamma_eta_phi_bx_isolated_plus1 ptmin_20_gev/denominator_egamma_isolated",
       "Ratio_L1TEGamma_ptmin_20_gev_BX_Isolated_plus2 'Ratio for L1TEGamma isolated bunch for BX=+2' ptmin_20_gev/egamma_eta_phi_bx_isolated_plus2 ptmin_20_gev/denominator_egamma_isolated",
       "Ratio_L1TEGamma_ptmin_30_gev_BX_Isolated_minus2 'Ratio for L1TEGamma isolated bunch for BX=-2' ptmin_30_gev/egamma_eta_phi_bx_isolated_minus2 ptmin_30_gev/denominator_egamma_isolated",
       "Ratio_L1TEGamma_ptmin_30_gev_BX_Isolated_minus1 'Ratio for L1TEGamma isolated bunch for BX=-1' ptmin_30_gev/egamma_eta_phi_bx_isolated_minus1 ptmin_30_gev/denominator_egamma_isolated",
       "Ratio_L1TEGamma_ptmin_30_gev_BX_Isolated_0 'Ratio for L1TEGamma isolated bunch for BX=0' ptmin_30_gev/egamma_eta_phi_bx_isolated_0 ptmin_30_gev/denominator_egamma_isolated",
       "Ratio_L1TEGamma_ptmin_30_gev_BX_Isolated_plus1 'Ratio for L1TEGamma isolated bunch for BX=+1' ptmin_30_gev/egamma_eta_phi_bx_isolated_plus1 ptmin_30_gev/denominator_egamma_isolated",
       "Ratio_L1TEGamma_ptmin_30_gev_BX_Isolated_plus2 'Ratio for L1TEGamma isolated bunch for BX=+2' ptmin_30_gev/egamma_eta_phi_bx_isolated_plus2 ptmin_30_gev/denominator_egamma_isolated"
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)


l1tTauFirstBunchMinus2RatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TTau/timing/First_bunch_minus_2/'),
    efficiency = cms.vstring(
       "Ratio_L1TTaus_BX_FirstBunchMinus2_minus2 'Ratio for L1TTau two buckets before train for BX=-2' tau_eta_phi_bx_firstbunchminus2_minus2 denominator_tau_firstbunchminus2",
       "Ratio_L1TTaus_BX_FirstBunchMinus2_minus1 'Ratio for L1TTau two buckets before train for BX=-1' tau_eta_phi_bx_firstbunchminus2_minus1 denominator_tau_firstbunchminus2",
       "Ratio_L1TTaus_BX_FirstBunchMinus2_0 'Ratio for L1TTau two buckets before train for BX=0' tau_eta_phi_bx_firstbunchminus2_0 denominator_tau_firstbunchminus2",
       "Ratio_L1TTaus_BX_FirstBunchMinus2_plus1 'Ratio for L1TTau two buckets before train for BX=+1' tau_eta_phi_bx_firstbunchminus2_plus1 denominator_tau_firstbunchminus2",
       "Ratio_L1TTaus_BX_FirstBunchMinus2_plus2 'Ratio for L1TTau two buckets before train for BX=+2' tau_eta_phi_bx_firstbunchminus2_plus2 denominator_tau_firstbunchminus2",
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tTauFirstBunchMinus1RatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TTau/timing/First_bunch_minus_1/'),
    efficiency = cms.vstring(
       "Ratio_L1TTaus_BX_FirstBunchMinus1_minus2 'Ratio for L1TTau one bucket before train for BX=-2' tau_eta_phi_bx_firstbunchminus1_minus2 denominator_tau_firstbunchminus1",
       "Ratio_L1TTaus_BX_FirstBunchMinus1_minus1 'Ratio for L1TTau one bucket before train for BX=-1' tau_eta_phi_bx_firstbunchminus1_minus1 denominator_tau_firstbunchminus1",
       "Ratio_L1TTaus_BX_FirstBunchMinus1_0 'Ratio for L1TTau one bucket before train for BX=0' tau_eta_phi_bx_firstbunchminus1_0 denominator_tau_firstbunchminus1",
       "Ratio_L1TTaus_BX_FirstBunchMinus1_plus1 'Ratio for L1TTau one bucket before train for BX=+1' tau_eta_phi_bx_firstbunchminus1_plus1 denominator_tau_firstbunchminus1",
       "Ratio_L1TTaus_BX_FirstBunchMinus1_plus2 'Ratio for L1TTau one bucket before train for BX=+2' tau_eta_phi_bx_firstbunchminus1_plus2 denominator_tau_firstbunchminus1",
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tTauFirstBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TTau/timing/First_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TTau_BX_FirstBunch_minus2 'Ratio for L1TTau first bunch for BX=-2' tau_eta_phi_bx_firstbunch_minus2 denominator_tau_firstbunch",
       "Ratio_L1TTau_BX_FirstBunch_minus1 'Ratio for L1TTau first bunch for BX=-1' tau_eta_phi_bx_firstbunch_minus1 denominator_tau_firstbunch",
       "Ratio_L1TTau_BX_FirstBunch_0 'Ratio for L1TTau first bunch for BX=0' tau_eta_phi_bx_firstbunch_0 denominator_tau_firstbunch"
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tTauLastBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TTau/timing/Last_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TTau_BX_LastBunch_0 'Ratio for L1TTau last bunch for BX=0' tau_eta_phi_bx_lastbunch_0 denominator_tau_lastbunch",
       "Ratio_L1TTau_BX_LastBunch_plus1 'Ratio for L1TTau last bunch for BX=+1' tau_eta_phi_bx_lastbunch_plus1 denominator_tau_lastbunch",
       "Ratio_L1TTau_BX_LastBunch_plus2 'Ratio for L1TTau last bunch for BX=+2' tau_eta_phi_bx_lastbunch_plus2 denominator_tau_lastbunch"
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tTauIsolatedBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TTau/timing/Isolated_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TTau_BX_Isolated_minus2 'Ratio for L1TTau isolated bunch for BX=-2' tau_eta_phi_bx_isolated_minus2 denominator_tau_isolated",
       "Ratio_L1TTau_BX_Isolated_minus1 'Ratio for L1TTau isolated bunch for BX=-1' tau_eta_phi_bx_isolated_minus1 denominator_tau_isolated",
       "Ratio_L1TTau_BX_Isolated_0 'Ratio for L1TTau isolated bunch for BX=0' tau_eta_phi_bx_isolated_0 denominator_tau_isolated",
       "Ratio_L1TTau_BX_Isolated_plus1 'Ratio for L1TTau isolated bunch for BX=+1' tau_eta_phi_bx_isolated_plus1 denominator_tau_isolated",
       "Ratio_L1TTau_BX_Isolated_plus2 'Ratio for L1TTau isolated bunch for BX=+2' tau_eta_phi_bx_isolated_plus2 denominator_tau_isolated"
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)


l1tEtSumFirstBunchMinus2RatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEtSum/timing/First_bunch_minus_2/'),
    efficiency = cms.vstring(
       "Ratio_L1TEtSum_BX_MET_FirstBunchMinus2_minus2 'Ratio for L1TEtSum MET two buckets before train for BX=-2' etsum_phi_bx_MET_firstbunchminus2_minus2 denominator_etsum_firstbunchminus2_MET",
       "Ratio_L1TEtSum_BX_MET_FirstBunchMinus2_minus1 'Ratio for L1TEtSum MET two buckets before train for BX=-1' etsum_phi_bx_MET_firstbunchminus2_minus1 denominator_etsum_firstbunchminus2_MET",
       "Ratio_L1TEtSum_BX_MET_FirstBunchMinus2_0 'Ratio for L1TEtSum MET two buckets before train for BX=0' etsum_phi_bx_MET_firstbunchminus2_0 denominator_etsum_firstbunchminus2_MET",
       "Ratio_L1TEtSum_BX_MET_FirstBunchMinus2_plus1 'Ratio for L1TEtSum MET two buckets before train for BX=+1' etsum_phi_bx_MET_firstbunchminus2_plus1 denominator_etsum_firstbunchminus2_MET",
       "Ratio_L1TEtSum_BX_MET_FirstBunchMinus2_plus2 'Ratio for L1TEtSum MET two buckets before train for BX=+2' etsum_phi_bx_MET_firstbunchminus2_plus2 denominator_etsum_firstbunchminus2_MET",
       "Ratio_L1TEtSum_BX_METHF_FirstBunchMinus2_minus2 'Ratio for L1TEtSum METHF two buckets before train for BX=-2' etsum_phi_bx_METHF_firstbunchminus2_minus2 denominator_etsum_firstbunchminus2_METHF",
       "Ratio_L1TEtSum_BX_METHF_FirstBunchMinus2_minus1 'Ratio for L1TEtSum METHF two buckets before train for BX=-1' etsum_phi_bx_METHF_firstbunchminus2_minus1 denominator_etsum_firstbunchminus2_METHF",
       "Ratio_L1TEtSum_BX_METHF_FirstBunchMinus2_0 'Ratio for L1TEtSum METHF two buckets before train for BX=0' etsum_phi_bx_METHF_firstbunchminus2_0 denominator_etsum_firstbunchminus2_METHF",      
       "Ratio_L1TEtSum_BX_METHF_FirstBunchMinus2_plus1 'Ratio for L1TEtSum METHF two buckets before train for BX=+1' etsum_phi_bx_METHF_firstbunchminus2_plus1 denominator_etsum_firstbunchminus2_METHF",
       "Ratio_L1TEtSum_BX_METHF_FirstBunchMinus2_plus2 'Ratio for L1TEtSum METHF two buckets before train for BX=+2' etsum_phi_bx_METHF_firstbunchminus2_plus2 denominator_etsum_firstbunchminus2_METHF",
       "Ratio_L1TEtSum_BX_MHTHF_FirstBunchMinus2_minus2 'Ratio for L1TEtSum MHTHF two buckets before train for BX=-2' etsum_phi_bx_MHTHF_firstbunchminus2_minus2 denominator_etsum_firstbunchminus2_MHTHF",
       "Ratio_L1TEtSum_BX_MHTHF_FirstBunchMinus2_minus1 'Ratio for L1TEtSum MHTHF two buckets before train for BX=-1' etsum_phi_bx_MHTHF_firstbunchminus2_minus1 denominator_etsum_firstbunchminus2_MHTHF",
       "Ratio_L1TEtSum_BX_MHTHF_FirstBunchMinus2_0 'Ratio for L1TEtSum MHTHF two buckets before train for BX=0' etsum_phi_bx_MHTHF_firstbunchminus2_0 denominator_etsum_firstbunchminus2_MHTHF",
       "Ratio_L1TEtSum_BX_MHTHF_FirstBunchMinus2_plus1 'Ratio for L1TEtSum MHTHF two buckets before train for BX=+1' etsum_phi_bx_MHTHF_firstbunchminus2_plus1 denominator_etsum_firstbunchminus2_MHTHF",
       "Ratio_L1TEtSum_BX_MHTHF_FirstBunchMinus2_plus2 'Ratio for L1TEtSum MHTHF two buckets before train for BX=+2' etsum_phi_bx_MHTHF_firstbunchminus2_plus2 denominator_etsum_firstbunchminus2_MHTHF",
       "Ratio_L1TEtSum_BX_MHT_FirstBunchMinus2_minus2 'Ratio for L1TEtSum MHT two buckets before train for BX=-2' etsum_phi_bx_MHT_firstbunchminus2_minus2 denominator_etsum_firstbunchminus2_MHT",
       "Ratio_L1TEtSum_BX_MHT_FirstBunchMinus2_minus1 'Ratio for L1TEtSum MHT two buckets before train for BX=-1' etsum_phi_bx_MHT_firstbunchminus2_minus1 denominator_etsum_firstbunchminus2_MHT",
       "Ratio_L1TEtSum_BX_MHT_FirstBunchMinus2_0 'Ratio for L1TEtSum MHT two buckets before train for BX=0' etsum_phi_bx_MHT_firstbunchminus2_0 denominator_etsum_firstbunchminus2_MHT",
       "Ratio_L1TEtSum_BX_MHT_FirstBunchMinus2_plus1 'Ratio for L1TEtSum MHT two buckets before train for BX=+1' etsum_phi_bx_MHT_firstbunchminus2_plus1 denominator_etsum_firstbunchminus2_MHT",
       "Ratio_L1TEtSum_BX_MHT_FirstBunchMinus2_plus2 'Ratio for L1TEtSum MHT two buckets before train for BX=+2' etsum_phi_bx_MHT_firstbunchminus2_plus2 denominator_etsum_firstbunchminus2_MHT",
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tEtSumFirstBunchMinus1RatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEtSum/timing/First_bunch_minus_1/'),
    efficiency = cms.vstring(
       "Ratio_L1TEtSum_BX_MET_FirstBunchMinus1_minus2 'Ratio for L1TEtSum MET one bucket before train for BX=-2' etsum_phi_bx_MET_firstbunchminus1_minus2 denominator_etsum_firstbunchminus1_MET",
       "Ratio_L1TEtSum_BX_MET_FirstBunchMinus1_minus1 'Ratio for L1TEtSum MET one bucket before train for BX=-1' etsum_phi_bx_MET_firstbunchminus1_minus1 denominator_etsum_firstbunchminus1_MET",
       "Ratio_L1TEtSum_BX_MET_FirstBunchMinus1_0 'Ratio for L1TEtSum MET one bucket before train for BX=0' etsum_phi_bx_MET_firstbunchminus1_0 denominator_etsum_firstbunchminus1_MET",
       "Ratio_L1TEtSum_BX_MET_FirstBunchMinus1_plus1 'Ratio for L1TEtSum MET one bucket before train for BX=+1' etsum_phi_bx_MET_firstbunchminus1_plus1 denominator_etsum_firstbunchminus1_MET",
       "Ratio_L1TEtSum_BX_MET_FirstBunchMinus1_plus2 'Ratio for L1TEtSum MET one bucket before train for BX=+2' etsum_phi_bx_MET_firstbunchminus1_plus2 denominator_etsum_firstbunchminus1_MET",
       "Ratio_L1TEtSum_BX_METHF_FirstBunchMinus1_minus2 'Ratio for L1TEtSum METHF one bucket before train for BX=-2' etsum_phi_bx_METHF_firstbunchminus1_minus2 denominator_etsum_firstbunchminus1_METHF",
       "Ratio_L1TEtSum_BX_METHF_FirstBunchMinus1_minus1 'Ratio for L1TEtSum METHF one bucket before train for BX=-1' etsum_phi_bx_METHF_firstbunchminus1_minus1 denominator_etsum_firstbunchminus1_METHF",
       "Ratio_L1TEtSum_BX_METHF_FirstBunchMinus1_0 'Ratio for L1TEtSum METHF one bucket before train for BX=0' etsum_phi_bx_METHF_firstbunchminus1_0 denominator_etsum_firstbunchminus1_METHF",      
       "Ratio_L1TEtSum_BX_METHF_FirstBunchMinus1_plus1 'Ratio for L1TEtSum METHF one bucket before train for BX=+1' etsum_phi_bx_METHF_firstbunchminus1_plus1 denominator_etsum_firstbunchminus1_METHF",
       "Ratio_L1TEtSum_BX_METHF_FirstBunchMinus1_plus2 'Ratio for L1TEtSum METHF one bucket before train for BX=+2' etsum_phi_bx_METHF_firstbunchminus1_plus2 denominator_etsum_firstbunchminus1_METHF",
       "Ratio_L1TEtSum_BX_MHTHF_FirstBunchMinus1_minus2 'Ratio for L1TEtSum MHTHF one bucket before train for BX=-2' etsum_phi_bx_MHTHF_firstbunchminus1_minus2 denominator_etsum_firstbunchminus1_MHTHF",
       "Ratio_L1TEtSum_BX_MHTHF_FirstBunchMinus1_minus1 'Ratio for L1TEtSum MHTHF one bucket before train for BX=-1' etsum_phi_bx_MHTHF_firstbunchminus1_minus1 denominator_etsum_firstbunchminus1_MHTHF",
       "Ratio_L1TEtSum_BX_MHTHF_FirstBunchMinus1_0 'Ratio for L1TEtSum MHTHF one bucket before train for BX=0' etsum_phi_bx_MHTHF_firstbunchminus1_0 denominator_etsum_firstbunchminus1_MHTHF",
       "Ratio_L1TEtSum_BX_MHTHF_FirstBunchMinus1_plus1 'Ratio for L1TEtSum MHTHF one bucket before train for BX=+1' etsum_phi_bx_MHTHF_firstbunchminus1_plus1 denominator_etsum_firstbunchminus1_MHTHF",
       "Ratio_L1TEtSum_BX_MHTHF_FirstBunchMinus1_plus2 'Ratio for L1TEtSum MHTHF one bucket before train for BX=+2' etsum_phi_bx_MHTHF_firstbunchminus1_plus2 denominator_etsum_firstbunchminus1_MHTHF",
       "Ratio_L1TEtSum_BX_MHT_FirstBunchMinus1_minus2 'Ratio for L1TEtSum MHT one bucket before train for BX=-2' etsum_phi_bx_MHT_firstbunchminus1_minus2 denominator_etsum_firstbunchminus1_MHT",
       "Ratio_L1TEtSum_BX_MHT_FirstBunchMinus1_minus1 'Ratio for L1TEtSum MHT one bucket before train for BX=-1' etsum_phi_bx_MHT_firstbunchminus1_minus1 denominator_etsum_firstbunchminus1_MHT",
       "Ratio_L1TEtSum_BX_MHT_FirstBunchMinus1_0 'Ratio for L1TEtSum MHT one bucket before train for BX=0' etsum_phi_bx_MHT_firstbunchminus1_0 denominator_etsum_firstbunchminus1_MHT"
       "Ratio_L1TEtSum_BX_MHT_FirstBunchMinus1_plus1 'Ratio for L1TEtSum MHT one bucket before train for BX=+1' etsum_phi_bx_MHT_firstbunchminus1_plus1 denominator_etsum_firstbunchminus1_MHT",
       "Ratio_L1TEtSum_BX_MHT_FirstBunchMinus1_plus2 'Ratio for L1TEtSum MHT one bucket before train for BX=+2' etsum_phi_bx_MHT_firstbunchminus1_plus2 denominator_etsum_firstbunchminus1_MHT",
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tEtSumFirstBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEtSum/timing/First_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TEtSum_BX_MET_FirstBunch_minus2 'Ratio for L1TEtSum MET first bunch for BX=-2' etsum_phi_bx_MET_firstbunch_minus2 denominator_etsum_firstbunch_MET",
       "Ratio_L1TEtSum_BX_MET_FirstBunch_minus1 'Ratio for L1TEtSum MET first bunch for BX=-1' etsum_phi_bx_MET_firstbunch_minus1 denominator_etsum_firstbunch_MET",
       "Ratio_L1TEtSum_BX_MET_FirstBunch_0 'Ratio for L1TEtSum MET first bunch for BX=0' etsum_phi_bx_MET_firstbunch_0 denominator_etsum_firstbunch_MET",
       "Ratio_L1TEtSum_BX_METHF_FirstBunch_minus2 'Ratio for L1TEtSum METHF first bunch for BX=-2' etsum_phi_bx_METHF_firstbunch_minus2 denominator_etsum_firstbunch_METHF",
       "Ratio_L1TEtSum_BX_METHF_FirstBunch_minus1 'Ratio for L1TEtSum METHF first bunch for BX=-1' etsum_phi_bx_METHF_firstbunch_minus1 denominator_etsum_firstbunch_METHF",
       "Ratio_L1TEtSum_BX_METHF_FirstBunch_0 'Ratio for L1TEtSum METHF first bunch for BX=0' etsum_phi_bx_METHF_firstbunch_0 denominator_etsum_firstbunch_METHF",      
       "Ratio_L1TEtSum_BX_MHTHF_FirstBunch_minus2 'Ratio for L1TEtSum MHTHF first bunch for BX=-2' etsum_phi_bx_MHTHF_firstbunch_minus2 denominator_etsum_firstbunch_MHTHF",
       "Ratio_L1TEtSum_BX_MHTHF_FirstBunch_minus1 'Ratio for L1TEtSum MHTHF first bunch for BX=-1' etsum_phi_bx_MHTHF_firstbunch_minus1 denominator_etsum_firstbunch_MHTHF",
       "Ratio_L1TEtSum_BX_MHTHF_FirstBunch_0 'Ratio for L1TEtSum MHTHF first bunch for BX=0' etsum_phi_bx_MHTHF_firstbunch_0 denominator_etsum_firstbunch_MHTHF",
       "Ratio_L1TEtSum_BX_MHT_FirstBunch_minus2 'Ratio for L1TEtSum MHT first bunch for BX=-2' etsum_phi_bx_MHT_firstbunch_minus2 denominator_etsum_firstbunch_MHT",
       "Ratio_L1TEtSum_BX_MHT_FirstBunch_minus1 'Ratio for L1TEtSum MHT first bunch for BX=-1' etsum_phi_bx_MHT_firstbunch_minus1 denominator_etsum_firstbunch_MHT",
       "Ratio_L1TEtSum_BX_MHT_FirstBunch_0 'Ratio for L1TEtSum MHT first bunch for BX=0' etsum_phi_bx_MHT_firstbunch_0 denominator_etsum_firstbunch_MHT"
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tEtSumLastBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEtSum/timing/Last_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TEtSum_BX_MET_LastBunch_0 'Ratio for L1TEtSum MET last bunch for BX=0' etsum_phi_bx_MET_lastbunch_0 denominator_etsum_lastbunch_MET",
       "Ratio_L1TEtSum_BX_MET_LastBunch_plus1 'Ratio for L1TEtSum MET last bunch for BX=+1' etsum_phi_bx_MET_lastbunch_plus1 denominator_etsum_lastbunch_MET",
       "Ratio_L1TEtSum_BX_MET_LastBunch_plus2 'Ratio for L1TEtSum MET last bunch for BX=+2' etsum_phi_bx_MET_lastbunch_plus2 denominator_etsum_lastbunch_MET",
       "Ratio_L1TEtSum_BX_METHF_LastBunch_minus2 'Ratio for L1TEtSum METHF last bunch for BX=-2' etsum_phi_bx_METHF_lastbunch_minus2 denominator_etsum_lastbunch_METHF",
       "Ratio_L1TEtSum_BX_METHF_LastBunch_minus1 'Ratio for L1TEtSum METHF last bunch for BX=-1' etsum_phi_bx_METHF_lastbunch_minus1 denominator_etsum_lastbunch_METHF",
       "Ratio_L1TEtSum_BX_METHF_LastBunch_0 'Ratio for L1TEtSum METHF last bunch for BX=0' etsum_phi_bx_METHF_lastbunch_0 denominator_etsum_lastbunch_METHF",
       "Ratio_L1TEtSum_BX_MHTHF_LastBunch_0 'Ratio for L1TEtSum MHTHF last bunch for BX=0' etsum_phi_bx_MHTHF_lastbunch_0 denominator_etsum_lastbunch_MHTHF",
       "Ratio_L1TEtSum_BX_MHTHF_LastBunch_plus1 'Ratio for L1TEtSum MHTHF last bunch for BX=+1' etsum_phi_bx_MHTHF_lastbunch_plus1 denominator_etsum_lastbunch_MHTHF",
       "Ratio_L1TEtSum_BX_MHTHF_LastBunch_plus2 'Ratio for L1TEtSum MHTHF last bunch for BX=+2' etsum_phi_bx_MHTHF_lastbunch_plus2 denominator_etsum_lastbunch_MHTHF",
       "Ratio_L1TEtSum_BX_MHT_LastBunch_0 'Ratio for L1TEtSum MHT last bunch for BX=0' etsum_phi_bx_MHT_lastbunch_0 denominator_etsum_lastbunch_MHT",
       "Ratio_L1TEtSum_BX_MHT_LastBunch_plus1 'Ratio for L1TEtSum MHT last bunch for BX=+1' etsum_phi_bx_MHT_lastbunch_plus1 denominator_etsum_lastbunch_MHT",
       "Ratio_L1TEtSum_BX_MHT_LastBunch_plus2 'Ratio for L1TEtSum MHT last bunch for BX=+2' etsum_phi_bx_MHT_lastbunch_plus2 denominator_etsum_lastbunch_MHT"
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

l1tEtSumIsolatedBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEtSum/timing/Isolated_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TEtSum_BX_MET_Isolated_minus2 'Ratio for L1TEtSum MET isolated bunch for BX=-2' etsum_phi_bx_MET_isolated_minus2 denominator_etsum_isolated_MET",
       "Ratio_L1TEtSum_BX_MET_Isolated_minus1 'Ratio for L1TEtSum MET isolated bunch for BX=-1' etsum_phi_bx_MET_isolated_minus1 denominator_etsum_isolated_MET",
       "Ratio_L1TEtSum_BX_MET_Isolated_0 'Ratio for L1TEtSum MET isolated bunch for BX=0' etsum_phi_bx_MET_isolated_0 denominator_etsum_isolated_MET",
       "Ratio_L1TEtSum_BX_MET_Isolated_plus1 'Ratio for L1TEtSum MET isolated bunch for BX=+1' etsum_phi_bx_MET_isolated_plus1 denominator_etsum_isolated_MET",
       "Ratio_L1TEtSum_BX_MET_Isolated_plus2 'Ratio for L1TEtSum MET isolated bunch for BX=+2' etsum_phi_bx_MET_isolated_plus2 denominator_etsum_isolated_MET", 
       "Ratio_L1TEtSum_BX_METHF_Isolated_minus2 'Ratio for L1TEtSum METHF isolated bunch for BX=-2' etsum_phi_bx_METHF_isolated_minus2 denominator_etsum_isolated_METHF",
       "Ratio_L1TEtSum_BX_METHF_Isolated_minus1 'Ratio for L1TEtSum METHF isolated bunch for BX=-1' etsum_phi_bx_METHF_isolated_minus1 denominator_etsum_isolated_METHF",
       "Ratio_L1TEtSum_BX_METHF_Isolated_0 'Ratio for L1TEtSum METHF isolated bunch for BX=0' etsum_phi_bx_METHF_isolated_0 denominator_etsum_isolated_METHF",
       "Ratio_L1TEtSum_BX_METHF_Isolated_plus1 'Ratio for L1TEtSum METHF isolated bunch for BX=+1' etsum_phi_bx_METHF_isolated_plus1 denominator_etsum_isolated_METHF",
       "Ratio_L1TEtSum_BX_METHF_Isolated_plus2 'Ratio for L1TEtSum METHF isolated bunch for BX=+2' etsum_phi_bx_METHF_isolated_plus2 denominator_etsum_isolated_METHF",
       "Ratio_L1TEtSum_BX_MHTHF_Isolated_minus2 'Ratio for L1TEtSum MHTHF isolated bunch for BX=-2' etsum_phi_bx_MHTHF_isolated_minus2 denominator_etsum_isolated_MHTHF",
       "Ratio_L1TEtSum_BX_MHTHF_Isolated_minus1 'Ratio for L1TEtSum MHTHF isolated bunch for BX=-1' etsum_phi_bx_MHTHF_isolated_minus1 denominator_etsum_isolated_MHTHF",
       "Ratio_L1TEtSum_BX_MHTHF_Isolated_0 'Ratio for L1TEtSum MHTHF isolated bunch for BX=0' etsum_phi_bx_MHTHF_isolated_0 denominator_etsum_isolated_MHTHF",
       "Ratio_L1TEtSum_BX_MHTHF_Isolated_plus1 'Ratio for L1TEtSum MHTHF isolated bunch for BX=+1' etsum_phi_bx_MHTHF_isolated_plus1 denominator_etsum_isolated_MHTHF",
       "Ratio_L1TEtSum_BX_MHTHF_Isolated_plus2 'Ratio for L1TEtSum MHTHF isolated bunch for BX=+2' etsum_phi_bx_MHTHF_isolated_plus2 denominator_etsum_isolated_MHTHF",
       "Ratio_L1TEtSum_BX_MHT_Isolated_minus2 'Ratio for L1TEtSum MHT isolated bunch for BX=-2' etsum_phi_bx_MHT_isolated_minus2 denominator_etsum_isolated_MHT",
       "Ratio_L1TEtSum_BX_MHT_Isolated_minus1 'Ratio for L1TEtSum MHT isolated bunch for BX=-1' etsum_phi_bx_MHT_isolated_minus1 denominator_etsum_isolated_MHT",
       "Ratio_L1TEtSum_BX_MHT_Isolated_0 'Ratio for L1TEtSum MHT isolated bunch for BX=0' etsum_phi_bx_MHT_isolated_0 denominator_etsum_isolated_MHT",
       "Ratio_L1TEtSum_BX_MHT_Isolated_plus1 'Ratio for L1TEtSum MHT isolated bunch for BX=+1' etsum_phi_bx_MHT_isolated_plus1 denominator_etsum_isolated_MHT",
       "Ratio_L1TEtSum_BX_MHT_Isolated_plus2 'Ratio for L1TEtSum MHT isolated bunch for BX=+2' etsum_phi_bx_MHT_isolated_plus2 denominator_etsum_isolated_MHT"
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

# sequences
l1tObjectsTimingClient = cms.Sequence(
   l1tObjectsRatioPlots
+  l1tMuonFirstBunchMinus2RatioPlots
+  l1tMuonFirstBunchMinus1RatioPlots
+  l1tMuonFirstBunchRatioPlots
+  l1tMuonLastBunchRatioPlots 
+  l1tMuonIsolatedBunchRatioPlots
+  l1tJetFirstBunchMinus2RatioPlots
+  l1tJetFirstBunchMinus1RatioPlots
+  l1tJetFirstBunchRatioPlots
+  l1tJetLastBunchRatioPlots
+  l1tJetIsolatedBunchRatioPlots
+  l1tEGammaFirstBunchMinus2RatioPlots
+  l1tEGammaFirstBunchMinus1RatioPlots
+  l1tEGammaFirstBunchRatioPlots
+  l1tEGammaLastBunchRatioPlots
+  l1tEGammaIsolatedBunchRatioPlots
+  l1tTauFirstBunchMinus2RatioPlots
+  l1tTauFirstBunchMinus1RatioPlots
+  l1tTauFirstBunchRatioPlots
+  l1tTauLastBunchRatioPlots
+  l1tTauIsolatedBunchRatioPlots
+  l1tEtSumFirstBunchMinus2RatioPlots
+  l1tEtSumFirstBunchMinus1RatioPlots
+  l1tEtSumFirstBunchRatioPlots
+  l1tEtSumLastBunchRatioPlots
+  l1tEtSumIsolatedBunchRatioPlots

)
