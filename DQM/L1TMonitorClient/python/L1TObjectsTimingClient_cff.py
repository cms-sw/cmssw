import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

# directory path shortening
l1tobjectstimingDqmDir = 'L1T/L1TObjects/'

# L1TObjects Timing Ratio Plots
l1tMuonRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TMuon/timing/'),
    efficiency = cms.vstring(
       "Ratio_L1TMuons_BX_minus2 'Ratio for L1TMuons for BX=-2' muons_eta_phi_bx_minus2 den_muons_eta_phi_bx_minus2",
       "Ratio_L1TMuons_BX_minus1 'Ratio for L1TMuons for BX=-1' muons_eta_phi_bx_minus1 den_muons_eta_phi_bx_minus1",
       "Ratio_L1TMuons_BX_0 'Ratio for L1TMuons for BX=0' muons_eta_phi_bx_0 den_muons_eta_phi_bx_0",
       "Ratio_L1TMuons_BX_plus1 'Ratio for L1TMuons for BX=+1' muons_eta_phi_bx_plus1 den_muons_eta_phi_bx_plus1",
       "Ratio_L1TMuons_BX_plus2 'Ratio for L1TMuons for BX=+2' muons_eta_phi_bx_plus2 den_muons_eta_phi_bx_plus2"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tMuonFirstBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TMuon/timing/First_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TMuons_BX_FirstBunch_minus2 'Ratio for L1TMuons first bunch for BX=-2' muons_eta_phi_bx_firstbunch_minus2 den_muons_eta_phi_bx_firstbunch_minus2",
       "Ratio_L1TMuons_BX_FirstBunch_minus1 'Ratio for L1TMuons first bunch for BX=-1' muons_eta_phi_bx_firstbunch_minus1 den_muons_eta_phi_bx_firstbunch_minus1",
       "Ratio_L1TMuons_BX_FirstBunch_0 'Ratio for L1TMuons first bunch for BX=0' muons_eta_phi_bx_firstbunch_0 den_muons_eta_phi_bx_firstbunch_0"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tMuonLastBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TMuon/timing/Last_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TMuons_BX_LastBunch_0 'Ratio for L1TMuons last bunch for BX=0' muons_eta_phi_bx_lastbunch_0 den_muons_eta_phi_bx_lastbunch_0",
       "Ratio_L1TMuons_BX_LastBunch_plus1 'Ratio for L1TMuons last bunch for BX=+1' muons_eta_phi_bx_lastbunch_plus1 den_muons_eta_phi_bx_lastbunch_plus1",
       "Ratio_L1TMuons_BX_LastBunch_plus2 'Ratio for L1TMuons last bunch for BX=+2' muons_eta_phi_bx_lastbunch_plus2 den_muons_eta_phi_bx_lastbunch_plus2"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tMuonIsolatedBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TMuon/timing/Isolated_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TMuons_BX_Isolated_minus2 'Ratio for L1TMuons isolated bunch for BX=-2' muons_eta_phi_bx_isolated_minus2 den_muons_eta_phi_bx_isolated_minus2",
       "Ratio_L1TMuons_BX_Isolated_minus1 'Ratio for L1TMuons isolated bunch for BX=-1' muons_eta_phi_bx_isolated_minus1 den_muons_eta_phi_bx_isolated_minus1",
       "Ratio_L1TMuons_BX_Isolated_0 'Ratio for L1TMuons isolated bunch for BX=0' muons_eta_phi_bx_isolated_0 den_muons_eta_phi_bx_isolated_0",
       "Ratio_L1TMuons_BX_Isolated_plus1 'Ratio for L1TMuons isolated bunch for BX=+1' muons_eta_phi_bx_isolated_plus1 den_muons_eta_phi_bx_isolated_plus1",
       "Ratio_L1TMuons_BX_Isolated_plus2 'Ratio for L1TMuons isolated bunch for BX=+2' muons_eta_phi_bx_isolated_plus2 den_muons_eta_phi_bx_isolated_plus2"
       ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tJetRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TJet/timing/'),
    efficiency = cms.vstring(
       "Ratio_L1TJet_BX_minus2 'Ratio for L1TJet for BX=-2' jet_eta_phi_bx_minus2 den_jet_eta_phi_bx_minus2",
       "Ratio_L1TJet_BX_minus1 'Ratio for L1TJet for BX=-1' jet_eta_phi_bx_minus1 den_jet_eta_phi_bx_minus1",
       "Ratio_L1TJet_BX_0 'Ratio for L1TJet for BX=0' jet_eta_phi_bx_0 den_jet_eta_phi_bx_0",
       "Ratio_L1TJet_BX_plus1 'Ratio for L1TJet for BX=+1' jet_eta_phi_bx_plus1 den_jet_eta_phi_bx_plus1",
       "Ratio_L1TJet_BX_plus2 'Ratio for L1TJet for BX=+2' jet_eta_phi_bx_plus2 den_jet_eta_phi_bx_plus2"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)
 
l1tJetFirstBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TJet/timing/First_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TJet_BX_FirstBunch_minus2 'Ratio for L1TJet first bunch for BX=-2' jet_eta_phi_bx_firstbunch_minus2 den_jet_eta_phi_bx_firstbunch_minus2",
       "Ratio_L1TJet_BX_FirstBunch_minus1 'Ratio for L1TJet first bunch for BX=-1' jet_eta_phi_bx_firstbunch_minus1 den_jet_eta_phi_bx_firstbunch_minus1",
       "Ratio_L1TJet_BX_FirstBunch_0 'Ratio for L1TJet first bunch for BX=0' jet_eta_phi_bx_firstbunch_0 den_jet_eta_phi_bx_firstbunch_0"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tJetLastBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TJet/timing/Last_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TJet_BX_LastBunch_0 'Ratio for L1TJet last bunch for BX=0' jet_eta_phi_bx_lastbunch_0 den_jet_eta_phi_bx_lastbunch_0",
       "Ratio_L1TJet_BX_LastBunch_plus1 'Ratio for L1TJet last bunch for BX=+1' jet_eta_phi_bx_lastbunch_plus1 den_jet_eta_phi_bx_lastbunch_plus1",
       "Ratio_L1TJet_BX_LastBunch_plus2 'Ratio for L1TJet last bunch for BX=+2' jet_eta_phi_bx_lastbunch_plus2 den_jet_eta_phi_bx_lastbunch_plus2"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tJetIsolatedBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TJet/timing/Isolated_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TJet_BX_Isolated_minus2 'Ratio for L1TJet isolated bunch for BX=-2' jet_eta_phi_bx_isolated_minus2 den_jet_eta_phi_bx_isolated_minus2",
       "Ratio_L1TJet_BX_Isolated_minus1 'Ratio for L1TJet isolated bunch for BX=-1' jet_eta_phi_bx_isolated_minus1 den_jet_eta_phi_bx_isolated_minus1",
       "Ratio_L1TJet_BX_Isolated_0 'Ratio for L1TJet isolated bunch for BX=0' jet_eta_phi_bx_isolated_0 den_jet_eta_phi_bx_isolated_0",
       "Ratio_L1TJet_BX_Isolated_plus1 'Ratio for L1TJet isolated bunch for BX=+1' jet_eta_phi_bx_isolated_plus1 den_jet_eta_phi_bx_isolated_plus1",
       "Ratio_L1TJet_BX_Isolated_plus2 'Ratio for L1TJet isolated bunch for BX=+2' jet_eta_phi_bx_isolated_plus2 den_jet_eta_phi_bx_isolated_plus2"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tEGammaRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEGamma/timing/'),
    efficiency = cms.vstring(
       "Ratio_L1TEGamma_BX_minus2 'Ratio for L1TEGamma for BX=-2' egamma_eta_phi_bx_minus2 den_egamma_eta_phi_bx_minus2",
       "Ratio_L1TEGamma_BX_minus1 'Ratio for L1TEGamma for BX=-1' egamma_eta_phi_bx_minus1 den_egamma_eta_phi_bx_minus1",
       "Ratio_L1TEGamma_BX_0 'Ratio for L1TEGamma for BX=0' egamma_eta_phi_bx_0 den_egamma_eta_phi_bx_0",
       "Ratio_L1TEGamma_BX_plus1 'Ratio for L1TEGamma for BX=+1' egamma_eta_phi_bx_plus1 den_egamma_eta_phi_bx_plus1",
       "Ratio_L1TEGamma_BX_plus2 'Ratio for L1TEGamma for BX=+2' egamma_eta_phi_bx_plus2 den_egamma_eta_phi_bx_plus2"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tEGammaFirstBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEGamma/timing/First_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TEGamma_BX_FirstBunch_minus2 'Ratio for L1TEGamma first bunch for BX=-2' egamma_eta_phi_bx_firstbunch_minus2 den_egamma_eta_phi_bx_firstbunch_minus2",
       "Ratio_L1TEGamma_BX_FirstBunch_minus1 'Ratio for L1TEGamma first bunch for BX=-1' egamma_eta_phi_bx_firstbunch_minus1 den_egamma_eta_phi_bx_firstbunch_minus1",
       "Ratio_L1TEGamma_BX_FirstBunch_0 'Ratio for L1TEGamma first bunch for BX=0' egamma_eta_phi_bx_firstbunch_0 den_egamma_eta_phi_bx_firstbunch_0"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tEGammaLastBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEGamma/timing/Last_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TEGamma_BX_LastBunch_0 'Ratio for L1TEGamma last bunch for BX=0' egamma_eta_phi_bx_lastbunch_0 den_egamma_eta_phi_bx_lastbunch_0",
       "Ratio_L1TEGamma_BX_LastBunch_plus1 'Ratio for L1TEGamma last bunch for BX=+1' egamma_eta_phi_bx_lastbunch_plus1 den_egamma_eta_phi_bx_lastbunch_plus1",
       "Ratio_L1TEGamma_BX_LastBunch_plus2 'Ratio for L1TEGamma last bunch for BX=+2' egamma_eta_phi_bx_lastbunch_plus2 den_egamma_eta_phi_bx_lastbunch_plus2"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tEGammaIsolatedBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEGamma/timing/Isolated_bunch/'),
    efficiency = cms.vstring( 
      "Ratio_L1TEGamma_BX_Isolated_minus2 'Ratio for L1TEGamma isolated bunch for BX=-2' egamma_eta_phi_bx_isolated_minus2 den_egamma_eta_phi_bx_isolated_minus2",
       "Ratio_L1TEGamma_BX_Isolated_minus1 'Ratio for L1TEGamma isolated bunch for BX=-1' egamma_eta_phi_bx_isolated_minus1 den_egamma_eta_phi_bx_isolated_minus1",
       "Ratio_L1TEGamma_BX_Isolated_0 'Ratio for L1TEGamma isolated bunch for BX=0' egamma_eta_phi_bx_isolated_0 den_egamma_eta_phi_bx_isolated_0",
       "Ratio_L1TEGamma_BX_Isolated_plus1 'Ratio for L1TEGamma isolated bunch for BX=+1' egamma_eta_phi_bx_isolated_plus1 den_egamma_eta_phi_bx_isolated_plus1",
       "Ratio_L1TEGamma_BX_Isolated_plus2 'Ratio for L1TEGamma isolated bunch for BX=+2' egamma_eta_phi_bx_isolated_plus2 den_egamma_eta_phi_bx_isolated_plus2"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tTauRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TTau/timing/'),
    efficiency = cms.vstring(
       "Ratio_L1TTau_BX_minus2 'Ratio for L1TTau for BX=-2' tau_eta_phi_bx_minus2 den_tau_eta_phi_bx_minus2",
       "Ratio_L1TTau_BX_minus1 'Ratio for L1TTau for BX=-1' tau_eta_phi_bx_minus1 den_tau_eta_phi_bx_minus1",
       "Ratio_L1TTau_BX_0 'Ratio for L1TTau for BX=0' tau_eta_phi_bx_0 den_tau_eta_phi_bx_0",
       "Ratio_L1TTau_BX_plus1 'Ratio for L1TTau for BX=+1' tau_eta_phi_bx_plus1 den_tau_eta_phi_bx_plus1",
       "Ratio_L1TTau_BX_plus2 'Ratio for L1TTau for BX=+2' tau_eta_phi_bx_plus2 den_tau_eta_phi_bx_plus2"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tTauFirstBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TTau/timing/First_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TTau_BX_FirstBunch_minus2 'Ratio for L1TTau first bunch for BX=-2' tau_eta_phi_bx_firstbunch_minus2 den_tau_eta_phi_bx_firstbunch_minus2",
       "Ratio_L1TTau_BX_FirstBunch_minus1 'Ratio for L1TTau first bunch for BX=-1' tau_eta_phi_bx_firstbunch_minus1 den_tau_eta_phi_bx_firstbunch_minus1",
       "Ratio_L1TTau_BX_FirstBunch_0 'Ratio for L1TTau first bunch for BX=0' tau_eta_phi_bx_firstbunch_0 den_tau_eta_phi_bx_firstbunch_0"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tTauLastBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TTau/timing/Last_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TTau_BX_LastBunch_0 'Ratio for L1TTau last bunch for BX=0' tau_eta_phi_bx_lastbunch_0 den_tau_eta_phi_bx_lastbunch_0",
       "Ratio_L1TTau_BX_LastBunch_plus1 'Ratio for L1TTau last bunch for BX=+1' tau_eta_phi_bx_lastbunch_plus1 den_tau_eta_phi_bx_lastbunch_plus1",
       "Ratio_L1TTau_BX_LastBunch_plus2 'Ratio for L1TTau last bunch for BX=+2' tau_eta_phi_bx_lastbunch_plus2 den_tau_eta_phi_bx_lastbunch_plus2"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tTauIsolatedBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TTau/timing/Isolated_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TTau_BX_Isolated_minus2 'Ratio for L1TTau isolated bunch for BX=-2' tau_eta_phi_bx_isolated_minus2 den_tau_eta_phi_bx_isolated_minus2",
       "Ratio_L1TTau_BX_Isolated_minus1 'Ratio for L1TTau isolated bunch for BX=-1' tau_eta_phi_bx_isolated_minus1 den_tau_eta_phi_bx_isolated_minus1",
       "Ratio_L1TTau_BX_Isolated_0 'Ratio for L1TTau isolated bunch for BX=0' tau_eta_phi_bx_isolated_0 den_tau_eta_phi_bx_isolated_0",
       "Ratio_L1TTau_BX_Isolated_plus1 'Ratio for L1TTau isolated bunch for BX=+1' tau_eta_phi_bx_isolated_plus1 den_tau_eta_phi_bx_isolated_plus1",
       "Ratio_L1TTau_BX_Isolated_plus2 'Ratio for L1TTau isolated bunch for BX=+2' tau_eta_phi_bx_isolated_plus2 den_tau_eta_phi_bx_isolated_plus2"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tEtSumRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEtSum/timing/'),
    efficiency = cms.vstring(
       "Ratio_L1TEtSum_BX_MET_minus2 'Ratio for L1TEtSum MET for BX=-2' etsum_phi_bx_MET_minus2 den_etsum_phi_bx_MET_minus2",
       "Ratio_L1TEtSum_BX_MET_minus1 'Ratio for L1TEtSum MET for BX=-1' etsum_phi_bx_MET_minus1 den_etsum_phi_bx_MET_minus1",
       "Ratio_L1TEtSum_BX_MET_0 'Ratio for L1TEtSum MET for BX=0' etsum_phi_bx_MET_0 den_etsum_phi_bx_MET_0",
       "Ratio_L1TEtSum_BX_MET_plus1 'Ratio for L1TEtSum MET for BX=+1' etsum_phi_bx_MET_plus1 den_etsum_phi_bx_MET_plus1",
       "Ratio_L1TEtSum_BX_MET_plus2 'Ratio for L1TEtSum MET for BX=+2' etsum_phi_bx_MET_plus2 den_etsum_phi_bx_MET_plus2",
       "Ratio_L1TEtSum_BX_METHF_minus2 'Ratio for L1TEtSum METHF for BX=-2' etsum_phi_bx_METHF_minus2 den_etsum_phi_bx_METHF_minus2",
       "Ratio_L1TEtSum_BX_METHF_minus1 'Ratio for L1TEtSum METHF for BX=-1' etsum_phi_bx_METHF_minus1 den_etsum_phi_bx_METHF_minus1",
       "Ratio_L1TEtSum_BX_METHF_0 'Ratio for L1TEtSum METHF for BX=0' etsum_phi_bx_METHF_0 den_etsum_phi_bx_METHF_0",
       "Ratio_L1TEtSum_BX_METHF_plus1 'Ratio for L1TEtSum METHF for BX=+1' etsum_phi_bx_METHF_plus1 den_etsum_phi_bx_METHF_plus1",
       "Ratio_L1TEtSum_BX_METHF_plus2 'Ratio for L1TEtSum METHF for BX=+2' etsum_phi_bx_METHF_plus2 den_etsum_phi_bx_METHF_plus2",
       "Ratio_L1TEtSum_BX_MHTHF_minus2 'Ratio for L1TEtSum MHTHF for BX=-2' etsum_phi_bx_MHTHF_minus2 den_etsum_phi_bx_MHTHF_minus2",
       "Ratio_L1TEtSum_BX_MHTHF_minus1 'Ratio for L1TEtSum MHTHF for BX=-1' etsum_phi_bx_MHTHF_minus1 den_etsum_phi_bx_MHTHF_minus1",
       "Ratio_L1TEtSum_BX_MHTHF_0 'Ratio for L1TEtSum MHTHF for BX=0' etsum_phi_bx_MHTHF_0 den_etsum_phi_bx_MHTHF_0",
       "Ratio_L1TEtSum_BX_MHTHF_plus1 'Ratio for L1TEtSum MHTHF for BX=+1' etsum_phi_bx_MHTHF_plus1 den_etsum_phi_bx_MHTHF_plus1",
       "Ratio_L1TEtSum_BX_MHTHF_plus2 'Ratio for L1TEtSum MHTHF for BX=+2' etsum_phi_bx_MHTHF_plus2 den_etsum_phi_bx_MHTHF_plus2",
       "Ratio_L1TEtSum_BX_MHT_minus2 'Ratio for L1TEtSum MHT for BX=-2' etsum_phi_bx_MHT_minus2 den_etsum_phi_bx_MHT_minus2",
       "Ratio_L1TEtSum_BX_MHT_minus1 'Ratio for L1TEtSum MHT for BX=-1' etsum_phi_bx_MHT_minus1 den_etsum_phi_bx_MHT_minus1",
       "Ratio_L1TEtSum_BX_MHT_0 'Ratio for L1TEtSum MHT for BX=0' etsum_phi_bx_MHT_0 den_etsum_phi_bx_MHT_0",
       "Ratio_L1TEtSum_BX_MHT_plus1 'Ratio for L1TEtSum MHT for BX=+1' etsum_phi_bx_MHT_plus1 den_etsum_phi_bx_MHT_plus1",
       "Ratio_L1TEtSum_BX_MHT_plus2 'Ratio for L1TEtSum MHT for BX=+2' etsum_phi_bx_MHT_plus2 den_etsum_phi_bx_MHT_plus2"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tEtSumFirstBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEtSum/timing/First_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TEtSum_BX_MET_FirstBunch_minus2 'Ratio for L1TEtSum MET first bunch for BX=-2' etsum_phi_bx_MET_firstbunch_minus2 den_etsum_phi_bx_MET_firstbunch_minus2",
       "Ratio_L1TEtSum_BX_MET_FirstBunch_minus1 'Ratio for L1TEtSum MET first bunch for BX=-1' etsum_phi_bx_MET_firstbunch_minus1 den_etsum_phi_bx_MET_firstbunch_minus1",
       "Ratio_L1TEtSum_BX_MET_FirstBunch_0 'Ratio for L1TEtSum MET first bunch for BX=0' etsum_phi_bx_MET_firstbunch_0 den_etsum_phi_bx_MET_firstbunch_0",
       "Ratio_L1TEtSum_BX_METHF_FirstBunch_minus2 'Ratio for L1TEtSum METHF first bunch for BX=-2' etsum_phi_bx_METHF_firstbunch_minus2 den_etsum_phi_bx_METHF_firstbunch_minus2",
       "Ratio_L1TEtSum_BX_METHF_FirstBunch_minus1 'Ratio for L1TEtSum METHF first bunch for BX=-1' etsum_phi_bx_METHF_firstbunch_minus1 den_etsum_phi_bx_METHF_firstbunch_minus1",
       "Ratio_L1TEtSum_BX_METHF_FirstBunch_0 'Ratio for L1TEtSum METHF first bunch for BX=0' etsum_phi_bx_METHF_firstbunch_0 den_etsum_phi_bx_METHF_firstbunch_0",      
       "Ratio_L1TEtSum_BX_MHTHF_FirstBunch_minus2 'Ratio for L1TEtSum MHTHF first bunch for BX=-2' etsum_phi_bx_MHTHF_firstbunch_minus2 den_etsum_phi_bx_MHTHF_firstbunch_minus2",
       "Ratio_L1TEtSum_BX_MHTHF_FirstBunch_minus1 'Ratio for L1TEtSum MHTHF first bunch for BX=-1' etsum_phi_bx_MHTHF_firstbunch_minus1 den_etsum_phi_bx_MHTHF_firstbunch_minus1",
       "Ratio_L1TEtSum_BX_MHTHF_FirstBunch_0 'Ratio for L1TEtSum MHTHF first bunch for BX=0' etsum_phi_bx_MHTHF_firstbunch_0 den_etsum_phi_bx_MHTHF_firstbunch_0",
       "Ratio_L1TEtSum_BX_MHT_FirstBunch_minus2 'Ratio for L1TEtSum MHT first bunch for BX=-2' etsum_phi_bx_MHT_firstbunch_minus2 den_etsum_phi_bx_MHT_firstbunch_minus2",
       "Ratio_L1TEtSum_BX_MHT_FirstBunch_minus1 'Ratio for L1TEtSum MHT first bunch for BX=-1' etsum_phi_bx_MHT_firstbunch_minus1 den_etsum_phi_bx_MHT_firstbunch_minus1",
       "Ratio_L1TEtSum_BX_MHT_FirstBunch_0 'Ratio for L1TEtSum MHT first bunch for BX=0' etsum_phi_bx_MHT_firstbunch_0 den_etsum_phi_bx_MHT_firstbunch_0"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tEtSumLastBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEtSum/timing/Last_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TEtSum_BX_MET_LastBunch_0 'Ratio for L1TEtSum MET last bunch for BX=0' etsum_phi_bx_MET_lastbunch_0 den_etsum_phi_bx_MET_lastbunch_0",
       "Ratio_L1TEtSum_BX_MET_LastBunch_plus1 'Ratio for L1TEtSum MET last bunch for BX=+1' etsum_phi_bx_MET_lastbunch_plus1 den_etsum_phi_bx_MET_lastbunch_plus1",
       "Ratio_L1TEtSum_BX_MET_LastBunch_plus2 'Ratio for L1TEtSum MET last bunch for BX=+2' etsum_phi_bx_MET_lastbunch_plus2 den_etsum_phi_bx_MET_lastbunch_plus2",
       "Ratio_L1TEtSum_BX_METHF_LastBunch_minus2 'Ratio for L1TEtSum METHF last bunch for BX=-2' etsum_phi_bx_METHF_lastbunch_minus2 den_etsum_phi_bx_METHF_lastbunch_minus2",
       "Ratio_L1TEtSum_BX_METHF_LastBunch_minus1 'Ratio for L1TEtSum METHF last bunch for BX=-1' etsum_phi_bx_METHF_lastbunch_minus1 den_etsum_phi_bx_METHF_lastbunch_minus1",
       "Ratio_L1TEtSum_BX_METHF_LastBunch_0 'Ratio for L1TEtSum METHF last bunch for BX=0' etsum_phi_bx_METHF_lastbunch_0 den_etsum_phi_bx_METHF_lastbunch_0",
       "Ratio_L1TEtSum_BX_MHTHF_LastBunch_0 'Ratio for L1TEtSum MHTHF last bunch for BX=0' etsum_phi_bx_MHTHF_lastbunch_0 den_etsum_phi_bx_MHTHF_lastbunch_0",
       "Ratio_L1TEtSum_BX_MHTHF_LastBunch_plus1 'Ratio for L1TEtSum MHTHF last bunch for BX=+1' etsum_phi_bx_MHTHF_lastbunch_plus1 den_etsum_phi_bx_MHTHF_lastbunch_plus1",
       "Ratio_L1TEtSum_BX_MHTHF_LastBunch_plus2 'Ratio for L1TEtSum MHTHF last bunch for BX=+2' etsum_phi_bx_MHTHF_lastbunch_plus2 den_etsum_phi_bx_MHTHF_lastbunch_plus2",
       "Ratio_L1TEtSum_BX_MHT_LastBunch_0 'Ratio for L1TEtSum MHT last bunch for BX=0' etsum_phi_bx_MHT_lastbunch_0 den_etsum_phi_bx_MHT_lastbunch_0",
       "Ratio_L1TEtSum_BX_MHT_LastBunch_plus1 'Ratio for L1TEtSum MHT last bunch for BX=+1' etsum_phi_bx_MHT_lastbunch_plus1 den_etsum_phi_bx_MHT_lastbunch_plus1",
       "Ratio_L1TEtSum_BX_MHT_LastBunch_plus2 'Ratio for L1TEtSum MHT last bunch for BX=+2' etsum_phi_bx_MHT_lastbunch_plus2 den_etsum_phi_bx_MHT_lastbunch_plus2"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

l1tEtSumIsolatedBunchRatioPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(l1tobjectstimingDqmDir+'L1TEtSum/timing/Isolated_bunch/'),
    efficiency = cms.vstring(
       "Ratio_L1TEtSum_BX_MET_Isolated_minus2 'Ratio for L1TEtSum MET isolated bunch for BX=-2' etsum_phi_bx_MET_isolated_minus2 den_etsum_phi_bx_MET_isolated_minus2",
       "Ratio_L1TEtSum_BX_MET_Isolated_minus1 'Ratio for L1TEtSum MET isolated bunch for BX=-1' etsum_phi_bx_MET_isolated_minus1 den_etsum_phi_bx_MET_isolated_minus1",
       "Ratio_L1TEtSum_BX_MET_Isolated_0 'Ratio for L1TEtSum MET isolated bunch for BX=0' etsum_phi_bx_MET_isolated_0 den_etsum_phi_bx_MET_isolated_0",
       "Ratio_L1TEtSum_BX_MET_Isolated_plus1 'Ratio for L1TEtSum MET isolated bunch for BX=+1' etsum_phi_bx_MET_isolated_plus1 den_etsum_phi_bx_MET_isolated_plus1",
       "Ratio_L1TEtSum_BX_MET_Isolated_plus2 'Ratio for L1TEtSum MET isolated bunch for BX=+2' etsum_phi_bx_MET_isolated_plus2 den_etsum_phi_bx_MET_isolated_plus2", 
       "Ratio_L1TEtSum_BX_METHF_Isolated_minus2 'Ratio for L1TEtSum METHF isolated bunch for BX=-2' etsum_phi_bx_METHF_isolated_minus2 den_etsum_phi_bx_METHF_isolated_minus2",
       "Ratio_L1TEtSum_BX_METHF_Isolated_minus1 'Ratio for L1TEtSum METHF isolated bunch for BX=-1' etsum_phi_bx_METHF_isolated_minus1 den_etsum_phi_bx_METHF_isolated_minus1",
       "Ratio_L1TEtSum_BX_METHF_Isolated_0 'Ratio for L1TEtSum METHF isolated bunch for BX=0' etsum_phi_bx_METHF_isolated_0 den_etsum_phi_bx_METHF_isolated_0",
       "Ratio_L1TEtSum_BX_METHF_Isolated_plus1 'Ratio for L1TEtSum METHF isolated bunch for BX=+1' etsum_phi_bx_METHF_isolated_plus1 den_etsum_phi_bx_METHF_isolated_plus1",
       "Ratio_L1TEtSum_BX_METHF_Isolated_plus2 'Ratio for L1TEtSum METHF isolated bunch for BX=+2' etsum_phi_bx_METHF_isolated_plus2 den_etsum_phi_bx_METHF_isolated_plus2",
       "Ratio_L1TEtSum_BX_MHTHF_Isolated_minus2 'Ratio for L1TEtSum MHTHF isolated bunch for BX=-2' etsum_phi_bx_MHTHF_isolated_minus2 den_etsum_phi_bx_MHTHF_isolated_minus2",
       "Ratio_L1TEtSum_BX_MHTHF_Isolated_minus1 'Ratio for L1TEtSum MHTHF isolated bunch for BX=-1' etsum_phi_bx_MHTHF_isolated_minus1 den_etsum_phi_bx_MHTHF_isolated_minus1",
       "Ratio_L1TEtSum_BX_MHTHF_Isolated_0 'Ratio for L1TEtSum MHTHF isolated bunch for BX=0' etsum_phi_bx_MHTHF_isolated_0 den_etsum_phi_bx_MHTHF_isolated_0",
       "Ratio_L1TEtSum_BX_MHTHF_Isolated_plus1 'Ratio for L1TEtSum MHTHF isolated bunch for BX=+1' etsum_phi_bx_MHTHF_isolated_plus1 den_etsum_phi_bx_MHTHF_isolated_plus1",
       "Ratio_L1TEtSum_BX_MHTHF_Isolated_plus2 'Ratio for L1TEtSum MHTHF isolated bunch for BX=+2' etsum_phi_bx_MHTHF_isolated_plus2 den_etsum_phi_bx_MHTHF_isolated_plus2",
       "Ratio_L1TEtSum_BX_MHT_Isolated_minus2 'Ratio for L1TEtSum MHT isolated bunch for BX=-2' etsum_phi_bx_MHT_isolated_minus2 den_etsum_phi_bx_MHT_isolated_minus2",
       "Ratio_L1TEtSum_BX_MHT_Isolated_minus1 'Ratio for L1TEtSum MHT isolated bunch for BX=-1' etsum_phi_bx_MHT_isolated_minus1 den_etsum_phi_bx_MHT_isolated_minus1",
       "Ratio_L1TEtSum_BX_MHT_Isolated_0 'Ratio for L1TEtSum MHT isolated bunch for BX=0' etsum_phi_bx_MHT_isolated_0 den_etsum_phi_bx_MHT_isolated_0",
       "Ratio_L1TEtSum_BX_MHT_Isolated_plus1 'Ratio for L1TEtSum MHT isolated bunch for BX=+1' etsum_phi_bx_MHT_isolated_plus1 den_etsum_phi_bx_MHT_isolated_plus1",
       "Ratio_L1TEtSum_BX_MHT_Isolated_plus2 'Ratio for L1TEtSum MHT isolated bunch for BX=+2' etsum_phi_bx_MHT_isolated_plus2 den_etsum_phi_bx_MHT_isolated_plus2"
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string("output.root"),
    verbose = cms.untracked.uint32(0)
)

# sequences
l1tObjectsTimingClient = cms.Sequence(
   l1tMuonRatioPlots
+  l1tMuonFirstBunchRatioPlots
+  l1tMuonLastBunchRatioPlots 
+  l1tMuonIsolatedBunchRatioPlots
+  l1tJetRatioPlots
+  l1tJetFirstBunchRatioPlots
+  l1tJetLastBunchRatioPlots
+  l1tJetIsolatedBunchRatioPlots
+  l1tEGammaRatioPlots
+  l1tEGammaFirstBunchRatioPlots
+  l1tEGammaLastBunchRatioPlots
+  l1tEGammaIsolatedBunchRatioPlots
+  l1tTauRatioPlots
+  l1tTauFirstBunchRatioPlots
+  l1tTauLastBunchRatioPlots
+  l1tTauIsolatedBunchRatioPlots
+  l1tEtSumRatioPlots
+  l1tEtSumFirstBunchRatioPlots
+  l1tEtSumLastBunchRatioPlots
+  l1tEtSumIsolatedBunchRatioPlots

)
