from .adapt_to_new_backend import *
dqmitems={}

def shiftjetmetlayout(i, p, *rows): i["00 Shift/JetMET/" + p] = rows

shiftjetmetlayout(dqmitems, "01 Calo Jet Plots (for collisions)", [{ 'path': "JetMET/Jet/Cleanedak4CaloJets/PtFirst", 'draw': { 'withref': 'yes' }, 'description': "Distribution of Jet Pt for all cleaned jets (event primary vertex requirement, the distribution should be fast falling with no spikes) (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#jet_pt_plots>more</a>)" },
{ 'path': "JetMET/Jet/Cleanedak4CaloJets/Constituents", 'draw': { 'withref': 'yes' }, 'description': "Number of constituents towers in the jet.  There should not be any spikes in the distribution (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#jet_constit_plots>more</a>)" }],
[{ 'path': "JetMET/Jet/Cleanedak4CaloJets/Phi", 'draw': { 'withref': 'yes' }, 'description': "Phi distribution for all jets.  Should be smooth and without spikes (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#jet_phi_plots>more</a>)" },
{ 'path': "JetMET/Jet/Cleanedak4CaloJets/EtaFirst", 'draw': { 'withref': 'yes' }, 'description': "Eta distribution for all jets.  Should be smooth and without spikes (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#jet_eta_plots>more</a>)" }])


shiftjetmetlayout(dqmitems, "02 PF Jet Plots (for collisions)", [
{ 'path': "JetMET/Jet/Cleanedak4PFJets/PtFirst", 'draw': { 'withref': 'yes' }, 'description': "Distribution of Jet Pt for all Particle Flow jets (event primary vertex requirement, the distribution should be fast falling with no spikes) (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#jet_pt_plots>more</a>)" },
{ 'path': "JetMET/Jet/Cleanedak4PFJets/Eta", 'draw': { 'withref': 'yes' }, 'description': "Eta distribution for Particle Flow jets.  Should be smooth and without spikes (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#jet_eta_plots>more</a>)" }],
[{ 'path': "JetMET/Jet/Cleanedak4PFJets/Constituents", 'draw': { 'withref': 'yes' }, 'description': "Number of constituents in the jet.  There should not be any spikes in the distribution (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#jet_constit_plots>more</a>)" },
{ 'path': "JetMET/Jet/Cleanedak4PFJets/NJets", 'draw': { 'withref': 'yes' }, 'description': "Jet's multiplicity for Particle Flow jets.  Should not be excessively large (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#jet_multiplicity_plots</a>)" }])

shiftjetmetlayout(dqmitems, "03 PF Jet Plots (for collisions)", 
[{ 'path': "JetMET/Jet/Cleanedak4PFJets/Phi_Barrel", 'draw': { 'withref': 'yes' }, 'description': "Jets Phi in Barrel (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#jet_pt_plots>more</a>)" }],
[{ 'path': "JetMET/Jet/Cleanedak4PFJets/Phi_EndCap", 'draw': { 'withref': 'yes' }, 'description': "Jets Phi in EndCap (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#jet_phi_plots>more</a>)" },
{ 'path': "JetMET/Jet/Cleanedak4PFJets/Phi_Forward", 'draw': { 'withref': 'yes' }, 'description': "Jets Phi in HF (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#jet_phi_plots>more</a>)" }])

shiftjetmetlayout(dqmitems, "05 PF MET Plots (for collisions)", [{ 'path': "JetMET/MET/pfMet/Cleaned/MET", 'draw': { 'withref': 'yes' }, 'description': "The Particle Flow Missing ET distribution.  There should not be any spikes in the distribution (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#met_plots>more</a>)" },
{ 'path': "JetMET/MET/pfMet/Cleaned/METPhi", 'draw': { 'withref': 'yes' }, 'description': "The Particle Flow Missing ET phi distribution.  There should not be any spikes in the distribution (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#met_phi_plots>more</a>)" }],
[{ 'path': "JetMET/MET/pfMet/Cleaned/MEx", 'draw': { 'withref': 'yes' }, 'description': "The Particle Flow Missing Ex distribution.  There should not be any spikes in the distribution (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#met_x_plots>more</a>)" },
{ 'path': "JetMET/MET/pfMet/Cleaned/MEy", 'draw': { 'withref': 'yes' }, 'description': "The Particle Flow Missing Ey distribution.  There should not be any spikes in the distribution (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#met_y_plots>more</a>)" }])

shiftjetmetlayout(dqmitems, "06 PF MET Plots (for collisions)", [{ 'path': "JetMET/MET/pfMet/Cleaned/SumET", 'draw': { 'withref': 'yes' }, 'description': "The Particle Flow Scalar Sum ET distribution. There should not be any spikes in the distribution (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#met_sumet_plots>more</a>)" }],
[{ 'path': "JetMET/MET/pfMet/Cleaned/MET_logx", 'draw': { 'withref': 'yes' }, 'description': "The Particle Flow Missing ET distribution in log(X) scale.  There should not be any spikes in the distribution (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#MET_logx_plots>more</a>)" }])

shiftjetmetlayout(dqmitems, "07 PF MET Plots (for collisions)", 
[{ 'path': "JetMET/MET/pfMet/Cleaned/PfPhotonEtFraction", 'draw': { 'withref': 'yes' }, 'description': "photon energy fraction(<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#PhotonEtFraction_plots>more</a>)" },
{ 'path': "JetMET/MET/pfMet/Cleaned/PfNeutralHadronEtFraction", 'draw': { 'withref': 'yes' }, 'description': "Neutral hadron energy fraction (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#NeHadEtFraction_plots>more</a>)" }],
[{ 'path': "JetMET/MET/pfMet/Cleaned/h0Barrel_multiplicity_", 'draw': { 'withref': 'yes' }, 'description': "Neutral hadron multiplicity in Barrel (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#NeHadMultiplicityBarrel_plots>more</a>)" },
{ 'path': "JetMET/MET/pfMet/Cleaned/h0Endcap_multiplicity_", 'draw': { 'withref': 'yes' }, 'description': "Neutral hadron multiplicity in EndCap (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#NeHadMultiplicityEndcap_plots>more</a>)" }])

shiftjetmetlayout(dqmitems, "08 PF MET Plots (for collisions)", 
[{ 'path': "JetMET/MET/pfMet/Cleaned/PfHFEMEtFraction", 'draw': { 'withref': 'yes' }, 'description': "EM energy fraction in HF(<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#HFEMEnergyFraction_plots>more</a>)" },
{ 'path': "JetMET/MET/pfMet/Cleaned/PfHFHadronEtFraction", 'draw': { 'withref': 'yes' }, 'description': "Hadron energy fraction HF(<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#HFHadEnergyFraction_plots>more</a>)" }],
[{ 'path': "JetMET/MET/pfMet/Cleaned/PfChargedHadronEtFraction", 'draw': { 'withref': 'yes' }, 'description': "Charged hadron energy fraction (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#ChHadEtFraction_plots>more</a>)" }])

shiftjetmetlayout(dqmitems, "09 PF candidates map", [{ 'path': "JetMET/MET/pfMet/Cleaned/h_occupancy_", 'description': "Charged hadron occupancy (<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineJetMET#h_occupancy_plot>more</a>)" }])


apply_dqm_items_to_new_back_end(dqmitems, __file__)
