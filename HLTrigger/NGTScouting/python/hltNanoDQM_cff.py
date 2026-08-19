import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.nanoDQM_tools_cff import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hltNanoDQM = DQMEDAnalyzer("NanoAODDQM",
    folder = cms.string("HLT/NanoAODDQM"),
    vplots = cms.PSet(
        hltPixelTrack = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Plot1D('vx', 'vx', 50, -0.1, 0.1, 'track vx'),
                Plot1D('vy', 'vy', 50, -0.1, 0.1, 'track vy'),
                Plot1D('dXY', 'dXY', 40, -2, 2, 'track dXY'),
                Plot1D('phi', 'phi', 20, -3.14159, 3.14159, 'phi'),
                Plot1D('eta', 'eta', 20, -4., 4., 'eta'),
                Plot1D('vz', 'vz', 50, -25, 25, 'track vz'),
                Plot1D('dZ', 'dZ', 40, -20, 20, 'track dZ'),
                Plot1D('chi2', 'chi2', 20, 0, 50, 'track chi2'),
                Plot1D('pt', 'pt', 20, 0, 400, 'pt'),
                Plot1D('dxyBS', 'dxyBS', 40, -2, 2, 'track dxy wrt beamspot'),
                Plot1D('dzBS', 'dzBS', 40, -20, 20, 'track dz wrt beamspot'),
                Plot1D('ndof', 'ndof', 15, 0, 30, 'track ndof'),
                Plot1D('nPixelHits', 'nPixelHits', 10, 0, 10, 'track # pixel hits'),
                Plot1D('nTrkLays', 'nTrkLays', 20, 0, 20, 'track # tracker layers'),
                Plot1D('charge', 'charge', 5, -2, 3, 'track charge'),
                Plot1D('t0', 't0', 40, -1, 1, 'track t0')
            )
        ),
        hltGeneralTrack = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Plot1D('vx', 'vx', 50, -0.1, 0.1, 'track vx'),
                Plot1D('vy', 'vy', 50, -0.1, 0.1, 'track vy'),
                Plot1D('dXY', 'dXY', 40, -2, 2, 'track dXY'),
                Plot1D('phi', 'phi', 20, -3.14159, 3.14159, 'phi'),
                Plot1D('eta', 'eta', 20, -4., 4., 'eta'),
                Plot1D('vz', 'vz', 50, -25, 25, 'track vz'),
                Plot1D('dZ', 'dZ', 40, -20, 20, 'track dZ'),
                Plot1D('chi2', 'chi2', 20, 0, 50, 'track chi2'),
                Plot1D('pt', 'pt', 20, 0, 400, 'pt'),
                Plot1D('dxyBS', 'dxyBS', 40, -2, 2, 'track dxy wrt beamspot'),
                Plot1D('dzBS', 'dzBS', 40, -20, 20, 'track dz wrt beamspot'),
                Plot1D('ndof', 'ndof', 15, 0, 30, 'track ndof'),
                Plot1D('nPixelHits', 'nPixelHits', 10, 0, 10, 'track # pixel hits'),
                Plot1D('nTrkLays', 'nTrkLays', 20, 0, 20, 'track # tracker layers'),
                Plot1D('charge', 'charge', 5, -2, 3, 'track charge'),
                Plot1D('t0', 't0', 40, -1, 1, 'track t0')
            )
        ),
        hltPixelVertex = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Plot1D('zError', 'zError', 20, 0, 0.05, 'vertex z error'),
                Plot1D('sumpx', 'sumpx', 20, -50, 50, 'vertex sum px'),
                Plot1D('z', 'z', 50, -25, 25, 'vertex z'),
                Plot1D('sumpy', 'sumpy', 20, -50, 50, 'vertex sum py'),
                Plot1D('sumpt2', 'sumpt2', 20, 0, 2000, 'vertex sum pt2'),
                Plot1D('nTracks', 'nTracks', 25, 0, 100, 'vertex # tracks'),
                Plot1D('score', 'score', 20, 0, 2000, 'vertex score'),
                Plot1D('ndof', 'ndof', 20, 0, 200, 'vertex ndof'),
                Plot1D('xError', 'xError', 20, 0, 0.05, 'vertex x error'),
                Plot1D('yError', 'yError', 20, 0, 0.05, 'vertex y error'),
                Plot1D('x', 'x', 50, -0.1, 0.1, 'vertex x'),
                Plot1D('y', 'y', 50, -0.1, 0.1, 'vertex y'),
                Plot1D('chi2', 'chi2', 20, 0, 50, 'vertex chi2'),
                Plot1D('isGood', 'isGood', 2, 0, 2, 'vertex isGood')
            )
        ),
        hltPrimaryVertex = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Plot1D('zError', 'zError', 20, 0, 0.05, 'vertex z error'),
                Plot1D('z', 'z', 50, -25, 25, 'vertex z'),
                Plot1D('y', 'y', 50, -0.1, 0.1, 'vertex y'),
                Plot1D('sumpx', 'sumpx', 20, -50, 50, 'vertex sum px'),
                Plot1D('sumpy', 'sumpy', 20, -50, 50, 'vertex sum py'),
                Plot1D('x', 'x', 50, -0.1, 0.1, 'vertex x'),
                Plot1D('sumpt2', 'sumpt2', 20, 0, 2000, 'vertex sum pt2'),
                Plot1D('xError', 'xError', 20, 0, 0.05, 'vertex x error'),
                Plot1D('yError', 'yError', 20, 0, 0.05, 'vertex y error'),
                Plot1D('score', 'score', 20, 0, 2000, 'vertex score'),
                Plot1D('ndof', 'ndof', 20, 0, 200, 'vertex ndof'),
                Plot1D('chi2', 'chi2', 20, 0, 50, 'vertex chi2'),
                Plot1D('nTracks', 'nTracks', 25, 0, 100, 'vertex # tracks'),
                Plot1D('isGood', 'isGood', 2, 0, 2, 'vertex isGood')
            )
        ),
        hltSecondaryVertex = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Plot1D('dlenSig', 'dlenSig', 20, 0, 50, 'sv decay length significance'),
                Plot1D('dxySig', 'dxySig', 20, 0, 50, 'sv dxy significance'),
                Plot1D('pAngle', 'pAngle', 20, 0, 3.14159, 'sv pointing angle'),
                Plot1D('dlen', 'dlen', 20, 0, 5, 'sv decay length'),
                Plot1D('ndof', 'ndof', 20, 0, 20, 'sv ndof'),
                Plot1D('chi2', 'chi2', 20, 0, 50, 'sv chi2'),
                Plot1D('mass', 'mass', 20, 0, 10, 'sv mass'),
                Plot1D('dxy', 'dxy', 20, 0, 5, 'sv dxy'),
                Plot1D('eta', 'eta', 20, -2.5, 2.5, 'sv eta'),
                Plot1D('phi', 'phi', 20, -3.14159, 3.14159, 'sv phi'),
                Plot1D('pt', 'pt', 20, 0, 400, 'sv pt'),
                Plot1D('x', 'x', 40, -1, 1, 'sv x'),
                Plot1D('y', 'y', 40, -1, 1, 'sv y'),
                Plot1D('z', 'z', 50, -25, 25, 'sv z'),
                Plot1D('charge', 'charge', 11, -5, 6, 'sv charge'),
                Plot1D('ntracks', 'ntracks', 20, 0, 20, 'sv # tracks'),
                Plot1D('isGood', 'isGood', 2, 0, 2, 'sv isGood')
            )
        ),
        hltPhoton = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Plot1D('phi', 'phi', 20, -3.14159, 3.14159, 'photon phi'),
                Plot1D('m', 'm', 10, -0.001, 0.001, 'photon mass'),
                Plot1D('eta', 'eta', 20, -2.5, 2.5, 'photon eta'),
                Plot1D('ecalIso', 'ecalIso', 20, 0, 20, 'photon ecal iso'),
                Plot1D('seedId', 'seedId', 50, 0, 2000000000, 'photon seed detId'),
                Plot1D('sMaj', 'sMaj', 20, 0, 5, 'photon shower sMaj'),
                Plot1D('pt', 'pt', 20, 0, 400, 'photon pt'),
                Plot1D('hOverE', 'hOverE', 20, 0, 0.5, 'photon H/E'),
                Plot1D('sMin', 'sMin', 20, 0, 5, 'photon shower sMin'),
                Plot1D('sigmaIetaIeta', 'sigmaIetaIeta', 20, 0, 0.05, 'photon sigmaIetaIeta'),
                Plot1D('hcalIso', 'hcalIso', 20, 0, 20, 'photon hcal iso'),
                Plot1D('r9', 'r9', 20, 0, 1.2, 'photon r9')
            )
        ),
        hltElectron = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Plot1D('ooEMOop', 'ooEMOop', 20, -0.2, 0.2, 'electron 1/E - 1/p'),
                Plot1D('dEtaIn', 'dEtaIn', 20, -0.05, 0.05, 'electron dEtaIn'),
                Plot1D('dPhiIn', 'dPhiIn', 20, -0.2, 0.2, 'electron dPhiIn'),
                Plot1D('phi', 'phi', 20, -3.14159, 3.14159, 'electron phi'),
                Plot1D('m', 'm', 20, 0, 0.001, 'electron mass'),
                Plot1D('eta', 'eta', 20, -2.5, 2.5, 'electron eta'),
                Plot1D('pt', 'pt', 20, 0, 400, 'electron pt'),
                Plot1D('seedId', 'seedId', 50, 0, 2000000000, 'electron seed detId'),
                Plot1D('ecalIso', 'ecalIso', 20, 0, 20, 'electron ecal iso'),
                Plot1D('sigmaIetaIeta', 'sigmaIetaIeta', 20, 0, 0.05, 'electron sigmaIetaIeta'),
                Plot1D('sMaj', 'sMaj', 20, 0, 5, 'electron shower sMaj'),
                Plot1D('sMin', 'sMin', 20, 0, 5, 'electron shower sMin'),
                Plot1D('trackIso', 'trackIso', 20, 0, 20, 'electron track iso'),
                Plot1D('r9', 'r9', 20, 0, 1.2, 'electron r9'),
                Plot1D('hcalIso', 'hcalIso', 20, 0, 20, 'electron hcal iso'),
                Plot1D('hOverE', 'hOverE', 20, 0, 0.5, 'electron H/E'),
                Plot1D('missingHits', 'missingHits', 5, 0, 5, 'electron missing hits')
            )
        ),
        hltMuon = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Plot1D('dXY', 'dXY', 40, -2, 2, 'muon dXY'),
                Plot1D('dZ', 'dZ', 40, -20, 20, 'muon dZ'),
                Plot1D('eta', 'eta', 20, -2.5, 2.5, 'muon eta'),
                Plot1D('phi', 'phi', 20, -3.14159, 3.14159, 'muon phi'),
                Plot1D('pt', 'pt', 20, 0, 400, 'muon pt'),
                Plot1D('nMuHits', 'nMuHits', 20, 0, 40, 'muon # muon hits'),
                Plot1D('nPixelHits', 'nPixelHits', 10, 0, 10, 'muon # pixel hits'),
                Plot1D('nTrkLays', 'nTrkLays', 20, 0, 20, 'muon # tracker layers'),
                Plot1D('t0', 't0', 40, -1, 1, 'muon t0')
            )
        ),
        hltHpsPFTau = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Plot1D('dz', 'dz', 40, -20, 20, 'tau dz'),
                Plot1D('dzError', 'dzError', 20, 0, 2, 'tau dz error'),
                Plot1D('pt', 'pt', 20, 0, 400, 'tau pt'),
                Plot1D('vz', 'vz', 50, -25, 25, 'tau vz'),
                Plot1D('phi', 'phi', 20, -3.14159, 3.14159, 'tau phi'),
                Plot1D('eta', 'eta', 20, -2.5, 2.5, 'tau eta'),
                Plot1D('hcalTotOverPLead', 'hcalTotOverPLead', 20, 0, 5, 'tau hcalTotOverPLead'),
                Plot1D('dxy', 'dxy', 40, -2, 2, 'tau dxy'),
                Plot1D('ip3d', 'ip3d', 40, -2, 2, 'tau IP3D'),
                Plot1D('leadTkDeltaEta', 'leadTkDeltaEta', 20, -0.2, 0.2, 'tau leading track delta eta'),
                Plot1D('leadTkDeltaPhi', 'leadTkDeltaPhi', 20, -0.2, 0.2, 'tau leading track delta phi'),
                Plot1D('ip3d_error', 'ip3d_error', 20, 0, 0.5, 'tau IP3D error'),
                Plot1D('leadTkPtOverTauPt', 'leadTkPtOverTauPt', 20, 0, 1.5, 'tau leading track pt / tau pt'),
                Plot1D('mass', 'mass', 20, 0, 4, 'tau mass'),
                Plot1D('deepTauVSjet', 'deepTauVSjet', 20, 0, 1, 'deepTau vs jet'),
                Plot1D('emFraction', 'emFraction', 20, 0, 1, 'tau em fraction'),
                Plot1D('dxy_error', 'dxy_error', 20, 0, 0.5, 'tau dxy error'),
                Plot1D('deepTauVSe', 'deepTauVSe', 20, 0, 1, 'deepTau vs electron'),
                Plot1D('deepTauVSmu', 'deepTauVSmu', 20, 0, 1, 'deepTau vs muon'),
                Plot1D('decayMode', 'decayMode', 16, 0, 16, 'tau decay mode'),
                Plot1D('charge', 'charge', 5, -2, 3, 'tau charge'),
                Plot1D('pdgId', 'pdgId', 10, -20, 20, 'tau pdgId'),
                Plot1D('secondaryVertex_y', 'secondaryVertex_y', 40, -1, 1, 'tau secondary vertex y'),
                Plot1D('secondaryVertex_x', 'secondaryVertex_x', 40, -1, 1, 'tau secondary vertex x'),
                Plot1D('flightLengthSig', 'flightLengthSig', 20, 0, 20, 'tau flight length significance'),
                Plot1D('secondaryVertex_z', 'secondaryVertex_z', 40, -5, 5, 'tau secondary vertex z'),
                Plot1D('flightLength_z', 'flightLength_z', 40, -5, 5, 'tau flight length z'),
                Plot1D('flightLength_y', 'flightLength_y', 40, -1, 1, 'tau flight length y'),
                Plot1D('flightLength_x', 'flightLength_x', 40, -1, 1, 'tau flight length x'),
                Plot1D('signalConeSize', 'signalConeSize', 20, 0, 0.3, 'tau signal cone size'),
                Plot1D('hasSecondaryVertex', 'hasSecondaryVertex', 2, 0, 2, 'tau has secondary vertex'),
                Plot1D('vx', 'vx', 50, -0.1, 0.1, 'tau vx'),
                Plot1D('vy', 'vy', 50, -0.1, 0.1, 'tau vy'),
                Plot1D('jetIsValid', 'jetIsValid', 2, 0, 2, 'tau jet is valid')
            )
        ),
        hltAK4PuppiJet = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Plot1D('pt', 'pt', 20, 0, 400, 'jet pt'),
                Plot1D('eta', 'eta', 20, -2.5, 2.5, 'jet eta'),
                Plot1D('phi', 'phi', 20, -3.14159, 3.14159, 'jet phi'),
                Plot1D('mass', 'mass', 20, 0, 200, 'jet mass'),
                Plot1D('DeepFlavour_prob_bb', 'DeepFlavour_prob_bb', 20, 0, 1, 'jet DeepFlavour prob bb'),
                Plot1D('DeepFlavour_prob_b', 'DeepFlavour_prob_b', 20, 0, 1, 'jet DeepFlavour prob b'),
                Plot1D('DeepFlavour_prob_lepb', 'DeepFlavour_prob_lepb', 20, 0, 1, 'jet DeepFlavour prob lepb'),
                Plot1D('DeepFlavour_prob_g', 'DeepFlavour_prob_g', 20, 0, 1, 'jet DeepFlavour prob g'),
                Plot1D('DeepFlavour_prob_uds', 'DeepFlavour_prob_uds', 20, 0, 1, 'jet DeepFlavour prob uds'),
                Plot1D('DeepFlavour_prob_c', 'DeepFlavour_prob_c', 20, 0, 1, 'jet DeepFlavour prob c'),
                Plot1D('neEmEF', 'neEmEF', 20, 0, 1, 'jet neutral EM energy fraction'),
                Plot1D('chHEF', 'chHEF', 20, 0, 1, 'jet charged hadron energy fraction'),
                Plot1D('neHEF', 'neHEF', 20, 0, 1, 'jet neutral hadron energy fraction'),
                Plot1D('area', 'area', 20, 0, 1, 'jet area'),
                Plot1D('nCh', 'nCh', 25, 0, 50, 'jet # charged constituents'),
                Plot1D('nConstituents', 'nConstituents', 25, 0, 100, 'jet # constituents'),
                Plot1D('nNh', 'nNh', 20, 0, 20, 'jet # neutral hadrons'),
                Plot1D('nPhotons', 'nPhotons', 20, 0, 20, 'jet # photons'),
                Plot1D('nElectrons', 'nElectrons', 5, 0, 5, 'jet # electrons'),
                Plot1D('chEmEF', 'chEmEF', 20, 0, 1, 'jet charged EM energy fraction'),
                Plot1D('muEF', 'muEF', 20, 0, 1, 'jet muon energy fraction'),
                Plot1D('nMuons', 'nMuons', 5, 0, 5, 'jet # muons')
            )
        ),
        hltPFCandidate = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Plot1D('pt', 'pt', 20, 0, 400, 'pf candidate pt'),
                Plot1D('phi', 'phi', 20, -3.14159, 3.14159, 'pf candidate phi'),
                Plot1D('eta', 'eta', 20, -4., 4., 'pf candidate eta'),
                Plot1D('mass', 'mass', 20, 0, 5, 'pf candidate mass'),
                Plot1D('trackIndex', 'trackIndex', 20, 0, 255, 'pf candidate track index'),
                Plot1D('pdgId', 'pdgId', 50, -250, 250, 'pf candidate pdgId'),
                Plot1D('charge', 'charge', 5, -2, 3, 'pf candidate charge')
            )
        ),
        hltPFPuppiHT = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Plot1D('pt', 'pt', 50, 0, 2000, 'PF Puppi HT')
            )
        ),
        hltPFPuppiMET = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Plot1D('pt', 'pt', 20, 0, 400, 'PF Puppi MET')
            )
        ),
        GenPart = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Plot1D('pt', 'pt', 20, 0, 400, 'genParticle pt'),
                Plot1D('eta', 'eta', 20, -4., 4., 'genParticle eta'),
                Plot1D('phi', 'phi', 20, -3.14159, 3.14159, 'genParticle phi')
            )
        ),
        TriggerObject = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Plot1D('HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepFlavour_2p4', 'HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepFlavour_2p4', 2, 0, 2, 'matched HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepFlavour_2p4'),
                Plot1D('HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepFlavour_2p4', 'HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepFlavour_2p4', 2, 0, 2, 'matched HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepFlavour_2p4'),
                Plot1D('HLT_Ele30_WPTight_L1Seeded_LooseDeepTauPFTauHPS30_eta2p1_CrossL1', 'HLT_Ele30_WPTight_L1Seeded_LooseDeepTauPFTauHPS30_eta2p1_CrossL1', 2, 0, 2, 'matched HLT_Ele30_WPTight_L1Seeded_LooseDeepTauPFTauHPS30_eta2p1_CrossL1'),
                Plot1D('HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepFlavour_2p4', 'HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepFlavour_2p4', 2, 0, 2, 'matched HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepFlavour_2p4'),
                Plot1D('HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1', 'HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1', 2, 0, 2, 'matched HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1'),
                Plot1D('HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepCSV_2p4', 'HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepCSV_2p4', 2, 0, 2, 'matched HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepCSV_2p4'),
                Plot1D('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_FromL1TkMuon', 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_FromL1TkMuon', 2, 0, 2, 'matched HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_FromL1TkMuon'),
                Plot1D('HLT_DoubleMediumChargedIsoPFTauHPS40_eta2p1', 'HLT_DoubleMediumChargedIsoPFTauHPS40_eta2p1', 2, 0, 2, 'matched HLT_DoubleMediumChargedIsoPFTauHPS40_eta2p1'),
                Plot1D('HLT_Photon108EB_TightID_TightIso_Unseeded', 'HLT_Photon108EB_TightID_TightIso_Unseeded', 2, 0, 2, 'matched HLT_Photon108EB_TightID_TightIso_Unseeded'),
                Plot1D('HLT_Photon108EB_TightID_TightIso_L1Seeded', 'HLT_Photon108EB_TightID_TightIso_L1Seeded', 2, 0, 2, 'matched HLT_Photon108EB_TightID_TightIso_L1Seeded'),
                Plot1D('HLT_DoubleMediumDeepTauPFTauHPS35_eta2p1', 'HLT_DoubleMediumDeepTauPFTauHPS35_eta2p1', 2, 0, 2, 'matched HLT_DoubleMediumDeepTauPFTauHPS35_eta2p1'),
                Plot1D('HLT_PFPuppiMETTypeOne140_PFPuppiMHT140', 'HLT_PFPuppiMETTypeOne140_PFPuppiMHT140', 2, 0, 2, 'matched HLT_PFPuppiMETTypeOne140_PFPuppiMHT140'),
                Plot1D('HLT_DoubleEle25_CaloIdL_PMS2_Unseeded', 'HLT_DoubleEle25_CaloIdL_PMS2_Unseeded', 2, 0, 2, 'matched HLT_DoubleEle25_CaloIdL_PMS2_Unseeded'),
                Plot1D('HLT_DoubleEle25_CaloIdL_PMS2_L1Seeded', 'HLT_DoubleEle25_CaloIdL_PMS2_L1Seeded', 2, 0, 2, 'matched HLT_DoubleEle25_CaloIdL_PMS2_L1Seeded'),
                Plot1D('HLT_Diphoton30_23_IsoCaloId_Unseeded', 'HLT_Diphoton30_23_IsoCaloId_Unseeded', 2, 0, 2, 'matched HLT_Diphoton30_23_IsoCaloId_Unseeded'),
                Plot1D('HLT_Diphoton30_23_IsoCaloId_L1Seeded', 'HLT_Diphoton30_23_IsoCaloId_L1Seeded', 2, 0, 2, 'matched HLT_Diphoton30_23_IsoCaloId_L1Seeded'),
                Plot1D('HLT_TriMu_10_5_5_DZ_FromL1TkMuon', 'HLT_TriMu_10_5_5_DZ_FromL1TkMuon', 2, 0, 2, 'matched HLT_TriMu_10_5_5_DZ_FromL1TkMuon'),
                Plot1D('HLT_DoubleEle23_12_Iso_L1Seeded', 'HLT_DoubleEle23_12_Iso_L1Seeded', 2, 0, 2, 'matched HLT_DoubleEle23_12_Iso_L1Seeded'),
                Plot1D('HLT_Mu37_Mu27_FromL1TkMuon', 'HLT_Mu37_Mu27_FromL1TkMuon', 2, 0, 2, 'matched HLT_Mu37_Mu27_FromL1TkMuon'),
                Plot1D('HLT_Ele32_WPTight_Unseeded', 'HLT_Ele32_WPTight_Unseeded', 2, 0, 2, 'matched HLT_Ele32_WPTight_Unseeded'),
                Plot1D('HLT_Ele32_WPTight_L1Seeded', 'HLT_Ele32_WPTight_L1Seeded', 2, 0, 2, 'matched HLT_Ele32_WPTight_L1Seeded'),
                Plot1D('HLT_Ele115_NonIso_L1Seeded', 'HLT_Ele115_NonIso_L1Seeded', 2, 0, 2, 'matched HLT_Ele115_NonIso_L1Seeded'),
                Plot1D('HLT_IsoMu24_FromL1TkMuon', 'HLT_IsoMu24_FromL1TkMuon', 2, 0, 2, 'matched HLT_IsoMu24_FromL1TkMuon'),
                Plot1D('HLT_Ele26_WP70_Unseeded', 'HLT_Ele26_WP70_Unseeded', 2, 0, 2, 'matched HLT_Ele26_WP70_Unseeded'),
                Plot1D('HLT_Ele26_WP70_L1Seeded', 'HLT_Ele26_WP70_L1Seeded', 2, 0, 2, 'matched HLT_Ele26_WP70_L1Seeded'),
                Plot1D('HLT_Photon187_Unseeded', 'HLT_Photon187_Unseeded', 2, 0, 2, 'matched HLT_Photon187_Unseeded'),
                Plot1D('HLT_Photon187_L1Seeded', 'HLT_Photon187_L1Seeded', 2, 0, 2, 'matched HLT_Photon187_L1Seeded'),
                Plot1D('HLT_Mu50_FromL1TkMuon', 'HLT_Mu50_FromL1TkMuon', 2, 0, 2, 'matched HLT_Mu50_FromL1TkMuon'),
                Plot1D('HLT_AK4PFPuppiJet520', 'HLT_AK4PFPuppiJet520', 2, 0, 2, 'matched HLT_AK4PFPuppiJet520'),
                Plot1D('HLT_PFPuppiHT1070', 'HLT_PFPuppiHT1070', 2, 0, 2, 'matched HLT_PFPuppiHT1070'),
                Plot1D('mass', 'mass', 20, 0, 50, 'trigger object mass'),
                Plot1D('eta', 'eta', 20, -2.5, 2.5, 'trigger object eta'),
                Plot1D('phi', 'phi', 20, -3.14159, 3.14159, 'trigger object phi'),
                Plot1D('id', 'id', 20, -50, 50, 'trigger object type id'),
                Plot1D('pt', 'pt', 20, 0, 400, 'trigger object pt')
            )
        ),
        # Add more as needed
    )
)
