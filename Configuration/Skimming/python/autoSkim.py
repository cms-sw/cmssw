autoSkim = {
    'MinimumBias':'MuonTrack+BeamBkg+ValSkim+LogError+HSCPSD',
    'ZeroBias':'LogError',
    'Commissioning':'MuonDPG+LogError',
    'Cosmics':'CosmicSP+LogError',
    'Mu' : 'WMu+ZMu+HighMET+LogError',    
    'EG':'WElectron+ZElectron+HighMET+LogError',
    'Electron':'WElectron+ZElectron+HighMET+LogError',
    'Photon':'WElectron+ZElectron+HighMET+LogError+DiPhoton+DoublePhoton',
    'JetMETTau':'LogError+DiJet+Tau',
    'JetMET':'HighMET+LogError+DiJet',
    'BTau':'LogError+Tau',
    'Jet':'HighMET+LogError+DiJet',
    'METFwd':'HighMET+LogError',

    'SingleMu' : 'WMu+ZMu+HighMET+LogError+HWWMuMu',
    'DoubleMu' : 'WMu+ZMu+HighMET+LogError+HWWMuMu',
    'DoubleElectron' : 'LogError+HWWElEl',
    'MuEG' : 'LogError+HWWElMu',
    
    }
