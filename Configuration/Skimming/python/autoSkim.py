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

    'SingleMu' : 'WMu+ZMu+HighMET+LogError+HWWMuMu+DiTau',
    'DoubleMu' : 'WMu+ZMu+HighMET+LogError+HWWMuMu',
    'SingleElectron' : 'LogError+Tau',
    'DoubleElectron' : 'LogError+HWWElEl',
    'MuEG' : 'LogError+HWWElMu',

    'Tau': 'LogError',
    'PhotonHad': 'LogError',
    'MuHad': 'LogError',
    'METBTag': 'LogError',
    'MultiJet': 'LogError',
    'MuOnia': 'LogError',
    'ElectronHad': 'LogError',
    'TauPlusX': 'LogError',
    'HT': 'LogError'
    
    }


autoSkimPDWG = {
    
    }

autoSkimDPG = {

    }

def mergeMapping(map1,map2):
    merged={}
    for k in list(set(map1.keys()+map2.keys())):
        items=[]
        if k in map1: 
            items.append(map1[k])
        if k in map2:
            items.append(map2[k])
        merged[k]='+'.join(items)
    return merged
    
#autoSkim = mergeMapping(autoSkimPDWG,autoSkimDPG)
#print autoSkim
