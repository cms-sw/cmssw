autoSkim = {
 'BTagCSV' : 'LogError',
 'BTagMu' : 'LogError',
 'HTMHT' : 'LogError',
 'JetHT' : 'HighMET+LogError',
 'DisplacedJet' : 'LogError',
 'MET' : 'LogError',
 'SingleElectron' : 'LogError',
 'SinglePhoton' : 'LogError',
 'DoubleEG' : 'ZElectron+LogError',
 'Tau' : 'LogError',
 'SingleMuon' : 'ZMu+MuTau+LogError',
 'DoubleMuon' : 'LogError',
 'MuonEG' : 'TopMuEG+LogError',
 'DoubleMuonLowMass' : 'LogError',
 'MuOnia' : 'LogError',
 'Charmonium' : 'LogError',
 'NoBPTX' : 'LogError',
 'HcalHPDNoise' : 'LogError',
 'HcalNZS' : 'LogError',
 'HLTPhysics' : 'LogError',
 'ZeroBias' : 'LogError',
 'Commissioning' : 'LogError', #DT skim does not exist and was requested by none I know
 'Cosmics':'CosmicSP+CosmicTP+LogError'
}

autoSkimRunI = {
    'MinBias':'MuonTrack+BeamBkg+ValSkim+LogError+HSCPSD',
    'ZeroBias':'LogError',
    'Commissioning':'DT+LogError',
    'Cosmics':'CosmicSP+CosmicTP+LogError',
    'Mu' : 'WMu+ZMu+HighMET+LogError',    
    'EG':'WElectron+ZElectron+HighMET+LogError',
    'TopMuEG':'TopMuEG+LogError',
    'Electron':'WElectron+ZElectron+HighMET+LogError',
    'Photon':'WElectron+ZElectron+HighMET+LogError+DiPhoton+EXOHPTE',
    'JetMETTau':'LogError+Tau',
    'JetMET':'HighMET+LogError',
    'BTau':'LogError+Tau',
    'Jet':'HighMET+LogError',
    'METFwd':'HighMET+LogError',
    'SingleMu' : 'WMu+ZMu+HighMET+LogError+HWW+HZZ+DiTau+EXOHSCP',
    'DoubleMu' : 'WMu+ZMu+HighMET+LogError+HWW+HZZ+EXOHSCP',
    'SingleElectron' : 'WElectron+HighMET+LogError+HWW+HZZ+Tau',
    'DoubleElectron' : 'ZElectron+LogError+HWW+HZZ',
    'MuEG' : 'LogError+HWW+HZZ',
    'METBTag': 'HighMET+LogError+EXOHSCP',
    'BTag': 'LogError+EXOHSCP',
    'MET': 'HighMET+LogError+EXOHSCP',
    'HighMET': 'HighMET+LogError',

    'HT': 'HighMET+LogError',

    'Tau': 'LogError',
    'MuTau': 'MuTau+LogError',
    'PhotonHad': 'LogError',
    'MuHad': 'LogError',
    'MultiJet': 'LogError',
    'MuOnia': 'LogError',
    'ElectronHad': 'LogError',
    'TauPlusX': 'LogError',
    
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
