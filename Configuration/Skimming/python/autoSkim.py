autoSkim = {
 'BTagCSV' : 'LogError+LogErrorMonitor',
 'BTagMu' : 'LogError+LogErrorMonitor',
 'HTMHT' : 'LogError+LogErrorMonitor',
 'JetHT' : 'HighMET+LogError+LogErrorMonitor',
 'DisplacedJet' : 'LogError+LogErrorMonitor',
 'MET' : 'LogError+LogErrorMonitor',
 'SingleElectron' : 'LogError+LogErrorMonitor',
 'SinglePhoton' : 'LogError+LogErrorMonitor',
 'DoubleEG' : 'ZElectron+LogError+LogErrorMonitor',
 'Tau' : 'LogError+LogErrorMonitor',
 'SingleMuon' : 'ZMu+MuTau+LogError+LogErrorMonitor',
 'DoubleMuon' : 'LogError+LogErrorMonitor',
 'MuonEG' : 'TopMuEG+LogError+LogErrorMonitor',
 'DoubleMuonLowMass' : 'LogError+LogErrorMonitor',
 'MuOnia' : 'LogError+LogErrorMonitor',
 'Charmonium' : 'LogError+LogErrorMonitor',
 'NoBPTX' : 'LogError+LogErrorMonitor',
 'HcalHPDNoise' : 'LogError+LogErrorMonitor',
 'HcalNZS' : 'LogError+LogErrorMonitor',
 'HLTPhysics' : 'LogError+LogErrorMonitor',
 'ZeroBias' : 'LogError+LogErrorMonitor',
 'Commissioning' : 'EcalActivity+LogError+LogErrorMonitor',
 'Cosmics':'CosmicSP+CosmicTP+LogError+LogErrorMonitor'
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
