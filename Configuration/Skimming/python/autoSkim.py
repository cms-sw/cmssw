#The autoSkim should be reviewed for Run-3 when PDs are available
autoSkim = {
 'BTagCSV' : 'LogError+LogErrorMonitor',
 'BTagMu' : 'LogError+LogErrorMonitor',
 'HTMHT' : 'LogError+LogErrorMonitor',
 'JetHT' : 'JetHTJetPlusHOFilter+LogError+LogErrorMonitor',
 'DisplacedJet' : 'LogError+LogErrorMonitor',
 'MET' : 'HighMET+LogError+LogErrorMonitor',
 'SingleElectron' : 'LogError+LogErrorMonitor', #to be updated if we will have EGamma as Run-2 (2018), or splitting as 2016,2017
 'SinglePhoton' : 'SinglePhotonJetPlusHOFilter+EXOMONOPOLE+LogError+LogErrorMonitor', #to be updated if we will have EGamma as Run-2 (2018), or splitting as 2016,2017
 'DoubleEG' : 'ZElectron+LogError+LogErrorMonitor', #to be updated if we will have EGamma as Run-2 (2018), or splitting as 2016,2017
 'EGamma':'SinglePhotonJetPlusHOFilter+ZElectron+EXOMONOPOLE+LogError+LogErrorMonitor',
 'Tau' : 'LogError+LogErrorMonitor',
 'SingleMuon' : 'MuonPOGSkim+ZMu+MuTau+LogError+LogErrorMonitor',
 'DoubleMuon' : 'LogError+LogErrorMonitor',
 'MuonEG' : 'TopMuEG+LogError+LogErrorMonitor',
 'DoubleMuonLowMass' : 'BPHSkim+LogError+LogErrorMonitor',
 'MuOnia' : 'BPHSkim+LogError+LogErrorMonitor',
 'Charmonium' : 'MuonPOGJPsiSkim+BPHSkim+LogError+LogErrorMonitor',
 'NoBPTX' : 'EXONoBPTXSkim+LogError+LogErrorMonitor',
 'HcalHPDNoise' : 'LogError+LogErrorMonitor',
 'HcalNZS' : 'LogError+LogErrorMonitor',
 'HLTPhysics' : 'LogError+LogErrorMonitor',
 'ZeroBias' : 'LogError+LogErrorMonitor',
 'Commissioning' : 'EcalActivity+LogError+LogErrorMonitor',
 'Cosmics':'CosmicSP+CosmicTP+LogError+LogErrorMonitor',
 'ParkingBPH':'SkimBPark+LogError+LogErrorMonitor',
}

autoSkimRunII = {
 'BTagCSV' : 'LogError+LogErrorMonitor',
 'BTagMu' : 'LogError+LogErrorMonitor',
 'HTMHT' : 'LogError+LogErrorMonitor',
 'JetHT' : 'JetHTJetPlusHOFilter+LogError+LogErrorMonitor',
 'DisplacedJet' : 'LogError+LogErrorMonitor',
 'MET' : 'HighMET+EXOMONOPOLE+LogError+LogErrorMonitor',
 'SingleElectron' : 'LogError+LogErrorMonitor',
 'SinglePhoton' : 'SinglePhotonJetPlusHOFilter+EXOMONOPOLE+LogError+LogErrorMonitor',
 'DoubleEG' : 'ZElectron+EXOMONOPOLE+LogError+LogErrorMonitor',
 'EGamma':'SinglePhotonJetPlusHOFilter+ZElectron+EXOMONOPOLE+LogError+LogErrorMonitor',
 'Tau' : 'LogError+LogErrorMonitor',
 'SingleMuon' : 'MuonPOGSkim+ZMu+MuTau+LogError+LogErrorMonitor',
 'DoubleMuon' : 'LogError+LogErrorMonitor',
 'MuonEG' : 'TopMuEG+LogError+LogErrorMonitor',
 'DoubleMuonLowMass' : 'BPHSkim+LogError+LogErrorMonitor',
 'MuOnia' : 'BPHSkim+LogError+LogErrorMonitor',
 'Charmonium' : 'MuonPOGJPsiSkim+BPHSkim+LogError+LogErrorMonitor',
 'NoBPTX' : 'EXONoBPTXSkim+LogError+LogErrorMonitor',
 'HcalHPDNoise' : 'LogError+LogErrorMonitor',
 'HcalNZS' : 'LogError+LogErrorMonitor',
 'HLTPhysics' : 'LogError+LogErrorMonitor',
 'ZeroBias' : 'LogError+LogErrorMonitor',
 'Commissioning' : 'EcalActivity+LogError+LogErrorMonitor',
 'Cosmics':'CosmicSP+CosmicTP+LogError+LogErrorMonitor',
 'ParkingBPH':'SkimBPark+LogError+LogErrorMonitor',
}
#2018 EGamma is a merged datasets of SingleElectron, SinglePhoton, DoubleEG

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
