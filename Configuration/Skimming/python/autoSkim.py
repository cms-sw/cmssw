autoSkim = {

 # Skim 2023
 'BTagMu' : 'LogError+LogErrorMonitor',
 'DisplacedJet' : 'EXODisplacedJet+EXODelayedJet+EXODTCluster+EXOLLPJetHCAL+LogError+LogErrorMonitor',
 'JetMET0' : 'JetHTJetPlusHOFilter+EXOHighMET+EXODelayedJetMET+EXODisappTrk+LogError+LogErrorMonitor',
 'JetMET1' : 'JetHTJetPlusHOFilter+EXOHighMET+EXODelayedJetMET+EXODisappTrk+LogError+LogErrorMonitor',
 'EGamma0':'ZElectron+WElectron+EXOMONOPOLE+EXODisappTrk+LogError+LogErrorMonitor',
 'EGamma1':'ZElectron+WElectron+EXOMONOPOLE+EXODisappTrk+LogError+LogErrorMonitor',
 'Tau' : 'EXODisappTrk+LogError+LogErrorMonitor',
 'Muon0' : 'ZMu+EXODisappTrk+EXOCSCCluster+EXODisappMuon+LogError+LogErrorMonitor',
 'Muon1' : 'ZMu+EXODisappTrk+EXOCSCCluster+EXODisappMuon+LogError+LogErrorMonitor',
 'MuonEG' : 'TopMuEG+LogError+LogErrorMonitor',
 'NoBPTX' : 'EXONoBPTXSkim+LogError+LogErrorMonitor',
 'HcalNZS' : 'LogError+LogErrorMonitor',
 'HLTPhysics' : 'LogError+LogErrorMonitor',
 'ZeroBias' : 'LogError+LogErrorMonitor',
 'Commissioning' : 'EcalActivity+LogError+LogErrorMonitor',
 'Cosmics':'CosmicSP+CosmicTP+LogError+LogErrorMonitor',
 'ParkingDoubleMuonLowMass0': 'ReserveDMu+LogError+LogErrorMonitor',
 'ParkingDoubleMuonLowMass1': 'ReserveDMu+LogError+LogErrorMonitor',
 'ParkingDoubleMuonLowMass2': 'ReserveDMu+LogError+LogErrorMonitor',
 'ParkingDoubleMuonLowMass3': 'ReserveDMu+LogError+LogErrorMonitor',
 'ParkingDoubleMuonLowMass4': 'ReserveDMu+LogError+LogErrorMonitor',
 'ParkingDoubleMuonLowMass5': 'ReserveDMu+LogError+LogErrorMonitor',
 'ParkingDoubleMuonLowMass6': 'ReserveDMu+LogError+LogErrorMonitor',
 'ParkingDoubleMuonLowMass7': 'ReserveDMu+LogError+LogErrorMonitor',

 # These should be uncommented when 2022 data reprocessing
 # Dedicated skim for 2022
 #'JetMET' : 'JetHTJetPlusHOFilter+EXOHighMET+EXODelayedJetMET+EXODisappTrk+LogError+LogErrorMonitor',
 #'EGamma':'ZElectron+WElectron+EXOMONOPOLE+EXODisappTrk+LogError+LogErrorMonitor',
 #'Muon' : 'ZMu+EXODisappTrk+EXODisappMuon+LogError+LogErrorMonitor',
 #'DisplacedJet' : 'EXODisplacedJet+EXODelayedJet+EXODTCluster+EXOCSCCluster+EXOLLPJetHCAL+LogError+LogErrorMonitor',
 #'JetHT' : 'JetHTJetPlusHOFilter+LogError+LogErrorMonitor',
 #'MET' : 'EXOHighMET+EXODelayedJetMET+EXODisappTrk+LogError+LogErrorMonitor',
 #'SingleMuon' : 'ZMu+EXODisappTrk+EXODisappMuon+LogError+LogErrorMonitor',
 #'DoubleMuon' : 'LogError+LogErrorMonitor',

 # Used in unit test scenario ppEra_Run2_2018
 'SingleMuon': 'LogError+LogErrorMonitor',
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
