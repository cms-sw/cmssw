## autoSkim 2012 (7E33 HLT menu) --> starting from Run2012B, ... 
autoSkim = {
    'BJetPlusX' : 'LogError+LogErrorMonitor',
    'BTag' : 'LogError+LogErrorMonitor+HighLumi',
    'Commissioning' : 'LogError+LogErrorMonitor+EcalActivity',
    'Cosmics' : 'LogError+LogErrorMonitor+CosmicSP',
    'DoubleElectron' : 'LogError+LogErrorMonitor+ZElectron+DiTau+HZZ+HighLumi',
    'DoubleMu' : 'LogError+LogErrorMonitor+Zmmg+HZZ+EXOHSCP+HighLumi',
    'DoublePhoton' : 'LogError+LogErrorMonitor',
    'DoublePhotonHighPt' : 'LogError+LogErrorMonitor',
    'ElectronHad' : 'LogError+LogErrorMonitor',
    'HTMHT' : 'LogError+LogErrorMonitor+HighMET',
    'HcalNZS' : 'LogError+LogErrorMonitor',
    'JetHT' : 'LogError+LogErrorMonitor+EXOHSCP+HighLumi',
    'JetMon' : 'LogError+LogErrorMonitor+HighLumi',
    'MET' : 'LogError+LogErrorMonitor+ZHbb+HighMET+EXOHSCP',
    'MinimumBias' : 'LogError+LogErrorMonitor+HLTPhysics+HighLumi',
    'MuEG' : 'LogError+LogErrorMonitor+HZZ',
    'MuHad' : 'LogError+LogErrorMonitor',
    'MuOnia' : 'LogError+LogErrorMonitor+ChiB',
    'MultiJet' : 'LogError+LogErrorMonitor+HighMET+HighLumi',
    'NoBPTX' : 'LogError+LogErrorMonitor+EXOHSCP',
    'PhotonHad' : 'LogError+LogErrorMonitor+EXOMonoPhoton',
    'SingleElectron' : 'LogError+LogErrorMonitor+WElectron+HighMET+TOPElePlusJets+DiTau',
    'SingleMu' : 'LogError+LogErrorMonitor+ZMu+HighMET+EXOHSCP+TOPMuPlusJets+MuTau',
    'SinglePhoton' : 'LogError+LogErrorMonitor+EXODisplacedPhoton+HighMET+EXOMonoPhoton',
    'Tau' : 'LogError+LogErrorMonitor',
    'TauPlusX' : 'LogError+LogErrorMonitor+MuTauMET',
    }

## autoSkim 2012 (5E33 HLT menu) --> only Run2012A
"""
autoSkim = {
    'BTag' : 'LogError+LogErrorMonitor+HighLumi',
    'Commissioning' : 'LogError+LogErrorMonitor+EcalActivity',
    'Cosmics' : 'LogError+LogErrorMonitor+CosmicSP',    
    'DoubleElectron' : 'LogError+LogErrorMonitor+ZElectron+DiTau+HZZ+HighLumi',
    'DoubleMu' : 'LogError+LogErrorMonitor+HZZ+Zmmg+EXOHSCP+HighLumi',
    'ElectronHad' : 'LogError+LogErrorMonitor+EXOHSCP',
    'HcalNZS' : 'LogError+LogErrorMonitor',
    'HT' : 'LogError+LogErrorMonitor+EXOHSCP+HighMET',
    'Jet' : 'LogError+LogErrorMonitor+HighLumi',
    'MET' : 'LogError+LogErrorMonitor+ZHbb+EXOHSCP+HighMET',
    'MinimumBias' : 'LogError+LogErrorMonitor+HLTPhysics+HighLumi',    
    'MuEG' : 'LogError+LogErrorMonitor+HZZ',
    'MuHad' : 'LogError+LogErrorMonitor',
    'MuOnia' : 'LogError+LogErrorMonitor+ChiB',
    'MultiJet' : 'LogError+LogErrorMonitor+HighMET+HighLumi',
    'Photon' : 'LogError+LogErrorMonitor+HighMET+EXOMonoPhoton',
    'PhotonHad' : 'LogError+LogErrorMonitor+EXOMonoPhoton',
    'SingleElectron' : 'LogError+LogErrorMonitor+HighMET+DiTau+WElectron',
    'SingleMu' : 'LogError+LogErrorMonitor+ZMu+HighMET+EXOHSCP+MuTau',
    'Tau' : 'LogError+LogErrorMonitor',
    'TauPlusX' : 'LogError+LogErrorMonitor+MuTauMET',
    }
"""

# IMPORTANT NOTE for Run2012A :
# TOPElePlusJets should go in ElectronHad
# TOPMuPlusJets should go to MuHad
# However, the configurations in the release, used for prompt skimming
# starting from Run2012B, are not good for Run2012A (since trigger names changed).
# So the TOP skims above have been removed from the Run2012A skim matrix
# and cannot be produced at the moment for this run period


## autoSkim 2013 (pPb HLT menu) --> only pPb run (Jan-Feb)
"""
autoSkim = {
    'PAMuon' : 'PsiMuMuPA+UpsMuMuPA+ZMuMuPA+HighPtPA',
    'PAHighPt' : 'HighPtPA+FlowCorrPA',
    }
"""

########################################################
### 53X re-processing in 2013: pp data collected in 2012
### (Created on Jan 21th, 2013 by Francesco Santanastasio)
########################################################

## Run2012A (53X reprocessing)
"""
autoSkim = {
    'BTag' : 'LogError+LogErrorMonitor+HighLumi',
    'Commissioning' : 'LogError+LogErrorMonitor+EcalActivity',
    'Cosmics' : 'LogError+LogErrorMonitor+CosmicSP',
    'DoubleElectron' : 'LogError+LogErrorMonitor+ZElectron+DiTau+HZZ+HighLumi',
    'DoubleMu' : 'LogError+LogErrorMonitor+HZZ+Zmmg+EXOHSCP+HighLumi',
    'ElectronHad' : 'LogError+LogErrorMonitor+EXOHSCP',
    'HcalNZS' : 'LogError+LogErrorMonitor',
    'HT' : 'LogError+LogErrorMonitor+EXOHSCP+HighMET',
    'Jet' : 'LogError+LogErrorMonitor+HighLumi',
    'MET' : 'LogError+LogErrorMonitor+ZHbb+EXOHSCP+HighMET',
    'MinimumBias' : 'LogError+LogErrorMonitor+HLTPhysics+HighLumi',
    'MuEG' : 'LogError+LogErrorMonitor+HZZ',
    'MuHad' : 'LogError+LogErrorMonitor',
    'MuOnia' : 'LogError+LogErrorMonitor+ChiB',
    'MultiJet' : 'LogError+LogErrorMonitor+HighMET+HighLumi',
    'Photon' : 'LogError+LogErrorMonitor+HighMET+EXOMonoPhoton',
    'PhotonHad' : 'LogError+LogErrorMonitor+EXOMonoPhoton',
    'SingleElectron' : 'LogError+LogErrorMonitor+HighMET+DiTau+WElectron',
    'SingleMu' : 'LogError+LogErrorMonitor+ZMu+HighMET+EXOHSCP+MuTau',
    'Tau' : 'LogError+LogErrorMonitor',
    'TauPlusX' : 'LogError+LogErrorMonitor+MuTauMET',
    }
"""

## Run2012B and Run2012C (53X reprocessing)
"""
autoSkim = {
    'BJetPlusX' : 'LogError+LogErrorMonitor',
    'BTag' : 'LogError+LogErrorMonitor+HighLumi',
    'Commissioning' : 'LogError+LogErrorMonitor+EcalActivity',
    'Cosmics' : 'LogError+LogErrorMonitor+CosmicSP',
    'DoubleElectron' : 'LogError+LogErrorMonitor+ZElectron+DiTau+HZZ+HighLumi',
    'DoubleMuParked' : 'LogError+LogErrorMonitor+Zmmg+HZZ+EXOHSCP+HighLumi',
    'DoublePhoton' : 'LogError+LogErrorMonitor',
    'DoublePhotonHighPt' : 'LogError+LogErrorMonitor',
    'ElectronHad' : 'LogError+LogErrorMonitor',
    'HTMHTParked' : 'LogError+LogErrorMonitor+HighMET',
    'HcalNZS' : 'LogError+LogErrorMonitor',
    'JetHT' : 'LogError+LogErrorMonitor+EXOHSCP+HighLumi',
    'JetMon' : 'LogError+LogErrorMonitor+HighLumi',
    'MET' : 'LogError+LogErrorMonitor+ZHbb+HighMET+EXOHSCP',
    'MinimumBias' : 'LogError+LogErrorMonitor+HLTPhysics+HighLumi',
    'MuEG' : 'LogError+LogErrorMonitor+HZZ',
    'MuHad' : 'LogError+LogErrorMonitor',
    'MuOnia' : 'LogError+LogErrorMonitor+ChiB',
    'MuOniaParked' : 'LogError+LogErrorMonitor',
    'NoBPTX' : 'LogError+LogErrorMonitor+EXOHSCP',
    'PhotonHad' : 'LogError+LogErrorMonitor+EXOMonoPhoton',
    'SingleElectron' : 'LogError+LogErrorMonitor+WElectron+HighMET+TOPElePlusJets+DiTau',
    'SingleMu' : 'LogError+LogErrorMonitor+ZMu+HighMET+EXOHSCP+TOPMuPlusJets+MuTau',
    'SinglePhoton' : 'LogError+LogErrorMonitor+EXODisplacedPhoton+HighMET+EXOMonoPhoton',
    'TauParked' : 'LogError+LogErrorMonitor',
    'TauPlusX' : 'LogError+LogErrorMonitor+MuTauMET',
    'VBF1Parked' : 'LogError+LogErrorMonitor',
    }
"""

# --> NOTE for Run2012B,C: we have remove this
# 'MultiJet1Parked' : 'LogError+LogErrorMonitor+HighMET+HighLumi',
# since the re-reco already happened at the end of 2012.
# If those skims can run with RAW and AOD in input we could produce them later in 2013.


## Run2012D (53X reprocessing)
"""
autoSkim = {
    'BJetPlusX' : 'LogError+LogErrorMonitor',
    'BTag' : 'LogError+LogErrorMonitor+HighLumi',
    'Commissioning' : 'LogError+LogErrorMonitor+EcalActivity',
    'Cosmics' : 'LogError+LogErrorMonitor+CosmicSP',
    'DoubleElectron' : 'LogError+LogErrorMonitor+ZElectron+DiTau+HZZ+HighLumi',
    'DoubleMuParked' : 'LogError+LogErrorMonitor+Zmmg+HZZ+EXOHSCP+HighLumi',
    'DoublePhoton' : 'LogError+LogErrorMonitor',
    'DoublePhotonHighPt' : 'LogError+LogErrorMonitor',
    'ElectronHad' : 'LogError+LogErrorMonitor',
    'HTMHTParked' : 'LogError+LogErrorMonitor+HighMET',
    'HcalNZS' : 'LogError+LogErrorMonitor',
    'JetHT' : 'LogError+LogErrorMonitor+EXOHSCP+HighLumi',
    'JetMon' : 'LogError+LogErrorMonitor+HighLumi',
    'METParked' : 'LogError+LogErrorMonitor+ZHbb+HighMET+EXOHSCP',
    'MinimumBias' : 'LogError+LogErrorMonitor+HLTPhysics+HighLumi',
    'MuEG' : 'LogError+LogErrorMonitor+HZZ',
    'MuHad' : 'LogError+LogErrorMonitor',
    'MuOnia' : 'LogError+LogErrorMonitor+ChiB',
    'MuOniaParked' : 'LogError+LogErrorMonitor',
    'MultiJet1Parked' : 'LogError+LogErrorMonitor+HighMET+HighLumi',
    'NoBPTX' : 'LogError+LogErrorMonitor+EXOHSCP',
    'PhotonHad' : 'LogError+LogErrorMonitor+EXOMonoPhoton',
    'SingleElectron' : 'LogError+LogErrorMonitor+WElectron+HighMET+TOPElePlusJets+DiTau',
    'SingleMu' : 'LogError+LogErrorMonitor+ZMu+HighMET+EXOHSCP+TOPMuPlusJets+MuTau',
    'SinglePhotonParked' : 'LogError+LogErrorMonitor+EXODisplacedPhoton+HighMET+EXOMonoPhoton',
    'TauParked' : 'LogError+LogErrorMonitor',
    'TauPlusX' : 'LogError+LogErrorMonitor+MuTauMET',
    'VBF1Parked' : 'LogError+LogErrorMonitor',
    'HLTPhysicsParked' : 'LogError+LogErrorMonitor',
    'ZeroBiasParked' : 'LogError+LogErrorMonitor',
    }
"""

## autoSkim 2011
"""
    'MinimumBias':'MuonTrack+BeamBkg+ValSkim+LogError+HSCPSD',
    'ZeroBias':'LogError',
    'Commissioning':'DT+LogError',
    'Cosmics':'CosmicSP+LogError',
    'Mu' : 'WMu+ZMu+HighMET+LogError',    
    'EG':'WElectron+ZElectron+HighMET+LogError',
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

    'HT': 'HighMET+LogError',

    'Tau': 'LogError',
    'PhotonHad': 'LogError',
    'MuHad': 'LogError',
    'MultiJet': 'LogError',
    'MuOnia': 'LogError',
    'ElectronHad': 'LogError',
    'TauPlusX': 'LogError',
"""

## autoSkim 2011 used for legacy 53x reRECO
#  removed from original: Tau, HWW, WMu, DiPhoton, EXOHPTE, MuonTrack+BeamBkg+ValSkim+HSCPSD, DT 
"""
    'MinimumBias':'LogError',
    'ZeroBias':'LogError',
    'Commissioning':'LogError',
    'Cosmics':'CosmicSP+LogError',
    'Mu' : 'ZMu+HighMET+LogError',    
    'EG':'WElectron+ZElectron+HighMET+LogError',
    'Electron':'WElectron+ZElectron+HighMET+LogError',
    'Photon':'WElectron+ZElectron+HighMET+LogError',
    'JetMETTau':'LogError',
    'JetMET':'HighMET+LogError',
    'BTau':'LogError',
    'Jet':'HighMET+LogError',
    'METFwd':'HighMET+LogError',
    'SingleMu' : 'ZMu+HighMET+LogError+HZZ+DiTau+EXOHSCP',
    'DoubleMu' : 'ZMu+HighMET+LogError+HZZ+EXOHSCP',
    'SingleElectron' : 'WElectron+HighMET+LogError+HZZ',
    'DoubleElectron' : 'ZElectron+LogError+HZZ',
    'MuEG' : 'LogError+HZZ',
    'METBTag': 'HighMET+LogError+EXOHSCP',
    'BTag': 'LogError+EXOHSCP',
    'MET': 'HighMET+LogError+EXOHSCP',

    'HT': 'HighMET+LogError',

    'Tau': 'LogError',
    'PhotonHad': 'LogError',
    'MuHad': 'LogError',
    'MultiJet': 'LogError',
    'MuOnia': 'LogError',
    'ElectronHad': 'LogError',
    'TauPlusX': 'LogError',
"""

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
