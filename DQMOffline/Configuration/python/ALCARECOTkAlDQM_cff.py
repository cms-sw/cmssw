import FWCore.ParameterSet.Config as cms
import DQM.TrackingMonitor.TrackingMonitor_cfi

#Below all DQM modules for TrackerAlignment AlCaRecos are instanciated.
#TkAlZMuMu
ALCARECOTkAlZMuMuDQM = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
#TkAlJpsiMuMu
ALCARECOTkAlJpsiMuMuDQM = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
#TkAlUpsilonMuMu
ALCARECOTkAlUpsilonMuMuDQM = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
#TkAlBeamHalo
ALCARECOTkAlBeamHaloDQM = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
#TkAlMinBias
ALCARECOTkAlMinBiasDQM = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
#TkAlMuonIsolated
ALCARECOTkAlMuonIsolatedDQM = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
#names & designations  
ALCARECOTkAlZMuMuDQM.TrackProducer = 'ALCARECOTkAlZMuMu'
ALCARECOTkAlZMuMuDQM.AlgoName = 'ALCARECOTkAlZMuMu'
ALCARECOTkAlZMuMuDQM.FolderName = 'TkAlZMuMu'
#sizes
ALCARECOTkAlZMuMuDQM.TkSizeBin = 5
ALCARECOTkAlZMuMuDQM.TkSizeMin = 0
ALCARECOTkAlZMuMuDQM.TkSizeMax = 5
ALCARECOTkAlZMuMuDQM.TrackPtBin = 100
ALCARECOTkAlZMuMuDQM.TrackPtMin = 0
ALCARECOTkAlZMuMuDQM.TrackPtMax = 100
#names & designations    
ALCARECOTkAlJpsiMuMuDQM.TrackProducer = 'ALCARECOTkAlJpsiMuMu'
ALCARECOTkAlJpsiMuMuDQM.AlgoName = 'ALCARECOTkAlJpsiMuMu'
ALCARECOTkAlJpsiMuMuDQM.FolderName = 'TkAlJpsiMuMu'
#sizes  
ALCARECOTkAlJpsiMuMuDQM.TkSizeBin = 5
ALCARECOTkAlJpsiMuMuDQM.TkSizeMin = 0
ALCARECOTkAlJpsiMuMuDQM.TkSizeMax = 5
ALCARECOTkAlJpsiMuMuDQM.TrackPtBin = 100
ALCARECOTkAlJpsiMuMuDQM.TrackPtMin = 0
ALCARECOTkAlJpsiMuMuDQM.TrackPtMax = 30
#names & designations  
ALCARECOTkAlUpsilonMuMuDQM.TrackProducer = 'ALCARECOTkAlUpsilonMuMu'
ALCARECOTkAlUpsilonMuMuDQM.AlgoName = 'ALCARECOTkAlUpsilonMuMu'
ALCARECOTkAlUpsilonMuMuDQM.FolderName = 'TkAlUpsilonMuMu'
#sizes
ALCARECOTkAlUpsilonMuMuDQM.TkSizeBin = 5
ALCARECOTkAlUpsilonMuMuDQM.TkSizeMin = 0
ALCARECOTkAlUpsilonMuMuDQM.TkSizeMax = 5
ALCARECOTkAlUpsilonMuMuDQM.TrackPtBin = 100
ALCARECOTkAlUpsilonMuMuDQM.TrackPtMin = 0
ALCARECOTkAlUpsilonMuMuDQM.TrackPtMax = 30
#names & designations  
ALCARECOTkAlBeamHaloDQM.TrackProducer = 'ALCARECOTkAlBeamHalo'
ALCARECOTkAlBeamHaloDQM.AlgoName = 'ALCARECOTkAlBeamHalo'
ALCARECOTkAlBeamHaloDQM.FolderName = 'TkAlBeamHalo'
#sizes
ALCARECOTkAlBeamHaloDQM.TkSizeBin = 5
ALCARECOTkAlBeamHaloDQM.TkSizeMin = 0
ALCARECOTkAlBeamHaloDQM.TkSizeMax = 5
#names & designations  
ALCARECOTkAlMinBiasDQM.TrackProducer = 'ALCARECOTkAlMinBias'
ALCARECOTkAlMinBiasDQM.AlgoName = 'ALCARECOTkAlMinBias'
ALCARECOTkAlMinBiasDQM.FolderName = 'TkAlMinBias'
#sizes
ALCARECOTkAlMinBiasDQM.TkSizeBin = 70
ALCARECOTkAlMinBiasDQM.TkSizeMin = 0
ALCARECOTkAlMinBiasDQM.TkSizeMax = 70
ALCARECOTkAlMinBiasDQM.TrackPtBin = 100
ALCARECOTkAlMinBiasDQM.TrackPtMin = 0
ALCARECOTkAlMinBiasDQM.TrackPtMax = 30
#names & designations  
ALCARECOTkAlMuonIsolatedDQM.TrackProducer = 'ALCARECOTkAlMuonIsolated'
ALCARECOTkAlMuonIsolatedDQM.AlgoName = 'ALCARECOTkAlMuonIsolated'
ALCARECOTkAlMuonIsolatedDQM.FolderName = 'TkAlMuonIsolated'
#sizes			    
ALCARECOTkAlMuonIsolatedDQM.TkSizeBin = 5
ALCARECOTkAlMuonIsolatedDQM.TkSizeMin = 0
ALCARECOTkAlMuonIsolatedDQM.TkSizeMax = 5
ALCARECOTkAlMuonIsolatedDQM.TrackPtBin = 100
ALCARECOTkAlMuonIsolatedDQM.TrackPtMin = 0
ALCARECOTkAlMuonIsolatedDQM.TrackPtMax = 100
###### DQM modules for cosmic data taking ######
### TkAlCosmicsCTF0T ###
ALCARECOTkAlCosmicsCTF0TDQM = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
# names & designations  
ALCARECOTkAlCosmicsCTF0TDQM.TrackProducer = 'ALCARECOTkAlCosmicsCTF0T'
ALCARECOTkAlCosmicsCTF0TDQM.AlgoName = 'ALCARECOTkAlCosmicsCTF0T'
ALCARECOTkAlCosmicsCTF0TDQM.FolderName = 'TkAlCosmics'
# sizes			    
ALCARECOTkAlCosmicsCTF0TDQM.TkSizeBin = 100
ALCARECOTkAlCosmicsCTF0TDQM.TkSizeMin = 0
ALCARECOTkAlCosmicsCTF0TDQM.TkSizeMax = 100
ALCARECOTkAlCosmicsCTF0TDQM.TrackPtBin = 500
ALCARECOTkAlCosmicsCTF0TDQM.TrackPtMin = 0
ALCARECOTkAlCosmicsCTF0TDQM.TrackPtMax = 500
### TkAlCosmicsCosmicTF0T ###
ALCARECOTkAlCosmicsCosmicTF0TDQM = ALCARECOTkAlCosmicsCTF0TDQM.clone()
ALCARECOTkAlCosmicsCosmicTF0TDQM.TrackProducer = 'ALCARECOTkAlCosmicsCosmicTF0T'
ALCARECOTkAlCosmicsCosmicTF0TDQM.AlgoName = 'ALCARECOTkAlCosmicsCosmicTF0T'
#TkAlCosmicsRS0T
ALCARECOTkAlCosmicsRS0TDQM = ALCARECOTkAlCosmicsCTF0TDQM.clone()
ALCARECOTkAlCosmicsRS0TDQM.TrackProducer = 'ALCARECOTkAlCosmicsRS0T'
ALCARECOTkAlCosmicsRS0TDQM.AlgoName = 'ALCARECOTkAlCosmicsRS0T'
### TkAlCosmicsCTF ###
ALCARECOTkAlCosmicsCTFDQM = ALCARECOTkAlCosmicsCTF0TDQM.clone()
ALCARECOTkAlCosmicsCTFDQM.TrackProducer = 'ALCARECOTkAlCosmicsCTF'
ALCARECOTkAlCosmicsCTFDQM.AlgoName = 'ALCARECOTkAlCosmicsCTF'
### TkAlCosmicsCosmicTF ###
ALCARECOTkAlCosmicsCosmicTFDQM = ALCARECOTkAlCosmicsCTFDQM.clone()
ALCARECOTkAlCosmicsCosmicTFDQM.TrackProducer = 'ALCARECOTkAlCosmicsCosmicTF'
ALCARECOTkAlCosmicsCosmicTFDQM.AlgoName = 'ALCARECOTkAlCosmicsCosmicTF'
#TkAlCosmicsRS
ALCARECOTkAlCosmicsRSDQM = ALCARECOTkAlCosmicsCTFDQM.clone()
ALCARECOTkAlCosmicsRSDQM.TrackProducer = 'ALCARECOTkAlCosmicsRS'
ALCARECOTkAlCosmicsRSDQM.AlgoName = 'ALCARECOTkAlCosmicsRS'
