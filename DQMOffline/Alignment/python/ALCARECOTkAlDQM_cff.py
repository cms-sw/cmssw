import FWCore.ParameterSet.Config as cms
import DQM.TrackingMonitor.TrackingMonitor_cfi
import DQMOffline.Alignment.TkAlCaRecoMonitor_cfi

#Below all DQM modules for TrackerAlignment AlCaRecos are instanciated.
#############---  TkAlZMuMu ---#######################
ALCARECOTkAlZMuMuTrackingDQM = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
ALCARECOTkAlZMuMuTkAlDQM =  DQMOffline.Alignment.TkAlCaRecoMonitor_cfi.TkAlCaRecoMonitor.clone()
ALCARECOTkAlZMuMuDQM = cms.Sequence( ALCARECOTkAlZMuMuTrackingDQM + ALCARECOTkAlZMuMuTkAlDQM )

#names & designations  
selectionName = 'TkAlZMuMu'
ALCARECOTkAlZMuMuTrackingDQM.TrackProducer = 'ALCARECO'+selectionName
ALCARECOTkAlZMuMuTrackingDQM.AlgoName = 'ALCARECO'+selectionName
ALCARECOTkAlZMuMuTrackingDQM.FolderName = selectionName
ALCARECOTkAlZMuMuTkAlDQM.TrackProducer = 'ALCARECO'+selectionName
ALCARECOTkAlZMuMuTkAlDQM.AlgoName = 'ALCARECO'+selectionName
ALCARECOTkAlZMuMuTkAlDQM.FolderName = selectionName
#TkAlDQM settings
ALCARECOTkAlZMuMuTkAlDQM.fillInvariantMass = cms.bool(True)
ALCARECOTkAlZMuMuTkAlDQM.MassBin = 300
ALCARECOTkAlZMuMuTkAlDQM.MassMin = 50.0
ALCARECOTkAlZMuMuTkAlDQM.MassMax = 150.0

#sizes
ALCARECOTkAlZMuMuTrackingDQM.TkSizeBin = 6
ALCARECOTkAlZMuMuTrackingDQM.TkSizeMin = -0.5
ALCARECOTkAlZMuMuTrackingDQM.TkSizeMax = 5.5
ALCARECOTkAlZMuMuTrackingDQM.TrackPtBin = 100
ALCARECOTkAlZMuMuTrackingDQM.TrackPtMin = 0
ALCARECOTkAlZMuMuTrackingDQM.TrackPtMax = 100

#############---  TkAlJpsiMuMu ---#######################
ALCARECOTkAlJpsiMuMuTrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone()
ALCARECOTkAlJpsiMuMuTkAlDQM = ALCARECOTkAlZMuMuTkAlDQM.clone()
ALCARECOTkAlJpsiMuMuDQM = cms.Sequence( ALCARECOTkAlJpsiMuMuTrackingDQM + ALCARECOTkAlJpsiMuMuTkAlDQM )

#names & designations    
selectionName = 'TkAlJpsiMuMu'
ALCARECOTkAlJpsiMuMuTrackingDQM.TrackProducer = 'ALCARECO'+selectionName
ALCARECOTkAlJpsiMuMuTrackingDQM.AlgoName = 'ALCARECO'+selectionName
ALCARECOTkAlJpsiMuMuTrackingDQM.FolderName = selectionName
ALCARECOTkAlJpsiMuMuTkAlDQM.TrackProducer = 'ALCARECO'+selectionName
ALCARECOTkAlJpsiMuMuTkAlDQM.AlgoName = 'ALCARECO'+selectionName
ALCARECOTkAlJpsiMuMuTkAlDQM.FolderName = selectionName

#TkAlDQM settings
ALCARECOTkAlJpsiMuMuTkAlDQM.MassMin = 2.5
ALCARECOTkAlJpsiMuMuTkAlDQM.MassMax = 4.0

#sizes  
ALCARECOTkAlJpsiMuMuTrackingDQM.TrackPtMax = 30

#############---  TkAlUpsilonMuMu ---#######################
ALCARECOTkAlUpsilonMuMuTrackingDQM = ALCARECOTkAlJpsiMuMuTrackingDQM.clone()
ALCARECOTkAlUpsilonMuMuTkAlDQM = ALCARECOTkAlZMuMuTkAlDQM.clone()
ALCARECOTkAlUpsilonMuMuDQM = cms.Sequence( ALCARECOTkAlUpsilonMuMuTrackingDQM + ALCARECOTkAlUpsilonMuMuTkAlDQM)

selectionName = 'TkAlUpsilonMuMu'
ALCARECOTkAlUpsilonMuMuTrackingDQM.TrackProducer = 'ALCARECO'+selectionName
ALCARECOTkAlUpsilonMuMuTrackingDQM.AlgoName = 'ALCARECO'+selectionName
ALCARECOTkAlUpsilonMuMuTrackingDQM.FolderName = selectionName
ALCARECOTkAlUpsilonMuMuTkAlDQM.TrackProducer = 'ALCARECO'+selectionName
ALCARECOTkAlUpsilonMuMuTkAlDQM.AlgoName = 'ALCARECO'+selectionName
ALCARECOTkAlUpsilonMuMuTkAlDQM.FolderName = selectionName

#TkAlDQM settings
ALCARECOTkAlUpsilonMuMuTkAlDQM.MassMin = 9.5
ALCARECOTkAlUpsilonMuMuTkAlDQM.MassMax = 10

#############---  TkAlBeamHalo ---#######################
ALCARECOTkAlBeamHaloTrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone()
ALCARECOTkAlBeamHaloDQM = cms.Sequence( ALCARECOTkAlBeamHaloTrackingDQM )

#names & designations  
ALCARECOTkAlBeamHaloTrackingDQM.TrackProducer = 'ALCARECOTkAlBeamHalo'
ALCARECOTkAlBeamHaloTrackingDQM.AlgoName = 'ALCARECOTkAlBeamHalo'
ALCARECOTkAlBeamHaloTrackingDQM.FolderName = 'TkAlBeamHalo'

#############---  TkAlMinBias ---#######################
ALCARECOTkAlMinBiasTrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone()
ALCARECOTkAlMinBiasTkAlDQM = ALCARECOTkAlZMuMuTkAlDQM.clone()
ALCARECOTkAlMinBiasDQM = cms.Sequence( ALCARECOTkAlMinBiasTrackingDQM + ALCARECOTkAlMinBiasTkAlDQM)

#names & designations  
selectionName = 'TkAlMinBias'
ALCARECOTkAlMinBiasTrackingDQM.TrackProducer = 'ALCARECO'+selectionName
ALCARECOTkAlMinBiasTrackingDQM.AlgoName = 'ALCARECO'+selectionName
ALCARECOTkAlMinBiasTrackingDQM.FolderName = selectionName
ALCARECOTkAlMinBiasTkAlDQM.TrackProducer = 'ALCARECO'+selectionName
ALCARECOTkAlMinBiasTkAlDQM.AlgoName = 'ALCARECO'+selectionName
ALCARECOTkAlMinBiasTkAlDQM.FolderName = selectionName

#TkAlDQM settings
ALCARECOTkAlMinBiasTkAlDQM.fillInvariantMass = cms.bool(False)

#sizes
ALCARECOTkAlMinBiasTrackingDQM.TkSizeBin = 71
ALCARECOTkAlMinBiasTrackingDQM.TkSizeMin = -0.5
ALCARECOTkAlMinBiasTrackingDQM.TkSizeMax = 70.5
ALCARECOTkAlMinBiasTrackingDQM.TrackPtMax = 30

#############---  TkAlMuonIsolated ---#######################
ALCARECOTkAlMuonIsolatedTrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone()
ALCARECOTkAlMuonIsolatedTkAlDQM = ALCARECOTkAlMinBiasTkAlDQM.clone()
ALCARECOTkAlMuonIsolatedDQM = cms.Sequence( ALCARECOTkAlMuonIsolatedTrackingDQM + ALCARECOTkAlMuonIsolatedTkAlDQM)

#names & designations  
selectionName = 'TkAlMuonIsolated'
ALCARECOTkAlMuonIsolatedTrackingDQM.TrackProducer = 'ALCARECO'+selectionName
ALCARECOTkAlMuonIsolatedTrackingDQM.AlgoName = 'ALCARECO'+selectionName
ALCARECOTkAlMuonIsolatedTrackingDQM.FolderName = selectionName
ALCARECOTkAlMuonIsolatedTkAlDQM.TrackProducer = 'ALCARECO'+selectionName
ALCARECOTkAlMuonIsolatedTkAlDQM.AlgoName = 'ALCARECO'+selectionName
ALCARECOTkAlMuonIsolatedTkAlDQM.FolderName = selectionName


###### DQM modules for cosmic data taking ######
### TkAlCosmicsCTF0T ###
ALCARECOTkAlCosmicsCTF0TTrackingDQM = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
ALCARECOTkAlCosmicsCTF0TDQM = cms.Sequence( ALCARECOTkAlCosmicsCTF0TTrackingDQM )

# names & designations  
ALCARECOTkAlCosmicsCTF0TTrackingDQM.TrackProducer = 'ALCARECOTkAlCosmicsCTF0T'
ALCARECOTkAlCosmicsCTF0TTrackingDQM.AlgoName = 'ALCARECOTkAlCosmicsCTF0T'
ALCARECOTkAlCosmicsCTF0TTrackingDQM.FolderName = 'TkAlCosmics'
# sizes			    
ALCARECOTkAlCosmicsCTF0TTrackingDQM.TkSizeBin = 100
ALCARECOTkAlCosmicsCTF0TTrackingDQM.TkSizeMin = 0
ALCARECOTkAlCosmicsCTF0TTrackingDQM.TkSizeMax = 100
ALCARECOTkAlCosmicsCTF0TTrackingDQM.TrackPtBin = 500
ALCARECOTkAlCosmicsCTF0TTrackingDQM.TrackPtMin = 0
ALCARECOTkAlCosmicsCTF0TTrackingDQM.TrackPtMax = 500

### TkAlCosmicsCosmicTF0T ###
ALCARECOTkAlCosmicsCosmicTF0TTrackingDQM = ALCARECOTkAlCosmicsCTF0TTrackingDQM.clone()
ALCARECOTkAlCosmicsCosmicTF0TDQM = cms.Sequence( ALCARECOTkAlCosmicsCosmicTF0TTrackingDQM )
ALCARECOTkAlCosmicsCosmicTF0TTrackingDQM.TrackProducer = 'ALCARECOTkAlCosmicsCosmicTF0T'
ALCARECOTkAlCosmicsCosmicTF0TTrackingDQM.AlgoName = 'ALCARECOTkAlCosmicsCosmicTF0T'
#TkAlCosmicsRS0T
ALCARECOTkAlCosmicsRS0TTrackingDQM = ALCARECOTkAlCosmicsCTF0TTrackingDQM.clone()
ALCARECOTkAlCosmicsRS0TDQM = cms.Sequence( ALCARECOTkAlCosmicsRS0TTrackingDQM )
ALCARECOTkAlCosmicsRS0TTrackingDQM.TrackProducer = 'ALCARECOTkAlCosmicsRS0T'
ALCARECOTkAlCosmicsRS0TTrackingDQM.AlgoName = 'ALCARECOTkAlCosmicsRS0T'
### TkAlCosmicsCTF ###
ALCARECOTkAlCosmicsCTFTrackingDQM = ALCARECOTkAlCosmicsCTF0TTrackingDQM.clone()
ALCARECOTkAlCosmicsCTFDQM = cms.Sequence( ALCARECOTkAlCosmicsCTFTrackingDQM )
ALCARECOTkAlCosmicsCTFTrackingDQM.TrackProducer = 'ALCARECOTkAlCosmicsCTF'
ALCARECOTkAlCosmicsCTFTrackingDQM.AlgoName = 'ALCARECOTkAlCosmicsCTF'
### TkAlCosmicsCosmicTF ###
ALCARECOTkAlCosmicsCosmicTFTrackingDQM = ALCARECOTkAlCosmicsCTFTrackingDQM.clone()
ALCARECOTkAlCosmicsCosmicTFDQM = cms.Sequence( ALCARECOTkAlCosmicsCosmicTFTrackingDQM )
ALCARECOTkAlCosmicsCosmicTFTrackingDQM.TrackProducer = 'ALCARECOTkAlCosmicsCosmicTF'
ALCARECOTkAlCosmicsCosmicTFTrackingDQM.AlgoName = 'ALCARECOTkAlCosmicsCosmicTF'
#TkAlCosmicsRS
ALCARECOTkAlCosmicsRSTrackingDQM = ALCARECOTkAlCosmicsCTFTrackingDQM.clone()
ALCARECOTkAlCosmicsRSDQM = cms.Sequence( ALCARECOTkAlCosmicsRSTrackingDQM )
ALCARECOTkAlCosmicsRSTrackingDQM.TrackProducer = 'ALCARECOTkAlCosmicsRS'
ALCARECOTkAlCosmicsRSTrackingDQM.AlgoName = 'ALCARECOTkAlCosmicsRS'
