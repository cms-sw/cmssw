import FWCore.ParameterSet.Config as cms

import copy
from DQM.TrackingMonitor.TrackingMonitor_cfi import *
#Below all DQM modules for TrackerAlignment AlCaRecos are instanciated.
#TkAlZMuMu
ALCARECOTkAlZMuMuDQM = copy.deepcopy(TrackMon)
import copy
from DQM.TrackingMonitor.TrackingMonitor_cfi import *
#TkAlJpsiMuMu
ALCARECOTkAlJpsiMuMuDQM = copy.deepcopy(TrackMon)
import copy
from DQM.TrackingMonitor.TrackingMonitor_cfi import *
#TkAlUpsilonMuMu
ALCARECOTkAlUpsilonMuMuDQM = copy.deepcopy(TrackMon)
import copy
from DQM.TrackingMonitor.TrackingMonitor_cfi import *
#TkAlBeamHalo
ALCARECOTkAlBeamHaloDQM = copy.deepcopy(TrackMon)
import copy
from DQM.TrackingMonitor.TrackingMonitor_cfi import *
#TkAlMinBias
ALCARECOTkAlMinBiasDQM = copy.deepcopy(TrackMon)
import copy
from DQM.TrackingMonitor.TrackingMonitor_cfi import *
#TkAlMuonIsolated
ALCARECOTkAlMuonIsolatedDQM = copy.deepcopy(TrackMon)
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

