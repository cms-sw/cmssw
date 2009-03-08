import FWCore.ParameterSet.Config as cms
import DQM.TrackingMonitor.TrackingMonitor_cfi as _TrackMon
import DQMOffline.Alignment.muonAlignment_cfi as _MuonAl
#from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagator_cfi import *
from Configuration.StandardSequences.ReconstructionCosmics_cff import *

#Below all DQM modules for MuonAlignment AlCaRecos are instanciated.

###############MuAlStandAloneCosmics######################
ALCARECOMuAlStandAloneCosmicsTrackingDQM = _TrackMon.TrackMon.clone(
#names & designations  
TrackProducer = 'ALCARECOMuAlStandAloneCosmics',
AlgoName = 'ALCARECOMuAlStandAloneCosmics',
FolderName = 'MuAlStandAloneCosmics',
#sizes
TkSizeBin = 5,
TkSizeMin = 0,
TkSizeMax = 5,
TrackPtBin = 100,
TrackPtMin = 0,
TrackPtMax = 100
)

ALCARECOMuAlStandAloneCosmicsMuAlDQM = _MuonAl.muonAlignment.clone(
    doSummary = cms.untracked.bool(True),
    FolderName ='MuAlStandAloneCosmics',
    MuonCollection = 'ALCARECOMuAlStandAloneCosmics'
 )

ALCARECOMuAlStandAloneCosmicsDQM = cms.Sequence(ALCARECOMuAlStandAloneCosmicsTrackingDQM  + ALCARECOMuAlStandAloneCosmicsMuAlDQM )

#MuAlGlobalCosmics
ALCARECOMuAlGlobalCosmicsDQM = _TrackMon.TrackMon.clone(
#names & designations    
TrackProducer = 'ALCARECOMuAlGlobalCosmics:GlobalMuon',
AlgoName = 'ALCARECOMuAlGlobalCosmics',
FolderName = 'MuAlGlobalCosmics',
#sizes  
TkSizeBin = 5,
TkSizeMin = 0,
TkSizeMax = 5,
TrackPtBin = 100,
TrackPtMin = 0,
TrackPtMax = 30
)


###################MuAlBeamHalo################################
ALCARECOMuAlBeamHaloDQM = _TrackMon.TrackMon.clone(
#names & designations  
TrackProducer = 'ALCARECOMuAlBeamHalo',
AlgoName = 'ALCARECOMuAlBeamHalo',
FolderName = 'MuAlBeamHalo',
#sizes
TkSizeBin = 5,
TkSizeMin = 0,
TkSizeMax = 5,
TrackPtBin = 100,
TrackPtMin = 0,
TrackPtMax = 30
)

##################MuAlBeamHaloOverlaps####################
ALCARECOMuAlBeamHaloOverlapsDQM = _TrackMon.TrackMon.clone(
#names & designations  
TrackProducer = 'ALCARECOMuAlBeamHaloOverlaps',
AlgoName = 'ALCARECOMuAlBeamHaloOverlaps',
FolderName = 'MuAlBeamHaloOverlaps',
#sizes
TkSizeBin = 70,
TkSizeMin = 0,
TkSizeMax = 70,
TrackPtBin = 100,
TrackPtMin = 0,
TrackPtMax = 30
)

##################MuAlCalIsolatedMu##########################
ALCARECOMuAlCalIsolatedMuTrackingDQM = _TrackMon.TrackMon.clone(
#names & designations  
TrackProducer = 'ALCARECOMuAlCalIsolatedMu:StandAlone',
AlgoName = 'ALCARECOMuAlCalIsolatedMu',
FolderName = 'MuAlCalIsolatedMu',
#sizes			    
TkSizeBin = 5,
TkSizeMin = 0,
TkSizeMax = 5,
TrackPtBin = 100,
TrackPtMin = 0,
TrackPtMax = 100
)

ALCARECOMuAlCalIsolatedMuMuAlDQM = _MuonAl.muonAlignment.clone(
    doSummary = cms.untracked.bool(True),
    FolderName ='MuAlCalIsolatedMu',
    MuonCollection = 'ALCARECOMuAlCalIsolatedMu:StandAlone'
 )

ALCARECOMuAlCalIsolatedMuDQM = cms.Sequence( ALCARECOMuAlCalIsolatedMuTrackingDQM + ALCARECOMuAlCalIsolatedMuMuAlDQM )

######################MuAlOverlaps##########################
ALCARECOMuAlOverlapsDQM = _TrackMon.TrackMon.clone(
#names & designations  
TrackProducer = 'ALCARECOMuAlOverlaps',
AlgoName = 'ALCARECOMuAlOverlaps',
FolderName = 'MuAlOverlaps',
#sizes                      
TkSizeBin = 70,
TkSizeMin = 0,
TkSizeMax = 70,
TrackPtBin = 100,
TrackPtMin = 0,
TrackPtMax = 30
)
