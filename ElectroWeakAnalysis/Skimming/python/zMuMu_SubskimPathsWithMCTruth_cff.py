import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.Skimming.dimuonsHLTFilter_cfi import *
from ElectroWeakAnalysis.Skimming.patCandidatesForZMuMuSubskim_cff import *
from ElectroWeakAnalysis.Skimming.zMuMuMuonUserData import *
from ElectroWeakAnalysis.Skimming.dimuons_cfi import *
dimuons.decay = cms.string('userDataMuons@+ userDataMuons@-')
from ElectroWeakAnalysis.Skimming.dimuonsOneTrack_cfi import *
dimuonsOneTrack.decay = cms.string('userDataMuons@+ userDataTracks@-') 
from ElectroWeakAnalysis.Skimming.dimuonsGlobal_cfi import *
dimuonsGlobal.src = cms.InputTag("userDataDimuons")
from ElectroWeakAnalysis.Skimming.dimuonsOneStandAloneMuon_cfi import *
dimuonsOneStandAloneMuon.src = cms.InputTag("userDataDimuons") 
from ElectroWeakAnalysis.Skimming.dimuonsOneTrackerMuon_cfi import *
dimuonsOneTrackerMuon.src = cms.InputTag("userDataDimuons")
from ElectroWeakAnalysis.Skimming.dimuonsFilter_cfi import *
dimuonsFilter.src = cms.InputTag("userDataDimuons") 
from ElectroWeakAnalysis.Skimming.dimuonsOneTrackFilter_cfi import *
dimuonsOneTrackFilter.src = cms.InputTag("userDataDimuonsOneTrack")

# MC matching sequence
from ElectroWeakAnalysis.Skimming.zMuMu_MCTruth_cfi import *
userDataDimuons.zGenParticlesMatch = cms.InputTag("allDimuonsMCMatch")
userDataDimuonsOneTrack.zGenParticlesMatch = cms.InputTag("allDimuonsOneTrackMCMatch")



dimuonsPath = cms.Path(dimuonsHLTFilter *
                       goodMuonRecoForDimuon *
                       userDataMuons*
                       dimuons *
                       mcTruthForDimuons *
                       userDataDimuons*
                       dimuonsGlobal *
                       dimuonsOneStandAloneMuon *
                       dimuonsOneTrackerMuon *
                       dimuonsFilter    
                       )

dimuonsOneTrackPath = cms.Path(dimuonsHLTFilter *
                               goodMuonRecoForDimuon*
                               userDataMuons*
                               userDataTracks*
                               dimuonsOneTrack*
                               mcTruthForDimuonsOneTrack *
                               userDataDimuonsOneTrack*
                               dimuonsOneTrackFilter
                               )


