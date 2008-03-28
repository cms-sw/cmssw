import FWCore.ParameterSet.Config as cms

from PhysicsTools.RecoAlgos.allElectrons_cfi import *
from PhysicsTools.RecoAlgos.allTracks_cfi import *
from PhysicsTools.RecoAlgos.allSuperClusters_cfi import *
from PhysicsTools.IsolationAlgos.allElectronIsolations_cfi import *
from PhysicsTools.IsolationAlgos.allTrackIsolations_cfi import *
from PhysicsTools.IsolationAlgos.allSuperClusterIsolations_cfi import *
from PhysicsTools.HepMCCandAlgos.allElectronsGenParticlesMatch_cfi import *
from PhysicsTools.HepMCCandAlgos.allTracksGenParticlesLeptonMatch_cfi import *
from PhysicsTools.HepMCCandAlgos.allSuperClustersGenParticlesLeptonMatch_cfi import *
from ElectroWeakAnalysis.ZReco.zToEE_cfi import *
from ElectroWeakAnalysis.ZReco.zToEEOneTrack_cfi import *
from ElectroWeakAnalysis.ZReco.zToEEOneSuperCluster_cfi import *
from ElectroWeakAnalysis.ZReco.zToEEGenParticlesMatch_cfi import *
from ElectroWeakAnalysis.ZReco.zToEEOneTrackGenParticlesMatch_cfi import *
from ElectroWeakAnalysis.ZReco.zToEEOneSuperClusterGenParticlesMatch_cfi import *
electronRecoForZToEE = cms.Sequence(allTracks+allElectrons+allSuperClusters+allElectronIsolations+allTrackIsolations+allSuperClusterIsolations)
zToEEReco = cms.Sequence(zToEE+zToEEOneTrack+zToEEOneSuperCluster)
zToEEAnalysis = cms.Sequence(electronRecoForZToEE+zToEEReco)
zToEEAnalysisSequenceData = cms.Sequence(electronRecoForZToEE+zToEEReco)
electronMCTruthForZToEE = cms.Sequence(allElectronsGenParticlesMatch+zToEEGenParticlesMatch)
electronMCTruthForZToEEOneTrack = cms.Sequence(allElectronsGenParticlesMatch+allTracksGenParticlesLeptonMatch+zToEEOneTrackGenParticlesMatch)
electronMCTruthForZToEEOneSuperCluster = cms.Sequence(allElectronsGenParticlesMatch+allSuperClustersGenParticlesLeptonMatch+zToEEOneSuperClusterGenParticlesMatch)
zToEEAnalysisSequence = cms.Sequence(zToEEAnalysisSequenceData+electronMCTruthForZToEE)
allSuperClusters.particleType = 'e-'

