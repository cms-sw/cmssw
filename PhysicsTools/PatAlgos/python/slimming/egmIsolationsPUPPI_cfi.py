import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationPUPPI_cff import egmPhotonIsolationMiniAODPUPPI as _egmPhotonPUPPIIsolationForPhotons

def makeInputForPUPPIIsolationEgm(process):
	process.load('RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationPUPPI_cff')
	process.egmPhotonPUPPIIsolationForPhotons = _egmPhotonPUPPIIsolationForPhotons.clone()
	process.egmPhotonPUPPIIsolationForPhotons.srcToIsolate = cms.InputTag("selectedPatPhotons")
	process.egmPhotonPUPPIIsolationForPhotons.srcForIsolationCone = cms.InputTag("packedPFCandidates")