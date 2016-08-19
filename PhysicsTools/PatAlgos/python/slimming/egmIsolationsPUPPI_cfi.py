import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationPUPPI_cff import egmPhotonIsolationMiniAODPUPPI as _egmPhotonPUPPIIsolationForPhotons
from RecoEgamma.EgammaIsolationAlgos.egmElectronIsolationPUPPI_cff import egmElectronIsolationMiniAODPUPPI as _egmElectronIsolationMiniAODPUPPI

def makeInputForPUPPIIsolationEgm(process):
	
	process.load('RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationPUPPI_cff')
	process.load('RecoEgamma.EgammaIsolationAlgos.egmElectronIsolationPUPPI_cff')

	process.egmPhotonPUPPIIsolationForPhotons = _egmPhotonPUPPIIsolationForPhotons.clone()
	process.egmPhotonPUPPIIsolationForPhotons.srcToIsolate = cms.InputTag("selectedPatPhotons")
	process.egmPhotonPUPPIIsolationForPhotons.srcForIsolationCone = cms.InputTag("packedPFCandidates")

	process.egmElectronPUPPIIsolationForElectrons = _egmElectronIsolationMiniAODPUPPI.clone()
	process.egmElectronPUPPIIsolationForElectrons.srcToIsolate = cms.InputTag("selectedPatElectrons")
	process.egmElectronPUPPIIsolationForElectrons.srcForIsolationCone = cms.InputTag("packedPFCandidates")
