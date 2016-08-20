import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationPUPPI_cff import egmPhotonIsolationMiniAODPUPPI as _egmPhotonPUPPIIsolationForPhotons
from RecoEgamma.EgammaIsolationAlgos.egmElectronIsolationPUPPI_cff import egmElectronIsolationMiniAODPUPPI as _egmElectronIsolationMiniAODPUPPI
from RecoEgamma.EgammaIsolationAlgos.egmElectronIsolationPUPPI_cff import egmElectronIsolationMiniAODPUPPINoLeptons as _egmElectronIsolationMiniAODPUPPINoLeptons

def makeInputForPUPPIIsolationEgm(process):
	
	process.load('RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationPUPPI_cff')
	process.load('RecoEgamma.EgammaIsolationAlgos.egmElectronIsolationPUPPI_cff')

	process.egmPhotonPUPPIIsolation = _egmPhotonPUPPIIsolationForPhotons.clone()
	process.egmPhotonPUPPIIsolation.srcToIsolate = cms.InputTag("selectedPatPhotons")
	process.egmPhotonPUPPIIsolation.srcForIsolationCone = cms.InputTag("packedPFCandidates")

	process.egmElectronPUPPIIsolation = _egmElectronIsolationMiniAODPUPPI.clone()
	process.egmElectronPUPPIIsolation.srcToIsolate = cms.InputTag("selectedPatElectrons")
	process.egmElectronPUPPIIsolation.srcForIsolationCone = cms.InputTag("packedPFCandidates")

	process.egmElectronPUPPINoLeptonsIsolation = _egmElectronIsolationMiniAODPUPPINoLeptons.clone()
	process.egmElectronPUPPINoLeptonsIsolation.srcToIsolate = cms.InputTag("selectedPatElectrons")
	process.egmElectronPUPPINoLeptonsIsolation.srcForIsolationCone = cms.InputTag("packedPFCandidates")