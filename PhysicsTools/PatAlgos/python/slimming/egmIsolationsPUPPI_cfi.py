import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationPUPPI_cff import egmPhotonIsolationMiniAODPUPPI as _egmPhotonPUPPIIsolationForPhotons
from RecoEgamma.EgammaIsolationAlgos.egmElectronIsolationPUPPI_cff import egmElectronIsolationMiniAODPUPPI as _egmElectronIsolationMiniAODPUPPI
from RecoEgamma.EgammaIsolationAlgos.egmElectronIsolationPUPPI_cff import egmElectronIsolationMiniAODPUPPINoLeptons as _egmElectronIsolationMiniAODPUPPINoLeptons

def makeInputForPUPPIIsolationEgm(process):

	process.egmPhotonPUPPIIsolation = _egmPhotonPUPPIIsolationForPhotons.clone()
	process.egmPhotonPUPPIIsolation.srcToIsolate = cms.InputTag("reducedEgamma","reducedGedPhotons")
	process.egmPhotonPUPPIIsolation.srcForIsolationCone = cms.InputTag("particleFlow")
	process.egmPhotonPUPPIIsolation.puppiValueMap = cms.InputTag('puppi')

	process.egmElectronPUPPIIsolation = _egmElectronIsolationMiniAODPUPPI.clone()
	process.egmElectronPUPPIIsolation.srcToIsolate = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
	process.egmElectronPUPPIIsolation.srcForIsolationCone = cms.InputTag("packedPFCandidates")

	process.egmElectronPUPPINoLeptonsIsolation = _egmElectronIsolationMiniAODPUPPINoLeptons.clone()
	process.egmElectronPUPPINoLeptonsIsolation.srcToIsolate = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
	process.egmElectronPUPPINoLeptonsIsolation.srcForIsolationCone = cms.InputTag("packedPFCandidates")
