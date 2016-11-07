import FWCore.ParameterSet.Config as cms
import PhysicsTools.PatAlgos.tools.helpers as configtools

from RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationPUPPI_cff import egmPhotonIsolationMiniAODPUPPI as _egmPhotonPUPPIIsolationForPhotons
from RecoEgamma.EgammaIsolationAlgos.egmElectronIsolationPUPPI_cff import egmElectronIsolationMiniAODPUPPI as _egmElectronIsolationMiniAODPUPPI
from RecoEgamma.EgammaIsolationAlgos.egmElectronIsolationPUPPI_cff import egmElectronIsolationMiniAODPUPPINoLeptons as _egmElectronIsolationMiniAODPUPPINoLeptons

def makeInputForPUPPIIsolationEgm(process):

	patAlgosToolsTask = configtools.getPatAlgosToolsTask(process)

	process.egmPhotonPUPPIIsolation = _egmPhotonPUPPIIsolationForPhotons.clone()
	patAlgosToolsTask.add(process.egmPhotonPUPPIIsolation)
	process.egmPhotonPUPPIIsolation.srcToIsolate = cms.InputTag("reducedEgamma","reducedGedPhotons")
	process.egmPhotonPUPPIIsolation.srcForIsolationCone = cms.InputTag("particleFlow")
	process.egmPhotonPUPPIIsolation.puppiValueMap = cms.InputTag('puppi')

	process.egmElectronPUPPIIsolation = _egmElectronIsolationMiniAODPUPPI.clone()
	patAlgosToolsTask.add(process.egmElectronPUPPIIsolation)
	process.egmElectronPUPPIIsolation.srcToIsolate = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
	process.egmElectronPUPPIIsolation.srcForIsolationCone = cms.InputTag("packedPFCandidates")

	process.egmElectronPUPPINoLeptonsIsolation = _egmElectronIsolationMiniAODPUPPINoLeptons.clone()
	patAlgosToolsTask.add(process.egmElectronPUPPINoLeptonsIsolation)
	process.egmElectronPUPPINoLeptonsIsolation.srcToIsolate = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
	process.egmElectronPUPPINoLeptonsIsolation.srcForIsolationCone = cms.InputTag("packedPFCandidates")
