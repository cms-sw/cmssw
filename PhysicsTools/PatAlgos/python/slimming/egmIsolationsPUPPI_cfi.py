import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask

from RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationPUPPI_cff import egmPhotonIsolationMiniAODPUPPI as _egmPhotonPUPPIIsolationForPhotons
from RecoEgamma.EgammaIsolationAlgos.egmElectronIsolationPUPPI_cff import egmElectronIsolationMiniAODPUPPI as _egmElectronIsolationMiniAODPUPPI
from RecoEgamma.EgammaIsolationAlgos.egmElectronIsolationPUPPI_cff import egmElectronIsolationMiniAODPUPPINoLeptons as _egmElectronIsolationMiniAODPUPPINoLeptons

def makeInputForPUPPIIsolationEgm(process):

	task = getPatAlgosToolsTask(process)

	addToProcessAndTask('egmPhotonPUPPIIsolation', _egmPhotonPUPPIIsolationForPhotons.clone(), process, task)
	process.egmPhotonPUPPIIsolation.srcToIsolate = cms.InputTag("reducedEgamma","reducedGedPhotons")
	process.egmPhotonPUPPIIsolation.srcForIsolationCone = cms.InputTag("particleFlow")
	process.egmPhotonPUPPIIsolation.puppiValueMap = cms.InputTag('puppi')

	addToProcessAndTask('egmElectronPUPPIIsolation', _egmElectronIsolationMiniAODPUPPI.clone(), process, task)
	process.egmElectronPUPPIIsolation.srcToIsolate = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
	process.egmElectronPUPPIIsolation.srcForIsolationCone = cms.InputTag("packedPFCandidates")

	addToProcessAndTask('egmElectronPUPPINoLeptonsIsolation', _egmElectronIsolationMiniAODPUPPINoLeptons.clone(), process, task)
	process.egmElectronPUPPINoLeptonsIsolation.srcToIsolate = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
	process.egmElectronPUPPINoLeptonsIsolation.srcForIsolationCone = cms.InputTag("packedPFCandidates")
