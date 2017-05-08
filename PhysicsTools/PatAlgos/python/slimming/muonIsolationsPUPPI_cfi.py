import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask

from RecoMuon.MuonIsolation.muonIsolationPUPPI_cff import muonIsolationMiniAODPUPPI as _muonIsolationMiniAODPUPPI
from RecoMuon.MuonIsolation.muonIsolationPUPPI_cff import muonIsolationMiniAODPUPPINoLeptons as _muonIsolationMiniAODPUPPINoLeptons

def makeInputForPUPPIIsolationMuon(process):

	task = getPatAlgosToolsTask(process)

	addToProcessAndTask('muonPUPPIIsolation', _muonIsolationMiniAODPUPPI.clone(), process, task)
	process.muonPUPPIIsolation.srcToIsolate = cms.InputTag("muons")
	process.muonPUPPIIsolation.srcForIsolationCone = cms.InputTag("packedPFCandidates")

	addToProcessAndTask('muonPUPPINoLeptonsIsolation', _muonIsolationMiniAODPUPPINoLeptons.clone(), process, task)
	process.muonPUPPINoLeptonsIsolation.srcToIsolate = cms.InputTag("muons")
	process.muonPUPPINoLeptonsIsolation.srcForIsolationCone = cms.InputTag("packedPFCandidates")
