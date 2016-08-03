import FWCore.ParameterSet.Config as cms


def makeInputForPUPPIIsolationEgm(process):
	process.load('RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationPUPPI_cff')
	process.egmPhotonIsolationMiniAODPUPPI.srcToIsolate = cms.InputTag("selectedPatPhotons")
	process.egmPhotonIsolationMiniAODPUPPI.srcForIsolationCone = cms.InputTag("packedPFCandidates")