import FWCore.ParameterSet.Config as cms


def makeInputForPUPPIIsolationEgm(process):
	process.load('RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationAOD_cff')
	process.egmPhotonIsolationAOD.srcToIsolate = cms.InputTag("selectedPatPhotons")