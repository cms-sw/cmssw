import FWCore.ParameterSet.Config as cms

def _addProcessModulesDevLST(process):
    process.hltESPModulesDevLST = cms.ESProducer('LSTModulesDevESProducer@alpaka',
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
modifyConfigurationForTrackingLSTModulesDevLST_ = trackingLST.makeProcessModifier(_addProcessModulesDevLST)
