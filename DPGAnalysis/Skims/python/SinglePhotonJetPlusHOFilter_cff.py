import FWCore.ParameterSet.Config as cms 

from DPGAnalysis.Skims.singlePhotonJetPlusHOFilter_cfi import singlePhotonJetPlusHOFilter as _singlePhotonJetPlusHOFilter

SinglePhotonJetPlusHOFilterSkim = _singlePhotonJetPlusHOFilter.clone(
    Photons = cms.InputTag("photons"),
    PFJets = cms.InputTag("ak4PFJets"),
    particleFlowClusterHO = cms.InputTag("particleFlowClusterHO"),
) 

SinglePhotonJetPlusHOFilterSequence = cms.Sequence(SinglePhotonJetPlusHOFilterSkim)

