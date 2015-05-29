import FWCore.ParameterSet.Config as cms

import RecoTauTag.RecoTau.RecoTauCleanerPlugins as cleaners

RecoTauCleaner = cms.EDProducer("RecoTauCleaner",
    src = cms.InputTag("combinatoricRecoTaus"),
    cleaners = cms.VPSet(
        # Reject taus that have charge == 3
        cleaners.charge,
        # Reject taus that are not within DR<0.1 of the jet axis
        #cleaners.matchingConeCut,
        # Reject taus that fail HPS selections
        cms.PSet(
            name = cms.string("HPS_Select"),
            plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
            src = cms.InputTag("hpsSelectionDiscriminator"),
        ),
        # CV: prefer 3-prong candidates over 2-prong candidates and 2-prong candidates over 1-prong candidates 
        cleaners.chargedHadronMultiplicity,                                    
        # CV: Take highest pT tau (use for testing of new high pT tau reconstruction and check if it can become the new default)
        cleaners.pt,
        # CV: in case two candidates have the same Pt,
        #     prefer candidates in which PFGammas are part of strips (rather than being merged with PFRecoTauChargedHadrons)
        cleaners.stripMultiplicity,
        # Take most isolated tau
        cleaners.combinedIsolation
    )
)
