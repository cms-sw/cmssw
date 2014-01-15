import FWCore.ParameterSet.Config as cms

import RecoTauTag.RecoTau.RecoTauCleanerPlugins as cleaners

RecoTauCleaner = cms.EDProducer(
        "RecoTauCleaner",
            src = cms.InputTag("combinatoricRecoTaus"),
            cleaners = cms.VPSet(
            # Prefer taus that dont' have charge == 3
            cleaners.unitCharge,
                     # Ignore taus reconstructed in pi0 decay modes in which the highest Pt ("leading") pi0 has pt below 2.5 GeV
                     # (in order to make decay mode reconstruction less sensitive to pile-up)
                     # NOTE: strips are sorted by decreasing pt
                    cms.PSet(
                name = cms.string("leadStripPtLt2_5"),
                            plugin = cms.string("RecoTauStringCleanerPlugin"),
                            selection = cms.string("signalPiZeroCandidates().size() = 0 | signalPiZeroCandidates()[0].pt > 2.5"),
                            selectionPassFunction = cms.string("0"),
                            selectionFailValue = cms.double(1e3)
                            ),
                    # Prefer taus that are within DR<0.1 of the jet axis
            #        cleaners.matchingConeCut,
                    # Prefer taus that pass HPS selections
                    cms.PSet(
                name = cms.string("HPS_Select"),
                            plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
                            src = cms.InputTag("hpsSelectionDiscriminator"),
                        ),
                    cleaners.combinedIsolation
                )
        )
