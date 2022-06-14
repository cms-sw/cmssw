import FWCore.ParameterSet.Config as cms

hltHpsPFTauCleaner8HitsMaxDeltaZWithOfflineVertices = cms.EDProducer("RecoTauCleaner",
    cleaners = cms.VPSet(
        cms.PSet(
            name = cms.string('Charge'),
            nprongs = cms.vuint32(1, 3),
            passForCharge = cms.int32(1),
            plugin = cms.string('RecoTauChargeCleanerPlugin'),
            selectionFailValue = cms.double(0),
            tolerance = cms.double(0)
        ),
        cms.PSet(
            name = cms.string('HPS_Select'),
            plugin = cms.string('RecoTauDiscriminantCleanerPlugin'),
            src = cms.InputTag("hltHpsPFTauSelectionDiscriminationByHPS8HitsMaxDeltaZWithOfflineVertices")
        ),
        cms.PSet(
            minTrackPt = cms.double(5.0),
            name = cms.string('killSoftTwoProngTaus'),
            plugin = cms.string('RecoTauSoftTwoProngTausCleanerPlugin'),
            tolerance = cms.double(0)
        ),
        cms.PSet(
            name = cms.string('leadTrackPt'),
            plugin = cms.string('RecoTauCleanerPluginHGCalWorkaround'),
            tolerance = cms.double(0)
        ),
        cms.PSet(
            name = cms.string('ChargedHadronMultiplicity'),
            plugin = cms.string('RecoTauChargedHadronMultiplicityCleanerPlugin'),
            tolerance = cms.double(0)
        ),
        cms.PSet(
            name = cms.string('Pt'),
            plugin = cms.string('RecoTauStringCleanerPlugin'),
            selection = cms.string('leadCand().isNonnull()'),
            selectionFailValue = cms.double(1000.0),
            selectionPassFunction = cms.string('-pt()'),
            tolerance = cms.double(0.01)
        ),
        cms.PSet(
            name = cms.string('StripMultiplicity'),
            plugin = cms.string('RecoTauStringCleanerPlugin'),
            selection = cms.string('leadCand().isNonnull()'),
            selectionFailValue = cms.double(1000.0),
            selectionPassFunction = cms.string('-signalPiZeroCandidates().size()'),
            tolerance = cms.double(0)
        ),
        cms.PSet(
            name = cms.string('ChargeIsolation'),
            plugin = cms.string('RecoTauStringCleanerPlugin'),
            selection = cms.string('leadCand().isNonnull()'),
            selectionFailValue = cms.double(1000.0),
            selectionPassFunction = cms.string('isolationPFChargedHadrCandsPtSum()'),
            tolerance = cms.double(0)
        )
    ),
    outputSelection = cms.string(''),
    src = cms.InputTag("hltHpsPFTauCombinatoricProducer8HitsMaxDeltaZWithOfflineVertices"),
    verbosity = cms.int32(0)
)
