import FWCore.ParameterSet.Config as cms

''' 

Plugins for ranking PFTau candidates

'''

# Prefer taus with charge == 1 (no three prongs with charge = 3)
unitCharge = cms.PSet(
    name = cms.string("UnitCharge"),
    plugin = cms.string("RecoTauStringCleanerPlugin"),
    # Only effects three prongs
    selection = cms.string("signalPFChargedHadrCands().size() = 3"),
    # As 1 is lower than 3, this will always prefer those with unit charge
    selectionPassFunction = cms.string("abs(charge())-1"),
    # If it is a one prong, consider it just as good as a 
    # three prong with unit charge
    selectionFailValue = cms.double(0),
)

# Prefer taus with pt greater 15
ptGt15 = cms.PSet(
    name = cms.string("PtGt15"),
    plugin = cms.string("RecoTauStringCleanerPlugin"),
    selection = cms.string("pt > 15."),
    selectionPassFunction = cms.string("0"),
    selectionFailValue = cms.double(1e3)
)

# Prefer taus that have higher TaNC output values
tanc = cms.PSet(
    name = cms.string("TaNC"),
    plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
    src = cms.InputTag("DISCRIMINATOR_SRC"),
)

leadPionFinding = cms.PSet(
    name = cms.string("LeadPion"),
    plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
    src = cms.InputTag("DISCRIMINATOR_SRC"),
)

combinedIsolation = cms.PSet(
    name = cms.string("CombinedIsolation"),
    plugin = cms.string("RecoTauStringCleanerPlugin"),
    # Require that cones were built by ensuring the a leadCand exits
    selection = cms.string("leadPFCand().isNonnull()"),
    selectionPassFunction = cms.string(
        "isolationPFChargedHadrCandsPtSum()+isolationPFGammaCandsEtSum()"),
    selectionFailValue = cms.double(1e3)
)

chargeIsolation = cms.PSet(
    name = cms.string("ChargeIsolation"),
    plugin = cms.string("RecoTauStringCleanerPlugin"),
    # Require that cones were built by ensuring the a leadCand exits
    selection = cms.string("leadPFCand().isNonnull()"),
    # Prefer lower isolation activity
    selectionPassFunction = cms.string("isolationPFChargedHadrCandsPtSum()"),
    selectionFailValue = cms.double(1e3)
)

ecalIsolation = cms.PSet(
    name = cms.string("GammaIsolation"),
    plugin = cms.string("RecoTauStringCleanerPlugin"),
    # Require that cones were built by ensuring the a leadCand exits
    selection = cms.string("leadPFCand().isNonnull()"),
    # Prefer lower isolation activity
    selectionPassFunction = cms.string("isolationPFGammaCandsEtSum()"),
    selectionFailValue = cms.double(1e3)
)

