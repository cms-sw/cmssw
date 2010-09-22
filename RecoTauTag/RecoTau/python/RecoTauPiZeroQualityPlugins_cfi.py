import FWCore.ParameterSet.Config as cms

'''

Configuration for Pi Zero producer plugins.

Author: Evan K. Friis, UC Davis


'''

# Get Builder parameters
import RecoTauTag.RecoTau.RecoTauPiZeroBuilderPlugins_cfi as builders

DELTA_M_PIZERO = 'abs(mass() - 0.13579)'

isInStrip = cms.PSet(
    name = cms.string('InStrip'),
    plugin = cms.string('RecoTauPiZeroStringQuality'),
    selection = cms.string('maxDeltaEta() < %(phi)f && maxDeltaEta() < %(eta)f'%
                           {'eta':builders.strips.stripEtaAssociationDistance.value(),
                            'phi':builders.strips.stripPhiAssociationDistance.value() }),
    selectionPassFunction = cms.string(DELTA_M_PIZERO),
    selectionFailValue = cms.double(1000)
)

nearPiZeroMass = cms.PSet(
    name = cms.string('nearPiZeroMass'),
    plugin = cms.string('RecoTauPiZeroStringQuality'),
    selection = cms.string('%s < 0.05' % DELTA_M_PIZERO),
    # Rank by closeness to piZero
    selectionPassFunction = cms.string(DELTA_M_PIZERO),
    selectionFailValue = cms.double(1000),
)

maximumMass = cms.PSet(
    name = cms.string('maxMass'),
    plugin = cms.string('RecoTauPiZeroStringQuality'),
    selection = cms.string('mass() < 0.2'),
    selectionPassFunction = cms.string(DELTA_M_PIZERO),
    selectionFailValue = cms.double(1000),
)


