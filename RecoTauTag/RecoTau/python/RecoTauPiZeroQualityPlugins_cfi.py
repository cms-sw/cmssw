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

nearPiZeroMassBarrel = cms.PSet(
    name = cms.string('nearPiZeroMass'),
    plugin = cms.string('RecoTauPiZeroStringQuality'),
    selection = cms.string('abs(eta()) < 1.5 & %s < 0.05' % DELTA_M_PIZERO),
    # Rank by closeness to piZero
    selectionPassFunction = cms.string(DELTA_M_PIZERO),
    selectionFailValue = cms.double(1000),
)

# Loose selection for endcap
nearPiZeroMassEndcap = cms.PSet(
    name = cms.string('nearPiZeroMass'),
    plugin = cms.string('RecoTauPiZeroStringQuality'),
    selection = cms.string('abs(eta()) > 1.5 & mass() < 0.2'),
    # Rank by closeness to piZero
    selectionPassFunction = cms.string(DELTA_M_PIZERO),
    selectionFailValue = cms.double(1000),
)
