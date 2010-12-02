import FWCore.ParameterSet.Config as cms

'''

Configuration for Pi Zero producer plugins.

Author: Evan K. Friis, UC Davis


'''

DELTA_M_PIZERO = 'abs(mass() - 0.13579)'

isInStrip = cms.PSet(
    name = cms.string('InStrip'),
    plugin = cms.string('RecoTauPiZeroStringQuality'),
    #selection = cms.string('maxDeltaPhi() < %(phi)f && maxDeltaEta() < %(eta)f'%
                           #{'eta':builders.strips.stripEtaAssociationDistance.value(),
                            #'phi':builders.strips.stripPhiAssociationDistance.value() }),
    # Mike pointed out the max value of the strip can be greater than what the
    # intial cuts are and still be consistent.  Until there is a good way to
    # deal with this just cut on the algo name.
    selection = cms.string('algoIs("kStrips")'),
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

legacyPFTauDecayModeSelection = cms.PSet(
    name = cms.string("PFTDM"),
    plugin = cms.string("RecoTauPiZeroStringQuality"),
    selection = cms.string('mass() < 0.2'),
    selectionPassFunction = cms.string(DELTA_M_PIZERO),
    selectionFailValue = cms.double(1000),
)

