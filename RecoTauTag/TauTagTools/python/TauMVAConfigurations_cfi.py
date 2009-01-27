import FWCore.ParameterSet.Config as cms

"""
Define names, locations, and decay mode mappings of different MVA configurations

Example:

   OneProngs = cms.PSet(                                        # name
      decayModeIndices = cms.vint32(0, 1, 2, 3, 4),             # Decay modes this MVA applies to.  In this case, decay modes with 1 track and 0, 1, 2, 3, 4 pi zeros.
                                                                # Please see DataFormats/TauReco/interface/PFTauDecayMode.h 
                                                                # for the definition of the decay mode indexing system.  [DecayMode# = (#ChargedPions-1) * 5 + #PiZeros]
      computerName     = cms.string('OneProngs')                # Identifier string.  Will be used to store the trainined MVA data in the database
                                                                # When training, the corresponding training xml file must be in RecoTauTag/TauTagTools/xml/[computerName].xml (i.e. OneProngs.xml)
      applyIsolation   = cms.bool(False)                        # Determines if isolation criteria is applied (for both training and final discrimination)
      )

"""

OneProngNoPiZero = cms.PSet(
      decayModeIndices = cms.vint32(0),
      computerName   = cms.string('OneProngNoPiZero'),
      applyIsolation = cms.bool(False)
      )

OneProngOnePiZero = cms.PSet(
      decayModeIndices = cms.vint32(1),
      computerName   = cms.string('OneProngOnePiZero'),
      applyIsolation = cms.bool(False)
      )

OneProngTwoPiZero = cms.PSet(
      decayModeIndices = cms.vint32(2),
      computerName   = cms.string('OneProngTwoPiZero'),
      applyIsolation = cms.bool(False)
      )

ThreeProngNoPiZero = cms.PSet(
      decayModeIndices = cms.vint32(10),
      computerName   = cms.string('ThreeProngNoPiZero'),
      applyIsolation = cms.bool(False)
      )

ThreeProngOnePiZero = cms.PSet(
      decayModeIndices = cms.vint32(11),
      computerName   = cms.string('ThreeProngOnePiZero'),
      applyIsolation = cms.bool(False)
      )

SingleNet = cms.PSet(
      decayModeIndices = cms.vint32(0, 1, 2, 10, 11),
      computerName   = cms.string('SingleNet'),
      applyIsolation = cms.bool(False)
      )

#ISOLATED versions

OneProngNoPiZeroIso = cms.PSet(
      decayModeIndices = cms.vint32(0),
      computerName   = cms.string('OneProngNoPiZeroIso'),
      applyIsolation = cms.bool(True)
      )

OneProngOnePiZeroIso = cms.PSet(
      decayModeIndices = cms.vint32(1),
      computerName   = cms.string('OneProngOnePiZeroIso'),
      applyIsolation = cms.bool(True)
      )

OneProngTwoPiZeroIso = cms.PSet(
      decayModeIndices = cms.vint32(2),
      computerName   = cms.string('OneProngTwoPiZeroIso'),
      applyIsolation = cms.bool(True)
      )

ThreeProngNoPiZeroIso = cms.PSet(
      decayModeIndices = cms.vint32(10),
      computerName   = cms.string('ThreeProngNoPiZeroIso'),
      applyIsolation = cms.bool(True)
      )

ThreeProngOnePiZeroIso = cms.PSet(
      decayModeIndices = cms.vint32(11),
      computerName   = cms.string('ThreeProngOnePiZeroIso'),
      applyIsolation = cms.bool(True)
      )

SingleNetIso = cms.PSet(
      decayModeIndices = cms.vint32(0, 1, 2, 10, 11),
      computerName   = cms.string('SingleNetIso'),
      applyIsolation = cms.bool(True)
      )
