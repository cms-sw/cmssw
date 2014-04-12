import FWCore.ParameterSet.Config as cms

"""
        TauMVAConfigurations_cfi.py
        Author: Evan K. Friis, UC Davis (friis@physics.ucdavis.edu)

            Define names, locations, and decay mode mappings of different MVA configurations

            At the bottom of this file the collections of individual neural nets are defined (eg TaNC).
            These collections are the inputs to the actual discriminator

Example:

   OneProngs = cms.PSet(                                        # name
      decayModeIndices = cms.vint32(0, 1, 2, 3, 4),             # Decay modes this MVA applies to.  In this case, decay modes with 1 track and 0, 1, 2, 3, 4 pi zeros.
                                                                # Please see DataFormats/TauReco/interface/PFTauDecayMode.h 
                                                                # for the definition of the decay mode indexing system.  [DecayMode# = (#ChargedPions-1) * 5 + #PiZeros]
      computerName     = cms.string('OneProngs')                # Identifier string.  Will be used to store the trainined MVA data in the database
                                                                # When training, the corresponding training xml file must be in RecoTauTag/TauTagTools/xml/[computerName].xml (i.e. OneProngs.xml)
      applyIsolation   = cms.bool(False),                       # Determines if isolation criteria is applied (for both training and final discrimination)

      cut              = cms.double(-10.)                       # Serves as a place holder for when the user wishes to specify the cuts they wish to use 
      )

"""

OneProngNoPiZero = cms.PSet(
      decayModeIndices = cms.vint32(0),
      computerName   = cms.string('OneProngNoPiZero'),
      applyIsolation = cms.bool(False),
      cut            = cms.double(-10.)
      )

OneProngOnePiZero = cms.PSet(
      decayModeIndices = cms.vint32(1),
      computerName   = cms.string('OneProngOnePiZero'),
      applyIsolation = cms.bool(False),
      cut            = cms.double(-10.)
      )

OneProngTwoPiZero = cms.PSet(
      decayModeIndices = cms.vint32(2),
      computerName   = cms.string('OneProngTwoPiZero'),
      applyIsolation = cms.bool(False),
      cut            = cms.double(-10.)
      )

ThreeProngNoPiZero = cms.PSet(
      decayModeIndices = cms.vint32(10),
      computerName   = cms.string('ThreeProngNoPiZero'),
      applyIsolation = cms.bool(False),
      cut            = cms.double(-10.)
      )

ThreeProngOnePiZero = cms.PSet(
      decayModeIndices = cms.vint32(11),
      computerName   = cms.string('ThreeProngOnePiZero'),
      applyIsolation = cms.bool(False),
      cut            = cms.double(-10.)
      )

SingleNet = cms.PSet(
      decayModeIndices = cms.vint32(0, 1, 2, 10, 11),
      computerName   = cms.string('SingleNet'),
      applyIsolation = cms.bool(False),
      cut            = cms.double(-10.)
      )

#ISOLATED versions

OneProngNoPiZeroIso = cms.PSet(
      decayModeIndices = cms.vint32(0),
      computerName   = cms.string('OneProngNoPiZeroIso'),
      applyIsolation = cms.bool(True),
      cut            = cms.double(-10.)
      )

OneProngOnePiZeroIso = cms.PSet(
      decayModeIndices = cms.vint32(1),
      computerName   = cms.string('OneProngOnePiZeroIso'),
      applyIsolation = cms.bool(True),
      cut            = cms.double(-10.)
      )

OneProngTwoPiZeroIso = cms.PSet(
      decayModeIndices = cms.vint32(2),
      computerName   = cms.string('OneProngTwoPiZeroIso'),
      applyIsolation = cms.bool(True),
      cut            = cms.double(-10.)
      )

ThreeProngNoPiZeroIso = cms.PSet(
      decayModeIndices = cms.vint32(10),
      computerName   = cms.string('ThreeProngNoPiZeroIso'),
      applyIsolation = cms.bool(True),
      cut            = cms.double(-10.)
      )

ThreeProngOnePiZeroIso = cms.PSet(
      decayModeIndices = cms.vint32(11),
      computerName   = cms.string('ThreeProngOnePiZeroIso'),
      applyIsolation = cms.bool(True),
      cut            = cms.double(-10.)
      )

SingleNetIso = cms.PSet(
      decayModeIndices = cms.vint32(0, 1, 2, 10, 11),
      computerName   = cms.string('SingleNetIso'),
      applyIsolation = cms.bool(True),
      cut            = cms.double(-10.)
      )

#Define collections of Neural nets
# Define vectors of the DecayMode->MVA implementaions associations you want to use
# Note: any decay mode not associated to an MVA will be marked as failing the MVA!
TaNC = cms.VPSet(
      OneProngNoPiZero,
      OneProngOnePiZero,
      OneProngTwoPiZero,
      ThreeProngNoPiZero,
      ThreeProngOnePiZero
      )

MultiNetIso = cms.VPSet(
      OneProngNoPiZeroIso,
      OneProngOnePiZeroIso,
      OneProngTwoPiZeroIso,
      ThreeProngNoPiZeroIso,
      ThreeProngOnePiZeroIso
      )

SingleNetBasedTauID = cms.VPSet(
      SingleNet
)
