import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUEZ2starSettings_cfi import *

#
# Example illustrating how to change particle properties/branching ratios
# with PYUPDA cards.
#
# This produces an H0, which decays to a pair of long-lived exotic particles
# which then each decay to a pair of light quarks.
#

generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(10),
    # Print event
    pythiaPylistVerbosity = cms.untracked.int32(3),
    # Print decay tables
#    pythiaPylistVerbosity = cms.untracked.int32(12),                         
    filterEfficiency = cms.untracked.double(1.0),
    comEnergy = cms.double(8000.0),
#    crossSection = cms.untracked.double(55000000000.),
    UseExternalGenerators = cms.untracked.bool(False),
#
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        pythiaMyParameters = cms.vstring(
# This is needed if you are introducing long-lived exotic particles.
          'MSTJ(22)=1    ! Decay ALL unstable particles',

          'MSEL=0',
# Request gg -> H0 production
          'MSUB(152)=1',
          'MWID(35)=2 ! Let me set H0 properties'
        ),
#
# This block controls how Pythia interacts with the PYUPDA cards.
#        
        PYUPDAParameters = cms.vstring(
# Either:
#     1) Specify the location of the PYUPDA table to be read in. 
         "PYUPDAFILE = 'Configuration/Generator/data/Pythia_H0_pyupda.in'"
#        Optionally ask to call PYUPDA after PYINIT. Don't do this unless you have to.
#        "PYUPDApostPYINIT"
# Or:
#     2) Write current PYUPDA parameters to file (so you can edit them and read back in).
#         "PYUPDAFILE = \'pyupda.out\'"
#         "PYUPDAWRITE"
        ),
#
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
                                    'pythiaMyParameters',
                                    'PYUPDAParameters')
    )
)

# N.B. If your PYUPDA tables introduces new exotic particles, you will need
# to include:
#
# genParticles.abortOnUnknownPDGCode = cms.untracked.bool(False)

# This is no longer necessary, as starting 34X cmsDriver takes care automatically
#
#ProductionFilterSequence = cms.Sequence(generator)
