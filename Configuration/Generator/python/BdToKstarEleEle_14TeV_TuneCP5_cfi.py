import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         comEnergy = cms.double(14000.0),
                         ##crossSection = cms.untracked.double(54000000000), # Given by PYTHIA after running
                         ##filterEfficiency = cms.untracked.double(0.004), # Given by PYTHIA after running
                         maxEventsToPrint = cms.untracked.int32(0),
                         ExternalDecays = cms.PSet(
        EvtGen130 = cms.untracked.PSet(
            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2014_NOLONGLIFE.DEC'),
            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt_2014.pdl'),
            user_decay_file = cms.vstring('GeneratorInterface/ExternalDecays/data/Bd_Kstarmumu_Kpi.dec'),
            list_forced_decays = cms.vstring('MyB0','Myanti-B0'),
            convertPythiaCodes = cms.untracked.bool(False),
            operates_on_particles = cms.vint32()
            ),
        parameterSets = cms.vstring('EvtGen130')
        ),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        ## check this (need extra parameters?)
        processParameters = cms.vstring('SoftQCD:nonDiffractive = on',
                                        'PTFilter:filter = on', # this turn on the filter
                                        'PTFilter:quarkToFilter = 5', # PDG id of q quark (can be any other)
                                        'PTFilter:scaleToFilter = 1.0'),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CP5Settings',
                                    'processParameters',
        )
        )
)


###########
# Filters #
###########

bfilter = cms.EDFilter(
    "PythiaFilter", 
    MaxEta = cms.untracked.double(9999.),
    MinEta = cms.untracked.double(-9999.),
    ParticleID = cms.untracked.int32(511)  ## Bd
    )

decayfilter = cms.EDFilter(
    "PythiaDauVFilter",
    verbose         = cms.untracked.int32(1),
    NumberDaughters = cms.untracked.int32(3),
    ParticleID      = cms.untracked.int32(511),
    DaughterIDs     = cms.untracked.vint32(-13, 13, 313),  ## mu+, mu-, K*^0(892)
    MinPt           = cms.untracked.vdouble(2.5, 2.5, -1.),
    MinEta          = cms.untracked.vdouble(-2.5, -2.5, -9999.),
    MaxEta          = cms.untracked.vdouble( 2.5,  2.5,  9999.)
    )

kstarfilter = cms.EDFilter(
    "PythiaDauVFilter",
    verbose         = cms.untracked.int32(1), 
    NumberDaughters = cms.untracked.int32(2), 
    MotherID        = cms.untracked.int32(511),  ## Bd
    ParticleID      = cms.untracked.int32(313),  ## K*^0(892)
    DaughterIDs     = cms.untracked.vint32(321, -211), ## K+, pi-
    MinPt           = cms.untracked.vdouble(0.4, 0.4), 
    MinEta          = cms.untracked.vdouble(-2.5, -2.5), 
    MaxEta          = cms.untracked.vdouble( 2.5,  2.5)
    )


ProductionFilterSequence = cms.Sequence(generator*bfilter*decayfilter*kstarfilter)
