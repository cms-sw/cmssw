import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *
from GeneratorInterface.EvtGenInterface.EvtGenSetting_cff import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.38e-3),
    crossSection = cms.untracked.double(540000000.),
    comEnergy = cms.double(14000.0),
    ExternalDecays = cms.PSet(
        EvtGen130 = cms.untracked.PSet(
            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2010.DEC'),
            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt.pdl'),
            user_decay_embedded= cms.vstring(
                '#',
                'Alias      MyBs    B_s0',
                'Alias      Myanti-Bs   anti-B_s0',
                'ChargeConj Myanti-Bs   MyBs',
                '#',
                'Decay MyBs',
                '1.000  e+       e-                     PHOTOS PHSP;',
                'Enddecay',
                '#',
                'Decay Myanti-Bs',
                '1.000  e+       e-                     PHOTOS PHSP;',
                'Enddecay',
                'End'
            ), 
            list_forced_decays = cms.vstring('MyBs','Myanti-Bs'),
            operates_on_particles = cms.vint32(),
        ),
        parameterSets = cms.vstring('EvtGen130')
    ),
    PythiaParameters = cms.PSet(pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,                        
        processParameters = cms.vstring(
            "SoftQCD:nonDiffractive = on",
            'PTFilter:filter = on', # this turn on the filter
            'PTFilter:quarkToFilter = 5', # PDG id of q quark
            'PTFilter:scaleToFilter = 1.0'
        ),
        parameterSets = cms.vstring('pythia8CommonSettings',
            'pythia8CP5Settings',                        
            'processParameters',
        )
    )
)

generator.PythiaParameters.processParameters.extend(EvtGenExtraParticles)

configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.0 $'),
    name = cms.untracked.string('$Source: /Configuration/Generator/python/BPH_BsMuMu_PhaseII_cfi.py $'),
    annotation = cms.untracked.string('PhaseII: Pythia8+EvtGen130 generation of Bs --> e+e-, 14TeV, Tune CP5')
)

bfilter = cms.EDFilter(
    "PythiaFilter",
    MaxEta = cms.untracked.double(9999.),
    MinEta = cms.untracked.double(-9999.),
    ParticleID = cms.untracked.int32(531)
    )

decayfilter = cms.EDFilter(
    "PythiaDauVFilter",
    verbose         = cms.untracked.int32(1), 
    NumberDaughters = cms.untracked.int32(2), 
    ParticleID      = cms.untracked.int32(531),  
    DaughterIDs     = cms.untracked.vint32(11, -11),
    MinPt           = cms.untracked.vdouble(2.0, 2.0), 
    MinEta          = cms.untracked.vdouble(-2.45, -2.45), 
    MaxEta          = cms.untracked.vdouble( 2.45,  2.45)
    )

ProductionFilterSequence = cms.Sequence(generator*bfilter*decayfilter)
