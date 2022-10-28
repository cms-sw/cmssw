import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *

generator = cms.EDFilter("Pythia8ConcurrentGeneratorFilter",
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         filterEfficiency = cms.untracked.double(0.037),
                         crossSection = cms.untracked.double(74310000.),
                         comEnergy = cms.double(8000.0),  # center of mass energy in GeV
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters = cms.vstring(
            'HardQCD:all = on',
            'PhaseSpace:pTHatMin = 30.',
            'PhaseSpace:pTHatMax = 80.'
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'processParameters',
                                    )
        )
                         )


genParticlesForFilter = cms.EDProducer("GenParticleProducer",
                                       saveBarCodes = cms.untracked.bool(True),
                                       src = cms.InputTag("generator", "unsmeared"),
                                       abortOnUnknownPDGCode = cms.untracked.bool(True)
                                       )

bctoefilter = cms.EDFilter("BCToEFilter",
                           filterAlgoPSet = cms.PSet(maxAbsEta = cms.double(2.5), 
                                                     eTThreshold = cms.double(1),
                                                     genParSource = cms.InputTag("genParticlesForFilter")
                                                     )
                           )


# enter below the configuration metadata (only a description is needed, the rest is filled in by cvs)
configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: Configuration/Generator/python/QCD_Pt_30_80_BCtoE_8TeV_TuneCUETP8M1_cfi.py $'),
    annotation = cms.untracked.string('b/c->e filtered QCD pthat 30-80, 8 TeV')
    )

# add your filters to this sequence
ProductionFilterSequence = cms.Sequence(generator * (genParticlesForFilter + bctoefilter))
