import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
from Configuration.Generator.Pythia8PowhegEmissionVetoSettings_cfi import *

# link to cards
# https://github.com/cms-sw/genproductions/blob/4927fb8fec2afc72cc98ff2e5fb5bf0db930f971/bin/Powheg/production/V2/13TeV/Higgs/gg_H_quark-mass-effects_JHUGen_HZZ4L_NNPDF30_13TeV/gg_H_quark-mass-effects_NNPDF30_13TeV_M125.input
# https://github.com/cms-sw/genproductions/blob/4927fb8fec2afc72cc98ff2e5fb5bf0db930f971/bin/Powheg/production/V2/13TeV/Higgs/gg_H_quark-mass-effects_JHUGen_HZZ4L_NNPDF30_13TeV/JHUGen_gg_H_ZZ4L_M125.input

externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/slc6_amd64_gcc481/13TeV/powheg/V2/gg_H_quark-mass-effects_NNPDF30_13TeV_M125_JHUGen_ZZ4L/v3/gg_H_quark-mass-effects_NNPDF30_13TeV_M125_JHUGen_ZZ4L_tarball.tar.gz'),
    nEvents = cms.untracked.uint32(5000),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh')
)


generator = cms.EDFilter("Pythia8HadronizerFilter",
                         maxEventsToPrint = cms.untracked.int32(1),
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         filterEfficiency = cms.untracked.double(1.0),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         comEnergy = cms.double(13000.),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        pythia8PowhegEmissionVetoSettingsBlock,
        processParameters = cms.vstring(
            'POWHEG:nFinal = 1',   ## Number of final state particles
                                   ## (BEFORE THE DECAYS) in the LHE
                                   ## other than emitted extra parton
          ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'pythia8PowhegEmissionVetoSettings',
                                    'processParameters'
                                    )
        )
                         )

ProductionFilterSequence = cms.Sequence(externalLHEProducer*generator)

