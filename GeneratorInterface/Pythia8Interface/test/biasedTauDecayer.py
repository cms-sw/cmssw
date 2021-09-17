#example gen fragment that takes a gridapack which produces
#ttH events with inclusive top decays and undecayed higgs at lhe level
#and selects resonance decays such that events have at least four leptons (electrons, muons, taus)

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

# link to cards:
# https://github.com/cms-sw/genproductions/blob/0d4b4288fa053d9a8aef5c6e123b66bf94c3aee8/bin/Powheg/production/V2/13TeV/Higgs/ttH_inclusive_NNPDF30_13TeV_M125/ttH_inclusive_NNPDF30_13TeV_M125.input

process.source = cms.Source("LHESource",
  fileNames = cms.untracked.vstring('file:powheg-Ztautau.lhe')
)

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
from Configuration.Generator.Pythia8PowhegEmissionVetoSettings_cfi import *

process.generator = cms.EDFilter("Pythia8HadronizerFilter",
                         maxEventsToPrint = cms.untracked.int32(20),
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         filterEfficiency = cms.untracked.double(1.0),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         comEnergy = cms.double(13000.),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters = cms.vstring(
            'BiasedTauDecayer:filter = on',
            'PartonLevel:MPI = off'
          ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'processParameters'
                                    )
        )
                         )

process.p = cms.Path(process.generator)


