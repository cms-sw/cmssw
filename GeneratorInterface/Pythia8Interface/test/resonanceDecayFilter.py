#example gen fragment that takes a gridapack which produces
#ttH events with inclusive top decays and undecayed higgs at lhe level
#and selects resonance decays such that events have at least four leptons (electrons, muons, taus)

import FWCore.ParameterSet.Config as cms

# link to cards:
# https://github.com/cms-sw/genproductions/blob/0d4b4288fa053d9a8aef5c6e123b66bf94c3aee8/bin/Powheg/production/V2/13TeV/Higgs/ttH_inclusive_NNPDF30_13TeV_M125/ttH_inclusive_NNPDF30_13TeV_M125.input

externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/slc6_amd64_gcc481/13TeV/powheg/V2/ttH_inclusive_NNPDF30_13TeV_M125/v1/ttH_inclusive_NNPDF30_13TeV_M125_tarball.tar.gz'),
    nEvents = cms.untracked.uint32(5000),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh')
)

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
from Configuration.Generator.Pythia8PowhegEmissionVetoSettings_cfi import *

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
            'POWHEG:nFinal = 3',   ## Number of final state particles
                                   ## (BEFORE THE DECAYS) in the LHE
                                   ## other than emitted extra parton
            '25:m0 = 125.0',
            '25:onMode = off',
            '25:onIfMatch = 15 -15',
            '25:onIfMatch = 23 23',
            '25:onIfMatch = 24 -24',
            '24:mMin = 0.05',
            '23:mMin = 0.05',
            'ResonanceDecayFilter:filter = on',
            'ResonanceDecayFilter:exclusive = off', #off: require at least the specified number of daughters, on: require exactly the specified number of daughters
            'ResonanceDecayFilter:eMuAsEquivalent    = off', #on: treat electrons and muons as equivalent
            'ResonanceDecayFilter:eMuTauAsEquivalent = on',  #on: treat electrons, muons , and taus as equivalent
            'ResonanceDecayFilter:allNuAsEquivalent  = on',  #on: treat all three neutrino flavours as equivalent
            'ResonanceDecayFilter:udscAsEquivalent   = off', #on: treat u,d,s,c quarks as equivalent
            'ResonanceDecayFilter:udscbAsEquivalent  = off', #on: treat u,d,s,c,b quarks as equivalent
            #'ResonanceDecayFilter:mothers =', #list of mothers not specified -> count all particles in hard process+resonance decays (better to avoid specifying mothers when including leptons from the lhe in counting, since intermediate resonances are not gauranteed to appear in general
            'ResonanceDecayFilter:daughters = 11,11,11,11',
          ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'pythia8PowhegEmissionVetoSettings',
                                    'processParameters'
                                    )
        )
                         )

ProductionFilterSequence = cms.Sequence(generator)


