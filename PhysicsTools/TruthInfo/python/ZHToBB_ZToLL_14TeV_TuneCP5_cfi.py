# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).
#
# Associated-Higgs (VH) GEN fragment: Higgs-strahlung q qbar -> Z H, with H -> b b
# and Z -> leptons. There is no VH sample in the standard relval matrix, so this
# minimal fragment produces a gallery/library example that exercises the 'vh'
# selection preset (seed the Higgs {25}, keep the recoiling Z as a production
# sibling).

import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *

generator = cms.EDFilter("Pythia8ConcurrentGeneratorFilter",
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         filterEfficiency = cms.untracked.double(1.0),
                         comEnergy = cms.double(14000.0),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        processParameters = cms.vstring(
            'HiggsSM:ffbar2HZ = on',  # q qbar -> H Z (Higgs-strahlung)
            '25:m0 = 125.0',
            '25:onMode = off',
            '25:onIfMatch = 5 -5',    # H -> b bbar
            '23:onMode = off',
            '23:onIfAny = 11 13 15',  # Z -> e / mu / tau
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CP5Settings',
                                    'processParameters',
                                    )
        )
                         )
ProductionFilterSequence = cms.Sequence(generator)
