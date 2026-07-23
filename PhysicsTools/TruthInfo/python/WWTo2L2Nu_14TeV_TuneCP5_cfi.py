# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).
#
# Diboson WW GEN fragment (q qbar -> W+ W-, both W -> leptons). There is no WW
# sample in the standard relval matrix, so this minimal fragment produces a
# gallery/library example that exercises the 'diboson' selection preset (seed the
# vector bosons {23, 24, -24}, keep the production system).

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
            'WeakDoubleBoson:ffbar2WW = on',  # q qbar -> W+ W-
            '24:onMode = off',
            '24:onIfAny = 11 13 15',          # W -> e / mu / tau (clean leptonic diboson)
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CP5Settings',
                                    'processParameters',
                                    )
        )
                         )
ProductionFilterSequence = cms.Sequence(generator)
