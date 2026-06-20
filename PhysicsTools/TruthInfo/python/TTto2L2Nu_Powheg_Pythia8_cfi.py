# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).
#
# ttbar -> dilepton (TTto2L2Nu) at NLO with POWHEG (hvq) showered by Pythia8, used
# to produce a more realistic ttbar gallery/library example than the LO Pythia8
# Top:gg2ttbar sample.
#
# WARNING: this is the Run3 13.6 TeV gridpack (and a Run3 13.6 TeV tune /
# PS-weights), but the truth-graph library/gallery is Phase-2 (Run4 D120) at
# 14 TeV. The GEN centre-of-mass energy (13.6 TeV) therefore does NOT match the
# Phase-2 detector/conditions it is simulated with - it is included only as a
# topology demonstration. For a physically consistent Phase-2 ttbar-POWHEG sample,
# swap the gridpack for a 14 TeV one and use a matching tune (e.g. CP5 14 TeV).

import FWCore.ParameterSet.Config as cms

externalLHEProducer = cms.EDProducer('ExternalLHEProducer',
    args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/PdmV/Run3Summer22/Powheg/TT/hvq_slc7_amd64_gcc10_CMSSW_12_4_8_TTto2L2Nu_powheg-pythia8.tgz'),
    nEvents = cms.untracked.uint32(5000),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh'),
    generateConcurrently = cms.untracked.bool(True),
)

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunesRun3ECM13p6TeV.PythiaCP5Settings_cfi import *
from Configuration.Generator.Pythia8PowhegEmissionVetoSettings_cfi import *
from Configuration.Generator.PSweightsPythia.PythiaPSweightsSettings_cfi import *

generator = cms.EDFilter("Pythia8ConcurrentHadronizerFilter",
    PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        pythia8PowhegEmissionVetoSettingsBlock,
        pythia8PSweightsSettingsBlock,
        processParameters = cms.vstring(
        'POWHEG:nFinal = 2',
        'TimeShower:mMaxGamma = 1.0'
        ),
        parameterSets = cms.vstring(
            'pythia8CommonSettings',
            'pythia8CP5Settings',
            'pythia8PowhegEmissionVetoSettings',
            'processParameters',
            'pythia8PSweightsSettings',
        )
    ),
    comEnergy = cms.double(13600),
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    pythiaPylistVerbosity = cms.untracked.int32(1),
)
