import FWCore.ParameterSet.Config as cms

# Standalone, fully resolved configuration of the Phase-2 L1 trigger emulation,
# modelled on HLTrigger/Configuration/python/HLT_75e33_cff.py.
#
# Scope: trigger primitives, track trigger (DTC, TTClusters/TTStubs, TTTracks),
# correlator/GTT/GMT/calo emulators and the Phase-2 Global Trigger (P2GT) menu.
#
# Conventions (see L1_Phase2/README.md):
#  - one object per file: L1_Phase2/<kind>/<label>_cfi.py
#  - no Era/Modifier logic: everything is resolved at generation time
#  - the fragment only loads the files and defines the schedule
#
# STATUS: skeleton only. Contents are to be produced by the generator
# (https://github.com/artlbv/cmssw/issues/10).

fragment = cms.ProcessFragment("L1")

### Services
# fragment.load("L1Trigger/Configuration/L1_Phase2/services/<label>_cfi")

### Event-setup (ES sources, ES producers, ES prefers)
# fragment.load("L1Trigger/Configuration/L1_Phase2/eventsetup/<label>_cfi")

### Shared PSets
# fragment.load("L1Trigger/Configuration/L1_Phase2/psets/<label>_cfi")

### Modules (producers, filters, analyzers)
# fragment.load("L1Trigger/Configuration/L1_Phase2/modules/<label>_cfi")

### Sequences and tasks
# fragment.load("L1Trigger/Configuration/L1_Phase2/sequences/<label>_cfi")

### Paths: track trigger, trigger primitives, correlator, GT/menu (p<X> = L1_<X>)
# fragment.load("L1Trigger/Configuration/L1_Phase2/paths/<label>_cfi")

fragment.schedule = cms.Schedule(*[
    # fragment.L1TrackTrigger_step,
    # fragment.L1Emulator_step,
    # fragment.L1P2GT_step,
])
