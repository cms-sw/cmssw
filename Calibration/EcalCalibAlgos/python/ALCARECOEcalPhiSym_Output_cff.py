import FWCore.ParameterSet.Config as cms
import copy

OutALCARECOEcalPhiSymByRun_noDrop = cms.PSet(
    SelectEvents=cms.untracked.PSet(
        SelectEvents=cms.vstring(
            'pathALCARECOEcalPhiSym*')
    ),
    outputCommands=cms.untracked.vstring()
)
OutALCARECOEcalPhiSymByRun = copy.deepcopy(OutALCARECOEcalPhiSymByRun_noDrop)
OutALCARECOEcalPhiSymByRun.outputCommands.insert(0, "drop *")

OutALCARECOEcalPhiSymByLumi_noDrop = copy.deepcopy(OutALCARECOEcalPhiSymByRun_noDrop)
OutALCARECOEcalPhiSymByLumi = copy.deepcopy(OutALCARECOEcalPhiSymByRun)
