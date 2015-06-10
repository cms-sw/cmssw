import FWCore.ParameterSet.Config as cms
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalIsolElectron_Output_cff import *
import copy

OutALCARECOEcalRecalElectron_noDrop=copy.deepcopy(OutALCARECOEcalCalElectron_noDrop)
OutALCARECOEcalRecalElectron_noDrop.outputCommands+=cms.untracked.vstring(
    'keep *_electronRecalibSCAssociator_*_*'
    )

OutALCARECOEcalRecalElectron=copy.deepcopy(OutALCARECOEcalRecalElectron_noDrop)
OutALCARECOEcalRecalElectron.outputCommands.insert(0, "drop *")
OutALCARECOEcalRecalElectron.outputCommands += cms.untracked.vstring(
    'drop *_gedGsfElectron*_*_*',
    'drop *_*Unclean*_*_*',
    'drop *_*unclean*_*_*',
    'drop *_*_*Unclean*_*',
    'drop *_*_*unclean*_*',
    )

OutALCARECOEcalRecalElectron.SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('pathALCARECOEcalRecalZElectron', 'pathALCARECOEcalRecalZSCElectron', 'pathALCARECOEcalRecalWElectron')
    #SelectEvents = cms.vstring('pathALCARECOEcalRecalZElectron', 'pathALCARECOEcalRecalWElectron')
    )

