import FWCore.ParameterSet.Config as cms

# module to select Electrons
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
#
selectedPatLowPtElectrons = cms.EDFilter("PATElectronSelector",
    src = cms.InputTag("patLowPtElectrons"),
    cut = cms.string("pt > 1. && electronID('ID') > -0.25"),
)

# Modifier for bParking (fully open selection)
from Configuration.Eras.Modifier_bParking_cff import bParking
bParking.toModify(selectedPatLowPtElectrons,cut = "pt > 1.")

# Modifiers for legacy AOD
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
from Configuration.Eras.Modifier_run2_miniAOD_94XFall17_cff import run2_miniAOD_94XFall17
(run2_miniAOD_80XLegacy | run2_miniAOD_94XFall17).toModify(selectedPatLowPtElectrons, 
                                                           cut = "pt>99999."
                                                           )
