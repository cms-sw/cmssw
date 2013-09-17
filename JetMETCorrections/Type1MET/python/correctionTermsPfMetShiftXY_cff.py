import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from JetMETCorrections.Type1MET.pfMETsysShiftCorrections_cfi import *

##____________________________________________________________________________||
corrPfMetShiftXY = pfMEtSysShiftCorr.clone()

##____________________________________________________________________________||
correctionTermsPfMetShiftXY = cms.Sequence(
    selectedVerticesForMEtCorr *
    corrPfMetShiftXY
    )

##____________________________________________________________________________||
