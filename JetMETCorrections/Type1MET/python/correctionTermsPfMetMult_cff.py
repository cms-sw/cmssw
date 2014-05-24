import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from JetMETCorrections.Type1MET.pfMETmultShiftCorrections_cfi import *

##____________________________________________________________________________||
corrPfMetMult = pfMEtMultShiftCorr.clone()

##____________________________________________________________________________||
correctionTermsPfMetMult = cms.Sequence(
#    selectedVerticesForMEtCorr *
    corrPfMetMult
    )

##____________________________________________________________________________||
