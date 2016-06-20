import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from JetMETCorrections.Type1MET.pfMETmultShiftCorrections_cfi import *

##____________________________________________________________________________||
corrPfMetXYMult = pfMEtMultShiftCorr.clone()
corrPfMetXYMultDB = pfMEtMultShiftCorrDB.clone()

##____________________________________________________________________________||
correctionTermsPfMetMult = cms.Sequence(
    corrPfMetXYMult
    )

##____________________________________________________________________________||
correctionTermsPfMetMultDB = cms.Sequence(
    corrPfMetXYMultDB
    )

##____________________________________________________________________________||
