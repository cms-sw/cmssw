import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from JetMETCorrections.Type1MET.pfMETmultShiftCorrections_cfi import *

##____________________________________________________________________________||
corrPfMetXYMult = pfMEtMultShiftCorr.clone()

##____________________________________________________________________________||
correctionTermsPfMetMult = cms.Sequence(
    corrPfMetXYMult
    )

##____________________________________________________________________________||
