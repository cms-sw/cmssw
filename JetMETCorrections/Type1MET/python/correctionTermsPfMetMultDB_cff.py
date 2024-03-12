import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from JetMETCorrections.Type1MET.pfMETmultShiftCorrectionsDB_cfi import *

##____________________________________________________________________________||
corrPfMetXYMultDB = pfMEtMultShiftCorrDB.clone()


##____________________________________________________________________________||
correctionTermsPfMetMultDB = cms.Sequence(
    corrPfMetXYMultDB
    )

##____________________________________________________________________________||
# foo bar baz
# 1sanf0BJqUXN1
# jAwgWy8U9MPiW
