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
