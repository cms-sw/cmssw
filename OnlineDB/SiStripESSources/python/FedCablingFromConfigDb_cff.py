import FWCore.ParameterSet.Config as cms

from OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi import *
FedCablingFromConfigDb = cms.ESSource("SiStripFedCablingBuilderFromDb",
    CablingSource = cms.untracked.string('UNDEFINED')
)

# foo bar baz
# CI99cUe2ao7O4
