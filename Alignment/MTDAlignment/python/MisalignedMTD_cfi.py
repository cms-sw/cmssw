# Include this file to produce a misaligned muon geometry
#
import FWCore.ParameterSet.Config as cms

import Alignment.MTDAlignment.Scenarios_cff as _MTDScenarios
MisalignedMTD = cms.EDAnalyzer("MTDMisalignedProducer",
                                saveToDbase = cms.untracked.bool(False),
                                scenario = _MTDScenarios.MTDNoMovementsScenario
                                )
