# Include this file to produce a misaligned muon geometry
#
import FWCore.ParameterSet.Config as cms

import Alignment.MuonAlignment.Scenarios_cff as _MuonScenarios
MisalignedMuon = cms.EDAnalyzer("MuonMisalignedProducer",
                                saveToDbase = cms.untracked.bool(False),
                                scenario = _MuonScenarios.MuonNoMovementsScenario
                                )
