import FWCore.ParameterSet.Config as cms

dimuonAcceptance = cms.PSet(filterType = cms.untracked.string("MultiCandGenEvtSelector"),
                            ptMin = cms.double(2.5),
                            etaMax = cms.double(2.5),
                            pdg = cms.int32(13),
                            status = cms.int32(1),
                            minimumCandidates = cms.int32(2)
                            )
