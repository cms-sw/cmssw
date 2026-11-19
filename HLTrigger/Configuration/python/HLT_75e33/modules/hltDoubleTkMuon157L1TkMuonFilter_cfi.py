import FWCore.ParameterSet.Config as cms

hltDoubleTkMuon157L1TkMuonFilter =  cms.EDFilter("HLTP2GTSingleObjectFilter",
                                                 saveTags = cms.bool(True),
                                                 l1GTAlgoBlockTag = cms.InputTag('l1tGTAlgoBlockProducer'),
                                                 minN = cms.uint32(2),
                                                 l1GTAlgos = cms.VPSet(
                                                     cms.PSet(
                                                         name = cms.string('pDoubleTkMuon15_7'),
                                                         collection = cms.PSet(
                                                             objectType = cms.string('GMTTkMuons'),
                                                             minPt = cms.double(0),
                                                             maxAbsEta = cms.double(9999.)
                                                         )
                                                     )
                                                 ))
