import FWCore.ParameterSet.Config as cms

chargedPackedCandsForTkMet = cms.EDFilter("CandPtrSelector",
                                          src=cms.InputTag("packedPFCandidates"),
                                          cut=cms.string("charge()!=0 && pvAssociationQuality()>=4 && vertexRef().key()==0")
                                      )
