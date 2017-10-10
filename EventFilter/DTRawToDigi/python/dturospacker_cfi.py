import FWCore.ParameterSet.Config as cms

dturospacker = cms.EDProducer("DTuROSDigiToRaw",
                                DTDigi_Source = cms.InputTag("simMuonDTDigis"),
                                feds     = cms.untracked.vint32( 1368, 1369, 1370,),
                                debug = cms.untracked.bool(True),
                             )
