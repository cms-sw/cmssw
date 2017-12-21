import FWCore.ParameterSet.Config as cms

dturospacker = cms.EDProducer("DTuROSDigiToRaw",
                                DTDigi_Source = cms.InputTag("simMuonDTDigis"),
                                feds     = cms.untracked.vint32( 1369, 1370, 1371,),
                                debug = cms.untracked.bool(True),
                             )
