import FWCore.ParameterSet.Config as cms

hcalBadDigiFilter = cms.EDFilter("HcalBadDigiFilter",
                                unpackerReportLabel  = cms.InputTag("hcalDigis"),
                                hbheRecHitsLabel = cms.InputTag("hbhereco"),
                                debug = cms.bool(False),
                                listOfFlags = cms.vstring('HBHERun3BadCapId',
                                                          'HBHERun3NonrotatingCapId',
                                                          'HBHERun3StuckADC',
                                                          'HBHERun3repeatedADCblock',),
                                minRecHitEnergies = cms.vdouble(-100., -100., 10., 10.), # in the same order as listOfFlags. < 0 means no cut
                                maxBadChannels = cms.uint32(5),
                                useBadChannelsTopology = cms.bool(False)
)
