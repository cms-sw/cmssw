import FWCore.ParameterSet.Config as cms

TrackerTrackHitFilter = cms.EDProducer("TrackerTrackHitFilter",
                              src = cms.InputTag("generalTracks"),
                              minimumHits =cms.uint32(3), ##min number of hits for refit
                              ## # layers to remove
                              commands = cms.vstring(
                                         "drop PXB",  "drop PXE"   ### same works for TIB, TID, TOB, TEC,
                                        #"drop TIB 3",  ## you can also drop specific layers/wheel/disks
                                        #"keep PXB 3",  ## you can also 'keep' some layer after
                                                        ##having dropped the whole structure
                               ),
                              
                              ###list of individual detids to turn off, in addition to the structures above
                              detsToIgnore = cms.vuint32( ),
                              
                              ### what to do with invalid hits
                              replaceWithInactiveHits =cms.bool(False), ## instead of removing hits replace
                                                                        ## them with inactive hits, so you still
                                                                        ## consider the multiple scattering
                              stripFrontInvalidHits   =cms.bool(False),   ## strip invalid & inactive hits from
                              stripBackInvalidHits    =cms.bool(False),   ## any end of the track
                              
                              stripAllInvalidHits = cms.bool(False), ##not sure if it's better 'true' or 'false'
                                                                     ## might be dangerous to turn on
                                                                     ## as you will forget about MS

                              ### hit quality cuts
                              rejectBadStoNHits = cms.bool(False),
                              CMNSubtractionMode = cms.string("Median"), ## "TT6"
                              StoNcommands = cms.vstring(
                                                         "TIB 1.0 ", "TOB 1.0 999.0"
                                                        ),
                              useTrajectories=cms.bool(False),
                              rejectLowAngleHits=cms.bool(False),
                              TrackAngleCut=cms.double(0.25),       ## in radians
                              tagOverlaps=cms.bool(False),
                              usePixelQualityFlag=cms.bool(False),
                              PxlTemplateProbXYCut=cms.double(0.000125), #recommended by experts
                              PxlTemplateProbXYChargeCut=cms.double(-99.), #recommended by experts
                              PxlTemplateqBinCut =cms.vint32(0, 3),       #recommended by experts
                              PxlCorrClusterChargeCut = cms.double(-999.0)   
                             )####end of module 
