import FWCore.ParameterSet.Config as cms

process = cms.Process("Alignment")
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.load("Alignment.TrackHitFilter.TrackHitFilter_cfi")
#process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi")
<<<<<<< align_tpl.py
#process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
=======
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
>>>>>>> 1.5
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

# "including" common configuration
<COMMON>

process.source = cms.Source("PoolSource",
							useCSA08Kludge = cms.untracked.bool(True),
							fileNames = cms.untracked.vstring(<FILE>)
)

# parameters for TrackSelector
##process.AlignmentTrackSelector.src = '<SKIM>'
process.AlignmentTrackSelector.src = 'TrackRefitter1'
# "including" selection for this track sample
<SELECTION>

# parameters for TrackHitFilter
##process.TrackHitFilter.src = 'AlignmentTrackSelector'
process.TrackHitFilter.src = '<SKIM>'
#process.TrackHitFilter.hitSelection = "SiStripOnly"
#process.TrackHitFilter.hitSelection = "TOBandTIBOnly"
process.TrackHitFilter.hitSelection = "All"        
process.TrackHitFilter.rejectBadStoNHits = True
process.TrackHitFilter.theStoNthreshold = 18
process.TrackHitFilter.minHitsForRefit = 6


# parameters for TrackRefitter
#process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter1 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone()
process.TrackRefitter1.src = 'TrackHitFilter'
process.TrackRefitter1.TrajectoryInEvent = False
process.TrackRefitter1.TTRHBuilder = "WithTrackAngle"

process.TrackRefitter2 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone()
process.TrackRefitter2.src = 'AlignmentTrackSelector'
process.TrackRefitter2.TrajectoryInEvent = True
process.TrackRefitter2.TTRHBuilder = "WithTrackAngle"


# parameters for HIP
process.AlignmentProducer.tjTkAssociationMapTag = 'TrackRefitter2'
process.AlignmentProducer.algoConfig.outpath = ''
process.AlignmentProducer.algoConfig.uvarFile = '<PATH>/IOUserVariables.root'
###process.AlignmentProducer.algoConfig.uvarFile = './IOUserVariables.root'
process.AlignmentProducer.algoConfig.eventPrescale= 1
process.AlignmentProducer.algoConfig.fillTrackMonitoring=False
process.AlignmentProducer.algoConfig.outfile =  '<PATH>/HIPAlignmentEvents.root'
process.AlignmentProducer.algoConfig.outfile2 = '<PATH>/HIPAlignmentAlignables.root'
process.AlignmentProducer.algoConfig.applyAPE = True

process.AlignmentProducer.algoConfig.apeParam = cms.VPSet(cms.PSet(
															function = cms.string('linear'),
															#apeRPar = cms.vdouble(0.01, 0.0, 100.0),
															#apeSPar = cms.vdouble(0.0100, 0.00, 100.0),
															apeRPar = cms.vdouble(0.010, 0.010, 100.0),
															apeSPar = cms.vdouble(0.10, 0.10, 100.0),
															Selector = cms.PSet(
																				alignParams = cms.vstring('TrackerTPBModule,000000')
																				)
															), 
												   cms.PSet(
															function = cms.string('linear'),
															#apeRPar = cms.vdouble(0.01, 0.0, 100.0),
															#apeSPar = cms.vdouble(0.0600, 0.00, 100.0),
															apeRPar = cms.vdouble(0.010, 0.010, 100.0),
															apeSPar = cms.vdouble(0.10, 0.10, 100.0),
															Selector = cms.PSet(
																				alignParams = cms.vstring('TrackerTPEModule,000000')
																				)
															), 
												   cms.PSet(
															function = cms.string('linear'),
															apeRPar = cms.vdouble(0.003, 0.003, 100.0),
															apeSPar = cms.vdouble(0.0300, 0.0300, 100.0),
                                                                                                                        #
                                                                                                                        #decrease by 1 mrad every 20 iter and by 200 um every 20 iter
															#apeRPar = cms.vdouble(0.005, 0.001, 20.0),
															#apeSPar = cms.vdouble(0.0900, 0.0200, 20.0),
															Selector = cms.PSet(
																				alignParams = cms.vstring('TIBDets,000000')
																				)
															), 
												   cms.PSet(
															function = cms.string('linear'),
															apeRPar = cms.vdouble(0.006, 0.006, 100.0),
															apeSPar = cms.vdouble(0.0600, 0.0600, 100.0),
															#apeRPar = cms.vdouble(0.010, 0.002, 20.0),
															#apeSPar = cms.vdouble(0.0900, 0.0200, 20.0),
															Selector = cms.PSet(
																				alignParams = cms.vstring('TIDDets,000000')
																				)
															), 
												   cms.PSet(
															function = cms.string('linear'),
															apeRPar = cms.vdouble(0.003, 0.003, 100.0),
															apeSPar = cms.vdouble(0.0300, 0.0300, 100.0),
															#apeRPar = cms.vdouble(0.005, 0.001, 20.0),
															#apeSPar = cms.vdouble(0.0700, 0.0150, 20.0),
														Selector = cms.PSet(
																				alignParams = cms.vstring('TOBDets,000000')
																				)
															), 
												   cms.PSet(
															function = cms.string('linear'),
															apeRPar = cms.vdouble(0.006, 0.006, 100.0),
															apeSPar = cms.vdouble(0.0600, 0.0600, 100.0),
															#apeRPar = cms.vdouble(0.010, 0.002, 20.0),
															#apeSPar = cms.vdouble(0.1000, 0.0200, 20.0),
															Selector = cms.PSet(
																				alignParams = cms.vstring('TECDets,000000')
																			    )
															)
												   )	

#process.p = cms.Path(process.offlineBeamSpot*process.AlignmentTrackSelector*process.TrackHitFilter*process.TrackRefitter1)
process.p = cms.Path(process.offlineBeamSpot*process.TrackHitFilter*process.TrackRefitter1*process.AlignmentTrackSelector*process.TrackRefitter2)
###process.p = cms.Path(process.offlineBeamSpot*process.TrackHitFilter*process.TrackRefitter)
