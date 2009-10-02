import FWCore.ParameterSet.Config as cms

process = cms.Process("Alignment")
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
#process.load("Alignment.TrackHitFilter.TrackHitFilter_cfi")
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
#process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi")
#process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

# "including" common configuration
<COMMON>

process.source = cms.Source("PoolSource",
							useCSA08Kludge = cms.untracked.bool(True),
							fileNames = cms.untracked.vstring(<FILE>)
)

# parameters for TrackRefitter
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter1 = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone()
#process.TrackRefitter1 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone()
process.TrackRefitter1.src = '<SKIM>'
process.TrackRefitter1.TrajectoryInEvent = True
process.TrackRefitter1.TTRHBuilder = "WithAngleAndTemplate" #"WithTrackAngle"


####  new FILTER
#-- new track hit filter
# TrackerTrackHitFilter takes as input the tracks/trajectories coming out from TrackRefitter1
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
process.TrackerTrackHitFilter.src = 'TrackRefitter1'
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 6
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [ ]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 8.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= False
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= False
#process.TrackerTrackHitFilter.PxlCorrClusterChargeCut=10000.0

# track producer to be run after the alignment hit filter
#import RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff
process.ctfProducerCustomised = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff.ctfWithMaterialTracks.clone()
process.ctfProducerCustomised.src = 'TrackerTrackHitFilter'
##process.ctfProducerCustomised.beamspot='offlineBeamSpot'
process.ctfProducerCustomised.TTRHBuilder = 'WithAngleAndTemplate'
process.ctfProducerCustomised.TrajectoryInEvent = True

# parameters for TrackSelector
###process.AlignmentTrackSelector.src = '<SKIM>'
process.AlignmentTrackSelector.src = 'ctfProducerCustomised'
#process.AlignmentTrackSelector.src = 'HitFilteredTracks'
# "including" selection for this track sample
<SELECTION>

process.TrackRefitter2 = process.TrackRefitter1.clone(
        src = 'AlignmentTrackSelector'
        )

#process.TrackRefitter2 = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone()
#process.TrackRefitter2 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone()
#process.TrackRefitter2.src = 'AlignmentTrackSelector'
#process.TrackRefitter2.TrajectoryInEvent = True
#process.TrackRefitter2.TTRHBuilder = "WithAngleAndTemplate"


# parameters for HIP
process.AlignmentProducer.tjTkAssociationMapTag = 'TrackRefitter2'
process.AlignmentProducer.algoConfig.outpath = ''
process.AlignmentProducer.algoConfig.uvarFile = '<PATH>/IOUserVariables.root'
###process.AlignmentProducer.algoConfig.uvarFile = './IOUserVariables.root'
process.AlignmentProducer.algoConfig.eventPrescale= 1
process.AlignmentProducer.algoConfig.fillTrackMonitoring=True
process.AlignmentProducer.algoConfig.outfile =  '<PATH>/HIPAlignmentEvents.root'
process.AlignmentProducer.algoConfig.outfile2 = '<PATH>/HIPAlignmentAlignables.root'
process.AlignmentProducer.algoConfig.applyAPE = True

process.AlignmentProducer.algoConfig.apeParam = cms.VPSet(cms.PSet(
															function = cms.string('linear'),
															apeRPar = cms.vdouble(0.001, 0.001, 100.0),
															apeSPar = cms.vdouble(0.0500, 0.0500, 100.0),
															Selector = cms.PSet(
																				alignParams = cms.vstring('TrackerTPBModule,000000')
																				)
															), 
												   cms.PSet(
															function = cms.string('linear'),
															apeRPar = cms.vdouble(0.001, 0.001, 100.0),
															apeSPar = cms.vdouble(0.0500, 0.0500, 100.0),
															Selector = cms.PSet(
																				alignParams = cms.vstring('TrackerTPEModule,000000')
																				)
															), 
												   cms.PSet(
															function = cms.string('linear'),
															apeRPar = cms.vdouble(0.001, 0.001, 100.0),
															apeSPar = cms.vdouble(0.0300, 0.0300, 100.0),
                                                                                                                        Selector = cms.PSet(
																				alignParams = cms.vstring('TIBDets,000000')
																				)
															), 
												   cms.PSet(
															function = cms.string('linear'),
															apeRPar = cms.vdouble(0.001, 0.001, 100.0),
															apeSPar = cms.vdouble(0.0300, 0.0300, 100.0),
															Selector = cms.PSet(
																				alignParams = cms.vstring('TIDDets,000000')
																				)
															), 
												   cms.PSet(
															function = cms.string('linear'),
															apeRPar = cms.vdouble(0.001, 0.001, 100.0),
															apeSPar = cms.vdouble(0.0300, 0.0300, 100.0),
                                                                                                                        Selector = cms.PSet(
																				alignParams = cms.vstring('TOBDets,000000')
																				)
															), 
												   cms.PSet(
															function = cms.string('linear'),
															apeRPar = cms.vdouble(0.001, 0.001, 100.0),
															apeSPar = cms.vdouble(0.0300, 0.0300, 100.0),
														
															Selector = cms.PSet(
																				alignParams = cms.vstring('TECDets,000000')
																			    )
															)
												   )	


process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter1*process.TrackerTrackHitFilter*process.ctfProducerCustomised*process.AlignmentTrackSelector*process.TrackRefitter2)
#process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter1*process.TrackerTrackHitFilter*process.ctfProducerCustomised)


