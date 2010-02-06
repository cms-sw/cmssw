import FWCore.ParameterSet.Config as cms

process = cms.Process("Alignment")
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
#process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")


# "including" common configuration
<COMMON>

process.source = cms.Source("PoolSource",
#							useCSA08Kludge = cms.untracked.bool(True),
							fileNames = cms.untracked.vstring(<FILE>)
)

process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi")
process.offlinePrimaryVertices.TrackLabel = cms.InputTag("TrackRefitter1")

# "including" selection for this track sample
<SELECTION>


# parameters for HIP
process.AlignmentProducer.tjTkAssociationMapTag = 'TrackRefitter2'
process.AlignmentProducer.HitPrescaleMap = '' #if this is not empty, turn on the usage of prescaled hits
process.AlignmentProducer.algoConfig.outpath = ''
process.AlignmentProducer.algoConfig.uvarFile = '<PATH>/IOUserVariables.root'
###process.AlignmentProducer.algoConfig.uvarFile = './IOUserVariables.root'
process.AlignmentProducer.algoConfig.eventPrescale= 100
process.AlignmentProducer.algoConfig.fillTrackMonitoring=False
#process.AlignmentProducer.algoConfig.outfile =  '<PATH>/HIPAlignmentEvents.root'
#process.AlignmentProducer.algoConfig.outfile2 = '<PATH>/HIPAlignmentAlignables.root'
process.AlignmentProducer.algoConfig.applyAPE = True

process.AlignmentProducer.algoConfig.apeParam = cms.VPSet(cms.PSet(
															function = cms.string('linear'),
															apeRPar = cms.vdouble(0.001, 0.001, 100.0),
															apeSPar = cms.vdouble(0.100, 0.100, 100.0),
															Selector = cms.PSet(
																				alignParams = cms.vstring('TrackerTPBModule,000000')
																				)
															), 
												   cms.PSet(
															function = cms.string('linear'),
                                                                                                                        apeRPar = cms.vdouble(0.003, 0.003, 100.0),
															apeSPar = cms.vdouble(0.0600, 0.0600, 100.0),
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
															apeSPar = cms.vdouble(0.0600, 0.0600, 100.0),
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
															apeSPar = cms.vdouble(0.0600, 0.0600, 100.0),
															Selector = cms.PSet(
																				alignParams = cms.vstring('TECDets,000000')
																			    )
															)
												   )	


#no constraints
#process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter1*process.TrackerTrackHitFilter*process.ctfProducerCustomised*process.AlignmentTrackSelector*process.TrackRefitter2)

#constraints
if 'MBVertex'=='<FLAG>':
    process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter1*process.offlinePrimaryVertices*process.TrackerTrackHitFilter*process.ctfProducerCustomised*process.AlignmentTrackSelector*process.doConstraint*process.TrackRefitter2)
elif 'MB'=='<FLAG>':
    process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter1*process.TrackerTrackHitFilter*process.ctfProducerCustomised*process.AlignmentTrackSelector*process.TrackRefitter2)
elif 'COSMICS' =='<FLAG>':
    process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter1*process.TrackerTrackHitFilter*process.ctfProducerCustomised*process.AlignmentTrackSelector*process.TrackRefitter2)
else :
    process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter1*process.TrackerTrackHitFilter*process.ctfProducerCustomised*process.AlignmentTrackSelector*process.TrackRefitter2)
