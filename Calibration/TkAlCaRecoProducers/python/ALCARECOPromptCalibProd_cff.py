import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOTkAlMinBiasFilterForBS = copy.deepcopy(hltHighLevel)
ALCARECOTkAlMinBiasFilterForBS.HLTPaths = ['pathALCARECOTkAlMinBias']
ALCARECOTkAlMinBiasFilterForBS.throw = True ## dont throw on unknown path names
ALCARECOTkAlMinBiasFilterForBS.TriggerResultsTag = cms.InputTag("TriggerResults","","RECO")
#process.TkAlMinBiasFilterForBS.eventSetupPathsKey = 'pathALCARECOTkAlMinBias:RECO'
#ALCARECODtCalibHLTFilter.andOr = True ## choose logical OR between Triggerbits


# ------------------------------------------------------------------------------
# configure the beam-spot production
from Calibration.TkAlCaRecoProducers.AlcaBeamSpotProducer_cff import *
alcaBeamSpotProducer.BeamFitter.TrackCollection = 'ALCARECOTkAlMinBias'
alcaBeamSpotProducer.BeamFitter.MinimumTotalLayers = 6
alcaBeamSpotProducer.BeamFitter.MinimumPixelLayers = -1
alcaBeamSpotProducer.BeamFitter.MaximumNormChi2 = 10
alcaBeamSpotProducer.BeamFitter.MinimumInputTracks = 50
alcaBeamSpotProducer.BeamFitter.MinimumPt = 1.0
alcaBeamSpotProducer.BeamFitter.MaximumImpactParameter = 1.0
alcaBeamSpotProducer.BeamFitter.TrackAlgorithm =  cms.untracked.vstring()
#alcaBeamSpotProducer.BeamFitter.Debug = True
alcaBeamSpotProducer.PVFitter.Apply3DFit = True
alcaBeamSpotProducer.PVFitter.minNrVerticesForFit = 10 

# fit as function of lumi sections
alcaBeamSpotProducer.AlcaBeamSpotProducerParameters.fitEveryNLumi = 1
alcaBeamSpotProducer.AlcaBeamSpotProducerParameters.resetEveryNLumi = 1

# alcaBeamSpotProducer.AlcaBeamSpotProducerParameters.TrackCollection = cms.untracked.InputTag('ALCARECOTkAlMinBias')
# alcaBeamSpotProducer.AlcaBeamSpotProducerParameters.fitEveryNLumi = cms.untracked.int32(1)
# alcaBeamSpotProducer.AlcaBeamSpotProducerParameters.resetEveryNLumi = cms.untracked.int32(1)
# alcaBeamSpotProducer.AlcaBeamSpotProducerParameters.TrackCollection = cms.untracked.InputTag('ALCARECOTkAlMinBias')
# alcaBeamSpotProducer.BeamFitter.TrackCollection = cms.untracked.InputTag('ALCARECOTkAlMinBias')

# ------------------------------------------------------------------------------
# this is for filtering on L1 technical trigger bit
# Set the HLT paths
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOHltFilterForBS = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
##     HLTPaths = [
##     #Minimum Bias
##     "HLT_MinBias*"
##     ],
    eventSetupPathsKey = 'PromptCalibProd',
    throw = False # tolerate triggers stated above, but not available
    )


#from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff import *
#from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import hltLevel1GTSeed
#L1CollTrigger = hltLevel1GTSeed.clone()

#L1CollTrigger.L1TechTriggerSeeding = cms.bool(True)
#L1CollTrigger.L1SeedsLogicalExpression = cms.string('0 AND ( 40 OR 41 ) AND NOT (36 OR 37 OR 38 OR 39)')


# ------------------------------------------------------------------------------
# configuration to reproduce offlinePrimaryVertices
# FIXME: needs to be moved in the TkAlMinBias definition
# from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *
# offlineBeamSpotForBS = offlineBeamSpot.clone()
# from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import *

# # overwrite some defaults for PV producer
# offlinePrimaryVerticesForBS = offlinePrimaryVertices.clone()
# offlinePrimaryVerticesForBS.TrackLabel = cms.InputTag("ALCARECOTkAlMinBias")
# offlinePrimaryVerticesForBS.beamSpotLabel = cms.InputTag("offlineBeamSpotForBS")
# offlinePrimaryVerticesForBS.PVSelParameters.maxDistanceToBeam = 2
# offlinePrimaryVerticesForBS.TkFilterParameters.maxNormalizedChi2 = 20
# offlinePrimaryVerticesForBS.TkFilterParameters.minSiliconLayersWithHits = 5
# offlinePrimaryVerticesForBS.TkFilterParameters.maxD0Significance = 100
# offlinePrimaryVerticesForBS.TkFilterParameters.minPixelLayersWithHits = 1
# offlinePrimaryVerticesForBS.TkClusParameters.TkGapClusParameters.zSeparation = 1
# alcaBeamSpotProducer.PVFitter.VertexCollection = "offlinePrimaryVerticesForBS"
                     
seqALCARECOPromptCalibProd = cms.Sequence(ALCARECOTkAlMinBiasFilterForBS *
                                          ALCARECOHltFilterForBS *
#                                           offlineBeamSpotForBS +
#                                           offlinePrimaryVerticesForBS +
                                          alcaBeamSpotProducer)

#process.bsProductionPath = cms.Path(process.TkAlMinBiasFilterForBS+process.alcaBeamSpotProducer)
