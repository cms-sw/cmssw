import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOTkAlMinBiasFilterForBSHP = copy.deepcopy(hltHighLevel)
ALCARECOTkAlMinBiasFilterForBSHP.HLTPaths = ['pathALCARECOTkAlMinBias']
ALCARECOTkAlMinBiasFilterForBSHP.throw = True ## dont throw on unknown path names
ALCARECOTkAlMinBiasFilterForBSHP.TriggerResultsTag = cms.InputTag("TriggerResults","","RECO")
#process.TkAlMinBiasFilterForBS.eventSetupPathsKey = 'pathALCARECOTkAlMinBias:RECO'
#ALCARECODtCalibHLTFilter.andOr = True ## choose logical OR between Triggerbits


# ------------------------------------------------------------------------------
# configure the beam-spot production
from Calibration.TkAlCaRecoProducers.AlcaBeamSpotProducerHP_cff import *
#alcaBeamSpotProducerHP.BeamFitter.TrackCollection = 'ALCARECOTkAlMinBias'
#alcaBeamSpotProducerHP.BeamFitter.MinimumTotalLayers = 6
#alcaBeamSpotProducerHP.BeamFitter.MinimumPixelLayers = -1
#alcaBeamSpotProducerHP.BeamFitter.MaximumNormChi2 = 10
#alcaBeamSpotProducerHP.BeamFitter.MinimumInputTracks = 50
#alcaBeamSpotProducerHP.BeamFitter.MinimumPt = 1.0
#alcaBeamSpotProducerHP.BeamFitter.MaximumImpactParameter = 1.0
#alcaBeamSpotProducerHP.BeamFitter.TrackAlgorithm =  cms.untracked.vstring()
#alcaBeamSpotProducerHP.BeamFitter.Debug = True
#alcaBeamSpotProducerHP.PVFitter.Apply3DFit = True
#alcaBeamSpotProducerHP.PVFitter.minNrVerticesForFit = 10 

# fit as function of lumi sections
#alcaBeamSpotProducerHP.AlcaBeamSpotProducerParameters.fitEveryNLumi = 1
#alcaBeamSpotProducerHP.AlcaBeamSpotProducerParameters.resetEveryNLumi = 1


# ------------------------------------------------------------------------------
# this is for filtering on L1 technical trigger bit
# Set the HLT paths
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOHltFilterForBSHP = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
##     HLTPaths = [
##     #Minimum Bias
##     "HLT_MinBias*"
##     ],
    eventSetupPathsKey = 'PromptCalibProd',
    throw = False # tolerate triggers stated above, but not available
    )

seqALCARECOPromptCalibProd = cms.Sequence(ALCARECOTkAlMinBiasFilterForBSHP *
                                          ALCARECOHltFilterForBSHP *
                                          alcaBeamSpotProducerHP)
