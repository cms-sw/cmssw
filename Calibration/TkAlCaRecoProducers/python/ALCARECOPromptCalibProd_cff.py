import FWCore.ParameterSet.Config as cms


# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOTkAlMinBiasFilterForBS = copy.deepcopy(hltHighLevel)
ALCARECOTkAlMinBiasFilterForBS.HLTPaths = ['pathALCARECOTkAlMinBias']
ALCARECOTkAlMinBiasFilterForBS.throw = True ## dont throw on unknown path names
ALCARECOTkAlMinBiasFilterForBS.TriggerResultsTag = cms.InputTag("TriggerResults","","RECO")
#process.TkAlMinBiasFilterForBS.eventSetupPathsKey = 'pathALCARECOTkAlMinBias:RECO'
#ALCARECODtCalibHLTFilter.andOr = True ## choose logical OR between Triggerbits

from RecoVertex.BeamSpotProducer.AlcaBeamSpotProducer_cff import *
alcaBeamSpotProducer.AlcaBeamSpotProducerParameters.fitEveryNLumi = cms.untracked.int32(1)
alcaBeamSpotProducer.AlcaBeamSpotProducerParameters.resetEveryNLumi = cms.untracked.int32(1)
alcaBeamSpotProducer.AlcaBeamSpotProducerParameters.TrackCollection = cms.untracked.InputTag('ALCARECOTkAlMinBias')
alcaBeamSpotProducer.BeamFitter.TrackCollection = cms.untracked.InputTag('ALCARECOTkAlMinBias')

seqALCARECOPromptCalibProd = cms.Sequence(ALCARECOTkAlMinBiasFilterForBS * alcaBeamSpotProducer)

#process.bsProductionPath = cms.Path(process.TkAlMinBiasFilterForBS+process.alcaBeamSpotProducer)
