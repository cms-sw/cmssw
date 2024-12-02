import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.triggerResultsFilterFromDB_cfi
ALCARECOEcalESAlignHLT = HLTrigger.HLTfilters.triggerResultsFilterFromDB_cfi.triggerResultsFilterFromDB.clone(
    eventSetupPathsKey = 'EcalESAlign',
    usePathStatus = False,
    hltResults = 'TriggerResults::HLT',
    l1tResults = '', # leaving empty (not interested in L1T results)
    throw = False # tolerate triggers stated above, but not available
)

# this imports the module that produces a reduced collections for ES alignment
#from Calibration.EcalAlCaRecoProducers.EcalAlCaESAlignTrackReducer_cfi import *

# this imports the filter that skims the events requiring a min number of selected tracks

esSelectedTracks = cms.EDFilter("TrackSelector",
                                src = cms.InputTag('generalTracks'),
                                cut = cms.string("abs(eta)>1.65 && pt>1 && numberOfValidHits>=10")
                                )

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ecalAlCaESAlignTrackReducer = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src = cms.InputTag('esSelectedTracks'),
    filter = True, ##do not store empty events
    applyBasicCuts = False,
    ptMin = 1.0, ##GeV 
    etaMin = -3.5,
    etaMax = 3.5,
    nHitMin = 0
)

esMinTrackNumberFilter = cms.EDFilter("TrackCountFilter",
                                      src = cms.InputTag('ecalAlCaESAlignTrackReducer'),
                                      minNumber = cms.uint32(10)
                                      )

EcalESAlignTracksSkimSeq = cms.Sequence( esSelectedTracks * ecalAlCaESAlignTrackReducer * esMinTrackNumberFilter) 

seqEcalESAlign = cms.Sequence(ALCARECOEcalESAlignHLT * EcalESAlignTracksSkimSeq)

