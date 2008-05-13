import FWCore.ParameterSet.Config as cms

# HLT setup ############################
# include "HLTrigger/Configuration/data/common/HLTSetup.cff"
from HLTrigger.special.HLTFullRecoForSpecial_cff import *
import HLTrigger.HLTfilters.hltLevel1GTSeed_cfi
# l1 seed filter #############
l1sIsolTrack = HLTrigger.HLTfilters.hltLevel1GTSeed_cfi.hltLevel1GTSeed.clone()
import HLTrigger.HLTcore.hltPrescaler_cfi
# PRESCALERS ###################
preIsolTrack = HLTrigger.HLTcore.hltPrescaler_cfi.hltPrescaler.clone()
# CALO #######################
#include "RecoLocalCalo/EcalRecProducers/data/getEcalConditions_frontier.cff"
#include "RecoLocalCalo/Configuration/data/ecalLocalRecoSequence_frontier.cff"
# TRACKER LOCAL ################
from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
# ISOLATED PIXEL CANDS ##########################
from Calibration.HcalIsolatedTrackReco.isolPixelTrackProd_cfi import *
# ISOLATED PIXEL CAND FILTER ####################
from HLTrigger.special.isolPixelTrackFilter_cfi import *
# REGIONAL FED COLLECTIONS PRODUCERS ############
from Calibration.HcalIsolatedTrackReco.subdetFED_cfi import *
from Calibration.HcalIsolatedTrackReco.stripFED_cfi import *
from Calibration.HcalIsolatedTrackReco.ecalFED_cfi import *
#
l1SeedFilter = cms.Sequence(l1sIsolTrack)
doPixelReco = cms.Sequence(cms.SequencePlaceholder("doLocalPixel"))
doPixTrackInput = cms.Sequence(pixelTrackingForIsol*isolPixelTrackProd)
pixelTrackFilter = cms.Sequence(isolPixelTrackFilter)
regFED = cms.Sequence(subdetFED*stripFED*ecalFED)
pixelIsolFilter = cms.Sequence(doPixelReco*doPixTrackInput*pixelTrackFilter)
#
hltHcalIsolatedTrack = cms.Sequence(cms.SequencePlaceholder("hltBegin")*l1SeedFilter*preIsolTrack+pixelIsolFilter*regFED)
l1sIsolTrack.L1SeedsLogicalExpression = 'L1_SingleJet30 OR L1_SingleJet50 OR L1_SingleJet70 OR L1_SingleJet100 OR L1_SingleTauJet30 OR L1_SingleTauJet40 OR L1_SingleTauJet60 OR L1_SingleTauJet80 '
preIsolTrack.prescaleFactor = 1
isolPixelTrackProd.L1GTSeedLabel = 'l1sIsolTrack'
stripFED.regSeedLabel = 'isolPixelTrackFilter'
ecalFED.regSeedLabel = 'isolPixelTrackFilter'

