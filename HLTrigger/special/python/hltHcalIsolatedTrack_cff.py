import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
# HLT setup ############################
# include "HLTrigger/Configuration/data/common/HLTSetup.cff"
# include "HLTrigger/special/data/HLTFullRecoForSpecial.cff"
# l1 seed filter #############
l1sIsolTrack = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
# PRESCALERS ###################
preIsolTrack = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
preIsolTrackNoEcalIso = copy.deepcopy(hltPrescaler)
# CALO #######################
#include "RecoLocalCalo/EcalRecProducers/data/getEcalConditions_frontier.cff"
#include "RecoLocalCalo/Configuration/data/ecalLocalRecoSequence_frontier.cff"
# TRACKER LOCAL ################
#include "RecoLocalTracker/Configuration/data/RecoLocalTracker.cff"
# include "RecoPixelVertexing/PixelTrackFitting/data/PixelTracks.cff"
# ECAL ISOLATION PRODUCER ##############
from Calibration.HcalIsolatedTrackReco.ecalIsolPartProd_cfi import *
# ECAL ISOLATION FILTER ##########################
from HLTrigger.special.ecalIsolFilter_cfi import *
# ISOLATED PIXEL CANDS ##########################
from Calibration.HcalIsolatedTrackReco.isolPixelTrackProd_cfi import *
# ISOLATED PIXEL CAND FILTER ####################
from HLTrigger.special.isolPixelTrackFilter_cfi import *
#
l1SeedFilter = cms.Sequence(l1sIsolTrack)
doEcal = cms.Sequence(cms.SequencePlaceholder("doLocalEcal"))
doEcalIsolInput = cms.Sequence(ecalIsolPartProd)
doPixelReco = cms.Sequence(cms.SequencePlaceholder("doLocalPixel"))
doPixTrackInput = cms.Sequence(cms.SequencePlaceholder("pixelTrackingForIsol")*isolPixelTrackProd)
pixelTrackFilter = cms.Sequence(isolPixelTrackFilter)
# L2 filtering on isolation in ECAL ##############################
l2EcalIsolFilter = cms.Sequence(doEcal*doEcalIsolInput*ecalIsolFilter)
# L3 filtering on isolation in pixel detector ####################
l3PixelIsolFilter = cms.Sequence(doPixelReco*doPixTrackInput*pixelTrackFilter)
#
hltHcalIsolatedTrack = cms.Sequence(cms.SequencePlaceholder("hltBegin")*l1SeedFilter*preIsolTrack*l2EcalIsolFilter+l3PixelIsolFilter)
hltHcalIsolatedTrackNoEcalIsol = cms.Sequence(cms.SequencePlaceholder("hltBegin")*l1SeedFilter*preIsolTrackNoEcalIso+l3PixelIsolFilter)
l1sIsolTrack.L1SeedsLogicalExpression = 'L1_SingleJet100 OR L1_SingleTauJet100'
preIsolTrack.prescaleFactor = 1
preIsolTrackNoEcalIso.prescaleFactor = 1
ecalIsolPartProd.L1GTSeedLabel = 'l1sIsolTrack'
isolPixelTrackProd.L1GTSeedLabel = 'l1sIsolTrack'

