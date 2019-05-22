import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL isotrk:
#------------------------------------------------
import HLTrigger.HLTfilters.hltHighLevel_cfi
from Calibration.HcalAlCaRecoProducers.alcaisotrk_cfi import *

isoHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#    HLTPaths = ['HLT_IsoTrack'],
    eventSetupPathsKey='HcalCalIsoTrk',
    throw = False #dont throw except on unknown path name

)

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
TkAlIsoProd = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
TkAlIsoProd.filter = False
TkAlIsoProd.applyBasicCuts = False
TkAlIsoProd.applyMultiplicityFilter = False
TkAlIsoProd.applyNHighestPt = False
TkAlIsoProd.applyIsolationCut = False
TkAlIsoProd.applyChargeCheck = False
#TkAlIsoProd.src = 'generalTracks'

seqALCARECOHcalCalIsoTrk = cms.Sequence(isoHLT*alcaisotrk*TkAlIsoProd)




