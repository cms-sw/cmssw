import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL isotrk:
#------------------------------------------------

from Calibration.HcalAlCaRecoProducers.alcaIsoTracksFilter_cfi import *

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
TkAlIsoProdFilter = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
TkAlIsoProdFilter.filter = False
TkAlIsoProdFilter.applyBasicCuts = False
TkAlIsoProdFilter.applyMultiplicityFilter = False
TkAlIsoProdFilter.applyNHighestPt = False
TkAlIsoProdFilter.applyIsolationCut = False
TkAlIsoProdFilter.applyChargeCheck = False

seqALCARECOHcalCalIsoTrkFilter = cms.Sequence(AlcaIsoTracksFilter*TkAlIsoProdFilter)




