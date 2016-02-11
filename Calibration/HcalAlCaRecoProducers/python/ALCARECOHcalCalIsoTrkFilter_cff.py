import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL isotrk:
#------------------------------------------------

from Calibration.HcalAlCaRecoProducers.alcaIsoTracksFilter_cfi import *

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
TkAlIsoProdFilter = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
TkAlIsoProd.filter = False
TkAlIsoProd.applyBasicCuts = False
TkAlIsoProd.applyMultiplicityFilter = False
TkAlIsoProd.applyNHighestPt = False
TkAlIsoProd.applyIsolationCut = False
TkAlIsoProd.applyChargeCheck = False

seqALCARECOHcalCalIsoTrkFilter = cms.Sequence(AlcaIsoTracksFilter*TkAlIsoProdFilter)




