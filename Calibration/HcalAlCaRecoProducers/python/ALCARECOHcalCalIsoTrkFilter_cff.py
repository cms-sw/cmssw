import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL isotrk:
#------------------------------------------------

from Calibration.HcalAlCaRecoProducers.alcaIsoTracksFilter_cfi import *

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
TkAlIsoProd = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
TkAlIsoProd.filter = False
TkAlIsoProd.applyBasicCuts = False
TkAlIsoProd.applyMultiplicityFilter = False
TkAlIsoProd.applyNHighestPt = False
TkAlIsoProd.applyIsolationCut = False
TkAlIsoProd.applyChargeCheck = False

seqALCARECOHcalCalIsoTrk = cms.Sequence(AlcaIsoTracksFilter*TkAlIsoProd)




