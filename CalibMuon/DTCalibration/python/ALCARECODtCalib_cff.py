import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *

from CalibMuon.DTCalibration.DTCalibMuonSelection_cfi import *

# AlCaReco for DT calibration
ALCARECODtCalibHLTFilter = copy.deepcopy(hltHighLevel)

ALCARECODtCalibHLTFilter.andOr = True ## choose logical OR between Triggerbits
ALCARECODtCalibHLTFilter.throw = False ## dont throw on unknown path names

ALCARECODtCalibHLTFilter.HLTPaths = ['HLT_L1MuOpen', 'HLT_L1Mu']

# Configuration for DT 4D segments without the wire propagation correction
from RecoLocalMuon.DTSegment.DTCombinatorialPatternReco2DAlgo_LinearDriftFromDB_cfi import *
from RecoLocalMuon.DTRecHit.DTLinearDriftFromDBAlgo_cfi import *

DTCombinatorialPatternReco4DAlgo_LinearDriftFromDB_NoWire = cms.PSet(
    Reco4DAlgoName = cms.string('DTCombinatorialPatternReco4D'),
    Reco4DAlgoConfig = cms.PSet(
        # this are the RecSegment2D algo parameters!
        DTCombinatorialPatternReco2DAlgo_LinearDriftFromDB,
        # Parameters for the updator
        # this is the RecHit1D algo!!
        DTLinearDriftFromDBAlgo,
        debug = cms.untracked.bool(False),
        # Parameters for the cleaner
        nUnSharedHitsMin = cms.int32(2),
        # the input type. 
        # If true the instructions in setDTRecSegment2DContainer will be schipped and the 
        # theta segment will be recomputed from the 1D rechits
        # If false the theta segment will be taken from the Event. Caveat: in this case the
        # event must contain the 2D segments!
        AllDTRecHits = cms.bool(True),
        # Parameters for  T0 fit segment in the Updator 
        performT0SegCorrection = cms.bool(False),
        hit_afterT0_resolution = cms.double(0.03),
        performT0_vdriftSegCorrection = cms.bool(False),
        doWirePropCorrection = cms.bool(False)
    )
)

# The actual 4D uncorrected segments build
dt4DSegmentsNoWire = cms.EDProducer("DTRecSegment4DProducer",
    # The reconstruction algo and its parameter set
    DTCombinatorialPatternReco4DAlgo_LinearDriftFromDB_NoWire,
    # debuggin opt
    debug = cms.untracked.bool(False),
    # name of the rechit 1D collection in the event
    recHits1DLabel = cms.InputTag("dt1DRecHits"),
    # name of the rechit 2D collection in the event
    recHits2DLabel = cms.InputTag("dt2DSegments")
)

seqALCARECODtCalib = cms.Sequence(ALCARECODtCalibHLTFilter*DTCalibMuonSelection*dt4DSegmentsNoWire) 
