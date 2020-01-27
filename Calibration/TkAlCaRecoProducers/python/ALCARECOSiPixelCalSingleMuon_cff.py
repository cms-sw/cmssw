import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# AlCaReco for track based calibration using min. bias events
ALCARECOSiPixelCalSingleMuonHLTFilter = copy.deepcopy(hltHighLevel)
#seqALCARECOSiPixelCalSingleMuon = cms.Sequence(ALCARECOSiPixelCalSingleMuonHLTFilter)
ALCARECOSiPixelCalSingleMuonHLTFilter.andOr = True ## choose logical OR between Triggerbits
ALCARECOSiPixelCalSingleMuonHLTFilter.throw = False ## dont throw on unknown path names
ALCARECOSiPixelCalSingleMuonHLTFilter.eventSetupPathsKey = 'SiPixelCalSingleMuon'

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOSiPixelCalSingleMuon = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()

ALCARECOSiPixelCalSingleMuon.filter         = True ##do not store empty events
ALCARECOSiPixelCalSingleMuon.applyBasicCuts = True
ALCARECOSiPixelCalSingleMuon.ptMin = 3.0 #GeV
ALCARECOSiPixelCalSingleMuon.etaMin = -3.5
ALCARECOSiPixelCalSingleMuon.etaMax = 3.5

# Sequence #
seqALCARECOSiPixelCalSingleMuon = cms.Sequence(ALCARECOSiPixelCalSingleMuonHLTFilter+ALCARECOSiPixelCalSingleMuon)
