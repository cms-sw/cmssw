import FWCore.ParameterSet.Config as cms

##################################################################
# AlCaReco for track based calibration using single muon events
##################################################################
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOSiPixelCalSingleMuonHLTFilter = hltHighLevel.clone()
ALCARECOSiPixelCalSingleMuonHLTFilter.andOr = True ## choose logical OR between Triggerbits
ALCARECOSiPixelCalSingleMuonHLTFilter.throw = False ## dont throw on unknown path names
ALCARECOSiPixelCalSingleMuonHLTFilter.eventSetupPathsKey = 'SiPixelCalSingleMuon'

##################################################################
# Basic track selection
##################################################################
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOSiPixelCalSingleMuon = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOSiPixelCalSingleMuon.filter         = True ##do not store empty events
ALCARECOSiPixelCalSingleMuon.applyBasicCuts = True
ALCARECOSiPixelCalSingleMuon.ptMin = 3.0 #GeV
ALCARECOSiPixelCalSingleMuon.etaMin = -3.5
ALCARECOSiPixelCalSingleMuon.etaMax = 3.5

##################################################################
# Loose Sequence
##################################################################
seqALCARECOSiPixelCalSingleMuon = cms.Sequence(ALCARECOSiPixelCalSingleMuonHLTFilter+ALCARECOSiPixelCalSingleMuon)

## customizations for the pp_on_AA eras
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(ALCARECOSiPixelCalSingleMuonHLTFilter,
                  eventSetupPathsKey='SiPixelCalSingleMuonHI'
)
