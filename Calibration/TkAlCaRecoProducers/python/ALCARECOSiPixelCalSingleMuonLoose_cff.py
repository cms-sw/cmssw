import FWCore.ParameterSet.Config as cms

##################################################################
# AlCaReco for track based calibration using single muon events
##################################################################
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOSiPixelCalSingleMuonLooseHLTFilter = hltHighLevel.clone()
ALCARECOSiPixelCalSingleMuonLooseHLTFilter.andOr = True ## choose logical OR between Triggerbits
ALCARECOSiPixelCalSingleMuonLooseHLTFilter.throw = False ## dont throw on unknown path names
ALCARECOSiPixelCalSingleMuonLooseHLTFilter.eventSetupPathsKey = 'SiPixelCalSingleMuon'

##################################################################
# Basic track selection
##################################################################
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOSiPixelCalSingleMuonLoose = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOSiPixelCalSingleMuonLoose.filter         = True ##do not store empty events
ALCARECOSiPixelCalSingleMuonLoose.applyBasicCuts = True
ALCARECOSiPixelCalSingleMuonLoose.ptMin = 3.0 #GeV
ALCARECOSiPixelCalSingleMuonLoose.etaMin = -3.5
ALCARECOSiPixelCalSingleMuonLoose.etaMax = 3.5

##################################################################
# Prescale events
##################################################################
import CalibTracker.SiStripCommon.prescaleEvent_cfi
scalerForSiPixelCalSingleMuonLoose = CalibTracker.SiStripCommon.prescaleEvent_cfi.prescaleEvent.clone(prescale = 10)

##################################################################
# Loose Sequence
##################################################################
seqALCARECOSiPixelCalSingleMuonLoose = cms.Sequence(ALCARECOSiPixelCalSingleMuonLooseHLTFilter+
                                                    scalerForSiPixelCalSingleMuonLoose+
                                                    ALCARECOSiPixelCalSingleMuonLoose)

## customizations for the pp_on_AA eras
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(ALCARECOSiPixelCalSingleMuonLooseHLTFilter,
                  eventSetupPathsKey='SiPixelCalSingleMuonHI'
)
