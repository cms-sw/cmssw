import FWCore.ParameterSet.Config as cms

from HLTrigger.special.HLTBackwardBSC_cff import *
from HLTrigger.special.HLTForwardBSC_cff import *
from HLTrigger.special.HLTCSCBeamHalo_cff import *
from HLTrigger.special.HLTCSCBeamHaloOverlapRing1_cff import *
from HLTrigger.special.HLTCSCBeamHaloOverlapRing2_cff import *
from HLTrigger.special.HLTCSCBeamHaloRing2or3_cff import *
from HLTrigger.special.HLTTrackerCosmics_cff import *
from HLTrigger.special.AlcastreamEcalPi0_cff import *
from HLTrigger.special.AlcastreamEcalPhiSym_cff import *
from HLTrigger.special.AlcastreamHcalPhiSym_cff import *
from HLTrigger.special.hltHcalIsolatedTrack_cff import *
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
from HLTrigger.special.MinBias_With_Pixel_Tracks_cff import *
from HLTrigger.special.MBForAlignment_cff import *
from HLTrigger.special.SimpleBias_cff import *
from HLTrigger.special.TriggerTypeSequence_cff import *
CandHLTBackwardBSC = cms.Path(hltBackwardBSC+cms.SequencePlaceholder("hltEnd"))
CandHLTForwardBSC = cms.Path(hltForwardBSC+cms.SequencePlaceholder("hltEnd"))
CandHLTCSCBeamHalo = cms.Path(hltCSCBeamHalo+cms.SequencePlaceholder("hltEnd"))
CandHLTCSCBeamHaloOverlapRing1 = cms.Path(hltCSCBeamHaloOverlapRing1+cms.SequencePlaceholder("hltEnd"))
CandHLTCSCBeamHaloOverlapRing2 = cms.Path(hltCSCBeamHaloOverlapRing2+cms.SequencePlaceholder("hltEnd"))
CandHLTCSCBeamHaloRing2or3 = cms.Path(hltCSCBeamHaloRing2or3+cms.SequencePlaceholder("hltEnd"))
CandHLTTrackerCosmics = cms.Path(hltTrackerCosmics+cms.SequencePlaceholder("hltEnd"))
CandHLTEcalPi0 = cms.Path(hltEcalPi0+cms.SequencePlaceholder("hltEnd"))
CandHLTEcalPhiSym = cms.Path(alcaEcalPhiSymSequence+cms.SequencePlaceholder("hltEnd"))
CandHLTHcalPhiSym = cms.Path(alcaHcalPhiSymSequence+cms.SequencePlaceholder("hltEnd"))
HLTHcalIsolatedTrack = cms.Path(hltHcalIsolatedTrack+cms.SequencePlaceholder("hltEnd"))
CandHLTHcalIsolatedTrackNoEcalIsol = cms.Path(hltHcalIsolatedTrackNoEcalIsol+cms.SequencePlaceholder("hltEnd"))
HLTMinBiasPixel = cms.Path(hltMinBiasPixel+cms.SequencePlaceholder("hltEnd"))
CandHLTMinBiasForAlignment = cms.Path(hltMBForAlignment+cms.SequencePlaceholder("hltEnd"))
HLTMinBias = cms.Path(hltMinBias+cms.SequencePlaceholder("hltEnd"))
HLTZeroBias = cms.Path(hltZeroBias+cms.SequencePlaceholder("hltEnd"))
HLTriggerType = cms.Path(sequenceTriggerType+cms.SequencePlaceholder("hltEnd"))

