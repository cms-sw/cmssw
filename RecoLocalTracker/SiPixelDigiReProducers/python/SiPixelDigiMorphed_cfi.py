import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelDigiReProducers.siPixelDigiMorphing_cfi import siPixelDigiMorphing as _siPixelDigiMorphing

siPixelDigisMorphed = _siPixelDigiMorphing.clone()
