import FWCore.ParameterSet.Config as cms

from RecoLocalMuon.GEMCSCSegment.gemcscSegments_cfi import gemcscSegments

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify(gemcscSegments, enableME21GE21=True)
