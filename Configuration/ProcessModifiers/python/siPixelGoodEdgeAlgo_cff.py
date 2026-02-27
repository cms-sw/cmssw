import FWCore.ParameterSet.Config as cms

# This modifier enables the good edge algorithm in pixel hit reconstruction that handles broken/truncated pixel cluster caused by radiation damage
siPixelGoodEdgeAlgo = cms.Modifier()
