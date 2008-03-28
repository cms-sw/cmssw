import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.JetSkims.onejet_HLTPaths_cfi import *
onejetHLTFilter = cms.Sequence(onejetpe0HLTFilter+onejetpe1HLTFilter+onejetpe3HLTFilter+onejetpe5HLTFilter+onejetpe7HLTFilter)

