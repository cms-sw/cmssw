import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.JetSkims.onejet_Sequences_cff import *
onejetHLTPath = cms.Path(onejetpe0HLTFilter)
onejetpe1HLTPath = cms.Path(onejetpe1HLTFilter)
onejetpe3HLTPath = cms.Path(onejetpe3HLTFilter)
onejetpe5HLTPath = cms.Path(onejetpe5HLTFilter)
onejetpe7HLTPath = cms.Path(onejetpe7HLTFilter)

