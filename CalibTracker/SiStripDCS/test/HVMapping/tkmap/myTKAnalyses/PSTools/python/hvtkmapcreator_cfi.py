import FWCore.ParameterSet.Config as cms

hvtkmapcreator = cms.EDAnalyzer('HVTkMapCreator',
                                   hvChannelFile = cms.string("Allresults.dat"),
                                   hvReassChannelFile = cms.string("Allmap.txt"),
)	
