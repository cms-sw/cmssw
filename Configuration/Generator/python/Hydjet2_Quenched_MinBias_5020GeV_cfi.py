import FWCore.ParameterSet.Config as cms

from GeneratorInterface.Hydjet2Interface.hydjet2DefaultParameters_cff import *

generator = cms.EDFilter("Hydjet2GeneratorFilter",
                         collisionParameters5020GeV,
                         qgpParametersLHC,
                         hydjet2Parameters,
                         fNhsel 	= cms.int32(2), # Flag to include jet (J)/jet quenching (JQ) and hydro (H) state production, fNhsel (0 H on & J off, 1 H/J on & JQ off, 2 H/J/HQ on, 3 J on & H/JQ off, 4 H off & J/JQ on)
                         PythiaParameters = cms.PSet(PythiaDefaultBlock,
                                                     parameterSets = cms.vstring(
                                                         'ProQ2Otune',
                                                         'hydjet2PythiaDefault',
                                                     )
                                                 ),

                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(0),

                         fIfb 	= cms.int32(1), 	# Flag of type of centrality generation, fBfix (=0 is fixed by fBfix, >0 distributed [fBfmin, fBmax])
                         fBmin 	= cms.double(0.),	# Minimal impact parameter, fBmin (fm)
                         fBmax	= cms.double(21.), 	# Maximal impact parameter, fBmax (fm)
                         fBfix 	= cms.double(0.), 	# Fixed impact parameter, fBfix (fm)
                     )
