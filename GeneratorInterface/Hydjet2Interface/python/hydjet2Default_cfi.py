import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

from GeneratorInterface.Hydjet2Interface.hydjet2DefaultParameters_cff import *

generator = cms.EDFilter("Hydjet2GeneratorFilter",
	collisionParameters5020GeV,
	qgpParametersLHC,
	hydjet2Parameters,
	fNhsel 	= cms.int32(2), # Flag to include jet (J)/jet quenching (JQ) and hydro (H) state production, fNhsel (0 H on & J off, 1 H/J on & JQ off, 2 H/J/HQ on, 3 J on & H/JQ off, 4 H off & J/JQ on)
	PythiaParameters = cms.PSet(PythiaDefaultBlock,
		parameterSets = cms.vstring(
			#'pythiaUESettings',
			'ProQ2Otune',
			'hydjet2PythiaDefault',
			#'pythiaJets',
			#'pythiaPromptPhotons',
			#'myParameters',
			#'pythiaZjets',
			#'pythiaBottomoniumNRQCD',
			#'pythiaCharmoniumNRQCD',
			#'pythiaQuarkoniaSettings',
			#'pythiaWeakBosons',
			#'TDB'
		)
	),
	
	maxEventsToPrint = cms.untracked.int32(0),
	pythiaPylistVerbosity = cms.untracked.int32(0),

	fIfb 	= cms.int32(1), 	# Flag of type of centrality generation, fBfix (=0 is fixed by fBfix, >0 distributed [fBfmin, fBmax])
	fBmin 	= cms.double(0.),	# Minimal impact parameter, fBmin (fm)
	fBmax	= cms.double(21.), 	# Maximal impact parameter, fBmax (fm)
	fBfix 	= cms.double(0.), 	# Fixed impact parameter, fBfix (fm)

)
'''
RA(Pb) ~= 6.813740957 fm
RA(Au) ~= 6.691445048 fm

		RHIC		   LHC
% centrality    b/RA(Au)     	   b/RA(Pb)
---------------------------------------------
0               0                  0
---------------------------------------------
5               0.5                0.51
---------------------------------------------
6               0.55               0.57
---------------------------------------------
10              0.72               0.74
---------------------------------------------
12              0.79               0.81
---------------------------------------------
15        	0.89               0.91
---------------------------------------------
20              1.02               1.05
---------------------------------------------
25        	1.15               1.18
---------------------------------------------
30              1.26               1.29
---------------------------------------------
35              1.36               1.39
---------------------------------------------
40              1.46               1.49
---------------------------------------------
45        	1.55               1.58
---------------------------------------------
50        	1.63               1.67
---------------------------------------------
55        	1.71               1.75
---------------------------------------------
60              1.79               1.83
---------------------------------------------
65        	1.86               1.90
---------------------------------------------
70        	1.93               1.97
---------------------------------------------
75        	2.01               2.06

'''
