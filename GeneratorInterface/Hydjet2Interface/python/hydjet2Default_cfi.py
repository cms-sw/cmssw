import FWCore.ParameterSet.Config as cms
source = cms.Source("EmptySource")
from GeneratorInterface.Hydjet2Interface.hydjet2DefaultParameters_cff import *

generator = cms.EDFilter("Hydjet2GeneratorFilter",
	collisionParameters5100GeV,
	qgpParameters,
	hydjet2Parameters,
	fNhsel 	= cms.int32(2), 	# Flag to include jet (J)/jet quenching (JQ) and hydro (H) state production, fNhsel (0 H on & J off, 1 H/J on & JQ off, 2 H/J/HQ on, 3 J on & H/JQ off, 4 H off & J/JQ on)
	PythiaParameters = cms.PSet(PythiaDefaultBlock,
		parameterSets = cms.vstring(
			#'pythiaUESettings',
			'hydjet2PythiaDefault',
			'ProQ2Otune',
			#'pythiaJets',
			#'pythiaPromptPhotons'

			#'myParameters',
			#'pythiaZjets',
			#'pythiaBottomoniumNRQCD',
			#'pythiaCharmoniumNRQCD',
			#'pythiaQuarkoniaSettings',
			#'pythiaWeakBosons'
		)
	),
	
	maxEventsToPrint = cms.untracked.int32(0),
	pythiaPylistVerbosity = cms.untracked.int32(0),

	fIfb 	= cms.int32(1), 	# Flag of type of centrality generation, fBfix (=0 is fixed by fBfix, >0 distributed [fBfmin, fBmax])
	fBmin 	= cms.double(0.),	# Minimum impact parameter, fBmin
	fBmax	= cms.double(3.47500770746), 	# Maximum impact parameter, fBmax
	fBfix 	= cms.double(0.), 	# Fixed impact parameter, fBfix

)
'''
RA(Pb) ~= 6.813740957 fm

% cent	b/RA 
0            0          
5           0.51
6           0.57
10          0.74 
12          0.81
15	0.91
20	1.05
25	1.18 
30          1.29
35          1.39
40          1.49
45	1.58
50	1.67
55	1.75  
60          1.83
65	1.90
70	1.97 
75	2.06
'''
