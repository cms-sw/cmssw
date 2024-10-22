import FWCore.ParameterSet.Config as cms

l1tPhase2CaloJetEmulator = cms.EDProducer("Phase2L1CaloJetEmulator",
			gctFullTowers = cms.InputTag("l1tPhase2L1CaloEGammaEmulator","GCTFullTowers"),
			hgcalTowers = cms.InputTag("l1tHGCalTowerProducer","HGCalTowerProcessor"),
			hcalDigis = cms.InputTag("simHcalTriggerPrimitiveDigis"),
			nHits_to_nvtx_params = cms.VPSet(
				cms.PSet(
					fit = cms.string( "hgcalEM" ),
					nHits_params = cms.vdouble( 157.522, 0.090 )
				),
				cms.PSet(
					fit = cms.string( "hgcalHad" ),
					nHits_params = cms.vdouble( 159.295, 0.178 )
				),
				cms.PSet(
					fit = cms.string( "hf" ),
					nHits_params = cms.vdouble( 165.706, 0.153 )
				),
			),
			nvtx_to_PU_sub_params = cms.VPSet(
				cms.PSet(
					calo = cms.string( "hgcalEM" ),
					iEta = cms.string( "er1p4to1p8" ),
					nvtx_params = cms.vdouble( -0.011772, 0.004142 )
				),
				cms.PSet(
					calo = cms.string( "hgcalEM" ),
					iEta = cms.string( "er1p8to2p1" ),
					nvtx_params = cms.vdouble( -0.015488, 0.005410 )
				),
				cms.PSet(
					calo = cms.string( "hgcalEM" ),
					iEta = cms.string( "er2p1to2p4" ),
					nvtx_params = cms.vdouble( -0.021150, 0.006078 )
				),
				cms.PSet(
					calo = cms.string( "hgcalEM" ),
					iEta = cms.string( "er2p4to2p7" ),
					nvtx_params = cms.vdouble( -0.015705, 0.005339 )
				),
				cms.PSet(
					calo = cms.string( "hgcalEM" ),
					iEta = cms.string( "er2p7to3p1" ),
					nvtx_params = cms.vdouble( -0.018492, 0.005620 )
				),
				cms.PSet(
					calo = cms.string( "hgcalHad" ),
					iEta = cms.string( "er1p4to1p8" ),
					nvtx_params = cms.vdouble( 0.005675, 0.000615 )
				),
				cms.PSet(
					calo = cms.string( "hgcalHad" ),
					iEta = cms.string( "er1p8to2p1" ),
					nvtx_params = cms.vdouble( 0.004560, 0.001099 )
				),
				cms.PSet(
					calo = cms.string( "hgcalHad" ),
					iEta = cms.string( "er2p1to2p4" ),
					nvtx_params = cms.vdouble( 0.000036, 0.001608 )
				),
				cms.PSet(
					calo = cms.string( "hgcalHad" ),
					iEta = cms.string( "er2p4to2p7" ),
					nvtx_params = cms.vdouble( 0.000869, 0.001754 )
				),
				cms.PSet(
					calo = cms.string( "hgcalHad" ),
					iEta = cms.string( "er2p7to3p1" ),
					nvtx_params = cms.vdouble( -0.006574, 0.003134 )
				),
				cms.PSet(
					calo = cms.string( "hf" ),
					iEta = cms.string( "er29to33" ),
					nvtx_params = cms.vdouble( -0.203291, 0.044096 )
				),
				cms.PSet(
					calo = cms.string( "hf" ),
					iEta = cms.string( "er34to37" ),
					nvtx_params = cms.vdouble( -0.210922, 0.045628 )
				),
				cms.PSet(
					calo = cms.string( "hf" ),
					iEta = cms.string( "er38to41" ),
					nvtx_params = cms.vdouble( -0.229562, 0.050560 )
				),
			),
			# Calibrations derived 26 June 2024 on 14_0_0_pre3 131X QCD sample
			jetPtBins = cms.vdouble([ 0.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,65.0,70.0,75.0,80.0,85.0,90.0,95.0,100.0,110.0,120.0,130.0,140.0,150.0,160.0,170.0,180.0,190.0,200.0,225.0,250.0,275.0,300.0,325.0,400.0,500.0]),
			absEtaBinsBarrel = cms.vdouble([ 0.00,0.30,0.60,1.00,1.50]),
			jetCalibrationsBarrel = cms.vdouble([
				2.256, 2.007, 1.932, 1.869, 1.816, 1.770, 1.731, 1.697, 1.668, 1.643, 1.622, 1.603, 1.587, 1.573, 1.561, 1.550, 1.540, 1.527, 1.513, 1.500, 1.490, 1.481, 1.472, 1.464, 1.457, 1.449, 1.442, 1.431, 1.414, 1.397, 1.381, 1.365, 1.332, 1.276,
				1.847, 1.733, 1.695, 1.661, 1.630, 1.601, 1.576, 1.552, 1.531, 1.512, 1.495, 1.479, 1.464, 1.451, 1.439, 1.428, 1.418, 1.404, 1.388, 1.375, 1.363, 1.353, 1.344, 1.336, 1.329, 1.323, 1.317, 1.308, 1.296, 1.285, 1.275, 1.265, 1.247, 1.215,
				1.934, 1.775, 1.726, 1.683, 1.646, 1.613, 1.585, 1.560, 1.538, 1.519, 1.502, 1.488, 1.474, 1.462, 1.452, 1.442, 1.434, 1.422, 1.409, 1.398, 1.388, 1.379, 1.371, 1.363, 1.356, 1.349, 1.342, 1.331, 1.316, 1.300, 1.285, 1.270, 1.240, 1.188,
				2.731, 2.353, 2.244, 2.153, 2.076, 2.012, 1.958, 1.912, 1.873, 1.840, 1.811, 1.787, 1.766, 1.747, 1.731, 1.716, 1.703, 1.686, 1.666, 1.649, 1.634, 1.620, 1.607, 1.594, 1.582, 1.570, 1.558, 1.537, 1.509, 1.480, 1.451, 1.423, 1.365, 1.265,
				]),
			absEtaBinsHGCal = cms.vdouble([ 1.50,1.90,2.40,3.00]),
			jetCalibrationsHGCal = cms.vdouble([
				2.413, 1.721, 1.617, 1.556, 1.519, 1.497, 1.484, 1.476, 1.471, 1.467, 1.465, 1.463, 1.462, 1.460, 1.459, 1.458, 1.458, 1.456, 1.454, 1.453, 1.451, 1.449, 1.448, 1.446, 1.444, 1.442, 1.441, 1.438, 1.433, 1.429, 1.425, 1.420, 1.412, 1.397,
				1.555, 1.307, 1.240, 1.187, 1.145, 1.112, 1.086, 1.066, 1.050, 1.038, 1.030, 1.023, 1.018, 1.015, 1.014, 1.013, 1.013, 1.014, 1.018, 1.023, 1.028, 1.034, 1.041, 1.047, 1.054, 1.061, 1.068, 1.080, 1.098, 1.115, 1.133, 1.150, 1.186, 1.247,
				3.097, 1.989, 1.721, 1.518, 1.365, 1.250, 1.163, 1.099, 1.051, 1.016, 0.991, 0.973, 0.960, 0.952, 0.947, 0.944, 0.944, 0.945, 0.951, 0.958, 0.967, 0.977, 0.987, 0.997, 1.008, 1.018, 1.029, 1.048, 1.074, 1.101, 1.128, 1.154, 1.208, 1.301,
				]),
			absEtaBinsHF = cms.vdouble([ 3.00,3.60,6.00]),
			jetCalibrationsHF = cms.vdouble([
				4.353, 3.282, 2.979, 2.730, 2.525, 2.355, 2.215, 2.098, 2.001, 1.920, 1.853, 1.796, 1.748, 1.707, 1.672, 1.642, 1.616, 1.582, 1.546, 1.517, 1.492, 1.469, 1.449, 1.430, 1.412, 1.395, 1.378, 1.348, 1.307, 1.267, 1.226, 1.186, 1.104, 0.963,
				2.155, 1.822, 1.722, 1.637, 1.564, 1.502, 1.449, 1.403, 1.364, 1.331, 1.302, 1.277, 1.256, 1.237, 1.221, 1.206, 1.194, 1.178, 1.160, 1.146, 1.134, 1.124, 1.116, 1.108, 1.101, 1.094, 1.087, 1.077, 1.062, 1.048, 1.034, 1.020, 0.992, 0.943,
				]),
			# Calibrations derived 6 June 2024 on 14_0_0_pre3 131X VBFHiggsTauTau sample
			tauPtBins = cms.vdouble([ 0.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,70.0,80.0,100.0,150.0,200.0]),
			tauAbsEtaBinsBarrel = cms.vdouble([ 0.00,0.30,0.60,1.00,1.50]),
			tauCalibrationsBarrel = cms.vdouble([
				1.936, 1.643, 1.508, 1.408, 1.332, 1.276, 1.235, 1.204, 1.180, 1.163, 1.150, 1.137, 1.126, 1.118, 1.113, 1.112,
				1.905, 1.637, 1.510, 1.413, 1.338, 1.281, 1.237, 1.204, 1.178, 1.159, 1.144, 1.128, 1.114, 1.103, 1.096, 1.095,
				1.941, 1.663, 1.531, 1.430, 1.353, 1.294, 1.249, 1.215, 1.188, 1.168, 1.153, 1.136, 1.122, 1.111, 1.104, 1.103,
				2.692, 2.109, 1.849, 1.659, 1.520, 1.419, 1.346, 1.292, 1.253, 1.224, 1.203, 1.182, 1.166, 1.154, 1.148, 1.147,
				]),
			tauAbsEtaBinsHGCal = cms.vdouble([ 1.50,1.90,2.40,3.00]),
			tauCalibrationsHGCal = cms.vdouble([
				3.047, 2.104, 1.762, 1.550, 1.418, 1.336, 1.285, 1.254, 1.234, 1.222, 1.215, 1.208, 1.205, 1.203, 1.202, 1.202,
				3.401, 2.104, 1.718, 1.513, 1.403, 1.345, 1.313, 1.297, 1.288, 1.283, 1.281, 1.279, 1.278, 1.278, 1.278, 1.278,
				4.381, 2.648, 2.012, 1.613, 1.364, 1.208, 1.110, 1.049, 1.010, 0.986, 0.971, 0.959, 0.951, 0.947, 0.946, 0.946,
				]),
)