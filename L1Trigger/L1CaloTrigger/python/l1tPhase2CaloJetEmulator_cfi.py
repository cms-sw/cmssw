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
			# Calibrations derived 21 March 2024 on 14_0_0_pre3 131X QCD sample
			jetPtBins = cms.vdouble([ 0.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,65.0,70.0,75.0,80.0,85.0,90.0,95.0,100.0,110.0,120.0,130.0,140.0,150.0,160.0,170.0,180.0,190.0,200.0,225.0,250.0,275.0,300.0,325.0,400.0,500.0]),
			absEtaBinsBarrel = cms.vdouble([ 0.00,0.30,0.60,1.00,1.50]),
			jetCalibrationsBarrel = cms.vdouble([
				2.220, 2.008, 1.942, 1.884, 1.833, 1.788, 1.749, 1.715, 1.685, 1.658, 1.635, 1.614, 1.595, 1.579, 1.564, 1.551, 1.540, 1.524, 1.507, 1.493, 1.481, 1.471, 1.461, 1.453, 1.446, 1.439, 1.432, 1.421, 1.407, 1.393, 1.380, 1.367, 1.340, 1.294,
				1.883, 1.759, 1.718, 1.681, 1.647, 1.616, 1.588, 1.563, 1.540, 1.519, 1.501, 1.483, 1.468, 1.454, 1.441, 1.429, 1.418, 1.403, 1.387, 1.372, 1.360, 1.350, 1.341, 1.333, 1.326, 1.319, 1.313, 1.304, 1.293, 1.283, 1.274, 1.265, 1.247, 1.218,
				2.025, 1.815, 1.754, 1.702, 1.657, 1.620, 1.588, 1.561, 1.537, 1.517, 1.500, 1.485, 1.472, 1.460, 1.450, 1.441, 1.433, 1.423, 1.410, 1.400, 1.391, 1.382, 1.374, 1.367, 1.360, 1.353, 1.346, 1.334, 1.318, 1.301, 1.285, 1.269, 1.236, 1.180,
				2.893, 2.497, 2.377, 2.274, 2.186, 2.110, 2.045, 1.989, 1.941, 1.899, 1.863, 1.831, 1.803, 1.779, 1.757, 1.738, 1.721, 1.699, 1.674, 1.653, 1.635, 1.619, 1.605, 1.591, 1.578, 1.566, 1.555, 1.535, 1.508, 1.481, 1.454, 1.427, 1.374, 1.282,
				]),
			absEtaBinsHGCal = cms.vdouble([ 1.50,1.90,2.40,3.00]),
			jetCalibrationsHGCal = cms.vdouble([
				2.426, 2.223, 2.157, 2.097, 2.044, 1.996, 1.953, 1.915, 1.881, 1.850, 1.822, 1.797, 1.775, 1.755, 1.737, 1.721, 1.707, 1.687, 1.666, 1.648, 1.634, 1.622, 1.612, 1.604, 1.597, 1.591, 1.585, 1.578, 1.569, 1.561, 1.555, 1.549, 1.538, 1.519,
				1.391, 1.374, 1.368, 1.362, 1.356, 1.351, 1.345, 1.340, 1.335, 1.331, 1.326, 1.322, 1.318, 1.314, 1.310, 1.307, 1.303, 1.298, 1.292, 1.287, 1.283, 1.278, 1.275, 1.272, 1.269, 1.267, 1.265, 1.263, 1.263, 1.264, 1.268, 1.273, 1.288, 1.325,
				1.713, 1.654, 1.632, 1.611, 1.590, 1.570, 1.551, 1.533, 1.515, 1.498, 1.481, 1.465, 1.450, 1.435, 1.421, 1.407, 1.394, 1.375, 1.352, 1.330, 1.311, 1.293, 1.277, 1.262, 1.249, 1.237, 1.227, 1.212, 1.196, 1.186, 1.182, 1.183, 1.197, 1.252,
				]),
			absEtaBinsHF = cms.vdouble([ 3.00,3.60,6.00]),
			jetCalibrationsHF = cms.vdouble([
				4.682, 3.448, 3.109, 2.833, 2.609, 2.425, 2.276, 2.153, 2.051, 1.968, 1.898, 1.840, 1.791, 1.750, 1.714, 1.683, 1.657, 1.622, 1.584, 1.552, 1.524, 1.498, 1.474, 1.451, 1.429, 1.407, 1.386, 1.349, 1.296, 1.244, 1.191, 1.139, 1.035, 0.852,
				2.085, 1.767, 1.672, 1.591, 1.522, 1.464, 1.414, 1.371, 1.334, 1.303, 1.276, 1.253, 1.233, 1.216, 1.201, 1.187, 1.176, 1.161, 1.145, 1.132, 1.122, 1.113, 1.105, 1.098, 1.091, 1.085, 1.079, 1.070, 1.056, 1.044, 1.031, 1.018, 0.993, 0.949,
				]),
			# Calibrations derived 21 March 2024 on 14_0_0_pre3 131X VBFHiggsTauTau sample
			tauPtBins = cms.vdouble([ 0.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,70.0,80.0,100.0,150.0,200.0]),
			tauAbsEtaBinsBarrel = cms.vdouble([ 0.00,0.30,0.60,1.00,1.50]),
			tauCalibrationsBarrel = cms.vdouble([
				1.974, 1.659, 1.516, 1.411, 1.333, 1.276, 1.233, 1.202, 1.179, 1.162, 1.149, 1.136, 1.126, 1.119, 1.115, 1.114,
				1.912, 1.641, 1.512, 1.414, 1.338, 1.281, 1.237, 1.203, 1.177, 1.158, 1.143, 1.127, 1.113, 1.102, 1.095, 1.094,
				1.988, 1.683, 1.541, 1.433, 1.353, 1.292, 1.246, 1.211, 1.185, 1.166, 1.151, 1.135, 1.122, 1.113, 1.107, 1.106,
				2.753, 2.137, 1.865, 1.667, 1.524, 1.421, 1.346, 1.292, 1.253, 1.224, 1.204, 1.183, 1.167, 1.156, 1.150, 1.149,
				]),
			tauAbsEtaBinsHGCal = cms.vdouble([ 1.50,1.90,2.40,3.00]),
			tauCalibrationsHGCal = cms.vdouble([
				4.029, 2.692, 2.185, 1.859, 1.650, 1.516, 1.430, 1.374, 1.339, 1.316, 1.301, 1.289, 1.281, 1.277, 1.275, 1.275,
				3.274, 2.537, 2.207, 1.966, 1.790, 1.661, 1.568, 1.499, 1.449, 1.412, 1.386, 1.359, 1.338, 1.323, 1.315, 1.314,
				2.467, 2.198, 2.044, 1.907, 1.787, 1.680, 1.586, 1.503, 1.429, 1.364, 1.307, 1.233, 1.153, 1.065, 0.951, 0.891,
				]),
)