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
			# Calibrations derived 4 April 2024 on 14_0_0_pre3 131X QCD sample
			jetPtBins = cms.vdouble([ 0.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,65.0,70.0,75.0,80.0,85.0,90.0,95.0,100.0,110.0,120.0,130.0,140.0,150.0,160.0,170.0,180.0,190.0,200.0,225.0,250.0,275.0,300.0,325.0,400.0,500.0]),
			absEtaBinsBarrel = cms.vdouble([ 0.00,0.30,0.60,1.00,1.50]),
			jetCalibrationsBarrel = cms.vdouble([
				2.221, 2.009, 1.943, 1.884, 1.833, 1.789, 1.749, 1.715, 1.685, 1.658, 1.635, 1.614, 1.595, 1.579, 1.564, 1.551, 1.540, 1.524, 1.507, 1.493, 1.481, 1.470, 1.461, 1.453, 1.446, 1.439, 1.432, 1.421, 1.407, 1.393, 1.380, 1.366, 1.340, 1.294,
				1.886, 1.760, 1.718, 1.680, 1.646, 1.615, 1.587, 1.562, 1.539, 1.519, 1.500, 1.483, 1.467, 1.453, 1.440, 1.429, 1.418, 1.404, 1.387, 1.373, 1.361, 1.351, 1.342, 1.334, 1.327, 1.321, 1.315, 1.306, 1.294, 1.284, 1.274, 1.265, 1.247, 1.217,
				2.032, 1.818, 1.755, 1.702, 1.657, 1.619, 1.587, 1.560, 1.536, 1.516, 1.499, 1.484, 1.471, 1.460, 1.450, 1.441, 1.433, 1.423, 1.411, 1.400, 1.391, 1.383, 1.375, 1.367, 1.360, 1.353, 1.346, 1.335, 1.318, 1.301, 1.285, 1.268, 1.236, 1.178,
				2.762, 2.411, 2.304, 2.213, 2.135, 2.067, 2.009, 1.959, 1.916, 1.878, 1.846, 1.817, 1.792, 1.770, 1.750, 1.732, 1.717, 1.696, 1.673, 1.653, 1.635, 1.620, 1.606, 1.592, 1.580, 1.568, 1.556, 1.536, 1.508, 1.481, 1.454, 1.427, 1.373, 1.279,
				]),
			absEtaBinsHGCal = cms.vdouble([ 1.50,1.90,2.40,3.00]),
			jetCalibrationsHGCal = cms.vdouble([
				2.620, 2.126, 2.008, 1.919, 1.852, 1.802, 1.764, 1.735, 1.712, 1.695, 1.682, 1.671, 1.663, 1.657, 1.651, 1.647, 1.643, 1.638, 1.633, 1.628, 1.624, 1.620, 1.617, 1.613, 1.609, 1.606, 1.602, 1.596, 1.587, 1.578, 1.569, 1.560, 1.542, 1.511,
				6.766, 1.454, 1.273, 1.221, 1.207, 1.204, 1.204, 1.205, 1.207, 1.209, 1.210, 1.212, 1.214, 1.215, 1.217, 1.219, 1.221, 1.223, 1.226, 1.230, 1.233, 1.237, 1.240, 1.243, 1.247, 1.250, 1.253, 1.259, 1.268, 1.276, 1.285, 1.293, 1.310, 1.340,
				3.635, 2.281, 1.968, 1.737, 1.566, 1.441, 1.349, 1.281, 1.232, 1.197, 1.172, 1.154, 1.142, 1.133, 1.128, 1.125, 1.124, 1.125, 1.128, 1.133, 1.139, 1.145, 1.152, 1.159, 1.166, 1.174, 1.181, 1.193, 1.211, 1.229, 1.247, 1.265, 1.301, 1.364,
				]),
			absEtaBinsHF = cms.vdouble([ 3.00,3.60,6.00]),
			jetCalibrationsHF = cms.vdouble([
				4.632, 3.392, 3.056, 2.784, 2.563, 2.385, 2.239, 2.121, 2.024, 1.944, 1.877, 1.822, 1.776, 1.737, 1.703, 1.674, 1.649, 1.617, 1.580, 1.549, 1.522, 1.497, 1.474, 1.451, 1.429, 1.408, 1.386, 1.349, 1.297, 1.244, 1.192, 1.140, 1.036, 0.853,
				2.092, 1.771, 1.675, 1.593, 1.524, 1.465, 1.414, 1.371, 1.335, 1.303, 1.276, 1.253, 1.233, 1.216, 1.201, 1.187, 1.176, 1.161, 1.145, 1.132, 1.122, 1.113, 1.105, 1.098, 1.091, 1.085, 1.080, 1.070, 1.057, 1.044, 1.031, 1.019, 0.994, 0.950,
				]),
			# Calibrations derived 4 April 2024 on 14_0_0_pre3 131X VBFHiggsTauTau sample
			tauPtBins = cms.vdouble([ 0.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,70.0,80.0,100.0,150.0,200.0]),
			tauAbsEtaBinsBarrel = cms.vdouble([ 0.00,0.30,0.60,1.00,1.50]),
			tauCalibrationsBarrel = cms.vdouble([
				1.978, 1.661, 1.517, 1.411, 1.333, 1.276, 1.233, 1.202, 1.179, 1.162, 1.150, 1.137, 1.127, 1.119, 1.115, 1.115,
				1.912, 1.641, 1.512, 1.414, 1.338, 1.281, 1.237, 1.203, 1.177, 1.158, 1.143, 1.127, 1.113, 1.102, 1.095, 1.094,
				1.989, 1.683, 1.541, 1.434, 1.353, 1.292, 1.246, 1.211, 1.185, 1.166, 1.151, 1.135, 1.122, 1.113, 1.107, 1.106,
				2.734, 2.129, 1.860, 1.665, 1.523, 1.420, 1.346, 1.292, 1.252, 1.224, 1.203, 1.182, 1.166, 1.155, 1.149, 1.148,
				]),
			tauAbsEtaBinsHGCal = cms.vdouble([ 1.50,1.90,2.40,3.00]),
			tauCalibrationsHGCal = cms.vdouble([
				3.528, 2.418, 2.003, 1.740, 1.573, 1.466, 1.399, 1.356, 1.329, 1.312, 1.301, 1.292, 1.286, 1.283, 1.282, 1.282,
				3.196, 2.328, 1.990, 1.770, 1.625, 1.531, 1.470, 1.429, 1.403, 1.386, 1.374, 1.364, 1.358, 1.355, 1.353, 1.353,
				5.837, 3.089, 2.205, 1.704, 1.421, 1.260, 1.170, 1.118, 1.089, 1.073, 1.063, 1.056, 1.053, 1.052, 1.051, 1.051,
				]),
)