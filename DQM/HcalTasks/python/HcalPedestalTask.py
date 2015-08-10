import FWCore.ParameterSet.Config as cms 

import DQM.HcalCommon.HcalDQStandard as standard
StandardSet = standard.StandardSet.clone()

#	List of FEDs
lFEDs = [x+700 for x in range(32)] + [929, 1118, 1120, 1122]

moduleName = "HcalPedestalTask"
#	Modify whatever is in StandardSet importing
StandardSet.moduleParameters.name		= cms.untracked.string(moduleName)
StandardSet.EventsProcessed.path		= cms.untracked.string(
	"%s/" % moduleName)
StandardSet.EventsProcessedPerLS.path	= cms.untracked.string(
	"%s/" % moduleName)
StandardSet.Standard2DMap.path			= cms.untracked.string(
	"%s/" % moduleName)
StandardSet.Standard2DMap.desc			= cms.untracked.string(
	"Some Pedestal Task 2D Map")

strdesc = " Recorded Per Event " 
strdesc_summary = " Summary of All Events "

#	Main Task Description
hcalPedestalTask = cms.EDAnalyzer(
	moduleName,
	moduleParameters	= StandardSet.moduleParameters,
	MEs					= cms.untracked.PSet(
		EventsProcessed			= StandardSet.EventsProcessed,
		EventsProcessedPerLS	= StandardSet.EventsProcessedPerLS,

		#--------------------------------------------------------
		#	TH1D Pedestals recorded per each event. For Online Mon
		#--------------------------------------------------------
		HB_Pedestals			= cms.untracked.PSet(
			path		= cms.untracked.string("%s/HB" % moduleName),
			kind		= cms.untracked.string("TH1D"),
			desc		= cms.untracked.string(
				"HB 4Caps-averaged Pedestals." + strdesc),
			xaxis		= cms.untracked.PSet(
				edges		= cms.untracked.bool(False),
				nbins		= cms.untracked.int32(50),
				min			= cms.untracked.double(0),
				max			= cms.untracked.double(5),
				title		= cms.untracked.string("Ped. (Unlin. ADC)")
			),
			yaxis		= cms.untracked.PSet(
				log		= cms.untracked.bool(True)
			)
		),
		HE_Pedestals			= cms.untracked.PSet(
			path		= cms.untracked.string("%s/HE" % (moduleName)),
			kind		= cms.untracked.string("TH1D"),
			desc		= cms.untracked.string(
				"HE 4Caps-averaged Pedestals." + strdesc),
			xaxis		= cms.untracked.PSet(
				edges		= cms.untracked.bool(False),
				nbins		= cms.untracked.int32(50),
				min			= cms.untracked.double(0),
				max			= cms.untracked.double(5),
				title		= cms.untracked.string("Ped. (Unlin. ADC)")
			),
			yaxis		= cms.untracked.PSet(
				log		= cms.untracked.bool(True)
			)
		),
		HO_Pedestals			= cms.untracked.PSet(
			path		= cms.untracked.string("%s/HO" % (moduleName)),
			kind		= cms.untracked.string("TH1D"),
			desc		= cms.untracked.string(
				"HO 4Caps-averaged Pedestals." + strdesc),
			xaxis		= cms.untracked.PSet(
				edges		= cms.untracked.bool(False),
				nbins		= cms.untracked.int32(100),
				min			= cms.untracked.double(0),
				max			= cms.untracked.double(16),
				title		= cms.untracked.string("Ped. (Unlin. ADC)")
			),
			yaxis		= cms.untracked.PSet(
				log		= cms.untracked.bool(True)
			)
		),
		HF_Pedestals			= cms.untracked.PSet(
			path		= cms.untracked.string("%s/HF" % (moduleName)),
			kind		= cms.untracked.string("TH1D"),
			desc		= cms.untracked.string(
				"HF 4Caps-averaged Pedestals." + strdesc),
			xaxis		= cms.untracked.PSet(
				edges		= cms.untracked.bool(False),
				nbins		= cms.untracked.int32(50),
				min			= cms.untracked.double(0),
				max			= cms.untracked.double(5),
				title		= cms.untracked.string("Ped. (Unlin. ADC)")
			),
			yaxis		= cms.untracked.PSet(
				log		= cms.untracked.bool(True)
			)
		),

		#--------------------------------------------------------
		#	2D Profiles. Recorded for each event. Means/RMSs should be close
		#	to the values we obtain using HcalDQPedClass
		#--------------------------------------------------------
		HBHEHFD1_PedestalsMap		= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % (moduleName)),
			kind		= cms.untracked.string("PROF2D"),
			desc		= cms.untracked.string(
				"HBHEHF D1 4Caps-averaged Pedestals" + strdesc),
			xaxis		= StandardSet.ietaAxis.clone(),
			yaxis		= StandardSet.iphiAxis.clone(),
			zaxis		= cms.untracked.PSet(
				wnbins		= cms.untracked.bool(False),
				min			= cms.untracked.double(0),
				max			= cms.untracked.double(5)
			)
		),
		HBHEHFD2_PedestalsMap		= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % (moduleName)),
			kind		= cms.untracked.string("PROF2D"),
			desc		= cms.untracked.string(
				"HBHEHF D2 4Caps-averaged Pedestals" + strdesc),
			xaxis		= StandardSet.ietaAxis.clone(),
			yaxis		= StandardSet.iphiAxis.clone(),
			zaxis		= cms.untracked.PSet(
				wnbins		= cms.untracked.bool(False),
				min			= cms.untracked.double(0),
				max			= cms.untracked.double(5)
			)
		),
		HBHEHFD3_PedestalsMap		= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % (moduleName)),
			kind		= cms.untracked.string("PROF2D"),
			desc		= cms.untracked.string(
				"HBHEHF D3 4Caps-averaged Pedestals" + strdesc),
			xaxis		= StandardSet.ietaAxis.clone(),
			yaxis		= StandardSet.iphiAxis.clone(),
			zaxis		= cms.untracked.PSet(
				wnbins		= cms.untracked.bool(False),
				min			= cms.untracked.double(0),
				max			= cms.untracked.double(5)
			)
		),
		HOD4_PedestalsMap		= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % (moduleName)),
			kind		= cms.untracked.string("PROF2D"),
			desc		= cms.untracked.string(
				"HO D4 4Caps-averaged Pedestals" + strdesc),
			xaxis		= cms.untracked.PSet(
				edges		= cms.untracked.bool(False),
				nbins		= cms.untracked.int32(33),
				min			= cms.untracked.double(-16.5),
				max			= cms.untracked.double(16.5),
				title		= cms.untracked.string("ieta")
			),
			yaxis		= StandardSet.iphiAxis.clone(),
			zaxis		= cms.untracked.PSet(
				wnbins		= cms.untracked.bool(False),
				min			= cms.untracked.double(0),
				max			= cms.untracked.double(16)
			)
		),

		#--------------------------------------------------------
		#	1D Histos of Pedestal Means/RMSs as Obtained from HcalDQPedData
		#--------------------------------------------------------
		HB_PedMeans_Summary			= cms.untracked.PSet(
			path		= cms.untracked.string("%s/HB" % (moduleName)),
			kind		= cms.untracked.string("TH1D"),
			desc		= cms.untracked.string(
				"HB 4Caps-averaged Pedestal Means " + strdesc_summary),
			xaxis		= cms.untracked.PSet(
				edges		= cms.untracked.bool(False),
				nbins		= cms.untracked.int32(50),
				min			= cms.untracked.double(0),
				max			= cms.untracked.double(5),
				title		= cms.untracked.string("Ped. (Unlin. ADC)")
			),
			yaxis		= cms.untracked.PSet(
				log		= cms.untracked.bool(True)
			)
		),
		HE_PedMeans_Summary			= cms.untracked.PSet(
			path		= cms.untracked.string("%s/HE" % (moduleName)),
			kind		= cms.untracked.string("TH1D"),
			desc		= cms.untracked.string(
				"HE 4Caps-averaged Pedestal Means " + strdesc_summary),
			xaxis		= cms.untracked.PSet(
				edges		= cms.untracked.bool(False),
				nbins		= cms.untracked.int32(50),
				min			= cms.untracked.double(0),
				max			= cms.untracked.double(5),
				title		= cms.untracked.string("Ped. (Unlin. ADC)")
			),
			yaxis		= cms.untracked.PSet(
				log		= cms.untracked.bool(True)
			)
		),
		HF_PedMeans_Summary			= cms.untracked.PSet(
			path		= cms.untracked.string("%s/HF" % (moduleName)),
			kind		= cms.untracked.string("TH1D"),
			desc		= cms.untracked.string(
				"HF 4Caps-averaged Pedestal Means " + strdesc_summary),
			xaxis		= cms.untracked.PSet(
				edges		= cms.untracked.bool(False),
				nbins		= cms.untracked.int32(50),
				min			= cms.untracked.double(0),
				max			= cms.untracked.double(5),
				title		= cms.untracked.string("Ped. (Unlin. ADC)")
			),
			yaxis		= cms.untracked.PSet(
				log		= cms.untracked.bool(True)
			)
		),
		HO_PedMeans_Summary			= cms.untracked.PSet(
			path		= cms.untracked.string("%s/HO" % (moduleName)),
			kind		= cms.untracked.string("TH1D"),
			desc		= cms.untracked.string(
				"HO 4Caps-averaged Pedestal Means " + strdesc_summary),
			xaxis		= cms.untracked.PSet(
				edges		= cms.untracked.bool(False),
				nbins		= cms.untracked.int32(100),
				min			= cms.untracked.double(0),
				max			= cms.untracked.double(16),
				title		= cms.untracked.string("Ped. (Unlin. ADC)")
			),
			yaxis		= cms.untracked.PSet(
				log		= cms.untracked.bool(True)
			)
		),
		HB_PedRMSs_Summary			= cms.untracked.PSet(
			path		= cms.untracked.string("%s/HB" % (moduleName)),
			kind		= cms.untracked.string("TH1D"),
			desc		= cms.untracked.string(
				"HB 4Caps-averaged Pedestal RMSs " + strdesc_summary),
			xaxis		= cms.untracked.PSet(
				edges		= cms.untracked.bool(False),
				nbins		= cms.untracked.int32(50),
				min			= cms.untracked.double(0),
				max			= cms.untracked.double(2),
				title		= cms.untracked.string("Ped. (Unlin. ADC)")
			),
			yaxis		= cms.untracked.PSet(
				log		= cms.untracked.bool(True)
			)
		),
		HE_PedRMSs_Summary			= cms.untracked.PSet(
			path		= cms.untracked.string("%s/HE" % (moduleName)),
			kind		= cms.untracked.string("TH1D"),
			desc		= cms.untracked.string(
				"HE 4Caps-averaged Pedestal RMSs " + strdesc_summary),
			xaxis		= cms.untracked.PSet(
				edges		= cms.untracked.bool(False),
				nbins		= cms.untracked.int32(50),
				min			= cms.untracked.double(0),
				max			= cms.untracked.double(2),
				title		= cms.untracked.string("Ped. (Unlin. ADC)")
			),
			yaxis		= cms.untracked.PSet(
				log		= cms.untracked.bool(True)
			)
		),
		HF_PedRMSs_Summary			= cms.untracked.PSet(
			path		= cms.untracked.string("%s/HF" % (moduleName)),
			kind		= cms.untracked.string("TH1D"),
			desc		= cms.untracked.string(
				"HF 4Caps-averaged Pedestal RMSs " + strdesc_summary),
			xaxis		= cms.untracked.PSet(
				edges		= cms.untracked.bool(False),
				nbins		= cms.untracked.int32(50),
				min			= cms.untracked.double(0),
				max			= cms.untracked.double(2),
				title		= cms.untracked.string("Ped. (Unlin. ADC)")
			),
			yaxis		= cms.untracked.PSet(
				log		= cms.untracked.bool(True)
			)
		),
		HO_PedRMSs_Summary			= cms.untracked.PSet(
			path		= cms.untracked.string("%s/HO" % (moduleName)),
			kind		= cms.untracked.string("TH1D"),
			desc		= cms.untracked.string(
				"HO 4Caps-averaged Pedestal RMSs " + strdesc_summary),
			xaxis		= cms.untracked.PSet(
				edges		= cms.untracked.bool(False),
				nbins		= cms.untracked.int32(50),
				min			= cms.untracked.double(0),
				max			= cms.untracked.double(2),
				title		= cms.untracked.string("Ped. (Unlin. ADC)")
			),
			yaxis		= cms.untracked.PSet(
				log		= cms.untracked.bool(True)
			)
		),

		#--------------------------------------------------------
		#	2D Maps of Pedestals as obtained from HcalDQPedData
		#--------------------------------------------------------
#		HBHEHFD1_PedestalsMap_Summary		= cms.untracked.PSet(
#			path		= cms.untracked.string("%s" % (moduleName)),
#			kind		= cms.untracked.string("TH2D"),
#			desc		= cms.untracked.string(
#				"HBHEHF D1 4Caps-averaged Pedestals " + strdesc_summary),
#			xaxis		= StandardSet.ietaAxis.clone(),
#			yaxis		= StandardSet.iphiAxis.clone(),
#		),
#		HBHEHFD2_PedestalsMap_Summary		= cms.untracked.PSet(
#			path		= cms.untracked.string("%s" % (moduleName)),
#			kind		= cms.untracked.string("TH2D"),
#			desc		= cms.untracked.string(
#				"HBHEHF D2 4Caps-averaged Pedestals " + strdesc_summary),
#			xaxis		= StandardSet.ietaAxis.clone(),
#			yaxis		= StandardSet.iphiAxis.clone(),
#		),
#		HBHEHFD3_PedestalsMap_Summary		= cms.untracked.PSet(
#			path		= cms.untracked.string("%s" % (moduleName)),
#			kind		= cms.untracked.string("TH2D"),
#			desc		= cms.untracked.string(
#				"HBHEHF D3 4Caps-averaged Pedestals " + strdesc_summary),
#			xaxis		= StandardSet.ietaAxis.clone(),
#			yaxis		= StandardSet.iphiAxis.clone(),
#		),
#		HOD4_PedestalsMap_Summary		= cms.untracked.PSet(
#			path		= cms.untracked.string("%s" % (moduleName)),
#			kind		= cms.untracked.string("TH2D"),
#			desc		= cms.untracked.string(
#				"HO D4 4Caps-averaged Pedestals " + strdesc_summary),
#			xaxis		= StandardSet.ietaAxis.clone(),
#			yaxis		= StandardSet.iphiAxis.clone(),
#		),
		
	)
)
