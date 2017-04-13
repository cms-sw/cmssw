import FWCore.ParameterSet.Config as cms 

import DQM.HcalCommon.HcalDQStandard as standard
StandardSet = standard.StandardSet.clone()

#	List of FEDs
lFEDs = [x+700 for x in range(32)] + [929, 1118, 1120, 1122]

moduleName = "HcalLEDTask"
#	Modify whatever is in StandardSet importing
StandardSet.moduleParameters.name		= cms.untracked.string(moduleName)
StandardSet.EventsProcessed.path		= cms.untracked.string(
	"%s/" % moduleName)
StandardSet.EventsProcessedPerLS.path	= cms.untracked.string(
	"%s/" % moduleName)
StandardSet.Standard2DMap.path			= cms.untracked.string(
	"%s/" % moduleName)
StandardSet.Standard2DMap.desc			= cms.untracked.string(
	"Some LED Task 2D Map")

strdesc_nocuts = " 3TS(Global) or 5TS(Local) Integral No Cuts Applied"

#	Main Task Description
hcalLEDTask = cms.EDAnalyzer(
	moduleName,
	moduleParameters	= StandardSet.moduleParameters,
	MEs					= cms.untracked.PSet(
		EventsProcessed			= StandardSet.EventsProcessed,
		EventsProcessedPerLS	= StandardSet.EventsProcessedPerLS,
		
		#---------------------------------------------------------
		#	1D Shapes
		#---------------------------------------------------------
		HB_Shape				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HB" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HB LED Shape"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("TS")
			)
		),
		HE_Shape				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HE LED Shape"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("TS")
			)
		),
		HO_Shape				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HO" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HO LED Shape"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("TS")
			)
		),
		HF_Shape				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HF LED Shape"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("TS")
			)
		),

		#---------------------------------------------------------
		#	TH1D signals recorded per each event. For Online Monitring
		#---------------------------------------------------------
		HB_Signal				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HB" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HB LED Signals" + strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(500),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(3000),
				title	= cms.untracked.string("Nominal fC")
			)
		),
		HE_Signal				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HE LED Signals " + strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(500),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(3000),
				title	= cms.untracked.string("Nominal fC")
			)
		),
		HO_Signal				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HO" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HO LED Signals " + strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(500),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(3000),
				title	= cms.untracked.string("Nominal fC")
			)
		),
		HF_Signal				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HF LED Signals " + strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(500),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(3000),
				title	= cms.untracked.string("Nominal fC")
			)
		),
		HB_Timing				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HB" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HB LED Timing " + strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(100),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10),
				title	= cms.untracked.string("TS")
			)
		),
		HE_Timing				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HE LED Timing " + strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(100),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10),
				title	= cms.untracked.string("TS")
			)
		),
		HO_Timing				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HO" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HO LED Timing " + strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(100),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10),
				title	= cms.untracked.string("TS")
			)
		),
		HF_Timing				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HF LED Timing " + strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(100),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10),
				title	= cms.untracked.string("TS")
			)
		),

		#---------------------------------------------------------
		#	2D Profiles. Recorded each Event. Means/RMSs to be
		#	close to the values we obtain using %sDQLedClass
		#---------------------------------------------------------
		HBHEHFD1_SignalMap			= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("PROF2D"),
			desc		= cms.untracked.string(
				"HBHEHF D1 LED Signal Map " + strdesc_nocuts
			),
			xaxis		= StandardSet.ietaAxis.clone(),
			yaxis		= StandardSet.iphiAxis.clone(),
			zaxis		= cms.untracked.PSet(
				wnbins	= cms.untracked.bool(False),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(4000)
			)
		),
		HBHEHFD2_SignalMap			= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("PROF2D"),
			desc		= cms.untracked.string(
				"HBHEHF D2 LED Signal Map " + strdesc_nocuts
			),
			xaxis		= StandardSet.ietaAxis.clone(),
			yaxis		= StandardSet.iphiAxis.clone(),
			zaxis		= cms.untracked.PSet(
				wnbins	= cms.untracked.bool(False),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(4000)
			)
		),
		HBHEHFD3_SignalMap			= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("PROF2D"),
			desc		= cms.untracked.string(
				"HBHEHF D3 LED Signal Map " + strdesc_nocuts
			),
			xaxis		= StandardSet.ietaAxis.clone(),
			yaxis		= StandardSet.iphiAxis.clone(),
			zaxis		= cms.untracked.PSet(
				wnbins	= cms.untracked.bool(False),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(4000)
			)
		),
		HOD4_SignalMap			= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("PROF2D"),
			desc		= cms.untracked.string(
				"HO D4 LED Signal Map " + strdesc_nocuts
			),
			xaxis		= StandardSet.ietaAxis.clone(),
			yaxis		= StandardSet.iphiAxis.clone(),
			zaxis		= cms.untracked.PSet(
				wnbins	= cms.untracked.bool(False),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(4000)
			)
		),
		HBHEHFD1_TimingMap			= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("PROF2D"),
			desc		= cms.untracked.string(
				"HBHEHF D1 LED Timing Map " + strdesc_nocuts
			),
			xaxis		= StandardSet.ietaAxis.clone(),
			yaxis		= StandardSet.iphiAxis.clone(),
			zaxis		= cms.untracked.PSet(
				wnbins	= cms.untracked.bool(False),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(10)
			)
		),
		HBHEHFD2_TimingMap			= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("PROF2D"),
			desc		= cms.untracked.string(
				"HBHEHF D2 LED Timing Map " + strdesc_nocuts
			),
			xaxis		= StandardSet.ietaAxis.clone(),
			yaxis		= StandardSet.iphiAxis.clone(),
			zaxis		= cms.untracked.PSet(
				wnbins	= cms.untracked.bool(False),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(10)
			)
		),
		HBHEHFD3_TimingMap			= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("PROF2D"),
			desc		= cms.untracked.string(
				"HBHEHF D3 LED Timing Map " + strdesc_nocuts
			),
			xaxis		= StandardSet.ietaAxis.clone(),
			yaxis		= StandardSet.iphiAxis.clone(),
			zaxis		= cms.untracked.PSet(
				wnbins	= cms.untracked.bool(False),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(10)
			)
		),
		HOD4_TimingMap			= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("PROF2D"),
			desc		= cms.untracked.string(
				"HO D4 LED Timing Map " + strdesc_nocuts
			),
			xaxis		= StandardSet.ietaAxis.clone(),
			yaxis		= StandardSet.iphiAxis.clone(),
			zaxis		= cms.untracked.PSet(
				wnbins	= cms.untracked.bool(False),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(10)
			)
		),

		#---------------------------------------------------------
		#	1D Histos of LED Signal/Timing Means/RMSs as obtained
		#	from %sDQLedData
		#---------------------------------------------------------
		HB_SignalMeans_Summary				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HB" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HB LED Signal Means Summary" 
				+ strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(500),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(3000),
				title	= cms.untracked.string("Nominal fC")
			)
		),
		HB_SignalRMSs_Summary				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HB" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HB LED Signal RMSs Summary" 
				+ strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(200),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(1000),
				title	= cms.untracked.string("Nominal fC")
			)
		),
		HB_TimingMeans_Summary				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HB" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HB LED Timing Means Summary" 
				+ strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(100),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10),
				title	= cms.untracked.string("TS")
			)
		),
		HB_TimingRMSs_Summary				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HB" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HB LED Timing RMSs Summary" 
				+ strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(50),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(5),
				title	= cms.untracked.string("TS")
			)
		),
		HE_SignalMeans_Summary				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HE LED Signal Means Summary" 
				+ strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(500),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(3000),
				title	= cms.untracked.string("Nominal fC")
			)
		),
		HE_SignalRMSs_Summary				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HE LED Signal RMSs Summary" 
				+ strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(200),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(1000),
				title	= cms.untracked.string("Nominal fC")
			)
		),
		HE_TimingMeans_Summary				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HE LED Timing Means Summary" 
				+ strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(100),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10),
				title	= cms.untracked.string("TS")
			)
		),
		HE_TimingRMSs_Summary				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HE LED Timing RMSs Summary" 
				+ strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(50),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(5),
				title	= cms.untracked.string("TS")
			)
		),
		HO_SignalMeans_Summary				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HO" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HO LED Signal Means Summary" 
				+ strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(500),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(3000),
				title	= cms.untracked.string("Nominal fC")
			)
		),
		HO_SignalRMSs_Summary				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HO" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HO LED Signal RMSs Summary" 
				+ strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(200),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(1000),
				title	= cms.untracked.string("Nominal fC")
			)
		),
		HO_TimingMeans_Summary				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HO" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HO LED Timing Means Summary" 
				+ strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(100),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10),
				title	= cms.untracked.string("TS")
			)
		),
		HO_TimingRMSs_Summary				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HO" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HO LED Timing RMSs Summary" 
				+ strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(50),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(5),
				title	= cms.untracked.string("TS")
			)
		),
		HF_SignalMeans_Summary				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HF LED Signal Means Summary" 
				+ strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(500),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(3000),
				title	= cms.untracked.string("Nominal fC")
			)
		),
		HF_SignalRMSs_Summary				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HF LED Signal RMSs Summary" 
				+ strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(200),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(1000),
				title	= cms.untracked.string("Nominal fC")
			)
		),
		HF_TimingMeans_Summary				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HF LED Timing Means Summary" 
				+ strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(100),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10),
				title	= cms.untracked.string("TS")
			)
		),
		HF_TimingRMSs_Summary				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HF LED Timing RMSs Summary" 
				+ strdesc_nocuts),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(50),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(5),
				title	= cms.untracked.string("TS")
			)
		),

	)
)
