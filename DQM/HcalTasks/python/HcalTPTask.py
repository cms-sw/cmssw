import FWCore.ParameterSet.Config as cms 

import DQM.HcalCommon.HcalDQStandard as standard
StandardSet = standard.StandardSet.clone()

#	List of FEDs
lFEDs = [x+700 for x in range(32)] + [929, 1118, 1120, 1122]

moduleName = "HcalTPTask"
#	Modify whatever is in StandardSet importing
StandardSet.moduleParameters.name		= cms.untracked.string(moduleName)
StandardSet.EventsProcessed.path		= cms.untracked.string(
	"%s/" % moduleName)
StandardSet.EventsProcessedPerLS.path	= cms.untracked.string(
	"%s/" % moduleName)
StandardSet.Standard2DMap.path			= cms.untracked.string(
	"%s/" % moduleName)
StandardSet.Standard2DMap.desc			= cms.untracked.string(
	"TP Digi Size")

cutstr_ZS = " DQM Et ZS Cut Applied(Et) "

lBits = ["Missing Data", "Missing Emul", "Size Error", "Presample Size Error",
	"MisMatch SOI Et", "MiMatch SOI FG", "MisMatch nonSOI Et", 
	"MisMatch nonSOI FG"]

#	Main Task Description
hcalTPTask = cms.EDAnalyzer(
	moduleName,
	moduleParameters	= StandardSet.moduleParameters,
	MEs					= cms.untracked.PSet(
		EventsProcessed			= StandardSet.EventsProcessed,
		EventsProcessedPerLS	= StandardSet.EventsProcessedPerLS,
		
		#	???????????????????????????????????????????//
		#	Do we need these 2 histos???
		#	???????????????????????????????????????????//
		HBHE_EtShape_Data				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HBHE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HBHE Compressed Et Shape Data"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("TS")
			)
		),
		HBHE_EtShape_Emul				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HBHE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HBHE Compressed Et Shape Emulator"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("TS")
			)
		),
		HF_EtShape_Data				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HF Compressed Et Shape Data"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("TS")
			)
		),
		HF_EtShape_Emul				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HF Compressed Et Shape Emulator"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("TS")
			)
		),
		HBHE_EtShape_Data_ZS				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HBHE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HBHE Compressed Et Shape Data" + 
				cutstr_ZS),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("TS")
			)
		),
		HBHE_EtShape_Emul_ZS				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HBHE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HBHE Compressed Et Shape Emul" + 
				cutstr_ZS),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("TS")
			)
		),
		HF_EtShape_Data_ZS				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HF Compressed Et Shape Data" + 
				cutstr_ZS),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("TS")
			)
		),
		HF_EtShape_Emul_ZS				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HF Compressed Et Shape Emul" + 
				cutstr_ZS),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("TS")
			)
		),

		#------------------------------------------------------
		#	SOI Et and FG
		#------------------------------------------------------
		HBHE_SOI_Et_Data				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HBHE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string(
				"HBHE SOI Compressed Et Distribution Data"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(256),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(256.),
				title	= cms.untracked.string("SOI Compressed Et")
			)
		),
		HBHE_SOI_Et_Emul				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HBHE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string(
				"HBHE SOI Compressed Et Distribution Emulator"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(256),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(256.),
				title	= cms.untracked.string("SOI Compressed Et")
			)
		),
		HF_SOI_Et_Data				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string(
				"HF SOI Compressed Et Distribution Data"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(256),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(256.),
				title	= cms.untracked.string("SOI Compressed Et")
			)
		),
		HF_SOI_Et_Emul				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string(
				"HF SOI Compressed Et Distribution Emulator"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(256),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(256.),
				title	= cms.untracked.string("SOI Compressed Et")
			)
		),

		#------------------------------------------------------
		#	Number of Presamples
		#------------------------------------------------------
		HF_Presamples_Data				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HF Number of Presamples Data"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("# Presamples")
			)
		),
		HBHE_Presamples_Data				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HBHE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HBHE Number of Presamples Data"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("# Presamples")
			)
		),
		HF_Presamples_Emul				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HF Number of Presamples Emul"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("# Presamples")
			)
		),
		HBHE_Presamples_Emul				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HBHE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HBHE Number of Presamples Emul"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("# Presamples")
			)
		),

		#------------------------------------------------------
		#	Occupancy Maps Data
		#------------------------------------------------------
		HBHEHF_TPOccupancyVSieta_Data				= cms.untracked.PSet(
			path	= cms.untracked.string("%s" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string(
				"HBHEHF TP Occupancy vs ieta Data"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(83),
				min		= cms.untracked.double(-41.5),
				max		= cms.untracked.double(41.5),
				title	= cms.untracked.string("ieta")
			)
		), 
		HBHEHF_TPOccupancyVSiphi_Data				= cms.untracked.PSet(
			path	= cms.untracked.string("%s" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string(
				"HBHEHF TP Occupancy vs iphi Data"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(72),
				min		= cms.untracked.double(0.5),
				max		= cms.untracked.double(72.5),
				title	= cms.untracked.string("iphi")
			)
		), 
		HF_TPOccupancyVSiphi_Data				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HF TP Occupancy vs iphi Data"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(72),
				min		= cms.untracked.double(0.5),
				max		= cms.untracked.double(72.5),
				title	= cms.untracked.string("iphi")
			)
		), 
		HBHE_TPOccupancyVSiphi_Data				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HBHE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HBHE TP Occupancy vs iphi Data"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(72),
				min		= cms.untracked.double(0.5),
				max		= cms.untracked.double(72.5),
				title	= cms.untracked.string("iphi")
			)
		), 
		
		HBHEHF_TPOccupancy_Data				= cms.untracked.PSet(
			path	= cms.untracked.string("%s" % moduleName),
			kind	= cms.untracked.string("TH2D"),
			desc	= cms.untracked.string("HBHEHF TP Occupancy Data"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(83),
				min		= cms.untracked.double(-41.5),
				max		= cms.untracked.double(41.5),
				title	= cms.untracked.string("ieta")
			),
			yaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(72),
				min		= cms.untracked.double(0.5),
				max		= cms.untracked.double(72.5),
				title	= cms.untracked.string("iphi")
			)
		),

		#------------------------------------------------------
		#	Occupancy Maps Emulator
		#------------------------------------------------------
		HBHEHF_TPOccupancyVSieta_Emul				= cms.untracked.PSet(
			path	= cms.untracked.string("%s" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string(
				"HBHEHF TP Occupancy vs ieta Emulator"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(83),
				min		= cms.untracked.double(-41.5),
				max		= cms.untracked.double(41.5),
				title	= cms.untracked.string("ieta")
			)
		), 
		HBHE_TPOccupancyVSieta_Data				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HBHE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string(
				"HBHE TP Occupancy vs ieta Data"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(83),
				min		= cms.untracked.double(-41.5),
				max		= cms.untracked.double(41.5),
				title	= cms.untracked.string("ieta")
			)
		), 
		HBHE_TPOccupancyVSieta_Emul				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HBHE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string(
				"HBHE TP Occupancy vs ieta Emulator"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(83),
				min		= cms.untracked.double(-41.5),
				max		= cms.untracked.double(41.5),
				title	= cms.untracked.string("ieta")
			)
		), 
		HF_TPOccupancyVSieta_Data				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string(
				"HF TP Occupancy vs ieta Data"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(83),
				min		= cms.untracked.double(-41.5),
				max		= cms.untracked.double(41.5),
				title	= cms.untracked.string("ieta")
			)
		), 
		HF_TPOccupancyVSieta_Emul				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string(
				"HF TP Occupancy vs ieta Emulator"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(83),
				min		= cms.untracked.double(-41.5),
				max		= cms.untracked.double(41.5),
				title	= cms.untracked.string("ieta")
			)
		), 
		HBHEHF_TPOccupancyVSiphi_Emul			= cms.untracked.PSet(
			path	= cms.untracked.string("%s" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string(
				"HBHEHF TP Occupancy vs iphi Emulator"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(72),
				min		= cms.untracked.double(0.5),
				max		= cms.untracked.double(72.5),
				title	= cms.untracked.string("iphi")
			)
		), 
		HF_TPOccupancyVSiphi_Emul				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string(
				"HF TP Occupancy vs iphi Emulator"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(72),
				min		= cms.untracked.double(0.5),
				max		= cms.untracked.double(72.5),
				title	= cms.untracked.string("iphi")
			)
		), 
		HBHE_TPOccupancyVSiphi_Emul				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HBHE" % moduleName),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("HBHE TP Occupancy vs iphi Emulator"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(72),
				min		= cms.untracked.double(0.5),
				max		= cms.untracked.double(72.5),
				title	= cms.untracked.string("iphi")
			)
		), 
		
		HBHEHF_TPOccupancy_Emul				= cms.untracked.PSet(
			path	= cms.untracked.string("%s" % moduleName),
			kind	= cms.untracked.string("TH2D"),
			desc	= cms.untracked.string("HBHEHF TP Occupancy Emulator"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(83),
				min		= cms.untracked.double(-41.5),
				max		= cms.untracked.double(41.5),
				title	= cms.untracked.string("ieta")
			),
			yaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(72),
				min		= cms.untracked.double(0.5),
				max		= cms.untracked.double(72.5),
				title	= cms.untracked.string("iphi")
			)
		), 

		HBHEHF_Missing_Data				= cms.untracked.PSet(
			path	= cms.untracked.string("%s" % moduleName),
			kind	= cms.untracked.string("TH2D"),
			desc	= cms.untracked.string("HBHEHF TP Missing from Data"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(83),
				min		= cms.untracked.double(-41.5),
				max		= cms.untracked.double(41.5),
				title	= cms.untracked.string("ieta")
			),
			yaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(72),
				min		= cms.untracked.double(0.5),
				max		= cms.untracked.double(72.5),
				title	= cms.untracked.string("iphi")
			)
		), 
		HBHEHF_Missing_Emul				= cms.untracked.PSet(
			path	= cms.untracked.string("%s" % moduleName),
			kind	= cms.untracked.string("TH2D"),
			desc	= cms.untracked.string("HBHEHF TP Missing from Emulator"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(83),
				min		= cms.untracked.double(-41.5),
				max		= cms.untracked.double(41.5),
				title	= cms.untracked.string("ieta")
			),
			yaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(72),
				min		= cms.untracked.double(0.5),
				max		= cms.untracked.double(72.5),
				title	= cms.untracked.string("iphi")
			)
		), 
		HBHEHF_Mismatch_SOIEt				= cms.untracked.PSet(
			path	= cms.untracked.string("%s" % moduleName),
			kind	= cms.untracked.string("TH2D"),
			desc	= cms.untracked.string("HBHEHF Mismatched SOI Et"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(83),
				min		= cms.untracked.double(-41.5),
				max		= cms.untracked.double(41.5),
				title	= cms.untracked.string("ieta")
			),
			yaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(72),
				min		= cms.untracked.double(0.5),
				max		= cms.untracked.double(72.5),
				title	= cms.untracked.string("iphi")
			)
		), 

		#------------------------------------------------------
		#	Compressed SOI Et Correlation
		#------------------------------------------------------
		HF_SOI_Et_Correlation				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH2D"),
			desc	= cms.untracked.string(
				"Correlation for HF SOI Compressed Et"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(128),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(256.),
				title	= cms.untracked.string("SOI Et Data")
			),
			yaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(128),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(256.),
				title	= cms.untracked.string("SOI Et Emul")
			),
		),
		HBHE_SOI_Et_Correlation				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HBHE" % moduleName),
			kind	= cms.untracked.string("TH2D"),
			desc	= cms.untracked.string(
				"Correlation for HBHE SOI Compressed Et"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(128),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(256.),
				title	= cms.untracked.string("SOI Et Data")
			),
			yaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(128),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(256.),
				title	= cms.untracked.string("SOI Et Emul")
			),
		),
		HF_SOI_FG_Correlation				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH2D"),
			desc	= cms.untracked.string(
				"Correlation for HF SOI Fine Grain Bit"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(2),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(1.5),
				title	= cms.untracked.string("SOI FG Data")
			),
			yaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(2),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(1.5),
				title	= cms.untracked.string("SOI FG Emul")
			),
		),
		HBHE_SOI_FG_Correlation				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HBHE" % moduleName),
			kind	= cms.untracked.string("TH2D"),
			desc	= cms.untracked.string(
				"Correlation for HBHE SOI Fine Grain Bit"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(2),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(1.5),
				title	= cms.untracked.string("SOI FG Data")
			),
			yaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(2),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(1.5),
				title	= cms.untracked.string("SOI FG Emul")
			),
		),
		HF_nonSOI_Et_Correlation				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH2D"),
			desc	= cms.untracked.string(
				"Correlation for HF nonSOI Compressed Et"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(50),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(256.),
				title	= cms.untracked.string("Et Data")
			),
			yaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(50),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(256.),
				title	= cms.untracked.string("Et Emul")
			),
		),
		HBHE_nonSOI_Et_Correlation				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HBHE" % moduleName),
			kind	= cms.untracked.string("TH2D"),
			desc	= cms.untracked.string(
				"Correlation for HBHE nonSOI Compressed Et"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(50),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(256.),
				title	= cms.untracked.string("Et Data")
			),
			yaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(50),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(256.),
				title	= cms.untracked.string("Et Emul")
			),
		),
		HF_nonSOI_FG_Correlation				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HF" % moduleName),
			kind	= cms.untracked.string("TH2D"),
			desc	= cms.untracked.string(
				"Correlation for HF nonSOI FG"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(2),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(1.5),
				title	= cms.untracked.string("FG Data")
			),
			yaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(2),
				min		= cms.untracked.double(0.5),
				max		= cms.untracked.double(1.5),
				title	= cms.untracked.string("FG Emul")
			),
		),
		HBHE_nonSOI_FG_Correlation				= cms.untracked.PSet(
			path	= cms.untracked.string("%s/HBHE" % moduleName),
			kind	= cms.untracked.string("TH2D"),
			desc	= cms.untracked.string(
				"Correlation for HBHE nonSOI FG"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(2),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(1.5),
				title	= cms.untracked.string("FG Data")
			),
			yaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(2),
				min		= cms.untracked.double(0.5),
				max		= cms.untracked.double(1.5),
				title	= cms.untracked.string("FG Emul")
			),
		),
		#------------------------------------------------------
		#	Summary Plots
		#------------------------------------------------------
		Summary_Flags				= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("TH2D"),
			desc		= cms.untracked.string("Summary Flags HBHEHF"),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(10),
				title	= cms.untracked.string("Flags"),
				labels	= cms.untracked.vstring(lBits)
			),
			yaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(2),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(2),
				title	= cms.untracked.string("Subsystem"),
				labels	= cms.untracked.vstring(["HBHE", "HF"])
			),
		),
		Summary_HBHE_FlagsVsLS				= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("TH2D"),
			desc		= cms.untracked.string("Summary HBHE Flags vs LS"),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(100),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(1000),
				title	= cms.untracked.string("LS")
			),
			yaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(10),
				title	= cms.untracked.string("Flags"),
				labels	= cms.untracked.vstring(lBits)
			),
		),
		Summary_HF_FlagsVsLS				= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("TH2D"),
			desc		= cms.untracked.string("Summary HBHE Flags vs LS"),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(100),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(1000),
				title	= cms.untracked.string("LS")
			),
			yaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(10),
				title	= cms.untracked.string("Flags"),
				labels	= cms.untracked.vstring(lBits)
			),
		),

		#------------------------------------------------------
		#	TP Digi Sizes
		#------------------------------------------------------
		HBHEHF_TPDigiSize_Data				= cms.untracked.PSet(
			path	= cms.untracked.string("%s" % moduleName),
			kind	= cms.untracked.string("TH2D"),
			desc	= cms.untracked.string(
				"TP Digi Size Data"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(2),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(1.5),
				title	= cms.untracked.string("Subsystem")
			),
			yaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(9.5),
				title	= cms.untracked.string("#TS")
			),
		),
		HBHEHF_TPDigiSize_Emul				= cms.untracked.PSet(
			path	= cms.untracked.string("%s" % moduleName),
			kind	= cms.untracked.string("TH2D"),
			desc	= cms.untracked.string(
				"TP Digi Size Emulator"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(2),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(1.5),
				title	= cms.untracked.string("Subsystem")
			),
			yaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(9.5),
				title	= cms.untracked.string("#TS")
			),
		),
	)
)
