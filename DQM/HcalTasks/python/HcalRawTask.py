import FWCore.ParameterSet.Config as cms 

import DQM.HcalCommon.HcalDQStandard as standard
StandardSet = standard.StandardSet.clone()

#	List of FEDs
lFEDs = [x+700 for x in range(32)] + [929, 1118, 1120, 1122]
utcaCrates = [24, 20, 21, 25, 31, 35, 37, 34, 30, 22, 29, 32]

moduleName = "HcalRawTask"
#	Modify whatever is in StandardSet importing
StandardSet.moduleParameters.name		= cms.untracked.string(moduleName)
StandardSet.EventsProcessed.path		= cms.untracked.string(
	"%s/" % moduleName)
StandardSet.EventsProcessedPerLS.path	= cms.untracked.string(
	"%s/" % moduleName)
StandardSet.Standard2DMap.path			= cms.untracked.string(
	"%s/" % moduleName)
StandardSet.Standard2DMap.desc			= cms.untracked.string(
	"Some Raw Task 2D Map")

HcalProblems = StandardSet.Standard2DMap.clone()
HcalProblems.path = cms.untracked.string("%s" % moduleName)
HcalProblems.desc = cms.untracked.string(
		"Hcal Problems Rate per LS for RecHits")
HcalProblems2 = StandardSet.Standard2DMap.clone()
vMEs = [HcalProblems, HcalProblems2]

lBits = ["MissingSlot", "MissingSpigot", "MissingCrate", "MissingFED",
	"MisMatch BcN", "MisMatch EvN", "MisMatch OrN"]
lUnpackerBits = ["ErrorFree", "UnmappedDigis", "UnmappedTPDigis",
	"SpigotFormatErrors", "BadQualityDigis", "TotalDigis",
	"TotalTPDigis", "EmptyEventSpigots", "OFWSpigots", "BusySpigots"]

#---------------------------------------------------------------
#	Here, we define the vectors of MEs
#---------------------------------------------------------------
nSpigots = 19
vecuTCA = [
	cms.untracked.PSet(
		name	= cms.untracked.string("uTCA_C%dS%d_Channels" % (x, y)),
		path	= cms.untracked.string("%s/uTCA/C%d/S%d" % (moduleName, x, y)),
		kind	= cms.untracked.string("TH2D"),
		desc	= cms.untracked.string("uTCA Crate %d Slot %d Channels" % (x, y)),
		xaxis		= cms.untracked.PSet(
			edges	= cms.untracked.bool(False),
			nbins	= cms.untracked.int32(30),
			min		= cms.untracked.double(-0.5),
			max		= cms.untracked.double(29.5),
			title	= cms.untracked.string("Fiber")
		),
		yaxis		= cms.untracked.PSet(
			edges	= cms.untracked.bool(False),
			nbins	= cms.untracked.int32(5),
			min		= cms.untracked.double(-0.5),
			max		= cms.untracked.double(4.5),
			title	= cms.untracked.string("Fiber Channels")
		)
	) for x in utcaCrates for y in range(1, 13)
]
vecuTCA_EvNComp = [
	cms.untracked.PSet(
		name	= cms.untracked.string("uTCA_C%dS%d_EvNComp" % (x, y)),
		path	= cms.untracked.string("%s/uTCA/C%d/S%d" % (moduleName, x, y)),
		kind	= cms.untracked.string("TH1D"),
		desc	= cms.untracked.string("uTCA Crate %d Slot %d EvN Comparison" % (
			x, y)),
		xaxis		= cms.untracked.PSet(
			edges	= cms.untracked.bool(False),
			nbins	= cms.untracked.int32(11),
			min		= cms.untracked.double(-5.5),
			max		= cms.untracked.double(5.5),
			title	= cms.untracked.string("uHTR EvN - AMC13 EvN")
		)
	) for x in utcaCrates for y in range(1, 13)
]
vecuTCA_ORNComp = [
	cms.untracked.PSet(
		name	= cms.untracked.string("uTCA_C%dS%d_ORNComp" % (x, y)),
		path	= cms.untracked.string("%s/uTCA/C%d/S%d" % (moduleName, x, y)),
		kind	= cms.untracked.string("TH1D"),
		desc	= cms.untracked.string("uTCA Crate %d Slot %d ORN Comparison" % (
			x, y)),
		xaxis		= cms.untracked.PSet(
			edges	= cms.untracked.bool(False),
			nbins	= cms.untracked.int32(11),
			min		= cms.untracked.double(-5.5),
			max		= cms.untracked.double(5.5),
			title	= cms.untracked.string("uHTR ORN - AMC13 ORN")
		)
	) for x in utcaCrates for y in range(1, 13)
]
vecuTCA_BcNComp = [
	cms.untracked.PSet(
		name	= cms.untracked.string("uTCA_C%dS%d_BcNComp" % (x, y)),
		path	= cms.untracked.string("%s/uTCA/C%d/S%d" % (moduleName, x, y)),
		kind	= cms.untracked.string("TH1D"),
		desc	= cms.untracked.string("uTCA Crate %d Slot %d BcN Comparison" % (
			x, y)),
		xaxis		= cms.untracked.PSet(
			edges	= cms.untracked.bool(False),
			nbins	= cms.untracked.int32(11),
			min		= cms.untracked.double(-5.5),
			max		= cms.untracked.double(5.5),
			title	= cms.untracked.string("uHTR BcN - AMC13 BcN")
		)
	) for x in utcaCrates for y in range(1, 13)
]
vecVME = [
	cms.untracked.PSet(
		name	= cms.untracked.string("VME_D%dS%d_Channels" % (x, y)),
		path	= cms.untracked.string("%s/VME/D%d/S%d" % (moduleName, x, y)),
		kind	= cms.untracked.string("TH2D"),
		desc	= cms.untracked.string("VME DCC %d Spigot %d Channels" % (x, y)),
		xaxis		= cms.untracked.PSet(
			edges	= cms.untracked.bool(False),
			nbins	= cms.untracked.int32(9),
			min		= cms.untracked.double(-0.5),
			max		= cms.untracked.double(8.5),
			title	= cms.untracked.string("Fiber")
		),
		yaxis		= cms.untracked.PSet(
			edges	= cms.untracked.bool(False),
			nbins	= cms.untracked.int32(5),
			min		= cms.untracked.double(-0.5),
			max		= cms.untracked.double(4.5),
			title	= cms.untracked.string("Fiber Channels")
		)
	) for x in range(32) for y in range(nSpigots)
]
vecVME_BcNComp = [
	cms.untracked.PSet(
		name	= cms.untracked.string("VME_D%dS%d_BcNComp" % (x, y)),
		path	= cms.untracked.string("%s/VME/D%d/S%d" % (moduleName, x, y)),
		kind	= cms.untracked.string("TH1D"),
		desc	= cms.untracked.string("VME Crate %d Slot %d BcN Comparison" % (
			x, y)),
		xaxis		= cms.untracked.PSet(
			edges	= cms.untracked.bool(False),
			nbins	= cms.untracked.int32(11),
			min		= cms.untracked.double(-5.5),
			max		= cms.untracked.double(5.5),
			title	= cms.untracked.string("HTR BcN - DCC BcN")
		)
	) for x in range(32) for y in range(nSpigots)
]
vecVME_EvNComp = [
	cms.untracked.PSet(
		name	= cms.untracked.string("VME_D%dS%d_EvNComp" % (x, y)),
		path	= cms.untracked.string("%s/VME/D%d/S%d" % (moduleName, x, y)),
		kind	= cms.untracked.string("TH1D"),
		desc	= cms.untracked.string("VME Crate %d Slot %d EvN Comparison" % (
			x, y)),
		xaxis		= cms.untracked.PSet(
			edges	= cms.untracked.bool(False),
			nbins	= cms.untracked.int32(11),
			min		= cms.untracked.double(-5.5),
			max		= cms.untracked.double(5.5),
			title	= cms.untracked.string("HTR EvN - DCC EvN")
		)
	) for x in range(32) for y in range(nSpigots)
]
vecVME_ORNComp = [
	cms.untracked.PSet(
		name	= cms.untracked.string("VME_D%dS%d_ORNComp" % (x, y)),
		path	= cms.untracked.string("%s/VME/D%d/S%d" % (moduleName, x, y)),
		kind	= cms.untracked.string("TH1D"),
		desc	= cms.untracked.string("VME Crate %d Slot %d ORN Comparison" % (
			x, y)),
		xaxis		= cms.untracked.PSet(
			edges	= cms.untracked.bool(False),
			nbins	= cms.untracked.int32(11),
			min		= cms.untracked.double(-5.5),
			max		= cms.untracked.double(5.5),
			title	= cms.untracked.string("HTR BcN - DCC BcN")
		)
	) for x in range(32) for y in range(nSpigots)
]

#	Define some useful strings and doubles

#	Main Task Description
hcalRawTask = cms.EDAnalyzer(
	moduleName,
	moduleParameters	= StandardSet.moduleParameters,
	MEs					= cms.untracked.PSet(
		EventsProcessed			= StandardSet.EventsProcessed,
		EventsProcessedPerLS	= StandardSet.EventsProcessedPerLS,

		#---------------------------------------------------------------
		#	Default Plots to show, which and how many feds are unpacked
		#---------------------------------------------------------------
		uTCA_FEDsUnpacked		= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("TH1D"),
			desc		= cms.untracked.string("Unpacked AMC13/uTCA FEDs"),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(30),
				min		= cms.untracked.double(1099.5),
				max		= cms.untracked.double(1129.5),
				title	= cms.untracked.string("FEDs")
			)
		),
		VME_FEDsUnpacked		= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("TH1D"),
			desc		= cms.untracked.string("Unpacked DCC/VME FEDs"),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(32),
				min		= cms.untracked.double(699.5),
				max		= cms.untracked.double(731.5),
				title	= cms.untracked.string("FEDs")
			)
		),
		NumFEDsUnpackedvsLS		= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("PROF"),
			desc		= cms.untracked.string(
				"Number of FEDs Unpacked Total. Should be constant vs LS"),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(500),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(500),
				title	= cms.untracked.string("LS")
			),
			yaxis		= cms.untracked.PSet(
				wnbins	= cms.untracked.bool(False),
				min		= cms.untracked.double(30),
				max		= cms.untracked.double(50),
				title	= cms.untracked.string("#FEDs")
			)
		),

		#---------------------------------------------------------------
		#	For DCC/VME Specifically
		#---------------------------------------------------------------
		VME_DCCvsSpigots		= cms.untracked.PSet(
			path		= cms.untracked.string("%s/VME" % (moduleName)),
			kind		= cms.untracked.string("TH2D"),
			desc		= cms.untracked.string("VME DCC vs Spigots"),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(19),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(18.5),
				title	= cms.untracked.string("Spigots")
			),
			yaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(50),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(49.5),
				title	= cms.untracked.string("DCC")
			)
		),
		vVME	= cms.untracked.VPSet(vecVME),
		vVME_EvNComp		= cms.untracked.VPSet(vecVME_EvNComp), 
		vVME_ORNComp		= cms.untracked.VPSet(vecVME_ORNComp), 
		vVME_BcNComp		= cms.untracked.VPSet(vecVME_BcNComp), 
		VME_DCCvsSpigots_dBcN		= cms.untracked.PSet(
			path		= cms.untracked.string("%s/VME" % (moduleName)),
			kind		= cms.untracked.string("TH2D"),
			desc		= cms.untracked.string(
				"VME BcN Comparison() DCC vs Spigots"),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(nSpigots),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(nSpigots-0.5),
				title	= cms.untracked.string("Spigot")
			),
			yaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(50),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(49.5),
				title	= cms.untracked.string("dcc")
			),
		),
		VME_DCCvsSpigots_dOrN		= cms.untracked.PSet(
			path		= cms.untracked.string("%s/VME" % (moduleName)),
			kind		= cms.untracked.string("TH2D"),
			desc		= cms.untracked.string(
				"VME OrN Comparison() DCC vs Spigots"),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(nSpigots),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(nSpigots - 0.5),
				title	= cms.untracked.string("Spigot")
			),
			yaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(50),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(49.5),
				title	= cms.untracked.string("dcc")
			),
		),
		VME_DCCvsSpigots_dEvN		= cms.untracked.PSet(
			path		= cms.untracked.string("%s/VME" % (moduleName)),
			kind		= cms.untracked.string("TH2D"),
			desc		= cms.untracked.string(
				"VME EvN Comparison() DCC vs Spigots"),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(nSpigots),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(nSpigots - 0.5),
				title	= cms.untracked.string("Spigot")
			),
			yaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(50),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(49.5),
				title	= cms.untracked.string("dcc")
			),
		),

		#---------------------------------------------------------------
		#	For AMC13/uTCA specifically
		#---------------------------------------------------------------
		uTCA_CratesVSslots		= cms.untracked.PSet(
			path		= cms.untracked.string("%s/uTCA" % (moduleName)),
			kind		= cms.untracked.string("TH2D"),
			desc		= cms.untracked.string("uTCA Crates vs Slots"),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(13),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(12.5),
				title	= cms.untracked.string("Slots")
			),
			yaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(50),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(49.5),
				title	= cms.untracked.string("Crates")
			)
		),
		uTCA_CratesVSslots_dBcN		= cms.untracked.PSet(
			path		= cms.untracked.string("%s/uTCA" % (moduleName)),
			kind		= cms.untracked.string("TH2D"),
			desc		= cms.untracked.string(
				"uTCA BcN Comparison() Crates vs Slots"),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(13),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(12.5),
				title	= cms.untracked.string("Slots")
			),
			yaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(50),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(49.5),
				title	= cms.untracked.string("Crates")
			),
		),
		uTCA_CratesVSslots_dEvN		= cms.untracked.PSet(
			path		= cms.untracked.string("%s/uTCA" % (moduleName)),
			kind		= cms.untracked.string("TH2D"),
			desc		= cms.untracked.string(
				"uTCA EvN Comparison() Crates vs Slots"),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(13),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(12.5),
				title	= cms.untracked.string("Slots")
			),
			yaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(50),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(49.5),
				title	= cms.untracked.string("Crates")
			),
		),
		uTCA_CratesVSslots_dOrN		= cms.untracked.PSet(
			path		= cms.untracked.string("%s/uTCA" % (moduleName)),
			kind		= cms.untracked.string("TH2D"),
			desc		= cms.untracked.string(
				"uTCA OrN Comparison() Crates vs Slots"),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(13),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(12.5),
				title	= cms.untracked.string("Slots")
			),
			yaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(50),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(49.5),
				title	= cms.untracked.string("Crates")
			),
		),
		vuTCA				= cms.untracked.VPSet(vecuTCA),
		vuTCA_EvNComp		= cms.untracked.VPSet(vecuTCA_EvNComp), 
		vuTCA_ORNComp		= cms.untracked.VPSet(vecuTCA_ORNComp), 
		vuTCA_BcNComp		= cms.untracked.VPSet(vecuTCA_BcNComp), 
		uTCA_DataSize		= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("TH1D"),
			desc		= cms.untracked.string("uHTR Data Size"),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(16),
				min		= cms.untracked.double(-0.5),
				max		= cms.untracked.double(15.5),
				title	= cms.untracked.string("kbytes")
			)
		),


		#---------------------------------------------------------------
		#	Summary Plots
		#---------------------------------------------------------------
		Summary_Flags			= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("TH2D"),
			desc		= cms.untracked.string("Summary Flags for All Crates"),
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
				title	= cms.untracked.string("Crate Type(VME or uTCA)"),
				labels	= cms.untracked.vstring("uTCA", "VME")
			)
		),
		Summary_FlagsVsLS			= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("TH2D"),
			desc		= cms.untracked.string(
				"Summary Flags for All Crates vs LS"),
			yaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(10),
				title	= cms.untracked.string("Flags"),
				labels	= cms.untracked.vstring(lBits)
			),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(100),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(1000),
				title	= cms.untracked.string("LS")
			)
		),
		Summary_UnpackerReportFlagsVsLS			= cms.untracked.PSet(
			path		= cms.untracked.string("%s" % moduleName),
			kind		= cms.untracked.string("TH2D"),
			desc		= cms.untracked.string(
				"Unpacker Report Summary Flags vs LS"),
			yaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(10),
				title	= cms.untracked.string("Flags"),
				labels	= cms.untracked.vstring(lUnpackerBits)
			),
			xaxis		= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(1000),
				min		= cms.untracked.double(0),
				max		= cms.untracked.double(1000),
				title	= cms.untracked.string("LS")
			)
		),
	)
)
