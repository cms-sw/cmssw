#-------------------------------------------------------------------
#	Standard Configuration File for HCAL DQM for both 
#	Local/Globla(Calib vs. Normal).
#	This file is configured as is for Global
#	All the modifications needed for either Local or Calib must be 
#	done in the actual cfg.
#-------------------------------------------------------------------

import FWCore.ParameterSet.Config as cms

#	Generate a list of FEDs
lFEDs = [x+700 for x in range(32)] + [1118, 1120, 1122]

subsystem = "Hcal"
StandardSet		= cms.untracked.PSet(
	moduleParameters	= cms.untracked.PSet(
		subsystem		= cms.untracked.string(subsystem),
		name			= cms.untracked.string("HcalDQStandard"),
		debug			= cms.untracked.int32(0),
		calibTypes		= cms.untracked.vint32(0),
		runType			= cms.untracked.string("TEST"),
		mtype			= cms.untracked.string("SOURCE"),
		FEDs			= cms.untracked.vint32(lFEDs),
		isGlobal		= cms.untracked.bool(True)
	),
	
	Labels				= cms.untracked.PSet(
		HBHEDigi		= cms.untracked.InputTag("hcalDigis"),
		HFDigi			= cms.untracked.InputTag("hcalDigis"),
		HODigi			= cms.untracked.InputTag("hcalDigis"),
		HCALTPD			= cms.untracked.InputTag("hcalDigis"),
		HCALTPE			= cms.untracked.InputTag("emulTPDigis"),
		HFDigiVME		= cms.untracked.InputTag("vmeDigis"),
		HBHEDigiuTCA	= cms.untracked.InputTag("utcaDigis"),
		RAW				= cms.untracked.InputTag("rawDataCollector"),
		HBHERecHit		= cms.untracked.InputTag("hbhereco"),
		HFRecHit		= cms.untracked.InputTag("hfreco"),
		HORecHit		= cms.untracked.InputTag("horeco"),
		L1GT			= cms.untracked.InputTag("l1GtUnpack"),
		HLTResults		= cms.untracked.InputTag("TriggerResults"),
		DCS				= cms.untracked.InputTag("scalersRawToDigi"),
		UnpackerReport	= cms.untracked.InputTag("hcalDigis"),
		HCALTBTrigger	= cms.untracked.InputTag("tbunpack")
	),

	EventsProcessed		= cms.untracked.PSet(
		path			= cms.untracked.string("HcalDQStandard/"),
		kind			= cms.untracked.string("INT"),
	#	desc			= cms.untracked.string("Processed Events Total"),
	),

	EventsProcessedPerLS = cms.untracked.PSet(
		path			= cms.untracked.string("HcalDQStandard/"),
		kind			= cms.untracked.string("INT"),
	#	desc			= cms.untracked.
	),

	Standard2DMap		= cms.untracked.PSet(
		path			= cms.untracked.string("HcalDQStandard/"),
		kind			= cms.untracked.string("TH2D"),
		desc			= cms.untracked.string("Standard 2D Map"),
		xaxis			= cms.untracked.PSet(
			edges		= cms.untracked.bool(False),
			nbins		= cms.untracked.int32(83),
			min			= cms.untracked.double(-41.5),
			max			= cms.untracked.double(41.5),
			title		= cms.untracked.string("ieta")
		),
		yaxis			= cms.untracked.PSet(
			edges		= cms.untracked.bool(False),
			nbins		= cms.untracked.int32(72),
			min			= cms.untracked.double(0.5),
			max			= cms.untracked.double(72.5),
			title		= cms.untracked.string("iphi")
		)
	),

	Standard2DSubSystem = cms.untracked.PSet(
		path			= cms.untracked.string("HcalDQStandard/"),
		kind			= cms.untracked.string("TH2D"),
		desc			= cms.untracked.string("Standard 2D SubSystem Map"),
		xaxis			= cms.untracked.PSet(
			edges		= cms.untracked.bool(False),
			nbins		= cms.untracked.int32(4),
			min			= cms.untracked.double(-0.5),
			max			= cms.untracked.double(3.5),
			title		= cms.untracked.string("Sub Detector")
		),
		yaxis			= cms.untracked.PSet(
			edges		= cms.untracked.bool(False),
			nbins		= cms.untracked.int32(20),
			min			= cms.untracked.double(0.5),
			max			= cms.untracked.double(20.5),
			title		= cms.untracked.string("Y-axis")
		)
	),

	Standard2DProf		= cms.untracked.PSet(
		path			= cms.untracked.string("HcalDQStandard/"),
		kind			= cms.untracked.string("PROF2D"),
		desc			= cms.untracked.string("Standard 2D Profile"),
		xaxis			= cms.untracked.PSet(
			edges		= cms.untracked.bool(False),
			nbins		= cms.untracked.int32(83),
			min			= cms.untracked.double(-41.5),
			max			= cms.untracked.double(41.5),
			title		= cms.untracked.string("ieta")
		),
		yaxis			= cms.untracked.PSet(
			edges		= cms.untracked.bool(False),
			nbins		= cms.untracked.int32(72),
			min			= cms.untracked.double(0.5),
			max			= cms.untracked.double(72.5),
			title		= cms.untracked.string("iphi")
		)
	),

	StandardPhiProf		= cms.untracked.PSet(
		path			= cms.untracked.string("HcalDQStandard/"),
		kind			= cms.untracked.string("PROF"),
		desc			= cms.untracked.string("Standard Phi Profile"),
		xaxis			= cms.untracked.PSet(
			edges		= cms.untracked.bool(False),
			nbins		= cms.untracked.int32(72),
			min			= cms.untracked.double(0.5),
			max			= cms.untracked.double(72.5),
			title		= cms.untracked.string("iphi")
		)
	),

	StandardEtaProf		= cms.untracked.PSet(
		path			= cms.untracked.string("HcalDQStandard/" ),
		kind			= cms.untracked.string("PROF"),
		desc			= cms.untracked.string("Standard Eta Profile"),
		xaxis			= cms.untracked.PSet(
			edges		= cms.untracked.bool(False),
			nbins		= cms.untracked.int32(83),
			min			= cms.untracked.double(-41.5),
			max			= cms.untracked.double(41.5),
			title		= cms.untracked.string("ieta")
		)
	),

	iphiAxis			= cms.untracked.PSet(
		edges	= cms.untracked.bool(False),
		nbins	= cms.untracked.int32(72),
		min		= cms.untracked.double(0.5),
		max		= cms.untracked.double(72.5),
		title	= cms.untracked.string("iphi")
	),
	ietaAxis			= cms.untracked.PSet(
		edges	= cms.untracked.bool(False),
		nbins	= cms.untracked.int32(83),
		min		= cms.untracked.double(-41.5),
		max		= cms.untracked.double(41.5),
		title	= cms.untracked.string("ieta")
	)
)

StandardSet.moduleParameters.Labels = StandardSet.Labels

