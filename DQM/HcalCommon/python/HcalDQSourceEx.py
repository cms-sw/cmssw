import FWCore.ParameterSet.Config as cms

hcalDQSourceEx = cms.EDAnalyzer(
	"HcalDQSourceEx",
	moduleParameters	= cms.untracked.PSet(
		name		= cms.untracked.string("HcalDQSourceEx"),
		debug		= cms.untracked.int32(10),
		eventType	= cms.untracked.string("TEST"),
		runType		= cms.untracked.string("TEST"),
		mtype		= cms.untracked.string("SOURCE"),
		Labels			= cms.untracked.PSet(
			HBHEDigi		= cms.untracked.InputTag("hcalDigis"),
			HFDigi			= cms.untracked.InputTag("utcaDigis"),
			HODigi			= cms.untracked.InputTag("hcalDigis"),
			RAW				= cms.untracked.InputTag("rawDataCollector"),
			HBHERecHit		= cms.untracked.InputTag("hbhereco"),
			HFRecHit		= cms.untracked.InputTag("hfreco"),
			HORecHit		= cms.untracked.InputTag("horeco"),
			L1GT			= cms.untracked.InputTag("l1GtUnpack"),
			HLTResults		= cms.untracked.InputTag("TriggerResults"),
			DCS				= cms.untracked.InputTag("scalersRawToDigi"),
			UnpackerReport	= cms.untracked.InputTag("hcalDigis")
		),
	),
	Labels				= cms.untracked.PSet(
		HBHEDigi		= cms.untracked.InputTag("hcalDigis"),
		HFDigi			= cms.untracked.InputTag("utcaDigis"),
		HODigi			= cms.untracked.InputTag("hcalDigis"),
		RAW				= cms.untracked.InputTag("rawDataCollector"),
		HBHERecHit		= cms.untracked.InputTag("hbhereco"),
		HFRecHit		= cms.untracked.InputTag("hfreco"),
		HORecHit		= cms.untracked.InputTag("horeco"),
		L1GT			= cms.untracked.InputTag("l1GtUnpack"),
		HLTResults		= cms.untracked.InputTag("TriggerResults"),
		DCS				= cms.untracked.InputTag("scalersRawToDigi"),
		UnpackerReport	= cms.untracked.InputTag("")
	),
	MEs					= cms.untracked.PSet(
		me1			= cms.untracked.PSet(
			path	= cms.untracked.string("Hcal/HcalDQSourceEx/"),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("Example ME1"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("me1")
			)
		),
		me2			= cms.untracked.PSet(
			path	= cms.untracked.string("Hcal/HcalDQSourceEx/"),
			kind	= cms.untracked.string("TH1D"),
			desc	= cms.untracked.string("Example ME2"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(10),
				min		= cms.untracked.double(0.),
				max		= cms.untracked.double(10.),
				title	= cms.untracked.string("me2")
			)	
		),
		me3			= cms.untracked.PSet(
			path	= cms.untracked.string("Hcal/HcalDQSourceEx/"),
			kind	= cms.untracked.string("TH2D"),
			desc	= cms.untracked.string("Example ME3"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(100),
				min		= cms.untracked.double(-50),
				max		= cms.untracked.double(50),
				title	= cms.untracked.string("me3-X")
			),
			yaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(100),
				min		= cms.untracked.double(-50),
				max		= cms.untracked.double(50),
				title	= cms.untracked.string("me3-Y")
			)
		),
		me4			= cms.untracked.PSet(
			path	= cms.untracked.string("Hcal/HcalDQSourceEx/"),
			kind	= cms.untracked.string("PROF"),
			desc	= cms.untracked.string("Example ME4"),
			xaxis	= cms.untracked.PSet(
				edges	= cms.untracked.bool(False),
				nbins	= cms.untracked.int32(200),
				min		= cms.untracked.double(-100),
				max		= cms.untracked.double(100),
				title	= cms.untracked.string("me4-X")
			),
			yaxis	= cms.untracked.PSet(
				wnbins	= cms.untracked.bool(True),
				nbins	= cms.untracked.int32(100),
				min		= cms.untracked.double(-50),
				max		= cms.untracked.double(50),
				title	= cms.untracked.string("me4-Y")
			)
		)
	)
)
