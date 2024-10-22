import FWCore.ParameterSet.Config as cms

process = cms.Process("NonTkPointSkim")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    files = cms.untracked.PSet(
        output = cms.untracked.PSet(
            extension = cms.untracked.string('txt')
        )
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring(
		'file:scratch/867948E3-6CC4-DD11-9DB5-0019B9E7CD78.root'
	)
)

process.filter = cms.EDFilter('EcalTangentFilter',
	MuLabel = cms.string("muonsBarrelOnly"),
   MuD0Min = cms.double(129),
	MuD0Max = cms.double(152),
	Verbose = cms.bool(False)
)

process.p1 = cms.Path(process.filter)

process.out = cms.OutputModule("PoolOutputModule",
	SelectEvents = cms.untracked.PSet(
		SelectEvents = cms.vstring('p1')
	),
	fileName = cms.untracked.string('EcalTangentSkim.root')
)

process.o = cms.EndPath(process.out)
