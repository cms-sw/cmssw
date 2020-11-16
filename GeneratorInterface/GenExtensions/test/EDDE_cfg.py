import FWCore.ParameterSet.Config as cms

#from Configuration.GenProduction.PythiaUESettings_cfi import "
from Configuration.Generator.PythiaUESettings_cfi import *

process = cms.Process("TEST")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
#process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("Configuration.StandardSequences.Services_cff")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

# The following three lines reduce the clutter of repeated printouts
# of the same exception message.
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = ['cerr']
process.MessageLogger.statistics = []


process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(50))

process.source = cms.Source("LHESource",
    fileNames = cms.untracked.vstring('file:edde.lhe')
)

process.generator = cms.EDFilter("Pythia6HadronizerFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    comEnergy = cms.double(10000.0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL=0         ! User defined processes', 
#=			'MSTJ(1)=1       ! Fragmentation/hadronization on or off',
			'MSTP(61)=1      ! Parton showering on or off',
#			'MSTU(53) = 0            ! no smearing energy',
#			'MSTU(54) = 3            ! massless jets',
			'MSTP(71) =1             ! Final-state QCD and QED radiation',
			'MSTP(81) =1             ! multiple interaction',
			'MSTP(111)=1             ! fragmentation and decay',
			'MSTP(122)=0             ! switch off X section print out',
#...Higgs decay definition...
			'MDME(210,1) =0           ! h0 -> d dbar',
			'MDME(211,1) =0           ! h0 -> u ubar',
                	'MDME(212,1) =0           ! h0 -> s sbar',
			'MDME(213,1) =0           ! h0 -> c cbar',
			'MDME(214,1) =1           ! h0 -> b bbar',
			'MDME(215,1) =0           ! h0 -> t tbar',
			'MDME(216,1) =-1          ! h0 -> bprime bbar',
			'MDME(217,1) =-1          ! h0 -> tprime tbar',
			'MDME(218,1) =0           ! h0 -> e+e-',
			'MDME(219,1) =0           ! h0 -> mu+mu-',
			'MDME(220,1) =0           ! h0 -> tau+tau-',
			'MDME(221,1) =-1          ! h0 -> tauprime+ tauprime-',
			'MDME(222,1) =0           ! h0 ->  gg',
			'MDME(223,1) =0           ! h0-> gamma gamma',
			'MDME(224,1) =0           ! h0 -> gamma Z0',
			'MDME(225,1) =0           ! h0 -> Z0 Z0',
			'MDME(226,1) =0           ! h0 -> W+W-'
	),
	# This is a vector of ParameterSet names to be read, in this order
	parameterSets = cms.vstring('pythiaUESettings', 'processParameters')
    )
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('EDDE.root')
)

process.p = cms.Path(process.generator)
process.p1 = cms.Path(process.randomEngineStateProducer)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.p1, process.outpath)
