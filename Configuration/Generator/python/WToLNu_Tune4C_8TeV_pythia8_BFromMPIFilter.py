import FWCore.ParameterSet.Config as cms



generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(8000.),
    PythiaParameters = cms.PSet(
        processParameters = cms.vstring(
   		'Main:timesAllowErrors = 10000',
    		'ParticleDecays:limitTau0 = on',
    		'ParticleDecays:tauMax = 10',
    		'Tune:ee 3',
    		'Tune:pp 5',
    
    		'WeakSingleBoson:ffbar2W = on',
    		'24:onMode = off',
    		'24:onIfAny = 11 12 13 14 15 16',
	),
        parameterSets = cms.vstring('processParameters')
    )
)


BFilter = cms.EDFilter("HeavyQuarkFromMPIFilter",
	filterAlgoPSet = cms.PSet(
		HQFlavour = cms.int32(5),
                CategoryToVeto= cms.int32(0),
	        genParSource = cms.InputTag("genParticlesForFilter")
	)
)

genParticlesForFilter = cms.EDProducer("GenParticleProducer",
    saveBarCodes = cms.untracked.bool(True),
    src = cms.InputTag("generator", "unsmeared"),
    abortOnUnknownPDGCode = cms.untracked.bool(False)
)


# enter below the configuration metadata (only a description is needed, the rest is filled in by cvs)
configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/Configuration/Generator/python/QCD_Pt_80_170_BCtoE_8TeV.py,v $'),
    annotation = cms.untracked.string('b-from-mpi filtered Pythia8 hadronizer')
)

#ProductionFilterSequence = cms.Sequence(generator )
ProductionFilterSequence = cms.Sequence(generator * (genParticlesForFilter + BFilter))
