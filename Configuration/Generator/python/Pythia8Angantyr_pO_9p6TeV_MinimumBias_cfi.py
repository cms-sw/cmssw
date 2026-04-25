import FWCore.ParameterSet.Config as cms

generator = cms.EDFilter("Pythia8ConcurrentGeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    AngantyrInitialState = cms.PSet(),
    comEnergy = cms.double(9600.),
    PythiaParameters = cms.PSet(
        processParameters = cms.vstring(
            # Op collisions (oxygen in the positive Z direction in CMS)
            'Beams:idA = 1000080160',
            'Beams:idB = 2212',
            'Beams:frameType = 2',
            'Beams:eA = 3400',
            'Beams:eB = 6800',
            
            # HI Angatyr setup (https://www.pythia.org/latest-manual/htmldoc/examples/main422.html)
            'SoftQCD:inelastic = on',
            'HeavyIon:SigFitErr = 0.02,0.02,0.1,0.05,0.05,0.0,0.1,0.0',
            'HeavyIon:SigFitNGen = 20',
            'HeavyIon:SigFitDefPar = 2.15,17.24,0.33',
            
            # Harmonic Oscillator Shell model (Oxygen geometry)
            'Angantyr:NucleusModelA = 3',
            
            #Forward tune (https://arxiv.org/pdf/2309.08604)
            'BeamRemnants:dampPopcorn=0',
            'BeamRemnants:hardRemnantBaryon=on',
            'BeamRemnants:aRemnantBaryon=0.36',
            'BeamRemnants:bRemnantBaryon=1.69',
            'BeamRemnants:primordialKTsoft=0.58',
            'BeamRemnants:primordialKThard=1.8',
            'BeamRemnants:halfScaleForKT=10',
            'BeamRemnants:halfMassForKT=1',
            'BeamRemnants:primordialKTremnant=0.58',
        ),
        parameterSets = cms.vstring('processParameters')
    )
)
