import FWCore.ParameterSet.Config as cms

generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(5360.),
    doProtonPhotonFlux = cms.untracked.bool(True),
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring('pythia8_example02'),
        pythia8_example02 = cms.vstring(
            'HardQCD:all = on',
            'PhaseSpace:pTHatMin = 10.',
            'PhotonParton:all = on',
            'MultipartonInteractions:pT0Ref = 3.0',
            'PDF:beamA2gamma = on',
            'PDF:proton2gammaSet = 0',
            'PDF:useHardNPDFB = on',
            'PDF:gammaFluxApprox2bMin = 13.272',
            'PDF:beam2gammaApprox = 2',
            'Photon:sampleQ2 = off',
            'Photon:ProcessType = 3'
        )
    )
)
ProductionFilterSequence = cms.Sequence(generator)
