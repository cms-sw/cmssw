import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8PhotonFluxSettings_cfi import PhotonFlux_PbPb

_generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(5360.),
    PhotonFlux = PhotonFlux_PbPb,
    PythiaParameters = cms.PSet(
        pythia8_example02 = cms.vstring('HardQCD:all = on',
                                        'PhaseSpace:pTHatMin = 10.',
                                        'PhotonParton:all = on',#Added from main70
                                        'MultipartonInteractions:pT0Ref = 3.0',#Added from main70
                                        'PDF:beamA2gamma = on',#Added from main70
                                        'PDF:proton2gammaSet = 0',#Added from main70
                                        'PDF:useHardNPDFB = on',
                                        'PDF:gammaFluxApprox2bMin = 13.272',
                                        'PDF:beam2gammaApprox = 2',
                                        'Photon:sampleQ2 = off'
                                    ), 
        parameterSets = cms.vstring('pythia8_example02')
    )
)


from GeneratorInterface.Core.ExternalGeneratorFilter import ExternalGeneratorFilter
generator = ExternalGeneratorFilter(_generator)
