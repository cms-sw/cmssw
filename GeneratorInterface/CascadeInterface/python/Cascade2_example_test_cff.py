import FWCore.ParameterSet.Config as cms

from GeneratorInterface.CascadeInterface.Cascade2Parameters_cfi import Cascade2Parameters as Cascade2ParametersRef

source = cms.Source("EmptySource")

generator = cms.EDFilter('Cascade2GeneratorFilter',
                         PythiaParameters = cms.PSet(
                               processParameters = cms.vstring('PMAS(4,1) = 1.6   ! charm mass',
                                                               'PMAS(5,1) = 4.75  ! bottom mass',
                                                               'PMAS(25,1) = 125. ! higgs mass',
                                                               'PARU(112) = 0.2 ! lambda QCD set A0',
                                                               'MSTU(111) = 1   ! = 0 : alpha_s is fixed at the value PARU(111), = 1 : first-order running alpha_s, = 2 : second-order running alpha_s',
                                                               'MSTU(112) = 4   ! nr of flavours wrt lambda_QCD',
                                                               'MSTU(113) = 3   ! min nr of flavours for alphas',
                                                               'MSTU(114) = 5   ! max nr of flavours for alphas',
                                                               'MSTJ(48) = 1    ! (D = 0) 0 = no max. angle, 1 = max angle def. in PARJ(85)'),
                               parameterSets = cms.vstring('processParameters')
                                                    ),
                         
                         comEnergy = cms.double(7000.0),
                         crossSection = cms.untracked.double(-1),
                         crossSectionError = cms.untracked.double(-1),
                         filterEfficiency = cms.untracked.double(1),
                         maxEventsToPrint = cms.untracked.int32(2),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         
                         Cascade2Parameters = Cascade2ParametersRef
                         )

ProductionFilterSequence = cms.Sequence(generator)
