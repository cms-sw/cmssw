import FWCore.ParameterSet.Config as cms

hcalrechitReflagger = cms.EDProducer('HcalRecHitReflagger',

                                   hfInputLabel       = cms.untracked.InputTag("hfreco"),
                                   hfFlagBit          = cms.untracked.int32(31),  # bit 0 is UserDefinedBit0; see RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h for full list
                                   hfBitAlwaysOn      = cms.untracked.bool(False),
                                   hfBitAlwaysOff     = cms.untracked.bool(False),
                                   hf_Algo3test       = cms.untracked.bool(True),
                                   hf_Algo2test       = cms.untracked.bool(False),
                                   hf_Algo3_AND_Algo2 = cms.untracked.bool(False),
                                   hf_Algo3_OR_Algo2  = cms.untracked.bool(False),
                                   
                                   # Parameterize energy, ET cuts as polynomials in |ieta|:  E> [0] + [1]*ieta + [2]*pow(ieta,2) + ...

                                   hf_PET_params  = cms.PSet( PET_EnergyThreshLong=cms.untracked.vdouble([162.4,-10.19,0.21]),
                                                              PET_ETThreshLong=cms.untracked.vdouble([0]),
                                                              PET_EnergyThreshShort=cms.untracked.vdouble([130,-6.61,0.1153]),
                                                              PET_ETThreshShort=cms.untracked.vdouble([0]),
                                                              PET_ratiocut = cms.vdouble([0.98]) # parameterized as polynomial of E, not ieta!!
                                                              ),
                                   
                                   hf_S9S1_params = cms.PSet( S9S1_EnergyThreshLong=cms.untracked.vdouble([162.4,-10.19,0.21]),
                                                              S9S1_ETThreshLong=cms.untracked.vdouble([0]),
                                                              # No explicit cuts on S9S1 energies?
                                                              S9S1_EnergyThreshShort=cms.untracked.vdouble([130,-6.61,0.1153]),
                                                              S9S1_ETThreshShort=cms.untracked.vdouble([0]),
                                                              S9S1_optimumslope=cms.vdouble([0.095, 0.062, 0.2974,-0.02498, 0.0005243]) # this is the parameterization of optimum slope vs. ieta.  First two values are the slopes at |ieta| = 40 & 41, followed by the polynomial parameterization of slope with ieta
                                                              ),
                                       
                                   debug          = cms.untracked.int32(0)
)
