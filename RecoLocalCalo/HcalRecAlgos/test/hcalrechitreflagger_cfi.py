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
                                                              PET_EnergyThreshShort=cms.untracked.vdouble([129.9,-6.61,0.1153]),
                                                              PET_ETThreshShort=cms.untracked.vdouble([0]),
                                                              PET_ratiocut = cms.vdouble([0.98]) # parameterized as polynomial of E, not ieta!!
                                                              ),
                                   
                                   hf_S9S1_params = cms.PSet( S9S1_EnergyThreshLong=cms.untracked.vdouble([162.4,-10.19,0.21]),
                                                              S9S1_ETThreshLong=cms.untracked.vdouble([0]),
                                                              # No explicit cuts on S9S1 energies?
                                                              S9S1_EnergyThreshShort=cms.untracked.vdouble([129.9,-6.61,0.1153]),
                                                              S9S1_ETThreshShort=cms.untracked.vdouble([0]),
                                                              S9S1_optimumslope=cms.vdouble([-9999,     # ieta=29
                                                                                             0.0164905, # ieta=30
                                                                                             0.0238698, # ieta=31
                                                                                             0.0321383,
                                                                                             0.041296,
                                                                                             0.0513428, # ieta=34
                                                                                             0.0622789,
                                                                                             0.0741041,
                                                                                             0.0868186, # ieta=37
                                                                                             0.100422,
                                                                                             0.135313,
                                                                                             0.136289,  # ieta=40
                                                                                             0.0589927  # ieta=41
                                                                                             ]),
                                                              ),

                                     
                                     debug          = cms.untracked.int32(0)
)
