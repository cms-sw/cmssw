import FWCore.ParameterSet.Config as cms

L1EGammaClusterEmuProducer = cms.EDProducer("L1EGCrystalClusterEmulatorProducer",
   ecalTPEB = cms.InputTag("simEcalEBTriggerPrimitiveDigis"),
   hcalTP = cms.InputTag("simHcalTriggerPrimitiveDigis"),
   calib = cms.PSet(

           etaBins = cms.vdouble( 0.087 , 0.174 , 0.261 , 0.348 , 0.435 , 0.522 , 0.609 , 0.696 , 0.783 , 0.870 , 0.957 , 1.044 , 1.131 , 1.218 , 1.305 , 1.392 , 1.479),
           ptBins = cms.vdouble( 12, 20, 30, 40, 55, 90, 1e6),
           scale = cms.vdouble( 
                  # pt < 12
                   1.18*1.1 ,1.17*1.1 ,1.19*1.1 ,1.18*1.1 ,1.19*1.1 ,1.19*1.1 ,1.19*1.1 ,1.18*1.1 ,1.19*1.1 ,1.18*1.1 ,1.19*1.1 ,1.19*1.1 ,1.19*1.1 ,1.20*1.1 ,1.19*1.1 ,1.20*1.1 ,1.19*1.1,
                  # pt < 20
                   1.14*1.03 ,1.13*1.03 ,1.13*1.03 ,1.13*1.03 ,1.13*1.03 ,1.13*1.03 ,1.13*1.03 ,1.14*1.03 ,1.14*1.03 ,1.13*1.03 ,1.13*1.03 ,1.14*1.03 ,1.13*1.03 ,1.13*1.03 ,1.14*1.03 ,1.14*1.03 ,1.12*1.03,
                  # pt < 30
                   1.11 ,1.11 ,1.11 ,1.11 ,1.11 ,1.11 ,1.11 ,1.11 ,1.11 ,1.11 ,1.11 ,1.11 ,1.11 ,1.11 ,1.11 ,1.11 ,1.10,
                  # pt < 40
                   1.09 ,1.09 ,1.09 ,1.09 ,1.09 ,1.09 ,1.09 ,1.09 ,1.09 ,1.09 ,1.09 ,1.09 ,1.09 ,1.09 ,1.09 ,1.09 ,1.09,
                   # pt < 55
                   1.07 ,1.07 ,1.07 ,1.07 ,1.07 ,1.07 ,1.07 ,1.08 ,1.07 ,1.07 ,1.08 ,1.08 ,1.07 ,1.08 ,1.08 ,1.08 ,1.08,
                   # pt < 90
                   1.06 ,1.06 ,1.06 ,1.06 ,1.05 ,1.05 ,1.06 ,1.06 ,1.06 ,1.06 ,1.06 ,1.06 ,1.06 ,1.06 ,1.06 ,1.06 ,1.06,
                   # pt < 1e6
                   1.04 ,1.04 ,1.04 ,1.04 ,1.05 ,1.04 ,1.05 ,1.05 ,1.05 ,1.05 ,1.05 ,1.05 ,1.05 ,1.05 ,1.05 ,1.05 ,1.05,
                  
           ),

   ),
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(L1EGammaClusterEmuProducer,
    ecalTPEB = cms.InputTag("DMEcalEBTriggerPrimitiveDigis"),
    hcalTP = cms.InputTag("DMHcalTriggerPrimitiveDigis"),
)
