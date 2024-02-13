import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.l1trig_cff import *


l1CaloTPTable = cms.EDProducer("CaloTPTableProducer",
                               ecalTPsSrc = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
                               ecalTPsName = cms.string("EcalUnpackedTPs"),
                               hcalTPsSrc = cms.InputTag("hcalDigis"),
                               hcalTPsName = cms.string("HcalUnpackedTPs")
)


l1EmulCaloTPTable = cms.EDProducer("CaloTPTableProducer",
    ecalTPsSrc = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    ecalTPsName = cms.string("EcalEmulTPs"),
    hcalTPsSrc = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    hcalTPsName = cms.string("HcalEmulTPs")
)


l1CaloTowerTable = cms.EDProducer("SimpleTriggerL1CaloTowerFlatTableProducer",
                                  src = cms.InputTag("caloStage2Digis","CaloTower"),
                                  minBX = cms.int32(0),
                                  maxBX = cms.int32(0),
                                  cut = cms.string("hwPt>0"),
                                  name = cms.string("L1UnpackedCaloTower"),
                                  doc = cms.string(""),
                                  extension = cms.bool(False),
                                  variables = cms.PSet(
                                      et = Var("pt()","int",doc=""),
                                      eta = Var("eta()","int",doc=""),
                                      phi = Var("phi()","int",doc=""),
                                      iet = Var("hwPt()","int",doc=""),
                                      ieta = Var("hwEta()","int",doc=""),
                                      iphi = Var("hwPhi()","int",doc=""),
                                      iem = Var("hwEtEm()","int",doc=""),
                                      ihad = Var("hwEtHad()","int",doc=""),
                                      iratio = Var("hwEtRatio()","int",doc=""),
                                      iqual = Var("hwQual()","int",doc="")
                                )
)

l1EmulCaloTowerTable = l1CaloTowerTable.clone(
    src = cms.InputTag("simCaloStage2Layer1Digis"),
    name = cms.string("L1EmulCaloTower")
)

l1EmulCaloClusterTable = cms.EDProducer("SimpleTriggerL1CaloClusterFlatTableProducer",
                                        src = cms.InputTag("simCaloStage2Digis", "MP"),
                                        minBX = cms.int32(0),
                                        maxBX = cms.int32(0),
                                        cut = cms.string(""),
                                        name= cms.string("L1EmulCaloCluster"),
                                        doc = cms.string(""),
                                        extension = cms.bool(False),
                                        variables = cms.PSet(
                                            et = Var("pt()","int",doc=""),
                                            eta = Var("eta()","int",doc=""),
                                            phi = Var("phi()","int",doc=""),
                                            iet = Var("hwPt()","int",doc=""),
                                            ieta = Var("hwEta()","int",doc=""),
                                            iphi = Var("hwPhi()","int",doc=""),
                                            iqual = Var("hwQual()","int",doc="")
                                        )
)



l1CaloTPsNanoTask = cms.Task(l1CaloTPTable)
l1CaloLayer1NanoTask = cms.Task(l1CaloTowerTable)
l1EmulCaloTPsNanoTask = cms.Task(l1EmulCaloTPTable)
l1EmulCaloLayer1NanoTask = cms.Task(l1EmulCaloTowerTable,l1EmulCaloClusterTable)


#Now L1 emulated objects 
#Cloning the L1 object tables producers used for central NANO (unpacked objects)

l1EmulMuTable = l1MuTable.clone(
    src = cms.InputTag("simGmtStage2Digis"),
    name= cms.string("L1EmulMu"),
)

l1EmulJetTable = l1JetTable.clone(
    src = cms.InputTag("simCaloStage2Digis"),
    name= cms.string("L1EmulJet"),
) 

l1EmulTauTable = l1TauTable.clone(
    src = cms.InputTag("simCaloStage2Digis"),
    name= cms.string("L1EmulTau"),
)

l1EmulEtSumTable = l1EtSumTable.clone(
    src = cms.InputTag("simCaloStage2Digis"),
    name= cms.string("L1EmulEtSum"),
)

l1EmulEGTable = l1EGTable.clone(
    src = cms.InputTag("simCaloStage2Digis"),
    name= cms.string("L1EmulEG"),
)

l1EmulObjTablesTask = cms.Task(l1EmulEGTable,l1EmulEtSumTable,l1EmulTauTable,l1EmulJetTable,l1EmulMuTable)
