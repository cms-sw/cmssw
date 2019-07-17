import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff import hgcalConcentratorProducer, hgcalBackEndLayer1Producer, hgcalBackEndLayer2Producer

# 1) concentrator: add STC-version
hgcalConcentratorProducerSTC = hgcalConcentratorProducer.clone()
hgcalConcentratorProducerSTC.ProcessorParameters.Method = 'superTriggerCellSelect'

# 2) layer1 (TC-based and TC-based)
hgcalBackEndLayer1ProducerTC = hgcalBackEndLayer1Producer.clone()
hgcalBackEndLayer1ProducerTC.ProcessorParameters.C2d_parameters.clusterType = 'dummyC2d'
hgcalBackEndLayer1ProducerTC.ProcessorParameters.C2d_parameters.threshold_scintillator = cms.double(0)
hgcalBackEndLayer1ProducerTC.ProcessorParameters.C2d_parameters.threshold_silicon      = cms.double(0)

hgcalBackEndLayer1ProducerSTC = hgcalBackEndLayer1ProducerTC.clone()
hgcalBackEndLayer1ProducerSTC.InputTriggerCells = cms.InputTag("hgcalConcentratorProducerSTC","HGCalConcentratorProcessorSelection")


# 3) layer 2 (TC-based and STC-based)
hgcalBackEndLayer2ProducerTC = hgcalBackEndLayer2Producer.clone()
hgcalBackEndLayer2ProducerTC.ProcessorParameters.C3d_parameters.type_multicluster = 'HistoMaxC3d'
hgcalBackEndLayer2ProducerTC.ProcessorParameters.C3d_parameters.dR_multicluster_byLayer = cms.vdouble(
        [0] + [0.010]*7 + [0.020]*7 + [0.030]*7 + [0.040]*7 +   [0.040]*6 + [0.050]*6  +  [0.050]*12 )
hgcalBackEndLayer2ProducerTC.ProcessorParameters.C3d_parameters.threshold_histo_multicluster = 20
hgcalBackEndLayer2ProducerTC.InputCluster = cms.InputTag("hgcalBackEndLayer1ProducerTC","HGCalBackendLayer1Processor2DClustering")

hgcalBackEndLayer2ProducerSTC = hgcalBackEndLayer2ProducerTC.clone()
hgcalBackEndLayer2ProducerSTC.InputCluster = cms.InputTag("hgcalBackEndLayer1ProducerSTC","HGCalBackendLayer1Processor2DClustering")

hgc3dClustersForPF_STC = cms.Sequence(hgcalConcentratorProducerSTC + hgcalBackEndLayer1ProducerSTC + hgcalBackEndLayer2ProducerSTC)
hgc3dClustersForPF_TC  = cms.Sequence(hgcalConcentratorProducer + hgcalBackEndLayer1ProducerTC + hgcalBackEndLayer2ProducerTC)

