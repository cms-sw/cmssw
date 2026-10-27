import FWCore.ParameterSet.Config as cms

# HGCal layer clustering (CLUEstering) for EE, FH (HSi) and BH (HSci).
#
# Each detector runs the same chain:
#   hltHgcalSoARecHitsProducer<Det>              (HGCRecHit -> HGCalSoARecHits SoA)
#   hltHgcalSoARecHitsLayerClustersProducer<Det> (CLUEstering clustering)
#   hltHgcalSoALayerClustersProducer<Det>        (build the CaloCluster SoA)
#
from ..modules.hltHgcalSoARecHitsProducer_cfi import *
from ..modules.hltHgcalSoARecHitsLayerClustersProducer_cfi import *
from ..modules.hltHgcalSoALayerClustersProducer_cfi import *
from ..modules.hltHgcalSoARecHitsProducerHSi_cfi import *
from ..modules.hltHgcalSoARecHitsLayerClustersProducerHSi_cfi import *
from ..modules.hltHgcalSoALayerClustersProducerHSi_cfi import *
from ..modules.hltHgcalSoARecHitsProducerHSci_cfi import *
from ..modules.hltHgcalSoARecHitsLayerClustersProducerHSci_cfi import *
from ..modules.hltHgcalSoALayerClustersProducerHSci_cfi import *

HLTHgcalLayerClustersSequence = cms.Sequence(
        hltHgcalSoARecHitsProducer+
        hltHgcalSoARecHitsLayerClustersProducer+
        hltHgcalSoALayerClustersProducer+
        hltHgcalSoARecHitsProducerHSi+
        hltHgcalSoARecHitsLayerClustersProducerHSi+
        hltHgcalSoALayerClustersProducerHSi+
        hltHgcalSoARecHitsProducerHSci+
        hltHgcalSoARecHitsLayerClustersProducerHSci+
        hltHgcalSoALayerClustersProducerHSci)

# Dedicated CPU ('SerialSync') instances, used by the alpakaValidationHLT
# GPU-vs-CPU comparison.
HLTHgcalLayerClustersSequenceSerialSync = cms.Sequence(
        hltHgcalSoARecHitsProducerSerialSync+
        hltHgcalSoARecHitsLayerClustersProducerSerialSync+
        hltHgcalSoALayerClustersProducerSerialSync+
        hltHgcalSoARecHitsProducerHSiSerialSync+
        hltHgcalSoARecHitsLayerClustersProducerHSiSerialSync+
        hltHgcalSoALayerClustersProducerHSiSerialSync+
        hltHgcalSoARecHitsProducerHSciSerialSync+
        hltHgcalSoARecHitsLayerClustersProducerHSciSerialSync+
        hltHgcalSoALayerClustersProducerHSciSerialSync)
