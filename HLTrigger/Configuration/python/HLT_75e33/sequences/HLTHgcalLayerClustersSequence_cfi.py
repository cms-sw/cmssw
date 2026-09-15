import FWCore.ParameterSet.Config as cms

# HGCal layer clustering (CLUEstering) for EE, FH (HSi) and BH (HSci).
#
# Each detector runs the same chain:
#   hltHgcalSoARecHitsProducer<Det>              (HGCRecHit -> HGCalSoARecHits SoA)
#   hltHgcalSoARecHitsLayerClustersProducer<Det> (CLUEstering clustering)
#   hltHgcalSoALayerClustersProducer<Det>        (build the CaloCluster SoA)
#   hltHgCalLayerClustersFromSoAProducer<Det>    (SoA -> legacy reco::CaloCluster)
#
from ..modules.hltHgcalSoARecHitsProducer_cfi import *
from ..modules.hltHgcalSoARecHitsLayerClustersProducer_cfi import *
from ..modules.hltHgcalSoALayerClustersProducer_cfi import *
from ..modules.hltHgcalLayerClustersFromSoAProducer_cfi import *
from ..modules.hltHgcalSoARecHitsProducerHSi_cfi import *
from ..modules.hltHgcalSoARecHitsLayerClustersProducerHSi_cfi import *
from ..modules.hltHgcalSoALayerClustersProducerHSi_cfi import *
from ..modules.hltHgCalLayerClustersFromSoAProducerHSi_cfi import *
from ..modules.hltHgcalSoARecHitsProducerHSci_cfi import *
from ..modules.hltHgcalSoARecHitsLayerClustersProducerHSci_cfi import *
from ..modules.hltHgcalSoALayerClustersProducerHSci_cfi import *
from ..modules.hltHgCalLayerClustersFromSoAProducerHSci_cfi import *

HLTHgcalLayerClustersSequence = cms.Sequence(
        hltHgcalSoARecHitsProducer+
        hltHgcalSoARecHitsLayerClustersProducer+
        hltHgcalSoALayerClustersProducer+
        hltHgCalLayerClustersFromSoAProducer+
        hltHgcalSoARecHitsProducerHSi+
        hltHgcalSoARecHitsLayerClustersProducerHSi+
        hltHgcalSoALayerClustersProducerHSi+
        hltHgCalLayerClustersFromSoAProducerHSi+
        hltHgcalSoARecHitsProducerHSci+
        hltHgcalSoARecHitsLayerClustersProducerHSci+
        hltHgcalSoALayerClustersProducerHSci+
        hltHgCalLayerClustersFromSoAProducerHSci)

# Dedicated CPU ('SerialSync') instances, used by the alpakaValidationHLT
# GPU-vs-CPU comparison.
HLTHgcalLayerClustersSequenceSerialSync = cms.Sequence(
        hltHgcalSoARecHitsProducerSerialSync+
        hltHgcalSoARecHitsLayerClustersProducerSerialSync+
        hltHgcalSoALayerClustersProducerSerialSync+
        hltHgCalLayerClustersFromSoAProducerSerialSync+
        hltHgcalSoARecHitsProducerHSiSerialSync+
        hltHgcalSoARecHitsLayerClustersProducerHSiSerialSync+
        hltHgcalSoALayerClustersProducerHSiSerialSync+
        hltHgCalLayerClustersFromSoAProducerHSiSerialSync+
        hltHgcalSoARecHitsProducerHSciSerialSync+
        hltHgcalSoARecHitsLayerClustersProducerHSciSerialSync+
        hltHgcalSoALayerClustersProducerHSciSerialSync+
        hltHgCalLayerClustersFromSoAProducerHSciSerialSync)
