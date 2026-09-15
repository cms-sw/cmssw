import FWCore.ParameterSet.Config as cms

from ..modules.hltHgcalSoARecHitsProducerL1Seeded_cfi import *
from ..modules.hltHgcalSoARecHitsLayerClustersProducerL1Seeded_cfi import *
from ..modules.hltHgcalSoALayerClustersProducerL1Seeded_cfi import *
from ..modules.hltHgCalLayerClustersFromSoAProducerL1Seeded_cfi import *
from ..modules.hltHgcalSoARecHitsProducerHSiL1Seeded_cfi import *
from ..modules.hltHgcalSoARecHitsLayerClustersProducerHSiL1Seeded_cfi import *
from ..modules.hltHgcalSoALayerClustersProducerHSiL1Seeded_cfi import *
from ..modules.hltHgCalLayerClustersFromSoAProducerHSiL1Seeded_cfi import *
from ..modules.hltHgcalSoARecHitsProducerHSciL1Seeded_cfi import *
from ..modules.hltHgcalSoARecHitsLayerClustersProducerHSciL1Seeded_cfi import *
from ..modules.hltHgcalSoALayerClustersProducerHSciL1Seeded_cfi import *
from ..modules.hltHgCalLayerClustersFromSoAProducerHSciL1Seeded_cfi import *

HLTHgcalLayerClustersL1SeededSequence = cms.Sequence(
        hltHgcalSoARecHitsProducerL1Seeded+
        hltHgcalSoARecHitsLayerClustersProducerL1Seeded+
        hltHgcalSoALayerClustersProducerL1Seeded+
        hltHgCalLayerClustersFromSoAProducerL1Seeded+
        hltHgcalSoARecHitsProducerHSiL1Seeded+
        hltHgcalSoARecHitsLayerClustersProducerHSiL1Seeded+
        hltHgcalSoALayerClustersProducerHSiL1Seeded+
        hltHgCalLayerClustersFromSoAProducerHSiL1Seeded+
        hltHgcalSoARecHitsProducerHSciL1Seeded+
        hltHgcalSoARecHitsLayerClustersProducerHSciL1Seeded+
        hltHgcalSoALayerClustersProducerHSciL1Seeded+
        hltHgCalLayerClustersFromSoAProducerHSciL1Seeded)
