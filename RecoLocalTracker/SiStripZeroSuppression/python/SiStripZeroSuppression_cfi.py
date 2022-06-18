import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *

siStripZeroSuppression = cms.EDProducer("SiStripZeroSuppression",

    Algorithms = DefaultAlgorithms,
    RawDigiProducersList = cms.VInputTag( cms.InputTag('siStripDigis','VirginRaw'), 
                                          cms.InputTag('siStripDigis','ProcessedRaw'),
                                          cms.InputTag('siStripDigis','ScopeMode')),
                                       #   cms.InputTag('siStripDigis','ZeroSuppressed')),

    storeCM = cms.bool(True), 
    fixCM= cms.bool(False),                # put -999 into CM collection for "inspected" APV

    produceRawDigis = cms.bool(True),
    produceCalculatedBaseline = cms.bool(False),
    produceBaselinePoints = cms.bool(False),
    storeInZScollBadAPV = cms.bool(True), # it selects if in the ZS collection the bad APVs are written. To be kept for ZS
    produceHybridFormat = cms.bool(False)
)

# The SiStripClusters are not used anymore in phase2 tracking
# This part has to be clean up when they will be officially removed from the entire flow
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(siStripZeroSuppression, # FIXME
  RawDigiProducersList = [ 'simSiStripDigis:VirginRaw',
                           'simSiStripDigis:ProcessedRaw',
                           'simSiStripDigis:ScopeMode' ]
)

siStripZeroSuppressionHLT = siStripZeroSuppression.clone(
    RawDigiProducersList =[("siStripDigisHLT","VirginRaw"), ("siStripDigisHLT","ProcessedRaw"), ("siStripDigisHLT","ScopeMode")]
)
    
