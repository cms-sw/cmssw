import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *

from  RecoLocalTracker.SiStripZeroSuppression.siStripZeroSuppression_cfi import siStripZeroSuppression
siStripZeroSuppression = siStripZeroSuppression.clone(
    Algorithms = DefaultAlgorithms,
    RawDigiProducersList = [ ("siStripDigis","VirginRaw"),
                             ("siStripDigis","ProcessedRaw"),
                             ("siStripDigis","ScopeMode"),
                             # ("siStripDigis","ZeroSuppressed")
                            ],
    storeCM = True,
    fixCM = False,                # put -999 into CM collection for "inspected" APV
    produceRawDigis = True,
    produceCalculatedBaseline = False,
    produceBaselinePoints = False,
    storeInZScollBadAPV = True, # it selects if in the ZS collection the bad APVs are written. To be kept for ZS
    produceHybridFormat = False
)

# The SiStripClusters are not used anymore in phase2 tracking
# This part has to be clean up when they will be officially removed from the entire flow
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(siStripZeroSuppression, # FIXME
  RawDigiProducersList = [ 'simSiStripDigis:VirginRaw',
                           'simSiStripDigis:ProcessedRaw',
                           'simSiStripDigis:ScopeMode' ]
)

# For the HI RAW' workflow
siStripZeroSuppressionHLT = siStripZeroSuppression.clone(
    RawDigiProducersList =[("hltSiStripRawToDigi","VirginRaw"), ("hltSiStripRawToDigi","ProcessedRaw"), ("hltSiStripRawToDigi","ScopeMode")]
)
    
