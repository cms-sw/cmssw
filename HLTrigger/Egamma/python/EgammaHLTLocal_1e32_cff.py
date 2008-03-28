import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1GeometryProducers.l1CaloGeometry_cfi import *
from L1TriggerConfig.L1GeometryProducers.l1CaloGeomRecordSource_cff import *
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1seedSingle = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1seedRelaxedSingle = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1seedDouble = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1seedRelaxedDouble = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1seedExclusiveDouble = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1seedSinglePrescaled = copy.deepcopy(hltLevel1GTSeed)
l1seedSingle.L1SeedsLogicalExpression = 'L1_SingleIsoEG12'
l1seedRelaxedSingle.L1SeedsLogicalExpression = 'L1_SingleEG15'
l1seedDouble.L1SeedsLogicalExpression = 'L1_DoubleIsoEG8'
l1seedRelaxedDouble.L1SeedsLogicalExpression = 'L1_DoubleEG10'
l1seedExclusiveDouble.L1SeedsLogicalExpression = 'L1_ExclusiveDoubleIsoEG6'
l1seedSinglePrescaled.L1SeedsLogicalExpression = 'L1_SingleIsoEG10'

