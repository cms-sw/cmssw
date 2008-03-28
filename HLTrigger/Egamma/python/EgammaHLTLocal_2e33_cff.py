import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1GeometryProducers.l1CaloGeometry_cfi import *
from L1TriggerConfig.L1GeometryProducers.l1CaloGeomRecordSource_cff import *
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1seedSingle = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
#module  l1seedRelaxedSingle = hltLevel1GTSeed from "HLTrigger/HLTfilters/data/hltLevel1GTSeed.cfi"
#replace l1seedRelaxedSingle.L1SeedsLogicalExpression = "L1_SingleEG15"
l1seedDouble = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1seedRelaxedDouble = copy.deepcopy(hltLevel1GTSeed)
l1seedSingle.L1SeedsLogicalExpression = 'L1_SingleIsoEG20'
l1seedDouble.L1SeedsLogicalExpression = 'L1_DoubleIsoEG10'
l1seedRelaxedDouble.L1SeedsLogicalExpression = 'L1_DoubleEG15'

