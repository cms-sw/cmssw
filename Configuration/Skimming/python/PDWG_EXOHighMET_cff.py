import FWCore.ParameterSet.Config as cms


## select events with at least one good PV
from RecoMET.METFilters.primaryVertexFilter_cfi import *

## select events with high MET dependent on PF and Calo MET Conditions
CondMETSelectorEXOHighMETSkim = cms.EDProducer(
   "CandViewShallowCloneCombiner",
   decay = cms.string("pfMet caloMetM"),
   cut = cms.string(" (daughter(0).pt > 200) || (daughter(0).pt/daughter(1).pt > 2 && daughter(1).pt > 150 ) || (daughter(1).pt/daughter(0).pt > 2 && daughter(0).pt > 150 )  " )
   )

CondMETCounterEXOHighMETSkim = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("CondMETSelectorEXOHighMETSkim"),
    minNumber = cms.uint32(1),
    )

EXOHighMETSequence = cms.Sequence(
                           primaryVertexFilter*
                           CondMETSelectorEXOHighMETSkim*
                           CondMETCounterEXOHighMETSkim
                           )
