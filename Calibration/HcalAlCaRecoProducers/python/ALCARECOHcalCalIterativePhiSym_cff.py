import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL Iterative Phi Symmetry:
#------------------------------------------------

from Calibration.HcalAlCaRecoProducers.alcaiterphisym_cfi import *

import HLTrigger.HLTfilters.triggerResultsFilterFromDB_cfi
hcalphisymHLT = HLTrigger.HLTfilters.triggerResultsFilterFromDB_cfi.triggerResultsFilterFromDB.clone(
    eventSetupPathsKey='HcalCalIterativePhiSym',
    usePathStatus = False,
    hltResults = 'TriggerResults::HLT',
    l1tResults = '', # leaving empty (not interested in L1T results)
    throw = False #dont throw except on unknown path name
)

seqALCARECOHcalCalIterativePhiSym = cms.Sequence(hcalphisymHLT*IterativePhiSymProd)

