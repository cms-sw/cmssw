import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL minbias:
#------------------------------------------------
import HLTrigger.HLTfilters.hltHighLevel_cfi
from Calibration.HcalAlCaRecoProducers.alcadijets_cfi import *


dijetsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#    HLTPaths = ['HLT_Jet30','HLT_Jet50','HLT_Jet80',     #1E31 Menu
#                'HLT_Jet15U','HLT_Jet30U','HLT_Jet50U'], #8E29 Menu
    eventSetupPathsKey='HcalCalDijets',
    throw = False
)


seqALCARECOHcalCalDijets = cms.Sequence(dijetsHLT*DiJetsProd)

