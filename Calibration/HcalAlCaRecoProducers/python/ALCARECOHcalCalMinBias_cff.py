import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL minbias:
#------------------------------------------------
import HLTrigger.HLTfilters.hltHighLevel_cfi

hcalminbiasHLT =  HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#    HLTPaths = ['AlCa_HcalPhiSym'],
    eventSetupPathsKey='HcalCalMinBias',
    throw = False #dont throw except on unknown path name
)

seqALCARECOHcalCalMinBias = cms.Sequence(hcalminbiasHLT)
