import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevel_cfi

ecalTrgHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#    eventSetupPathsKey='AlCa_EcalPhiSym*', # this is the HLT path that can be used                                                                                                          
     eventSetupPathsKey='EcalTrg',
     throw = cms.bool( False ),
)

seqALCARECOEcalTrg = cms.Sequence(ecalTrgHLT)
