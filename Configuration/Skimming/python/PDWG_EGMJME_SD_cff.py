import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
EGMJME = hlt.hltHighLevel.clone()
EGMJME.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
EGMJME.HLTPaths = cms.vstring(
    'HLT_Photon110EB_TightID_TightIso*',
    'HLT_Photon30EB_TightID_TightIso*',
    'HLT_Photon90_R9Id90_HE10_IsoM*', 
    'HLT_Photon75_R9Id90_HE10_IsoM*',
    'HLT_Photon50_R9Id90_HE10_IsoM*',
    'HLT_Photon200*')
EGMJME.andOr = cms.bool( True )
# we want to intentionally throw and exception
# in case it does not match one of the HLT Paths
EGMJME.throw = cms.bool( False )
