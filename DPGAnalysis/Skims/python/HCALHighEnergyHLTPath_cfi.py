
import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel as HCALHighEnergyHLTPath

HCALHighEnergyHLTPath.HLTPaths = cms.vstring( "HLTStoppedHSCPPath" )
