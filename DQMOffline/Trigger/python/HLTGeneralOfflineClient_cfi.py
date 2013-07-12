# $Id: FourVectorHLTOfflineClient_cfi.py,v 1.16 2010/08/04 14:27:51 rekovic Exp $
import FWCore.ParameterSet.Config as cms

hltGeneralClient = cms.EDAnalyzer("GeneralHLTClient",
    hltClientDir = cms.untracked.string('HLT/General/paths/'),
    hltSourceDir = cms.untracked.string('HLT/General/paths/'),
    prescaleLS = cms.untracked.int32(-1),
    prescaleEvt = cms.untracked.int32(1),
    processname = cms.untracked.string("HLT"),
    customEffDir = cms.untracked.string('custom-eff')


)

