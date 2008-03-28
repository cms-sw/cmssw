import FWCore.ParameterSet.Config as cms

# Standard KT Jets parameters
# $Id: ExtKtJetParameters.cfi,v 1.2 2007/02/08 01:46:11 fedor Exp $
ExtKtJetParameters = cms.PSet(
    dcut = cms.double(-1.0),
    KtRecom = cms.int32(1),
    njets = cms.int32(-1),
    KtAngle = cms.int32(2),
    PtMin = cms.double(1.0)
)

