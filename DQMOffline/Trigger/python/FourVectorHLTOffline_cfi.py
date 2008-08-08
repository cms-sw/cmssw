import FWCore.ParameterSet.Config as cms

# $Id: FourVectorHLTOffline.cfi,v 1.2 2008/06/24 19:35:43 berryhil Exp $
hltResults = cms.EDFilter("FourVectorHLTOffline",
    plotAll = cms.untracked.bool(False),
    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),
    filters = cms.VPSet(),
    # this is I think MC
    #InputTag triggerSummaryLabel = "hltTriggerSummaryAOD::HLT"
    # this is data (CRUZET I or II best guess)
    triggerSummaryLabel = cms.InputTag("triggerSummaryProducerAOD","","FU")
)



