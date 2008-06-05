import FWCore.ParameterSet.Config as cms

# $Id: FourVectorHLT.cfi,v 1.2 2008/05/26 07:59:36 wittich Exp $
hltResults = cms.EDFilter("FourVectorHLT",
    plotAll = cms.untracked.bool(False),
    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),
    filters = cms.VPSet(),
    # this is I think MC
    #InputTag triggerSummaryLabel = "hltTriggerSummaryAOD::HLT"
    # this is data (CRUZET I or II best guess)
    triggerSummaryLabel = cms.InputTag("triggerSummaryProducerAOD","","FU")
)


