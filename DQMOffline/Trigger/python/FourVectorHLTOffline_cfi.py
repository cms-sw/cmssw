import FWCore.ParameterSet.Config as cms

# $Id: FourVectorHLTOffline_cfi.py,v 1.1 2008/08/08 06:21:52 berryhil Exp $
hltResults = cms.EDFilter("FourVectorHLTOffline",
    plotAll = cms.untracked.bool(False),
    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),
    filters = cms.VPSet(
             cms.PSet(
              name = cms.InputTag("hlt1jet30","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              name = cms.InputTag("hlt1jet80","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(200.0)
             )

             ),
                          
    # this is I think MC
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT")
    # this is data (CRUZET I or II best guess)
    #triggerSummaryLabel = cms.InputTag("triggerSummaryProducerAOD","","FU")
)



