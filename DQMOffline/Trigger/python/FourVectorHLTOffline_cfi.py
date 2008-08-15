import FWCore.ParameterSet.Config as cms

# $Id: FourVectorHLTOffline_cfi.py,v 1.2 2008/08/15 17:50:32 berryhil Exp $
hltResults = cms.EDFilter("FourVectorHLTOffline",
    plotAll = cms.untracked.bool(False),
    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),
    paths = cms.VPSet(
             cms.PSet(
              pathname = cms.string("HLTJet30"),
              filtername = cms.InputTag("hlt1jet30","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLTJet50"),
              filtername = cms.InputTag("hlt1jet50","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLTJet80"),
              filtername = cms.InputTag("hlt1jet80","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLTJet110"),
              filtername = cms.InputTag("hlt1jet110","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLTJet180"),
              filtername = cms.InputTag("hlt1jet180regional","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLTJet250"),
              filtername = cms.InputTag("hlt1jet250","","HLT"),
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



