import FWCore.ParameterSet.Config as cms

# pt selection consistent with preselection on pfJets used by TauProducer
cmgTauSel = cms.EDFilter("CmgTauSelector",
                         src = cms.InputTag( "cmgTau" ),
                         cut = cms.string( "pt() > 15.0" )
                         )




