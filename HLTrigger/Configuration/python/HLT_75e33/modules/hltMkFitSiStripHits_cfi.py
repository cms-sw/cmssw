import FWCore.ParameterSet.Config as cms

# MkFitSiStripHits options
hltMkFitSiStripHits = cms.EDProducer("MkFitSiStripHitConverter",
        mightGet = cms.optional.untracked.vstring,
        minGoodStripCharge = cms.PSet(
            refToPSet_ = cms.string('SiStripClusterChargeCutLoose')
        ),
        rphiHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        stereoHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
)
