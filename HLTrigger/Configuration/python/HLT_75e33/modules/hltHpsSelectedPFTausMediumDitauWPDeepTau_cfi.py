import FWCore.ParameterSet.Config as cms

hltHpsSelectedPFTausMediumDitauWPDeepTau = cms.EDFilter( "PFTauSelector",
    src = cms.InputTag( "hltHpsPFTauProducer" ),
    cut = cms.string( "pt > 35 && abs(eta) < 2.1" ),
    discriminators = cms.VPSet( 
    ),
    discriminatorContainers = cms.VPSet( 
      cms.PSet(  discriminator = cms.InputTag( "hltHpsPFTauDeepTauProducer", "VSjet" ),
        rawValues = cms.vstring(  ),
        selectionCuts = cms.vdouble(  ),
        workingPoints = cms.vstring( 'double t1 = 0.649, t2 = 0.441, t3 = 0.05, x1 = 35, x2 = 100, x3 = 300; if (pt <= x1) return t1; if (pt >= x3) return t3; if (pt < x2) return (t2 - t1) / (x2 - x1) * (pt - x1) + t1; return (t3 - t2) / (x3 - x2) * (pt - x2) + t2;' )
      )
    )
)
