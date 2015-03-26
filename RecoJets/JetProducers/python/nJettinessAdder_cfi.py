import FWCore.ParameterSet.Config as cms

Njettiness = cms.EDProducer("NjettinessAdder",
                            src=cms.InputTag("ak8PFJetsCHS"),
                            Njets=cms.vuint32(1,2,3,4),          # compute 1-, 2-, 3-, 4- subjettiness
                            # variables for measure definition : 
                            measureDefinition = cms.uint32( 0 ), # CMS default is normalized measure
                            beta = cms.double(1.0),              # CMS default is 1
                            R0 = cms.double( 0.8 ),              # CMS default is jet cone size
                            Rcutoff = cms.double( -999.0),       # not used by default
                            # variables for axes definition :
                            axesDefinition = cms.uint32( 6 ),    # CMS default is 1-pass KT axes
                            nPass = cms.int32(-999),             # not used by default
                            akAxesR0 = cms.double(-999.0)        # not used by default
                            )
