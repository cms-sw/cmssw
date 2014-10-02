import FWCore.ParameterSet.Config as cms

Njettiness = cms.EDProducer("NjettinessAdder",
                            src=cms.InputTag("ak8PFJetsCHS"),
                            Njets=cms.vuint32(1,2,3),            # compute 1-, 2-, 3- subjettiness
                            # variables for measure definition : 
                            measureDefinition = cms.uint32( 1 ), # default is unnormalized measure
                            beta = cms.double(-999.0),           # not used by default
                            R0 = cms.double( -999.0 ),           # not used by default
                            Rcutoff = cms.double( -999.0),       # not used by default
                            # variables for axes definition :
                            axesDefinition = cms.uint32( 6 ),    # default is 1-pass KT axes
                            nPass = cms.int32(-999),             # not used by default
                            akAxesR0 = cms.double(-999.0)        # not used by default
                            )
