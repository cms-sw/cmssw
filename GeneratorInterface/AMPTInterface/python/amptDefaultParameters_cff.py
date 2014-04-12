import FWCore.ParameterSet.Config as cms

amptDefaultParameters = cms.PSet(
                        rotateEventPlane = cms.bool(True),
                        amptmode = cms.int32(1),
                        ntmax = cms.int32(150),
                        dt = cms.double(0.2),
                        stringFragA = cms.double(2.2),
                        stringFragB = cms.double(0.5),
                        popcornmode = cms.bool(True),
                        popcornpar = cms.double(1.0),
                        shadowingmode = cms.bool(True),
                        quenchingmode = cms.bool(False),
                        quenchingpar = cms.double(2.0),
                        pthard = cms.double(2.0),
                        mu = cms.double(3.2264),
                        izpc = cms.int32(0),
                        alpha = cms.double(0.47140452),
                        dpcoal = cms.double(1000000.0),
                        drcoal = cms.double(1000000.0),
                        ks0decay = cms.bool(False),
                        phidecay = cms.bool(True),
                        deuteronmode = cms.int32(0),
                        deuteronfactor = cms.int32(1),
                        deuteronxsec = cms.int32(1),          
                        minijetpt = cms.double(-7.0),
                        maxmiss = cms.int32(1000),               
                        doInitialAndFinalRadiation = cms.int32(3),
                        ktkick = cms.int32(1),
                        diquarkembedding = cms.int32(0),
                        diquarkpx = cms.double(7.0),
                        diquarkpy = cms.double(0.0),
                        diquarkx = cms.double(0.0),
                        diquarky = cms.double(0.0)
#                        nsembd = cms.int32(1.0),
#                        psembd = cms.double(5.0),
#                        tmaxembd = cms.double(0.0),
#                        shadowingmodflag = cms.bool(False),
#                        shadowingfactor = cms.double(0.0)
                    )




