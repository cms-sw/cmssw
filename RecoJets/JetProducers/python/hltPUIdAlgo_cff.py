import FWCore.ParameterSet.Config as cms

####################################################################################################################  
full_74x_wp  = cms.PSet(
                #4 Eta Categories  0-2.5 2.5-2.75 2.75-3.0 3.0-5.0

                #Tight Id
                Pt010_Tight    = cms.vdouble(-0.83,-0.81,-0.74,-0.81),
                Pt1020_Tight   = cms.vdouble(-0.83,-0.81,-0.74,-0.81),
                Pt2030_Tight   = cms.vdouble( 0.73, 0.05,-0.26,-0.42),
                Pt3050_Tight   = cms.vdouble( 0.73, 0.05,-0.26,-0.42),

                #Medium Id
                Pt010_Medium   = cms.vdouble(-0.83,-0.92,-0.90,-0.92),
                Pt1020_Medium  = cms.vdouble(-0.83,-0.92,-0.90,-0.92),
                Pt2030_Medium  = cms.vdouble( 0.10,-0.36,-0.54,-0.54),
                Pt3050_Medium  = cms.vdouble( 0.10,-0.36,-0.54,-0.54),

                #Loose Id
                Pt010_Loose    = cms.vdouble(-0.95,-0.96,-0.94,-0.95),
                Pt1020_Loose   = cms.vdouble(-0.95,-0.96,-0.94,-0.95),
                Pt2030_Loose   = cms.vdouble(-0.63,-0.60,-0.55,-0.45),
                Pt3050_Loose   = cms.vdouble(-0.63,-0.60,-0.55,-0.45),

                )


full_74x = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 tmvaWeights = cms.FileInPath("RecoJets/JetProducers/data/MVAJetPuID.weights_newNames.xml.gz"),
 tmvaMethod  = cms.string("BDTG"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "rho"     ,
    "nParticles"     , 
    "nCharged" , 
    "majW" , 
    "minW",	
    "frac01",
    "frac02",
    "frac03",
    "frac04",		 
    "ptD"      , 
    "beta"   , 
    "betaStar"   , 
    "dR2Mean"   , 
    "pull"   , 
    "jetR"   , 
    "jetRchg"	
    ),
 tmvaSpectators = cms.vstring(
    "jetEta",
    "jetPt",
    ),
 JetIdParams = full_74x_wp,
 label = cms.string("CATEv0")
 )




