import FWCore.ParameterSet.Config as cms

eidCutBasedClassBased = cms.EDProducer("EleIdCutBasedRef",

    src = cms.InputTag("gedGsfElectrons"),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    algorithm = cms.string('eIDClassBased'),

    electronQuality = cms.string('Eff95Cuts'),

   # Golden EB, BigBrem EB, Showering EB, Golden EE, BigBrem EE, Showering EE, Crack EB, Crack EE
    
   Eff95Cuts = cms.PSet(
    deltaEtaIn          = cms.vdouble(0.0128, 0.0029, 0.0169, 0.0314, 0.0065, 0.1825, 0.0119, 0.0137),
    sigmaIetaIetaMin    = cms.vdouble(-1e+30, -1e+30, -1e+30, -1e+30, -1e+30, -1e+30, -1e+30, -1e+30),
    sigmaIetaIetaMax    = cms.vdouble(0.0135, 0.0098, 0.0168, 0.0449, 0.0386, 0.0702, 0.0170, 0.0444),
    HoverE              = cms.vdouble(0.1863, 0.0448, 0.2903, 0.0631, 0.0407, 0.0903, 0.1273, 3.4388),
    EoverPOutMin        = cms.vdouble(-1e+30, -1e+30, 0.0834, -1e+30, -1e+30, -1e+30, -1e+30, 0.1480),
    EoverPOutMax        = cms.vdouble( 1e+30,  1e+30,  1e+30,  1e+30,  1e+30,  1e+30,  1e+30,  1e+30),
    deltaPhiInChargeMin = cms.vdouble(-0.0139,-1e+30, -1e+30,-0.0760,-0.0768, -1e+30, -1e+30, -1e+30),
    deltaPhiInChargeMax = cms.vdouble( 1e+30, 0.0575, 0.1267,  1e+30,  1e+30, 0.1664, 0.0586, 0.0889)
    ),

   Eff90Cuts = cms.PSet(
    deltaEtaIn          = cms.vdouble(0.0573, 0.0043, 0.0162, 0.0084,0.00525, 0.9018, 0.0115, 0.0137),
    sigmaIetaIetaMin    = cms.vdouble(-1e+30, -1e+30, -1e+30, -1e+30, -1e+30, -1e+30, -1e+30, -1e+30),
    sigmaIetaIetaMax    = cms.vdouble(0.0107, 0.0098, 0.0153, 0.0414, 0.0344, 0.0635, 0.0155, 0.0388),
    HoverE              = cms.vdouble(0.1916, 0.0547, 0.1493, 0.0631, 0.0407, 0.0302, 0.0730, 3.4431),
    EoverPOutMin        = cms.vdouble(-1e+30, -1e+30, 0.5070, -1e+30, -1e+30, -1e+30, -1e+30, 0.1175),
    EoverPOutMax        = cms.vdouble(2.4695,  1e+30,  1e+30, 15.179,  1e+30,  1e+30,  1e+30,  1e+30),
    deltaPhiInChargeMin = cms.vdouble(-0.0138,-1e+30, -1e+30,-0.0760,-0.2791, -1e+30, -1e+30, -1e+30),
    deltaPhiInChargeMax = cms.vdouble( 1e+30, 0.0121, 0.0789,  1e+30, 1e+30,  0.6220, 0.0387, 0.0545)
    )

)


