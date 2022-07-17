import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

# LuminosityBlock flat table producers
ecalPhiSymRecHitLumiTableEB = cms.EDProducer("EcalPhiSymRecHitFlatTableProducerLumi",
                                             src  = cms.InputTag("EcalPhiSymRecHitProducerLumi", "EB"),
                                             cut  = cms.string(""), 
                                             name = cms.string("EcalPhiSymEB"),
                                             doc  = cms.string("ECAL RecHits with information summed over a LS"),
                                             variables = cms.PSet(
                                                 id       = Var("rawId()", int, doc="ECAL PhiSym rechits: channel detector id"),
                                                 status   = Var("statusCode()", int, doc="ECAL PhiSym rechits: channel status"),
                                                 nHits    = Var("nHits()", int, doc="ECAL PhiSym rechits: number of recorded hits"),
                                                 sumEt    = Var("sumEt(0)", float, doc="ECAL PhiSym rechits: nominal et", precision=23),
                                                 sumEt2   = Var("sumEt2()", float, doc="ECAL PhiSym rechits: sum et^2", precision=23),
                                                 sumLC    = Var("lcSum()", float, doc="ECAL PhiSym rechits: laser correction", precision=23),
                                                 sumLC2   = Var("lc2Sum()", float, doc="ECAL PhiSym rechits: sum lc^2", precision=23)
                                             )
                                     )

ecalPhiSymRecHitLumiTableEE = ecalPhiSymRecHitLumiTableEB.clone(
    src  = cms.InputTag("EcalPhiSymRecHitProducerLumi", "EE"),
    name = cms.string("EcalPhiSymEE")
)                                                    
# iring is saved only for EE channels
setattr(ecalPhiSymRecHitLumiTableEE.variables, 'ring', Var("eeRing()", int, doc="ECAL PhiSym rechits: EE channel ring index"))

ecalPhiSymInfoLumiTable = cms.EDProducer("EcalPhiSymInfoFlatTableProducerLumi",
                                         src  = cms.InputTag("EcalPhiSymRecHitProducerLumi"),
                                         cut  = cms.string(""), 
                                         name = cms.string("EcalPhiSymInfo"),
                                         doc  = cms.string("Global phisym info with information summed over one or more LS"),
                                         variables = cms.PSet(
                                             hitsEB = Var("totHitsEB()", int, doc="Total number of rechits in EB"),
                                             hitsEE = Var("totHitsEE()", int, doc="Total number of rechits in EE"),
                                             nEvents = Var("nEvents()", int, doc="Total number of events recorded"),
                                             nLumis = Var("nLumis()", int, doc="Total number of lumis recorded"),
                                             fill = Var("fillNumber()", int, doc="LHC fill number"),
                                             delivLumi = Var("delivLumi()", float, doc="LHC delivered integrated luminosity"),
                                             recLumi = Var("recLumi()", float, doc="CMS recorded integrated luminosity"),
                                             nMis = Var("nMis()", float, doc="Number of mis-calibration steps injected (nominal value excluded)"),
                                             minMisEB = Var("minMisEB()", float, doc="Minimum mis-calibration value in EB"),
                                             maxMisEB = Var("maxMisEB()", float, doc="Maximum mis-calibration value in EB"),
                                             minMisEE = Var("minMisEE()", float, doc="Minimum mis-calibration value in EE"),
                                             maxMisEE = Var("maxMisEE()", float, doc="Maximum mis-calibration value in EE")))


# Run flat table producers
ecalPhiSymRecHitRunTableEB = cms.EDProducer("EcalPhiSymRecHitFlatTableProducerRun",
                                            src  = cms.InputTag("EcalPhiSymRecHitProducerRun", "EB"),
                                            cut  = cms.string(""), 
                                            name = cms.string("EcalPhiSymEB"),
                                            doc  = cms.string("ECAL RecHits with information summed over a Run"),
                                            variables = cms.PSet()
                                        )
ecalPhiSymRecHitRunTableEB.variables = ecalPhiSymRecHitLumiTableEB.variables

ecalPhiSymRecHitRunTableEE = ecalPhiSymRecHitRunTableEB.clone(
    src  = cms.InputTag("EcalPhiSymRecHitProducerRun", "EE"),
    name = cms.string("EcalPhiSymEE")
)                                                    
# iring is saved only for EE channels
setattr(ecalPhiSymRecHitRunTableEE.variables, 'ring', Var("eeRing()", int, doc="ECAL PhiSym rechits: EE channel ring index"))

ecalPhiSymInfoRunTable = cms.EDProducer("EcalPhiSymInfoFlatTableProducerRun",
                                        src  = cms.InputTag("EcalPhiSymRecHitProducerRun"),
                                        cut  = cms.string(""), 
                                        name = cms.string("EcalPhiSymInfo"),
                                        doc  = cms.string("Global phisym info with information summed over a run"),
                                        variables = cms.PSet()
                                    )

ecalPhiSymInfoRunTable.variables = ecalPhiSymInfoLumiTable.variables

