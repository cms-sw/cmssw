import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

# LuminosityBlock flat table producers
ecalPhiSymRecHitLumiTableEB = cms.EDProducer("EcalPhiSymRecHitFlatTableProducerLumi",
                                             src  = cms.InputTag("EcalPhiSymRecHitProducerLumi", "EB"),
                                             cut  = cms.string(""), 
                                             name = cms.string("EcalPhiSymEB"),
                                             doc  = cms.string("ECAL RecHits with information summed over a LS"),
                                             variables = cms.PSet(
                                                 id       = Var("GetRawId()", int, doc="ECAL PhiSym rechits: channel detector id"),
                                                 status   = Var("GetStatusCode()", int, doc="ECAL PhiSym rechits: channel status"),
                                                 nHits    = Var("GetNhits()", int, doc="ECAL PhiSym rechits: number of recorded hits"),
                                                 sumEt    = Var("GetSumEt(0)", float, doc="ECAL PhiSym rechits: nominal et", precision=23),
                                                 sumEt2   = Var("GetSumEt2()", float, doc="ECAL PhiSym rechits: sum et^2", precision=23),
                                                 sumLC    = Var("GetLCSum()", float, doc="ECAL PhiSym rechits: laser correction", precision=23),
                                                 sumLC2   = Var("GetLC2Sum()", float, doc="ECAL PhiSym rechits: sum lc^2", precision=23)
                                             )
                                     )

ecalPhiSymRecHitLumiTableEE = ecalPhiSymRecHitLumiTableEB.clone(
    src  = cms.InputTag("EcalPhiSymRecHitProducerLumi", "EE"),
    name = cms.string("EcalPhiSymEE")
)                                                    

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
                                             delivLumi = Var("delivLumi()", int, doc="LHC delivered integrated luminosity"),
                                             recLumi = Var("recLumi()", int, doc="CMS recorded integrated luminosity")))

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

ecalPhiSymInfoRunTable = cms.EDProducer("EcalPhiSymInfoFlatTableProducerRun",
                                        src  = cms.InputTag("EcalPhiSymRecHitProducerRun"),
                                        cut  = cms.string(""), 
                                        name = cms.string("EcalPhiSymInfo"),
                                        doc  = cms.string("Global phisym info with information summed over a run"),
                                        variables = cms.PSet()
                                    )

ecalPhiSymInfoRunTable.variables = ecalPhiSymInfoLumiTable.variables

