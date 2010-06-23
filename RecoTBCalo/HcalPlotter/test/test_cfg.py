import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 500

process.load("RecoTBCalo.HcalPlotter.hcal_tb10_cff")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring( 
"file:/data/data0/spool/EcalHcalCombined2010_00000495.0.root")
                            )

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('tb2_0000495.root')
)


process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi")
process.load("RecoTBCalo.HcalTBObjectUnpacker.HcalTBObjectUnpacker_Normal_cfi")

process.plotanal=cms.EDAnalyzer("HcalQLPlotAnal",
                                hbheRHtag = cms.untracked.InputTag("hbhereco"),
                                hoRHtag   = cms.untracked.InputTag("horeco"),
                                hfRHtag   = cms.untracked.InputTag("hfreco"),
                                hcalDigiTag = cms.untracked.InputTag("hcalDigis"),
                                hcalTrigTag = cms.untracked.InputTag("tbunpack"),
                                HistoParameters = cms.PSet(
        pedGeVlo   = cms.double(-5),
        pedGeVhi   = cms.double(5),
        pedADClo   = cms.double(0),
        pedADChi   = cms.double(49),
        ledGeVlo   = cms.double(-5),
        ledGeVhi   = cms.double(250),
        laserGeVlo = cms.double(-5),
        laserGeVhi = cms.double(350),
        otherGeVlo = cms.double(-5),
        otherGeVhi = cms.double(250),
        beamGeVlo  = cms.double(-5),
        beamGeVhi  = cms.double(350),
        timeNSlo   = cms.double(50),
        timeNShi   = cms.double(250)
      )
   )
 
process.p=cms.Path(process.hcalDigis+process.hbhereco+process.horeco+process.hfreco+process.tbunpack+process.plotanal)
