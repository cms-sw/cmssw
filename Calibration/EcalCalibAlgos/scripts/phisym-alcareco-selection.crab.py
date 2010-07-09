#
# Python config file to drive phi symmetry et sum accumulation
# From real or simulated alcaraw (output of hlt)
#
#

import FWCore.ParameterSet.Config as cms


process=cms.Process("PHISYM")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/StandardSequences/GeometryPilot2_cff')

process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')


process.load('FWCore/MessageService/MessageLogger_cfi')
process.MessageLogger.cerr = cms.untracked.PSet(placeholder =
cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100000) # every 100th only
#    limit = cms.untracked.int32(10)       # or limit to 10 printouts...
    ))



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/Commissioning10/AlCaPhiSymEcal/ALCARECO/v9/000/135/407/34806B9F-1F5E-DF11-926F-000423D98B5C.root'
)
)




process.phisymcalib = cms.EDAnalyzer("PhiSymmetryCalibration", 
    ecalRecHitsProducer = cms.string("ecalPhiSymCorrected"),
    barrelHitCollection = cms.string("phiSymEcalRecHitsEB"),
    endcapHitCollection = cms.string("phiSymEcalRecHitsEE"),
    eCut_barrel = cms.double(0.250),
    eCut_endcap = cms.double(0.000),
    ap = cms.double(-0.150),
    am = cms.double(-0.150),
    b  = cms.double( 0.600),
    eventSet = cms.int32(1),
    statusThreshold = cms.untracked.int32(0)
  )


process.L1T1coll=process.hltLevel1GTSeed.clone()
process.L1T1coll.L1TechTriggerSeeding = cms.bool(True)
process.L1T1coll.L1GtReadoutRecordTag = cms.InputTag("hltGtDigis")

#original
process.L1T1coll.L1SeedsLogicalExpression = cms.string('0 AND (40 OR 41  ) AND NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))')


process.GlobalTag.globaltag = 'GR10_P_V4::All' 

process.p = cms.Path(process.L1T1coll*process.phisymcalib)
