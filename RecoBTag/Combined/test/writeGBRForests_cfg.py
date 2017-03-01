import FWCore.ParameterSet.Config as cms

process = cms.Process("writeGBRForests")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1) # NB: needs to be set to 1 so that GBRForestWriter::analyze method gets called exactly once
)

process.source = cms.Source("EmptySource")

process.load('Configuration/StandardSequences/Services_cff')

process.gbrForestWriter = cms.EDAnalyzer("GBRForestWriter",
    jobs = cms.VPSet(
        cms.PSet(
            inputFileName = cms.FileInPath('RecoBTag/Combined/data/CombinedMVAV2_13_07_2015.weights.xml.gz'),
            inputFileType = cms.string("XML"),
            inputVariables = cms.vstring("Jet_CSV", "Jet_CSVIVF", "Jet_JP", "Jet_JBP", "Jet_SoftMu", "Jet_SoftEl"),
            spectatorVariables = cms.vstring(),
            methodName = cms.string("BDT"),
            outputFileType = cms.string("SQLLite"),
            outputRecord = cms.string("btag_CombinedMVAv2_BDT_TMVAv420_74X_v1")
        )
    )
)

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:btag_CombinedMVAv2_BDT_TMVAv420_GBRForest_74X_v1.db'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('btag_CombinedMVAv2_BDT_TMVAv420_74X_v1'),
            tag = cms.string('btag_CombinedMVAv2_BDT_TMVAv420_74X_v1'),
            label = cms.untracked.string('btag_CombinedMVAv2_BDT')
        )
    )
)

process.p = cms.Path(process.gbrForestWriter)
