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
            inputFileName = cms.FileInPath('RecoBTag/SoftLepton/data/SoftPFElectron_BDT.weights.xml.gz'),
            inputFileType = cms.string("XML"),
            inputVariables = cms.vstring("sip3d", "sip2d", "ptRel", "deltaR", "ratio", "mva_e_pi"),
            spectatorVariables = cms.vstring(),
            methodName = cms.string("BDT"),
            outputFileType = cms.string("SQLLite"),
            outputRecord = cms.string("btag_SoftPFElectron_BDT_TMVAv420_74X_v1")
        ),
        cms.PSet(
            inputFileName = cms.FileInPath('RecoBTag/SoftLepton/data/SoftPFMuon_BDT.weights.xml.gz'),
            inputFileType = cms.string("XML"),
            inputVariables = cms.vstring("TagInfo1.sip3d", "TagInfo1.sip2d", "TagInfo1.ptRel", "TagInfo1.deltaR", "TagInfo1.ratio"),
            spectatorVariables = cms.vstring(),
            methodName = cms.string("BDT"),
            outputFileType = cms.string("SQLLite"),
            outputRecord = cms.string("btag_SoftPFMuon_BDT_TMVAv420_74X_v1")
        )
    )
)

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:btag_SoftPFLepton_BDT_TMVAv420_GBRForest_74X_v1.db'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('btag_SoftPFElectron_BDT_TMVAv420_74X_v1'),
            tag = cms.string('btag_SoftPFElectron_BDT_TMVAv420_74X_v1'),
            label = cms.untracked.string('btag_SoftPFElectron_BDT')
        ),
        cms.PSet(
            record = cms.string('btag_SoftPFMuon_BDT_TMVAv420_74X_v1'),
            tag = cms.string('btag_SoftPFMuon_BDT_TMVAv420_74X_v1'),
            label = cms.untracked.string('btag_SoftPFMuon_BDT')
        )
    )
)

process.p = cms.Path(process.gbrForestWriter)
