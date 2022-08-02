import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatUtils.bJetOperatingPointsParameters_cfi import *

process = cms.Process("PatBTagAnalyzer")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/00BCD825-6E57-DD11-8C1F-000423D98EA8.root')
)


process.MessageLogger = cms.Service("MessageLogger")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string( autoCond[ 'phase1_2022_realistic' ] )
process.load("Configuration.StandardSequences.MagneticField_cff")


# PAT Layer 1
process.load("PhysicsTools.PatAlgos.patLayer0_cff") # need to load this
process.load("PhysicsTools.PatAlgos.patLayer1_cff") # even if we run only layer 1


process.TFileService = cms.Service("TFileService",
    fileName = cms.string('btagpatanalyzerpy.root')
)
 
# request a summary at the end of the file
process.options = cms.untracked.PSet(
     wantSummary = cms.untracked.bool(True)
)


process.PatBTagAnalyzerTC2 = cms.EDAnalyzer("PatBTagAnalyzer",
    BJetOperatingPointsParameters,
    jetTag = cms.untracked.InputTag("selectedLayer1Jets"),
    BjetTag = cms.PSet(
        verbose = cms.untracked.bool(True),
        tagger = cms.untracked.string('TC2'),
        purity = cms.string('Loose'),
        discriminator = cms.string('trackCountingHighEffBJetTags'),
        maxdiscriminatorcut = cms.untracked.double(30.0),
        mindiscriminatorcut = cms.untracked.double(-10.0)
    )
)


process.PatBTagAnalyzerTC3 = cms.EDAnalyzer("PatBTagAnalyzer",
    BJetOperatingPointsParameters,
    jetTag = cms.untracked.InputTag("selectedLayer1Jets"),
    BjetTag = cms.PSet(
        verbose = cms.untracked.bool(False),
        tagger = cms.untracked.string('TC3'),
        purity = cms.string('Loose'),
        discriminator = cms.string('trackCountingHighPurBJetTags'),
        maxdiscriminatorcut = cms.untracked.double(30.0),
        mindiscriminatorcut = cms.untracked.double(-10.0)
    )
)

process.PatBTagAnalyzerTP = cms.EDAnalyzer("PatBTagAnalyzer",
    BJetOperatingPointsParameters,
    jetTag = cms.untracked.InputTag("selectedLayer1Jets"),
    BjetTag = cms.PSet(
        verbose = cms.untracked.bool(False),
        tagger = cms.untracked.string('TP'),
        purity = cms.string('Loose'),
        discriminator = cms.string('jetProbabilityBJetTags'),
        maxdiscriminatorcut = cms.untracked.double(2.6),
        mindiscriminatorcut = cms.untracked.double(-0.1)
    )
)

process.PatBTagAnalyzerBTP = cms.EDAnalyzer("PatBTagAnalyzer",
    BJetOperatingPointsParameters,
    jetTag = cms.untracked.InputTag("selectedLayer1Jets"),
    BjetTag = cms.PSet(
        verbose = cms.untracked.bool(False),
        tagger = cms.untracked.string('BTP'),
        purity = cms.string('Loose'),
        discriminator = cms.string('jetBProbabilityBJetTags'),
        maxdiscriminatorcut = cms.untracked.double(8.1),
        mindiscriminatorcut = cms.untracked.double(-0.1)
    )
)
process.PatBTagAnalyzerSSV = cms.EDAnalyzer("PatBTagAnalyzer",
    BJetOperatingPointsParameters,
    jetTag = cms.untracked.InputTag("selectedLayer1Jets"),
    BjetTag = cms.PSet(
        verbose = cms.untracked.bool(False),
        tagger = cms.untracked.string('SSV'),
        purity = cms.string('Loose'),
        discriminator = cms.string('simpleSecondaryVertexBJetTags'),
        maxdiscriminatorcut = cms.untracked.double(8.0),
        mindiscriminatorcut = cms.untracked.double(0.0)
    )
)

process.PatBTagAnalyzerCSV = cms.EDAnalyzer("PatBTagAnalyzer",
    BJetOperatingPointsParameters,
   jetTag = cms.untracked.InputTag("selectedLayer1Jets"),
    BjetTag = cms.PSet(
        verbose = cms.untracked.bool(False),
        tagger = cms.untracked.string('CSV'),
        purity = cms.string('Loose'),
        discriminator = cms.string('combinedSecondaryVertexBJetTags'),
        maxdiscriminatorcut = cms.untracked.double(1.1),
        mindiscriminatorcut = cms.untracked.double(-0.1)
    )
)

process.PatBTagAnalyzerMSV = cms.EDAnalyzer("PatBTagAnalyzer",
    BJetOperatingPointsParameters,
    jetTag = cms.untracked.InputTag("selectedLayer1Jets"),
    BjetTag = cms.PSet(
        verbose = cms.untracked.bool(False),
        tagger = cms.untracked.string('MSV'),
        purity = cms.string('Loose'),
        discriminator = cms.string('combinedSecondaryVertexMVABJetTags'),
        maxdiscriminatorcut = cms.untracked.double(1.1),
        mindiscriminatorcut = cms.untracked.double(-0.1)
    )
)

process.PatBTagAnalyzerIPM = cms.EDAnalyzer("PatBTagAnalyzer",
    BJetOperatingPointsParameters,
    jetTag = cms.untracked.InputTag("selectedLayer1Jets"),
    BjetTag = cms.PSet(
        verbose = cms.untracked.bool(False),
        tagger = cms.untracked.string('IPM'),
        purity = cms.string('Loose'),
        discriminator = cms.string('impactParameterMVABJetTags'),
        maxdiscriminatorcut = cms.untracked.double(1.1),
        mindiscriminatorcut = cms.untracked.double(-0.1)
    )
)

process.PatBTagAnalyzerSET = cms.EDAnalyzer("PatBTagAnalyzer",
    BJetOperatingPointsParameters,
    jetTag = cms.untracked.InputTag("selectedLayer1Jets"),
    BjetTag = cms.PSet(
        verbose = cms.untracked.bool(False),
        tagger = cms.untracked.string('SET'),
        purity = cms.string('Loose'),
        discriminator = cms.string('softElectronBJetTags'),
        maxdiscriminatorcut = cms.untracked.double(1.1),
        mindiscriminatorcut = cms.untracked.double(-0.1)
    )
)

process.PatBTagAnalyzerSMT = cms.EDAnalyzer("PatBTagAnalyzer",
    BJetOperatingPointsParameters,
    jetTag = cms.untracked.InputTag("selectedLayer1Jets"),
    BjetTag = cms.PSet(
        verbose = cms.untracked.bool(False),
        tagger = cms.untracked.string('SMT'),
        purity = cms.string('Loose'),
        discriminator = cms.string('softMuonBJetTags'),
        maxdiscriminatorcut = cms.untracked.double(1.1),
        mindiscriminatorcut = cms.untracked.double(-0.1)
    )
)

process.PatBTagAnalyzerSMNIPT = cms.EDAnalyzer("PatBTagAnalyzer",
    BJetOperatingPointsParameters,
    jetTag = cms.untracked.InputTag("selectedLayer1Jets"),
    BjetTag = cms.PSet(
        verbose = cms.untracked.bool(False),
        tagger = cms.untracked.string('SMNIPT'),
        purity = cms.string('Loose'),
        discriminator = cms.string('softMuonNoIPBJetTags'),
        maxdiscriminatorcut = cms.untracked.double(1.1),
        mindiscriminatorcut = cms.untracked.double(-0.1)
    )
)


process.p = cms.Path(
    process.patLayer0 *
    process.patLayer1 *
    process.PatBTagAnalyzerTC2 *
    process.PatBTagAnalyzerTC3 *
    process.PatBTagAnalyzerBTP *
    process.PatBTagAnalyzerSSV *
    process.PatBTagAnalyzerCSV *
    process.PatBTagAnalyzerMSV *
    process.PatBTagAnalyzerIPM *
    process.PatBTagAnalyzerSET *
    process.PatBTagAnalyzerSMT *
    process.PatBTagAnalyzerSMNIPT *
    process.PatBTagAnalyzerTP
    )

