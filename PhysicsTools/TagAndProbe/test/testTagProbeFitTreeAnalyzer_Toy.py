import FWCore.ParameterSet.Config as cms

process = cms.Process("TagProbe")

process.load('FWCore.MessageService.MessageLogger_cfi')

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )    

process.TagProbeFitTreeAnalyzer = cms.EDAnalyzer("TagProbeFitTreeAnalyzer",
    InputFileNames = cms.vstring("testTagProbeFitTreeProducer_Toy.root"),
    InputDirectoryName = cms.string("Test"),
    InputTreeName = cms.string("fitter_tree"),
    OutputFileName = cms.string("testTagProbeFitTreeAnalyzer_Toy.root"),
    NumCPU = cms.uint32(1),
    SaveWorkspace = cms.bool(True),
    Variables = cms.PSet(
        mass = cms.vstring("Tag-Probe Mass", "2.6", "3.6", "GeV/c^{2}"),
        pt = cms.vstring("Probe p_{T}", "0", "100", "GeV/c"),
        eta = cms.vstring("Probe #eta", "-2.4", "2.4", ""),
    ),
    Categories = cms.PSet(
        mcTrue_idx = cms.vstring("MC true", "dummy[true=1,false=0]"),
        passing_idx = cms.vstring("isPassing", "dummy[true=1,false=0]"),
    ),
    PDFs = cms.PSet(
        gaussPlusLinear = cms.vstring(
            "Gaussian::signal(mass, mean[3.1,3.0,3.2], sigma[0.03,0.01,0.05])",
            "Chebychev::backgroundPass(mass, cPass[0,-1,1])",
            "Chebychev::backgroundFail(mass, cFail[0,-1,1])",
            "efficiency[0.9,0,1]",
            "signalFractionInPassing[0.9]"
        ),
    ),
    Efficiencies = cms.PSet(
        pt_eta = cms.PSet(
            EfficiencyCategoryAndState = cms.vstring("passing_idx","true"),
            UnbinnedVariables = cms.vstring("mass"),
            BinnedVariables = cms.PSet(
                pt = cms.vdouble(2.0, 4.0, 6.0, 8.0, 10.0),
                eta = cms.vdouble(-2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4),
            ),
            BinToPDFmap = cms.vstring("gaussPlusLinear"),
        ),
        pt_eta_mcTrue = cms.PSet(
            EfficiencyCategoryAndState = cms.vstring("passing_idx","true"),
            UnbinnedVariables = cms.vstring("mass"),
            BinnedVariables = cms.PSet(
                mcTrue_idx = cms.vstring("true"),
                pt = cms.vdouble(2.0, 4.0, 6.0, 8.0, 10.0),
                eta = cms.vdouble(-2.4, -1.2, 0.0, 1.2, 2.4),
            ),
        ),
    ),
)

process.fitness = cms.Path(
    process.TagProbeFitTreeAnalyzer
)

