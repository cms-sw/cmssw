import FWCore.ParameterSet.Config as cms

TagProbeFitTreeAnalyzer = cms.EDAnalyzer("TagProbeFitTreeAnalyzer",
    # IO parameters:
    InputFileName = cms.untracked.string("testNewWrite.root"),
    InputDirectoryName = cms.untracked.string("MakeHisto"),
    InputTreeName = cms.untracked.string("fitter_tree"),
    OutputFileName = cms.untracked.string("testNewAnalyzer.root"),
    # specifies wether to save the RooWorkspace containing the data for each bin and
    # the pdf object with the initial and final state snapshots
    SaveWorkspace = cms.untracked.bool(True),

    # defines all the real variables of the probes available in the input tree and intended for use in the efficiencies
    Variables = cms.untracked.PSet(
        mass = cms.untracked.vstring("Tag-Probe Mass", "2.5", "3.8", "GeV/c^{2}"),
        pt = cms.untracked.vstring("Probe p_{T}", "0", "1000", "GeV/c"),
        eta = cms.untracked.vstring("Probe #eta", "-2.5", "2.5", "")
    ),

    # defines all the discrete variables of the probes available in the input tree and intended for use in the efficiency calculations
    Categories = cms.untracked.PSet(
        mcTrue = cms.untracked.vstring("MC true","dummy[true=1,false=0]")
    ),

    # defines all the PDFs that will be available for the efficiency calculations; uses RooFit's "factory" syntax;
    # each pdf needs to define "signal", "backgroundPass", "backgroundFail" pdfs, "efficiency[0.9,0,1]" and "signalFractionInPassing[0.9]" are used for initial values  
    PDFs = cms.untracked.PSet(
        gaussPlusLinear = cms.untracked.vstring(
            "Gaussian::signal(mass, mean[3.1,3.0,3.2], sigma[0.03,0.01,0.05])",
            "Chebychev::backgroundPass(mass, cPass[0,-1,1])",
            "Chebychev::backgroundFail(mass, cFail[0,-1,1])",
            "efficiency[0.9,0,1]",
            "signalFractionInPassing[0.9]"
        )
    ),

    # defines a set of efficiency calculations, what PDF to use for fitting and how to bin the data;
    # there will be a separate output directory for each calculation that includes a simultaneous fit, side band subtraction and counting. 
    Efficiencies = cms.untracked.PSet(
        pt = cms.untracked.PSet(
            pdf = cms.untracked.string("gaussPlusLinear"),
            pt = cms.untracked.vdouble(3.5, 4.5, 6.0, 8.0, 50.0)
        ),
        pt_mcTrue = cms.untracked.PSet(
            pdf = cms.untracked.string("gaussPlusLinear"),
            mcTrue = cms.untracked.vstring("true"),
            pt = cms.untracked.vdouble(3.5, 4.5, 6.0, 8.0, 50.0)
        ),
        pt_eta = cms.untracked.PSet(
            pdf = cms.untracked.string("gaussPlusLinear"),
            pt = cms.untracked.vdouble(3.5, 4.5, 6.0, 8.0, 50.0),
            eta = cms.untracked.vdouble(-2.1, -1.2, 0.0, 1.2, 2.1)
        ),
        pt_eta_mcTrue = cms.untracked.PSet(
            pdf = cms.untracked.string("gaussPlusLinear"),
            mcTrue = cms.untracked.vstring("true"),
            pt = cms.untracked.vdouble(3.5, 4.5, 6.0, 8.0, 50.0),
            eta = cms.untracked.vdouble(-2.1, -1.2, 0.0, 1.2, 2.1)
        )
    )
)

