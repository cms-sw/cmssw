import FWCore.ParameterSet.Config as cms

TagProbeFitTreeAnalyzer = cms.EDAnalyzer("TagProbeFitTreeAnalyzer",
    # IO parameters:
    InputFileNames = cms.vstring("testNewWrite.root"),
    InputDirectoryName = cms.string("MakeHisto"),
    InputTreeName = cms.string("fitter_tree"),
    OutputFileName = cms.string("testNewAnalyzer.root"),
    #numbrer of CPUs to use for fitting
    NumCPU = cms.uint32(8),
    # specifies wether to save the RooWorkspace containing the data for each bin and
    # the pdf object with the initial and final state snapshots
    SaveWorkspace = cms.bool(True),

    # defines all the real variables of the probes available in the input tree and intended for use in the efficiencies
    Variables = cms.PSet(
        mass = cms.vstring("Tag-Probe Mass", "2.5", "3.8", "GeV/c^{2}"),
        pt = cms.vstring("Probe p_{T}", "0", "1000", "GeV/c"),
        eta = cms.vstring("Probe #eta", "-2.5", "2.5", "")
    ),

    # defines all the discrete variables of the probes available in the input tree and intended for use in the efficiency calculations
    Categories = cms.PSet(
        mcTrue = cms.vstring("MC true", "dummy[true=1,false=0]"),
        passing = cms.vstring("isMuon", "dummy[pass=1,fail=0]")
    ),

    # defines all the PDFs that will be available for the efficiency calculations; uses RooFit's "factory" syntax;
    # each pdf needs to define "signal", "backgroundPass", "backgroundFail" pdfs, "efficiency[0.9,0,1]" and "signalFractionInPassing[0.9]" are used for initial values  
    PDFs = cms.PSet(
        gaussPlusLinear = cms.vstring(
            "Gaussian::signal(mass, mean[3.1,3.0,3.2], sigma[0.03,0.01,0.05])",
            "Chebychev::backgroundPass(mass, cPass[0,-1,1])",
            "Chebychev::backgroundFail(mass, cFail[0,-1,1])",
            "efficiency[0.9,0,1]",
            "signalFractionInPassing[0.9]"
        ),
        gaussPlusQuadratic = cms.vstring(
            "Gaussian::signal(mass, mean[3.1,3.0,3.2], sigma[0.03,0.01,0.05])",
            "Chebychev::backgroundPass(mass, {cPass1[0,-1,1], cPass2[0,-1,1]})",
            "Chebychev::backgroundFail(mass, {cFail1[0,-1,1], cFail2[0,-1,1]})",
            "efficiency[0.9,0,1]",
            "signalFractionInPassing[0.9]"
        )
    ),

    # defines a set of efficiency calculations, what PDF to use for fitting and how to bin the data;
    # there will be a separate output directory for each calculation that includes a simultaneous fit, side band subtraction and counting. 
    Efficiencies = cms.PSet(
        #the name of the parameter set becomes the name of the directory
        pt = cms.PSet(
            #specifies the efficiency of which category and state to measure 
            EfficiencyCategoryAndState = cms.vstring("passing","pass"),
            #specifies what unbinned variables to include in the dataset, the mass is needed for the fit
            UnbinnedVariables = cms.vstring("mass"),
            #specifies the binning of parameters
            BinnedVariables = cms.PSet(
                pt = cms.vdouble(3.5, 4.5, 6.0, 8.0, 50.0)
            ),
            #first string is the default followed by binRegExp - PDFname pairs
            BinToPDFmap = cms.vstring("gaussPlusLinear", "*pt_bin0*", "gaussPlusQuadratic")
        ),
        pt_mcTrue = cms.PSet(
            EfficiencyCategoryAndState = cms.vstring("passing","pass"),
            UnbinnedVariables = cms.vstring("mass"),
            BinnedVariables = cms.PSet(
                mcTrue = cms.vstring("true"),
                pt = cms.vdouble(3.5, 4.5, 6.0, 8.0, 50.0)
            )
            #unspecified binToPDFmap means no fitting
        ),
        pt_eta = cms.PSet(
            EfficiencyCategoryAndState = cms.vstring("passing","pass"),
            UnbinnedVariables = cms.vstring("mass"),
            BinnedVariables = cms.PSet(
                pt = cms.vdouble(3.5, 4.5, 6.0, 8.0, 50.0),
                eta = cms.vdouble(-2.1, -1.2, 0.0, 1.2, 2.1)
            ),
            BinToPDFmap = cms.vstring("gaussPlusLinear", "*pt_bin0*", "gaussPlusQuadratic")
        ),
        pt_eta_mcTrue = cms.PSet(
            EfficiencyCategoryAndState = cms.vstring("passing","pass"),
            UnbinnedVariables = cms.vstring("mass"),
            BinnedVariables = cms.PSet(
                mcTrue = cms.vstring("true"),
                pt = cms.vdouble(3.5, 4.5, 6.0, 8.0, 50.0),
                eta = cms.vdouble(-2.1, -1.2, 0.0, 1.2, 2.1)
            )
        )
    )
)

