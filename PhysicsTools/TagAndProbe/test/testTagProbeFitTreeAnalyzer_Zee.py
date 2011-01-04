import FWCore.ParameterSet.Config as cms

process = cms.Process("TagProbe")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.destinations = ['cout', 'cerr']
process.MessageLogger.cerr.FwkReport.reportEvery = 1000


isMC = True
HLTDef = "probe_passingHLT"
PDFName = "pdfSignalPlusBackground"
if isMC:
    PDFName = ""



EfficiencyBinningSpecification = cms.PSet(
    #specifies what unbinned variables to include in the dataset, the mass is needed for the fit
    UnbinnedVariables = cms.vstring("mass"),
    #specifies the binning of parameters
    BinnedVariables = cms.PSet(
    probe_gsfEle_pt = cms.vdouble( 20, 30, 40, 50, 100 ),
    probe_gsfEle_eta = cms.vdouble( -2.5, -1.5, 0, 1.5, 2.5 )
    ),
    #first string is the default followed by binRegExp - PDFname pairs
    BinToPDFmap = cms.vstring(PDFName)
    #BinToPDFmap = cms.vstring()    
)



EfficiencyBinningSpecificationMC = cms.PSet(
    #specifies what unbinned variables to include in the dataset, the mass is needed for the fit
    UnbinnedVariables = cms.vstring("mass"),
    #specifies the binning of parameters
    BinnedVariables = cms.PSet(
    mcTrue = cms.vstring("true"),
    probe_gsfEle_pt = cms.vdouble( 20, 30, 40, 50, 100 ),
    probe_gsfEle_eta = cms.vdouble( -2.5, -1.5, 0, 1.5, 2.5 )
    ),
    BinToPDFmap = cms.vstring()  
)




process.TagProbeFitTreeAnalyzer = cms.EDAnalyzer("TagProbeFitTreeAnalyzer",
    # IO parameters:
    #InputFileNames = cms.vstring("testNewWrite.root"),
    #InputFileNames = cms.vstring("/uscms_data/d2/kalanand/allTPtrees_35invpb.root"),
    InputFileNames = cms.vstring("/uscms_data/d2/kalanand/allTPtrees_mc.root"),
    InputDirectoryName = cms.string("GsfElectronToIdToHLT"),
    InputTreeName = cms.string("fitter_tree"),
    OutputFileName = cms.string("testEfficiency.root"),
    #numbrer of CPUs to use for fitting
    NumCPU = cms.uint32(1),
    # specifies wether to save the RooWorkspace containing the data for each bin and
    # the pdf object with the initial and final state snapshots
    SaveWorkspace = cms.bool(True),
    floatShapeParameters = cms.bool(True),
    #fixVars = cms.vstring("mean"),
                                                 
    # defines all the real variables of the probes available in the input tree and intended for use in the efficiencies
    Variables = cms.PSet(
        mass = cms.vstring("Tag-Probe Mass", "60.0", "120.0", "GeV/c^{2}"),
        probe_gsfEle_pt = cms.vstring("Probe p_{T}", "0", "1000", "GeV/c"),
        probe_gsfEle_eta = cms.vstring("Probe #eta", "-2.5", "2.5", ""),        
        probe_eidCicVeryLoose = cms.vstring("isCicVeryLoose", "0", "15",""),
        probe_eidCicLoose = cms.vstring("isCicLoose", "0", "15",""),
        probe_eidCicMedium = cms.vstring("isCicMedium", "0", "15",""),
        probe_eidCicTight = cms.vstring("isCicTight", "0", "15",""),
        probe_eidCicSuperTight = cms.vstring("isCicSuperTight", "0", "15",""),
        probe_eidCicHyperTight1 = cms.vstring("isCicHyperTight1", "0", "15",""),
        probe_eidCicHyperTight2 = cms.vstring("isCicHyperTight2", "0", "15",""),
        probe_eidCicHyperTight3 = cms.vstring("isCicHyperTight3", "0", "15",""),
        probe_eidCicHyperTight4 = cms.vstring("isCicHyperTight4", "0", "15",""),  
        
    ),

    # defines all the discrete variables of the probes available in the input tree and intended for use in the efficiency calculations
    Categories = cms.PSet(
        mcTrue = cms.vstring("MC true", "dummy[true=1,false=0]"),
        #mcTrue = cms.vstring("MC true", "dummy[false=1,true=0]"),            
        probe_passingHLT = cms.vstring("probe_passingHLT", "dummy[pass=1,fail=0]"),               
        probe_passConvRej = cms.vstring("probe_passConvRej", "dummy[pass=1,fail=0]"),        
        probe_isWP95 = cms.vstring("probe_isWP95", "dummy[pass=1,fail=0]"),
        probe_isWP90 = cms.vstring("probe_isWP90", "dummy[pass=1,fail=0]"),
        probe_isWP85 = cms.vstring("probe_isWP85", "dummy[pass=1,fail=0]"),
        probe_isWP80 = cms.vstring("probe_isWP80", "dummy[pass=1,fail=0]"),
        probe_isWP70 = cms.vstring("probe_isWP70", "dummy[pass=1,fail=0]"),
        probe_isWP60 = cms.vstring("probe_isWP60", "dummy[pass=1,fail=0]"),
    ),
    Cuts = cms.PSet(
        isCicVeryLoose = cms.vstring("CiC VeryLoose", "probe_eidCicVeryLoose", "14.5"), # pass==15, fail<15
        isCicLoose = cms.vstring("CiC Loose", "probe_eidCicLoose", "14.5"), # pass==15, fail<15
        isCicMedium = cms.vstring("CiC Medium", "probe_eidCicMedium", "14.5"), # pass==15, fail<15
        isCicTight = cms.vstring("CiC Tight", "probe_eidCicTight", "14.5"), # pass==15, fail<15
        isCicSuperTight = cms.vstring("CiC SuperTight", "probe_eidCicSuperTight", "14.5"), # pass==15, fail<15
        isCicHyperTight1 = cms.vstring("CiC HyperTight1", "probe_eidCicHyperTight1", "14.5"), # pass==15, fail<15
        isCicHyperTight2 = cms.vstring("CiC HyperTight2", "probe_eidCicHyperTight2", "14.5"), # pass==15, fail<15
        isCicHyperTight3 = cms.vstring("CiC HyperTight3", "probe_eidCicHyperTight3", "14.5"), # pass==15, fail<15
        isCicHyperTight4 = cms.vstring("CiC HyperTight4", "probe_eidCicHyperTight4", "14.5"), # pass==15, fail<15          
    ),

    # defines all the PDFs that will be available for the efficiency calculations; uses RooFit's "factory" syntax;
    # each pdf needs to define "signal", "backgroundPass", "backgroundFail" pdfs, "efficiency[0.9,0,1]" and "signalFractionInPassing[0.9]" are used for initial values  
    PDFs = cms.PSet(
        pdfSignalPlusBackground = cms.vstring(
##     "CBExGaussShape::signalRes(mass, mean[2.0946e-01], sigma[8.5695e-04],alpha[3.8296e-04], n[6.7489e+00], sigma_2[2.5849e+00], frac[6.5704e-01])",  ### the signal function goes here
     "CBExGaussShape::signalRes(mass, mean[2.0946e-01, -5., 5.], sigma[8.5695e-04],alpha[3.8296e-04], n[6.7489e+00], sigma_2[2.5849e+00], frac[6.5704e-01])",  ### the signal function goes here     
    "ZGeneratorLineShape::signalPhy(mass)",
    "RooExponential::backgroundPass(mass, cPass[-0.02, -5, 0])",
    "RooExponential::backgroundFail(mass, cFail[-0.02, -5, 0])",
    "FCONV::signal(mass, signalPhy, signalRes)",
    "efficiency[0.9,0,1]",
    "signalFractionInPassing[1.0]"
    #"Gaussian::signal(mass, mean[91.2, 89.0, 93.0], sigma[2.3, 0.5, 10.0])",
    #"RooExponential::backgroundPass(mass, cPass[0,-10,10])",
    #"RooExponential::backgroundFail(mass, cFail[0,-10,10])",
    #"efficiency[0.9,0,1]",
    #"signalFractionInPassing[0.9]"
        ),
    ),

    # defines a set of efficiency calculations, what PDF to use for fitting and how to bin the data;
    # there will be a separate output directory for each calculation that includes a simultaneous fit, side band subtraction and counting. 
    Efficiencies = cms.PSet(
        #the name of the parameter set becomes the name of the directory
        WP95 = cms.PSet(
             EfficiencyBinningSpecification,
            #specifies the efficiency of which category and state to measure 
            EfficiencyCategoryAndState = cms.vstring("probe_isWP95","pass"),
        ),
        WP90 = cms.PSet(
            EfficiencyBinningSpecification,
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP90","pass"),
        ),
        WP85 = cms.PSet(
            EfficiencyBinningSpecification,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP85","pass"),
        ),
         WP80 = cms.PSet(
            EfficiencyBinningSpecification,   
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP80","pass"),
        ),
        WP70 = cms.PSet(
            EfficiencyBinningSpecification,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP70","pass"),
        ),      
        WP60 = cms.PSet(
            EfficiencyBinningSpecification,      
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP60","pass"),
        ),
        CicVeryLoose = cms.PSet(
            EfficiencyBinningSpecification,     
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicVeryLoose","above"),
        ),
        CicLoose = cms.PSet(
            EfficiencyBinningSpecification,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicLoose","above"),
        ),
        CicMedium = cms.PSet(
            EfficiencyBinningSpecification,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicMedium","above"),
        ),
        CicTight = cms.PSet(
            EfficiencyBinningSpecification,
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicTight","above"),
        ),
        CicSuperTight = cms.PSet(
            EfficiencyBinningSpecification,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicSuperTight","above"),
        ),        
        CicHyperTight1 = cms.PSet(
            EfficiencyBinningSpecification,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicHyperTight1","above"),
        ),
        CicHyperTight2 = cms.PSet(
            EfficiencyBinningSpecification,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicHyperTight2","above"),
        ),
        CicHyperTight3 = cms.PSet(
            EfficiencyBinningSpecification,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicHyperTight3","above"),
        ),
        CicHyperTight4 = cms.PSet(
            EfficiencyBinningSpecification, 
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicHyperTight4","above"),
        ),
        ############################################################################################
        WP95_AND_HLT = cms.PSet(
            EfficiencyBinningSpecification,
            EfficiencyCategoryAndState = cms.vstring("probe_isWP95","pass", HLTDef, "pass"),
        ),
        WP90_AND_HLT = cms.PSet(
            EfficiencyBinningSpecification,
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP90","pass", HLTDef, "pass"),
        ),
        WP85_AND_HLT = cms.PSet(
            EfficiencyBinningSpecification,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP85","pass", HLTDef, "pass"),
        ),
         WP80_AND_HLT = cms.PSet(
            EfficiencyBinningSpecification,   
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP80","pass", HLTDef, "pass"),
        ),
        WP70_AND_HLT = cms.PSet(
            EfficiencyBinningSpecification,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP70","pass", HLTDef, "pass"),
        ),      
        WP60_AND_HLT = cms.PSet(
            EfficiencyBinningSpecification,      
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP60","pass", HLTDef, "pass"),
        ),
        CicVeryLoose_AND_HLT = cms.PSet(
            EfficiencyBinningSpecification,     
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicVeryLoose","above", HLTDef, "pass"),
        ),
        CicLoose_AND_HLT = cms.PSet(
            EfficiencyBinningSpecification,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicLoose","above", HLTDef, "pass"),
        ),
        CicMedium_AND_HLT = cms.PSet(
            EfficiencyBinningSpecification,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicMedium","above", HLTDef, "pass"),
        ),
        CicTight_AND_HLT = cms.PSet(
            EfficiencyBinningSpecification,
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicTight","above", HLTDef, "pass"),
        ),
        CicSuperTight_AND_HLT = cms.PSet(
            EfficiencyBinningSpecification,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicSuperTight","above", HLTDef, "pass"),
        ),        
        CicHyperTight1_AND_HLT = cms.PSet(
            EfficiencyBinningSpecification,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicHyperTight1","above", HLTDef, "pass"),
        ),
        CicHyperTight2_AND_HLT = cms.PSet(
            EfficiencyBinningSpecification,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicHyperTight2","above", HLTDef, "pass"),
        ),
        CicHyperTight3_AND_HLT = cms.PSet(
            EfficiencyBinningSpecification,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicHyperTight3","above", HLTDef, "pass"),
        ),
        CicHyperTight4_AND_HLT = cms.PSet(
            EfficiencyBinningSpecification, 
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicHyperTight4","above", HLTDef, "pass"),
        ),        
        ############################################################################################
        ############################################################################################
        ############################################################################################
        ############################################################################################
        ############################################################################################
        ############################################################################################
        ############################################################################################
        ############################################################################################
        MCtruth_WP95 = cms.PSet(
            EfficiencyBinningSpecificationMC,
            EfficiencyCategoryAndState = cms.vstring("probe_isWP95","pass"),
        ),
        MCtruth_WP90 = cms.PSet(
            EfficiencyBinningSpecificationMC,
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP90","pass"),
        ),
        MCtruth_WP85 = cms.PSet(
            EfficiencyBinningSpecificationMC,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP85","pass"),
        ),
        MCtruth_WP80 = cms.PSet(
            EfficiencyBinningSpecificationMC,   
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP80","pass"),
        ),
        MCtruth_WP70 = cms.PSet(
            EfficiencyBinningSpecificationMC,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP70","pass"),
        ),      
        MCtruth_WP60 = cms.PSet(
            EfficiencyBinningSpecificationMC,      
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP60","pass"),
        ),
        MCtruth_CicVeryLoose = cms.PSet(
            EfficiencyBinningSpecificationMC,     
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicVeryLoose","above"),
        ),
        MCtruth_CicLoose = cms.PSet(
            EfficiencyBinningSpecificationMC,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicLoose","above"),
        ),
        MCtruth_CicMedium = cms.PSet(
            EfficiencyBinningSpecificationMC,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicMedium","above"),
        ),
        MCtruth_CicTight = cms.PSet(
            EfficiencyBinningSpecificationMC,
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicTight","above"),
        ),
        MCtruth_CicSuperTight = cms.PSet(
            EfficiencyBinningSpecificationMC,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicSuperTight","above"),
        ),        
        MCtruth_CicHyperTight1 = cms.PSet(
            EfficiencyBinningSpecificationMC,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicHyperTight1","above"),
        ),
        MCtruth_CicHyperTight2 = cms.PSet(
            EfficiencyBinningSpecificationMC,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicHyperTight2","above"),
        ),
        MCtruth_CicHyperTight3 = cms.PSet(
            EfficiencyBinningSpecificationMC,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicHyperTight3","above"),
        ),
        MCtruth_CicHyperTight4 = cms.PSet(
            EfficiencyBinningSpecificationMC, 
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicHyperTight4","above"),
        ),
        ############################################################################################
        MCtruth_WP95_AND_HLT = cms.PSet(
            EfficiencyBinningSpecificationMC,
            EfficiencyCategoryAndState = cms.vstring("probe_isWP95","pass", HLTDef, "pass"),
        ),
        MCtruth_WP90_AND_HLT = cms.PSet(
            EfficiencyBinningSpecificationMC,
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP90","pass", HLTDef, "pass"),
        ),
        MCtruth_WP85_AND_HLT = cms.PSet(
            EfficiencyBinningSpecificationMC,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP85","pass", HLTDef, "pass"),
        ),
        MCtruth_WP80_AND_HLT = cms.PSet(
            EfficiencyBinningSpecificationMC,   
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP80","pass", HLTDef, "pass"),
        ),
        MCtruth_WP70_AND_HLT = cms.PSet(
            EfficiencyBinningSpecificationMC,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP70","pass", HLTDef, "pass"),
        ),      
        MCtruth_WP60_AND_HLT = cms.PSet(
            EfficiencyBinningSpecificationMC,      
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","probe_isWP60","pass", HLTDef, "pass"),
        ),
        MCtruth_CicVeryLoose_AND_HLT = cms.PSet(
            EfficiencyBinningSpecificationMC,     
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicVeryLoose","above", HLTDef, "pass"),
        ),
        MCtruth_CicLoose_AND_HLT = cms.PSet(
            EfficiencyBinningSpecificationMC,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicLoose","above", HLTDef, "pass"),
        ),
        MCtruth_CicMedium_AND_HLT = cms.PSet(
            EfficiencyBinningSpecificationMC,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicMedium","above", HLTDef, "pass"),
        ),
        MCtruth_CicTight_AND_HLT = cms.PSet(
            EfficiencyBinningSpecificationMC,
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicTight","above", HLTDef, "pass"),
        ),
        MCtruth_CicSuperTight_AND_HLT = cms.PSet(
            EfficiencyBinningSpecificationMC,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicSuperTight","above", HLTDef, "pass"),
        ),        
        MCtruth_CicHyperTight1_AND_HLT = cms.PSet(
            EfficiencyBinningSpecificationMC,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicHyperTight1","above", HLTDef, "pass"),
        ),
        MCtruth_CicHyperTight2_AND_HLT = cms.PSet(
            EfficiencyBinningSpecificationMC,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicHyperTight2","above", HLTDef, "pass"),
        ),
        MCtruth_CicHyperTight3_AND_HLT = cms.PSet(
            EfficiencyBinningSpecificationMC,    
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicHyperTight3","above", HLTDef, "pass"),
        ),
        MCtruth_CicHyperTight4_AND_HLT = cms.PSet(
            EfficiencyBinningSpecificationMC, 
            EfficiencyCategoryAndState = cms.vstring("probe_passConvRej","pass","isCicHyperTight4","above", HLTDef, "pass"),
        ),                


    )
)


process.fit = cms.Path(process.TagProbeFitTreeAnalyzer)
