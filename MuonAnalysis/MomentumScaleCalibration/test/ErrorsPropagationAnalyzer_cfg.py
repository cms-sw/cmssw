import FWCore.ParameterSet.Config as cms

process = cms.Process("ERRORSANALYZER")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)

process.ErrorsPropagationAnalyzerModule = cms.EDAnalyzer(
    "ErrorsPropagationAnalyzer",

    InputFileName = cms.string("/home/castello/7TeV/CMSSW_3_8_5_patch3/src/Tree/Fall10/Tree_MCFall2010_INNtk_CRAFTRealistic_wGEN.root"),
    MaxEvents = cms.int32(-1),

    # Function parameters
    ResolFitType = cms.int32(20),


    # double ptPart = parval[2]*1./pt + pt/(pt+parval[3]) + pt*parval[9] + pt*pt*parval[10];

    # if(fabsEta<parval[0]) {
    #   // To impose continuity we require that the parval[0] of type11 is
    #   double par = parval[1] + parval[6]*fabs((parval[0]-parval[8])) + parval[7]*(parval[0]-parval[8])*(parval[0]-parval[8]) - (parval[4]*parval[0] + parval[5]*parval[0]*parval[0]);
    #   return( par + ptPart + parval[4]*fabsEta + parval[5]*eta*eta );
    # }
    # else {
    #   return( parval[1]+ ptPart + parval[6]*fabs((fabsEta-parval[8])) + parval[7]*(fabsEta-parval[8])*(fabsEta-parval[8]) );
    # }


    Parameters = cms.vdouble(),
    Errors = cms.vdouble(),
    ErrorFactors = cms.vint32(),

    OutputFileName = cms.string("test.root"),

    PtBins = cms.int32(50),
    PtMin = cms.double(0.),
    PtMax = cms.double(100.),
    
    EtaBins = cms.int32(100),
    EtaMin = cms.double(-3.),
    EtaMax = cms.double(3.),
    
    # Optionally configure cuts on pt and eta of the muons used to fill the histograms
    PtMinCut = cms.untracked.double(0.),
    PtMaxCut = cms.untracked.double(999999.),
    EtaMinCut = cms.untracked.double(0.),
    EtaMaxCut = cms.untracked.double(100.),

    Debug = cms.bool(False),
)

process.p1 = cms.Path(process.ErrorsPropagationAnalyzerModule)

