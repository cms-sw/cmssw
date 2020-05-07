import FWCore.ParameterSet.Config as cms
import os

source = cms.Source("EmptySource")

generator = cms.EDFilter("SherpaGeneratorFilter",
  maxEventsToPrint = cms.int32(0),
  filterEfficiency = cms.untracked.double(1.0),
  crossSection = cms.untracked.double(-1),
  SherpaProcess = cms.string('ZtoLL_2j_MEPSatNLO_13TeV'),
  SherpackLocation = cms.string('/cvmfs/cms.cern.ch/phys_generator/gridpacks/slc7_amd64_gcc820/13TeV/sherpa/2.2.8'),
  SherpackChecksum = cms.string('b0cdd4d30b6ddc1816f026831d6ccccf'),
  FetchSherpack = cms.bool(True),
  SherpaPath = cms.string('./'),
  SherpaPathPiece = cms.string('./'),
  SherpaResultDir = cms.string('Result'),
  SherpaDefaultWeight = cms.double(1.0),
  SherpaParameters = cms.PSet(parameterSets = cms.vstring(
                             "MPI_Cross_Sections",
                             "Run"),
                              MPI_Cross_Sections = cms.vstring(
				" MPIs in Sherpa, Model = Amisic:",
				" semihard xsec = 39.7318 mb,",
				" non-diffractive xsec = 17.0318 mb with nd factor = 0.3142."
                                                  ),
                              Run = cms.vstring(
				" (run){",
				" FSF:=1.; RSF:=1.; QSF:=1.;",
				" SCALES METS{FSF*MU_F2}{RSF*MU_R2}{QSF*MU_Q2};",
				" NJET:=2; LJET:=2,3,4; QCUT:=20.;",
				" ME_SIGNAL_GENERATOR Comix Amegic LOOPGEN;",
				" OL_PREFIX={0} ".format(os.environ['CMS_OPENLOOPS_PREFIX']),
				" LOOPGEN:=OpenLoops;",
				" PDF_LIBRARY LHAPDFSherpa;",
				" PDF_SET NNPDF31_nnlo_hessian_pdfas;",
				" USE_PDF_ALPHAS=1;",
				" BEAM_1 2212; BEAM_ENERGY_1 = 6500.;",
				" BEAM_2 2212; BEAM_ENERGY_2 = 6500.;",
				" EXCLUSIVE_CLUSTER_MODE 1;",
				" HEPMC_TREE_LIKE=1;",
				" PRETTY_PRINT=Off;",
				"}(run)",
				" (processes){",
				" Process 93 93 -> 90 90 93{NJET};",
				" Order (*,2); CKKW sqr(QCUT/E_CMS);",
				" NLO_QCD_Mode MC@NLO {LJET};",
				" ME_Generator Amegic {LJET};",
				" RS_ME_Generator Comix {LJET};",
				" Loop_Generator LOOPGEN {LJET};",
				" Integration_Error 0.02 {3,4};",
				" End process;",
				"}(processes)",
				" (selector){",
				" Mass 11 -11 50 E_CMS",
				" Mass 13 -13 50 E_CMS",
				" Mass 15 -15 50 E_CMS",
				"}(selector)"
                                                  ),
                             )
)

ProductionFilterSequence = cms.Sequence(generator)
