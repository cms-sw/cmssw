import FWCore.ParameterSet.Config as cms
import os

source = cms.Source("EmptySource")

generator = cms.EDFilter("SherpaGeneratorFilter",
  maxEventsToPrint = cms.int32(0),
  filterEfficiency = cms.untracked.double(1.0),
  crossSection = cms.untracked.double(-1),
  SherpaProcess = cms.string('ttbar_2j_MENLOPS_13TeV'),
  SherpackLocation = cms.string('/cvmfs/cms.cern.ch/phys_generator/gridpacks/slc7_amd64_gcc820/13TeV/sherpa/2.2.8'),
  SherpackChecksum = cms.string('4efdf38e0d189d58c65a554ef901d027'),
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
				" semihard xsec = 39.5554 mb,",
				" non-diffractive xsec = 17.0318 mb with nd factor = 0.3142."
                                                  ),
                              Run = cms.vstring(
				" (run){",
				" CORE_SCALE TTBar;",
				" METS_BBAR_MODE 5;",
				" NJET:=2; LJET:=2; QCUT:=20.;",
				" ME_SIGNAL_GENERATOR Comix Amegic LOOPGEN;",
				" OL_PREFIX={0} ".format(os.environ['CMS_OPENLOOPS_PREFIX']),
				" LOOPGEN:=OpenLoops;",
				" MI_HANDLER=Amisic;",
				" NLO_SMEAR_THRESHOLD 1;",
				" NLO_SMEAR_POWER 2;",
				" HARD_DECAYS On;",
				" HARD_SPIN_CORRELATIONS=1;",
				" SOFT_SPIN_CORRELATIONS=1;",
				" PDF_LIBRARY LHAPDFSherpa;",
				" PDF_SET NNPDF31_nnlo_hessian_pdfas;",
				" USE_PDF_ALPHAS=1;",
				" BEAM_1=2212; BEAM_ENERGY_1=6500;",
				" BEAM_2=2212; BEAM_ENERGY_2=6500;",
				" STABLE[6] 0; WIDTH[6] 0; STABLE[24] 0;",
				" EXCLUSIVE_CLUSTER_MODE 1;",
				" HEPMC_TREE_LIKE=1;",
				" PRETTY_PRINT=Off;",
				"}(run)",
				" (processes){",
				" Process : 93 93 -> 6 -6 93{NJET};",
				" Order (*,0); CKKW sqr(QCUT/E_CMS);",
				" NLO_QCD_Mode MC@NLO {LJET};",
				" ME_Generator Amegic {LJET};",
				" RS_ME_Generator Comix {LJET};",
				" Loop_Generator LOOPGEN {LJET};",
				" Integration_Error 0.05 {3,4};",
				" End process;",
				"}(processes)"
                                                  ),
                             )
)

ProductionFilterSequence = cms.Sequence(generator)
