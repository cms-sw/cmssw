import FWCore.ParameterSet.Config as cms
import os

source = cms.Source("EmptySource")

generator = cms.EDFilter("SherpaGeneratorFilter",
  maxEventsToPrint = cms.int32(0),
  filterEfficiency = cms.untracked.double(1.0),
  crossSection = cms.untracked.double(-1),
  SherpaProcess = cms.string('ZtoEE_0j_OpenLoops_13TeV'),
  SherpackLocation = cms.string('slc6_amd64_gcc491/sherpa/2.1.1/13TeV'),
  SherpackChecksum = cms.string('100ee88a091480befb360aef2e7f637d'),
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
				" semihard xsec = 74.0613 mb,",
				" non-diffractive xsec = 18.1593 mb with nd factor = 0.335."
                                                  ),
                              Run = cms.vstring(
				"(run){",
				" EVENTS 100; ERROR 0.99;",
				" MASSIVE_PS 4 5;",
				" FSF:=1.; RSF:=1.; QSF:=1.;",
				" SCALES METS{FSF*MU_F2}{RSF*MU_R2}{QSF*MU_Q2};",
				" NJET:=0; LJET:=2; QCUT:=20.;",
				" ME_SIGNAL_GENERATOR Comix Amegic LOOPGEN;",
				" OL_PREFIX={0} ".format(os.environ['CMS_OPENLOOPS_PREFIX']) ,
				" EVENT_GENERATION_MODE Weighted;",
				" LOOPGEN:=OpenLoops;",
				" MASSIVE[15] 1;",
				" BEAM_1 2212; BEAM_ENERGY_1 = 6500.;",
				" BEAM_2 2212; BEAM_ENERGY_2 = 6500.;",
				"}(run)",
				"(processes){",
				" Process 93 93 -> 11 -11 93{NJET};",
				" Order_EW 2; CKKW sqr(QCUT/E_CMS);",
				" NLO_QCD_Mode MC@NLO {LJET};",
				" ME_Generator Amegic {LJET};",
				" RS_ME_Generator Comix {LJET};",
				" Loop_Generator LOOPGEN {LJET};",
				" Integration_Error 0.02 {4};",
				" Scales LOOSE_METS{FSF*MU_F2}{RSF*MU_R2}{QSF*MU_Q2} {7,8};",
				" End process;",
				"}(processes)",
				"(isr){",
				" PDF_LIBRARY     = LHAPDFSherpa",
				" PDF_SET         = CT10",
				" PDF_SET_VERSION = 0",
				" PDF_GRID_PATH   = PDFsets",
				"}(isr)",
				"(selector){",
				" Mass 11 -11 66 E_CMS",
				" Mass 13 -13 66 E_CMS",
				"}(selector)"
                                                  ),
                             )
)

ProductionFilterSequence = cms.Sequence(generator)
