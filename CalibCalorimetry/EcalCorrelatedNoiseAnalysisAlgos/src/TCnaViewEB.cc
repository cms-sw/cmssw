//---------Author's Name: B.Fabbro DSM/DAPNIA/SPP CEA-Saclay
//---------Copyright: Those valid for CEA sofware
//---------Modified: 04/07/2007

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaViewEB.h"

#include "TGaxis.h"
#include "TRootCanvas.h"
#include "TH2D.h"
#include "TF1.h"
#include "TStyle.h"

ClassImp(TCnaViewEB)
//______________________________________________________________________________
//
// TCnaViewEB.
//
//==============> INTRODUCTION
//
//    This class provides methods which perform different kinds of plots:
//    1D, 2D and 3D histograms for different quantities (pedestals, noises,
//    correlations, .etc..). The data are read from files which has been
//    previously written by using the class TCnaRunEB (.root result files).
//    The reading is performed by appropriate methods of the class TCnaReadEB.
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//                     Class TCnaViewEB: Example of use:
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//      TCnaViewEB* MyView =  = new TCnaViewEB();
//      
//      //--> Gives the file parameters
//
//      TString AnalysisName   = "TB2006_1";
//      Int_t   RunNumber      = 16467;
//      Int_t   FirstTakenEvt  = 0;
//      Int_t   NbOfTakenEvts  = 150;
//      Int_t   SMNumber       = 6;
//
//      MyView->GetPathForResultsRootFiles("my_cna_config.cfg");
//  or: MyView->GetPathForResultsRootFiles();       // (see explanations after this example)
//
//      //--> Specify the file which has to be read (previously written by TCnaRunEB)
//
//      MyView->SetFile(AnalysisName, RunNumber, FirstTakenEvt, NbOfTakenEvts, SMNumber); 
//
//      //--> Plot covariance matrix between crystals of towers 10 and 33
//      //   (mean over samples)
//
//      Int_t SMtower_X = 10;
//      Int_t SMtower_Y = 33;
//
//      MyView->CovariancesBetweenCrystals(SMtower_X, SMtower_Y, "LEGO2Z");
//
//      //--> Plot correlation matrices between samples
//      //    for channel 21 (electronic channel number in tower)
//      //    and for the two towers 10 and 33
//
//      Int_t TowEcha = 21;
//
//      MyView->CorrelationsBetweenSamples(SMtower_X, TowEcha, "SURF1Z");
//      MyView->CorrelationsBetweenSamples(SMtower_Y, TowEcha, "SURF1Z");
//
//      //--> Plot histogram of the mean sigmas for tower 38 and channel 12
//            (electronic channel number in tower) as a function of time
//
//      MyView->GetPathForListOfRunFiles();       // (see explanations after this example)
//
//      Int_t SMtower = 38;
//      Int_t TowEcha = 12;
//      TString run_par_file_name = "CnP_SM6_evol_burst1.ascii";
//      MyView->HistimeCrystalMeanSigmas(run_par_file_name, SMtower, TowEcha, "SEVERAL");
//      
//      // the .ascii file "CnP_SM6_evol_burst1.ascii" must contain a list of
//      // the different runs with mention of their parameters (see below).
//
//      etc...
//
//------------------------------------------------------------------------------------------
// Explanations for the methods GetPathForResultsRootFiles(const TString),
// GetPathForResultsAsciiFiles(const TString) and GetPathForListOfRunFiles(const TString)
//
//  //.......... method GetPathForResultsRootFiles(const TString)
//
//    If the argument is specified, it is the name of a "cna-configuration" file.
//    This file must be present in the user's HOME directory and must have one line
//    which contains the path (without slash at the end) of the directory
//    where the .root result files (previously written with a program calling TCnaRunEB)
//    are located.
//  
//    If no argument is specified (DEFAULT), the name of the configuration file is considered
//    to be: "cna_results_root.cfg" and the file must be created in the user's HOME directory.
//
//    EXAMPLE of configuration file content (1 line): $HOME/scratch0/cna/results_root
//    (without slash at the end !)
//    The .root result files are located in the directory $HOME/scratch0/cna/results_root
// 
// //.......... method GetPathForResultsRootFiles(const TString)
//
//   Same as GetPathForResultsRootFiles(const TString). The name of the DEFAULT configuration
//   file is considered to be: "cna_results_ascii.cfg" and the file must be created in the
//   user's HOME directory.
//
//   EXAMPLE of configuration file content (1 line): $HOME/scratch0/cna/results_ascii
//   (without slash at the end !)
//   The .ascii result files are located in the directory $HOME/scratch0/cna/results_ascii
//
// //.......... method GetPathForListOfRunFiles(const TString)
//   The name of the DEFAULT configuration file is considered to be: "cna_stability.cfg"
//   and the file must be created in the user's HOME directory.
//
//   EXAMPLE of configuration file content (1 line): $HOME/scratch0/cna/runlist_time_evol
//   (without slash at the end !)
//   The .ascii "list of runs" files are located in the directory $HOME/scratch0/cna/runlist_time_evol
//
//   
//   //.......... SYNTAX OF THE FILE "CnP_SM6_evol_burst1.ascii" ("list of runs" file):
//
//   CnP_SM6_evol_burst1.ascii  <- 1rst line: comment (name of the file, for example)
//   5                          <- 2nd  line: nb of lines after this one
//   ped 73677 0 150 10         <- 3rd  line and others: run parameters
//   ped 73688 0 150 10                 (Analysis name, run number, first taken event, 
//   ped 73689 0 150 10                  number of taken events, SM number)
//   ped 73690 0 150 10 
//   ped 73692 0 150 10 
//
//---------------------------------------------------------------------------------------
//
//---------------------------------- LIST OF METHODS ------------------------------------
//
//  //................. (Eta,Phi) SM plots
//
//  void EtaPhiSuperModuleFoundEvents();
//  void EtaPhiSuperModuleMeanPedestals();
//  void EtaPhiSuperModuleMeanOfSampleSigmas();
//  void EtaPhiSuperModuleMeanOfCorss();
//  void EtaPhiSuperModuleSigmaPedestals();
//  void EtaPhiSuperModuleSigmaOfSampleSigmas();
//  void EtaPhiSuperModuleSigmaOfCorss();
//  void EtaPhiSuperModuleCorccMeanOverSamples();
//  
//  //................. Correlation and covariances matrices
//
//  draw_option = ROOT options for 2D histos: "COLZ", "LEGO2Z", "SURF1Z", "SURF4"
//
//  void CorrelationsBetweenTowers(const TString draw_option);
//  void CovariancesBetweenTowers (const TString draw_option));
//  void CorrelationsBetweenCrystals(const Int_t& towerX, const Int_t& towerY, const TString draw_option);
//  void CovariancesBetweenCrystals (const Int_t& towerX, const Int_t& towerY, const TString draw_option);
//  void CorrelationsBetweenSamples (const Int_t& tower, const Int_t& TowEcha, const TString draw_option);
//  void CovariancesBetweenSamples  (const Int_t& tower, const Int_t& TowEcha, const TString draw_option);
//
//  void CorrelationsBetweenSamples(const Int_t& SMtower);
//  void CovariancesBetweenSamples (const Int_t& SMtower);
//
//  //................. Channel and tower numbering plots
//
//  void TowerCrystalNumbering(const Int_t& SM_number, const Int_t& SMtower);
//  void SuperModuleTowerNumbering(const Int_t& SM_number);
//
//  //................. Histograms
//
//  plot_option:  "ONLYONE": only one plot,
//                "SEVERAL": superimposed plots (same as the option "SAME" in ROOT)
//
//  void HistoSuperModuleFoundEventsOfCrystals(const TString plot_option);
//  void HistoSuperModuleFoundEventsDistribution(const TString plot_option);
//  void HistoSuperModuleMeanPedestalsOfCrystals(const TString plot_option);
//  void HistoSuperModuleMeanPedestalsDistribution(const TString plot_option);
//
//       "OfCrystal": X axis = crystal number
//  void HistoSuperModuleMeanOfSampleSigmasOfCrystals(const TString plot_option);
//  void HistoSuperModuleMeanOfCorssOfCrystals(const TString plot_option);
//  void HistoSuperModuleSigmaPedestalsOfCrystals(const TString plot_option);
//  void HistoSuperModuleSigmaOfSampleSigmasOfCrystals(const TString plot_option);
//  void HistoSuperModuleSigmaOfCorssOfCrystals(const TString plot_option);
//
//       "Distribution": projection on Y axis
//  void HistoSuperModuleMeanOfSampleSigmasDistribution(const TString plot_option);
//  void HistoSuperModuleMeanOfCorssDistribution(const TString plot_option);
//  void HistoSuperModuleSigmaPedestalsDistribution(const TString plot_option);
//  void HistoSuperModuleSigmaOfSampleSigmasDistribution(const TString plot_option);
//  void HistoSuperModuleSigmaOfCorssDistribution(const TString plot_option);
//
//  void HistoCrystalExpectationValuesOfSamples
//               (const Int_t& Tower, const Int_t& TowEcha, const TString plot_option);
//  void HistoCrystalSigmasOfSamples
//               (const Int_t& Tower, const Int_t& TowEcha, const TString plot_option);
//  void HistoCrystalPedestalEventNumber
//               (const Int_t& Tower, const Int_t& TowEcha, const TString plot_option);
//
//  void HistoSampleEventDistribution (const Int_t& SMower, const Int_t& TowEcha,
//                                     const Int_t& sample, const TString plot_option);
//
//
//  //............... Histograms for evolution, stability
//
//  For these histograms, a "cna-configuration file" must be used (see the example above). 
//
//  HistimeCrystalMeanPedestals(const TString file_name, const Int_t&  SMtower,
//                              const Int_t&  TowEcha,   const TString plot_option)
//  HistimeCrystalMeanSigmas(const TString file_name, const Int_t&  SMtower,
//                           const Int_t&  TowEcha,   const TString plot_option)
//  HistimeCrystalMeanCorss(const TString file_name, const Int_t&  SMtower,
//                          const Int_t&  TowEcha,   const TString plot_option)
//
//
//-------------------------------------------------------------------------
//
//        For more details on other classes of the CNA package:
//
//                 http://www.cern.ch/cms-fabbro/cna
//
//-------------------------------------------------------------------------
//

//------------------------------ TCnaViewEB.cxx -----------------------------
//  
//   Creation (first version): 18 April 2005
//
//   For questions or comments, please send e-mail to Bernard Fabbro:
//             
//   fabbro@hep.saclay.cea.fr 
//
//------------------------------------------------------------------------


  //---------------> message de rappel pour l'auteur: 
  //
  //======= A T T E N T I O N ========= A T T E N T I O N ========= A T T E N T I O N ==============!!!!
  //      A EVITER ABSOLUMENT CAR LE cin >> BLOQUE EXCEED quand on est sous TCnaDialogEB
  //      puisqu'on n'a pas la main dans la fenetre de compte-rendu de la CNA
  //     {Int_t cintoto; cout << "taper 0 pour continuer" << endl; cin >> cintoto;}
  //                         *=============================================*
  //                         |                                             |
  //+++++++++++++++++++++++++|    A T T E N T I O N:  PAS DE cintoto ici!  |++++++++++++++++++++++++!!!!
  //                         |                                             |
  //                         *=============================================*

// 
  
TCnaViewEB::~TCnaViewEB()
{
  //destructor
  
  if (fT1DAnaType             != 0){delete [] fT1DAnaType;             fCdelete++;}
  if (fT1DRunNumber           != 0){delete [] fT1DRunNumber;           fCdelete++;}
  if (fT1DFirstEvt            != 0){delete [] fT1DFirstEvt;            fCdelete++;}
  if (fT1DNbOfEvts            != 0){delete [] fT1DNbOfEvts;            fCdelete++;}
  if (fT1DSuMoNumber          != 0){delete [] fT1DSuMoNumber;          fCdelete++;}
  if (fT1DResultsRootFilePath != 0){delete [] fT1DResultsRootFilePath; fCdelete++;}
  if (fT1DListOfRunsFilePath  != 0){delete [] fT1DListOfRunsFilePath;  fCdelete++;}

  if (fParameters             != 0){delete     fParameters;            fCdelete++;}
}

//===================================================================
//
//                   Constructor without arguments
//
//===================================================================
TCnaViewEB::TCnaViewEB()
{
  Init();
}

void  TCnaViewEB::Init()
{
//========================= GENERAL INITIALISATION 

  fCnew        = 0;
  fCdelete     = 0;
  fCnewRoot    = 0;
  fCdeleteRoot = 0;

  fCnaCommand  = 0;
  fCnaError    = 0;

  fgMaxCar = (Int_t)512;

  //------------------------------ initialisations ----------------------

  fTTBELL = '\007';

  fT1DAnaType             = 0;
  fT1DRunNumber           = 0;
  fT1DFirstEvt            = 0;
  fT1DNbOfEvts            = 0;
  fT1DSuMoNumber          = 0;
  fT1DResultsRootFilePath = 0;
  fT1DListOfRunsFilePath  = 0;
  //........................ init CNA paraneters

  fParameters = new TCnaParameters();                  fCnew++;
  fParameters->SetPeriodTitles();       // define the titles of the different periods of run

  //........................ init code plot type
  Int_t MaxCar = fgMaxCar;
  fOnlyOnePlot.Resize(MaxCar);
  fOnlyOnePlot = "ONLYONE";

  MaxCar = fgMaxCar;
  fSeveralPlot.Resize(MaxCar);
  fSeveralPlot = "SEVERAL";

  //.......... init flags Same plot
  fMemoPlotSMFoundEvtsGlobal = 0; fMemoPlotSMFoundEvtsProj = 0;
  fMemoPlotSMEvEvGlobal      = 0; fMemoPlotSMEvEvProj      = 0;
  fMemoPlotSMEvSigGlobal     = 0; fMemoPlotSMEvSigProj     = 0; 
  fMemoPlotSMEvCorssGlobal   = 0; fMemoPlotSMEvCorssProj   = 0;
  fMemoPlotSMSigEvGlobal     = 0; fMemoPlotSMSigEvProj     = 0; 
  fMemoPlotSMSigSigGlobal    = 0; fMemoPlotSMSigSigProj    = 0; 
  fMemoPlotSMSigCorssGlobal  = 0; fMemoPlotSMSigCorssProj  = 0; 
  fMemoPlotEv                = 0; fMemoPlotSigma           = 0;
  fMemoPlotEvts              = 0; fMemoPlotSampTime        = 0;
  fMemoPlotEvolEvEv          = 0; fMemoPlotEvolEvSig       = 0;
  fMemoPlotEvolEvCorss       = 0;

  //.......... init flags colors
  fMemoColorSMFoundEvtsGlobal = 0; fMemoColorSMFoundEvtsProj = 0;
  fMemoColorSMEvEvGlobal      = 0; fMemoColorSMEvEvProj      = 0;
  fMemoColorSMEvSigGlobal     = 0; fMemoColorSMEvSigProj     = 0; 
  fMemoColorSMEvCorssGlobal   = 0; fMemoColorSMEvCorssProj   = 0;
  fMemoColorSMSigEvGlobal     = 0; fMemoColorSMSigEvProj     = 0; 
  fMemoColorSMSigSigGlobal    = 0; fMemoColorSMSigSigProj    = 0; 
  fMemoColorSMSigCorssGlobal  = 0; fMemoColorSMSigCorssProj  = 0; 
  fMemoColorEv                = 0; fMemoColorSigma           = 0;
  fMemoColorEvts              = 0; fMemoColorSampTime        = 0;
  fMemoColorEvolEvEv          = 0; fMemoColorEvolEvSig       = 0;
  fMemoColorEvolEvCorss       = 0;

  //.......... init counter Same canvas
  fCanvSameSMFoundEvtsGlobal = 0; fCanvSameSMFoundEvtsProj = 0;
  fCanvSameSMEvEvGlobal      = 0; fCanvSameSMEvEvProj      = 0;
  fCanvSameSMEvSigGlobal     = 0; fCanvSameSMEvSigProj     = 0; 
  fCanvSameSMEvCorssGlobal   = 0; fCanvSameSMEvCorssProj   = 0;
  fCanvSameSMSigEvGlobal     = 0; fCanvSameSMSigEvProj     = 0; 
  fCanvSameSMSigSigGlobal    = 0; fCanvSameSMSigSigProj    = 0; 
  fCanvSameSMSigCorssGlobal  = 0; fCanvSameSMSigCorssProj  = 0; 
  fCanvSameEv                = 0; fCanvSameSigma           = 0;
  fCanvSameEvts              = 0; fCanvSameSampTime        = 0;
  fCanvSameEvolEvEv          = 0; fCanvSameEvolEvSig       = 0;
  fCanvSameEvolEvCorss       = 0;

  //......... init ymin,ymax histos
  // Default values for Ymin and Ymax

  PutYmin("SMFoundEvtsGlobal", (Double_t)0.);
  PutYmax("SMFoundEvtsGlobal", (Double_t)500.);

  PutYmin("SMFoundEvtsProj",   (Double_t)0.1);
  PutYmax("SMFoundEvtsProj",   (Double_t)1000.);

  PutYmin("SMEvEvGlobal",      (Double_t)0.);
  PutYmax("SMEvEvGlobal",      (Double_t)500.);

  PutYmin("SMEvEvProj",        (Double_t)0.1);
  PutYmax("SMEvEvProj",        (Double_t)1000.);

  PutYmin("SMEvSigGlobal",     (Double_t)0.);
  PutYmax("SMEvSigGlobal",     (Double_t)5.);

  PutYmin("SMEvSigProj",       (Double_t)0.1);
  PutYmax("SMEvSigProj",       (Double_t)1000.);

  PutYmin("SMEvCorssGlobal",   (Double_t)-1.);
  PutYmax("SMEvCorssGlobal",   (Double_t)1.);

  PutYmin("SMEvCorssProj",     (Double_t)0.1);
  PutYmax("SMEvCorssProj",     (Double_t)1000.);

  PutYmin("SMSigEvGlobal",     (Double_t)0.);
  PutYmax("SMSigEvGlobal",     (Double_t)1.);

  PutYmin("SMSigEvProj",       (Double_t)0.1);
  PutYmax("SMSigEvProj",       (Double_t)1000.);

  PutYmin("SMSigSigGlobal",    (Double_t)0.);
  PutYmax("SMSigSigGlobal",    (Double_t)1.);

  PutYmin("SMSigSigProj",      (Double_t)0.1);
  PutYmax("SMSigSigProj",      (Double_t)1000.);

  PutYmin("SMSigCorssGlobal",  (Double_t)(-1.));
  PutYmax("SMSigCorssGlobal",  (Double_t)1.);

  PutYmin("SMSigCorssProj",    (Double_t)0.1);
  PutYmax("SMSigCorssProj",    (Double_t)1000.);

  PutYmin("SMEvCorttMatrix",   (Double_t)(-1.));
  PutYmax("SMEvCorttMatrix",   (Double_t)1.);

  PutYmin("SMEvCovttMatrix",   (Double_t)0.);
  PutYmax("SMEvCovttMatrix",   (Double_t)10.);

  PutYmin("SMCorccInTowers",   (Double_t)-0.05);
  PutYmax("SMCorccInTowers",   (Double_t) 0.05);

  PutYmin("Ev",                (Double_t)0.);
  PutYmax("Ev",                (Double_t)500.);

  PutYmin("Sigma",             (Double_t)0.);
  PutYmax("Sigma",             (Double_t)5.);

  PutYmin("Evts",              (Double_t)0.);
  PutYmax("Evts",              (Double_t)500.);

  PutYmin("SampTime",          (Double_t)0.);
  PutYmax("SampTime",          (Double_t)500.);

  PutYmin("EvolEv",            (Double_t)0.);
  PutYmax("EvolEv",            (Double_t)500.);

  PutYmin("EvolSig",           (Double_t)0.);
  PutYmax("EvolSig",           (Double_t)5.);

  PutYmin("EvolCorss",         (Double_t)(-1.));
  PutYmax("EvolCorss",         (Double_t)1.);


  //................. Flag Scale X anf Y set to "LIN"

  MaxCar = fgMaxCar;
  fFlagScaleX.Resize(MaxCar);
  fFlagScaleX = "LIN";

  MaxCar = fgMaxCar;
  fFlagScaleY.Resize(MaxCar);
  fFlagScaleY = "LIN";

  //................. Init codes Options

  fOptScaleLiny = 31400;
  fOptScaleLogy = 31401;

  fOptVisLine  = 1101; 
  fOptVisPolm  = 1102;

  fOptMatCov   = 101;
  fOptMatCor   = 102;

  MaxCar = fgMaxCar;
  fOptMcc.Resize(MaxCar);
  fOptMcc      = "Crystal";

  MaxCar = fgMaxCar;
  fOptMss.Resize(MaxCar);
  fOptMss      = "Sample";

  MaxCar = fgMaxCar;
  fOptMtt.Resize(MaxCar);
  fOptMtt      = "Tower";  

  //.......................... text pave alignement

  fTextPaveAlign = 12;              // 1 = left adjusted, 2 = vertically centered
  fTextPaveFont  = 100;             // 10*10 = 10*(ID10 = Courier New)
  fTextPaveSize  = (Float_t)0.03;   // 0.95 = 95% of the pave size

  //......................... Init canvas/pad pointers
  
  fCanvSMFoundEvtsGlobal = 0;
  fCanvSMFoundEvtsProj   = 0;
  fCanvSMEvEvGlobal      = 0;
  fCanvSMEvEvProj        = 0;
  fCanvSMEvSigGlobal     = 0;   
  fCanvSMEvSigProj       = 0; 
  fCanvSMEvCorssGlobal   = 0; 
  fCanvSMEvCorssProj     = 0;
  fCanvSMSigEvGlobal     = 0;
  fCanvSMSigEvProj       = 0; 
  fCanvSMSigSigGlobal    = 0;   
  fCanvSMSigSigProj      = 0; 
  fCanvSMSigCorssGlobal  = 0; 
  fCanvSMSigCorssProj    = 0; 
  fCanvEv                = 0;
  fCanvSigma             = 0;  
  fCanvEvts              = 0;     
  fCanvSampTime          = 0;
  fCanvEvolEvEv          = 0;
  fCanvEvolEvSig         = 0;
  fCanvEvolEvCorss       = 0;
  
  fCurrentPad           = 0;

  fPadSMFoundEvtsGlobal = 0;
  fPadSMFoundEvtsProj   = 0;
  fPadSMEvEvGlobal      = 0;
  fPadSMEvEvProj        = 0;
  fPadSMEvSigGlobal     = 0;   
  fPadSMEvSigProj       = 0; 
  fPadSMEvCorssGlobal   = 0; 
  fPadSMEvCorssProj     = 0;
  fPadSMSigEvGlobal     = 0;
  fPadSMSigEvProj       = 0; 
  fPadSMSigSigGlobal    = 0;   
  fPadSMSigSigProj      = 0; 
  fPadSMSigCorssGlobal  = 0; 
  fPadSMSigCorssProj    = 0; 
  fPadEv                = 0;
  fPadSigma             = 0;
  fPadEvts              = 0;
  fPadSampTime          = 0;
  fPadEvolEvEv          = 0;
  fPadEvolEvSig         = 0;
  fPadEvolEvCorss       = 0;

  fPavTxtSMFoundEvtsGlobal = 0;
  fPavTxtSMFoundEvtsProj   = 0;
  fPavTxtSMEvEvGlobal      = 0;
  fPavTxtSMEvEvProj        = 0;
  fPavTxtSMEvSigGlobal     = 0;   
  fPavTxtSMEvSigProj       = 0; 
  fPavTxtSMEvCorssGlobal   = 0; 
  fPavTxtSMEvCorssProj     = 0;
  fPavTxtSMSigEvGlobal     = 0;
  fPavTxtSMSigEvProj       = 0; 
  fPavTxtSMSigSigGlobal    = 0;   
  fPavTxtSMSigSigProj      = 0; 
  fPavTxtSMSigCorssGlobal  = 0; 
  fPavTxtSMSigCorssProj    = 0; 
  fPavTxtEv                = 0;
  fPavTxtSigma             = 0;
  fPavTxtEvts              = 0;
  fPavTxtSampTime          = 0;
  fPavTxtEvolEvEv          = 0;
  fPavTxtEvolEvSig         = 0;
  fPavTxtEvolEvCorss       = 0;

  fImpSMFoundEvtsGlobal = 0;
  fImpSMFoundEvtsProj   = 0;
  fImpSMEvEvGlobal      = 0;
  fImpSMEvEvProj        = 0;
  fImpSMEvSigGlobal     = 0;   
  fImpSMEvSigProj       = 0; 
  fImpSMEvCorssGlobal   = 0; 
  fImpSMEvCorssProj     = 0;
  fImpSMSigEvGlobal     = 0;
  fImpSMSigEvProj       = 0; 
  fImpSMSigSigGlobal    = 0;   
  fImpSMSigSigProj      = 0; 
  fImpSMSigCorssGlobal  = 0; 
  fImpSMSigCorssProj    = 0; 
  fImpEv                = 0;
  fImpSigma             = 0;  
  fImpEvts              = 0;     
  fImpSampTime          = 0;
  fImpEvolEvEv          = 0;
  fImpEvolEvSig         = 0;
  fImpEvolEvCorss       = 0;

  fNbBinsProj   = 100;       // number of bins   for histos in option Projection
  fMaxNbColLine = 6;         // max number of colors for histos in option SAME 

  //.................................... Miscellaneous parameters

  fNbOfListFileEvolEvEv    = 0;
  fNbOfListFileEvolEvSig   = 0;
  fNbOfListFileEvolEvCorss = 0;

  fNbOfExistingRuns = 0;

  fFapNbOfRuns    = -1;                   // INIT NUMBER OF RUNS: set to -1
  fFapMaxNbOfRuns = -1;                   // INIT MAXIMUM NUMBER OF RUNS: set to -1 

  MaxCar = fgMaxCar;
  fFapFileRuns.Resize(MaxCar); 
  fFapFileRuns  = "(file with list of runs parameters: no info)";

  GetPathForResultsRootFiles();
  GetPathForListOfRunFiles();

} // end of Init()


//=============================================================================================
//
//                 Set lin or log scale on X or Y axis
//
//=============================================================================================
void TCnaViewEB::SetHistoScaleX(const TString  option_scale)
{
  fFlagScaleX = "LIN";
  if ( option_scale == "LOG" ){fFlagScaleX = "LOG";}
}
void TCnaViewEB::SetHistoScaleY(const TString  option_scale)
{
  fFlagScaleY = "LIN";
  if ( option_scale == "LOG" ){fFlagScaleY = "LOG";}
}

//=============================================================================================
//
//           View matrices
//
//=============================================================================================

//------------------------------------------------------------------------------ Tower x Tower

void TCnaViewEB::CorrelationsBetweenTowers(const TString  option_plot)
{
//Plot of correlation matrix between towers (mean over crystals and samples)

  Int_t   SMtower_X   = 0;
  Int_t   SMtower_Y   = 0;
  Int_t   element     = 0;

  Int_t   opt_cov_cor = fOptMatCor;
  TString SubsetCode  = fOptMtt;

  ViewMatrix(SMtower_X,   SMtower_Y,  element, opt_cov_cor, SubsetCode, option_plot);
}

void TCnaViewEB::CovariancesBetweenTowers(const TString option_plot)
{
//Plot of covariance matrix between towers (mean over crystals and samples)

  Int_t   SMtower_X   = 0;
  Int_t   SMtower_Y   = 0;
  Int_t   element     = 0;

  Int_t   opt_cov_cor = fOptMatCov;
  TString SubsetCode  = fOptMtt;

  ViewMatrix(SMtower_X,   SMtower_Y,  element, opt_cov_cor, SubsetCode, option_plot);
}


//------------------------------------------------------------------------------ Crystal x Crystal

void TCnaViewEB::CorrelationsBetweenCrystals(const Int_t&  SMtower_X, const Int_t& SMtower_Y,
					   const TString option_plot)
{
//Plot of correlation matrix between crystals of two towers (mean over samples)

  Int_t   element     = 0;
  Int_t   opt_cov_cor = fOptMatCor;
  TString SubsetCode  = fOptMcc;

  ViewMatrix(SMtower_X,   SMtower_Y,  element, opt_cov_cor, SubsetCode, option_plot);
}

void TCnaViewEB::CovariancesBetweenCrystals(const Int_t&  SMtower_X, const Int_t& SMtower_Y,
					  const TString option_plot)
{
//Plot of covariance matrix between crystals of two towers (mean over samples)

  Int_t   element     = 0;
  Int_t   opt_cov_cor = fOptMatCov;
  TString SubsetCode  = fOptMcc;

  ViewMatrix(SMtower_X,   SMtower_Y,  element, opt_cov_cor, SubsetCode, option_plot);
}

//------------------------------------------------------------------------------ Sample x Sample

void TCnaViewEB::CorrelationsBetweenSamples(const Int_t&  SMtower_X, const Int_t& TowEcha,
					  const TString option_plot)
{
//Plot of correlation matrix between samples for a given TowEcha

  Int_t   SMtower_Y   = 0;
  Int_t   opt_cov_cor = fOptMatCor;
  TString SubsetCode  = fOptMss;

  ViewMatrix(SMtower_X,  SMtower_Y,  TowEcha, opt_cov_cor, SubsetCode, option_plot);
}

//.............................................................................. Covariances
void TCnaViewEB::CovariancesBetweenSamples(const Int_t&  SMtower_X, const Int_t& TowEcha,
					 const TString option_plot)
{
//Plot of covariance matrix between samples for a given TowEcha

  Int_t   SMtower_Y   = 0;
  Int_t   opt_cov_cor = fOptMatCov;
  TString SubsetCode  = fOptMss;

  ViewMatrix(SMtower_X,  SMtower_Y,  TowEcha, opt_cov_cor, SubsetCode, option_plot);
}
//==============================================================================================
//
//                                       ViewMatrix
//     
//     Two types of call according to element:     element = sample number
//                                              OR element = TowEcha number
//
//     SMtower_X , SMtower_Y , sample number , opt_cov_cor, SubsetCode, option_plot)
//     Output:
//     Plot of the (TowEcha of SMtower_X, TowEcha of SMtower_Y) matrix for sample
//
//     SMtower_X , SMtower_Y (NOT USED) , TowEcha number, opt_cov_cor, SubsetCode, option_plot)
//     Output:
//     Plot of the (sample,sample) matrix for TowEcha of SMtower_X              
//
//
//==============================================================================================
void TCnaViewEB::ViewMatrix(const Int_t&  SMtower_X,  const Int_t& SMtower_Y,
			  const Int_t&  element,    const Int_t& opt_cov_cor,
			  const TString SubsetCode, const TString option_plot)
{
//Plot of matrix

  TCnaReadEB*  MyRootFile = new TCnaReadEB();                              fCnew++;
  MyRootFile->PrintNoComment();

  MyRootFile->GetReadyToReadRootFile(fFapAnaType,    fFapRunNumber, fFapFirstEvt, fFapNbOfEvts,
				     fFapSuMoNumber, fCfgResultsRootFilePath.Data());
  
  if ( MyRootFile->LookAtRootFile() == kTRUE )
    {
      TString fp_name_short = MyRootFile->GetRootFileNameShort();

      // cout << "*TCnaViewEB::ViewMatrix(...)> Data are analyzed from file ----> "
      //      << fp_name_short << endl;
      
      Int_t     nb_of_towers = MyRootFile->MaxTowInSM();
      TVectorD  vtow(nb_of_towers);
      vtow = MyRootFile->ReadTowerNumbers();
      
      if ( MyRootFile->DataExist() == kTRUE )
	{
	  Int_t tower_X_ok = 0;
	  Int_t tower_Y_ok = 0;
	  
	  if( SubsetCode == fOptMtt ){tower_X_ok = 1; tower_Y_ok = 1;}
	  if( SubsetCode == fOptMss ){tower_Y_ok = 1;}
	  
	  for (Int_t index_tow = 0; index_tow < nb_of_towers; index_tow++)
	    {
	      if ( vtow(index_tow) == SMtower_X ){tower_X_ok = 1;}
	      if ( vtow(index_tow) == SMtower_Y ){tower_Y_ok = 1;}
	    }
	  
	  if( tower_X_ok == 1 && tower_Y_ok == 1 )
	    {
	      fStartDate    = MyRootFile->GetStartDate();
	      fStopDate     = MyRootFile->GetStopDate();
	      
	      Int_t   MatSize  = -1; 
	      Int_t   TowEcha  = -1;
	      Int_t   sample   = -1;
	      
	      if( SubsetCode == fOptMtt ){
		MatSize = MyRootFile->MaxTowInSM();}      
	      if( SubsetCode == fOptMcc ){
		MatSize = MyRootFile->MaxCrysInTow(); sample = element;}
	      if( SubsetCode == fOptMss ){
		MatSize = MyRootFile->MaxSampADC();  TowEcha = element;}
	      
	      if( (  SubsetCode == fOptMtt                                                               ) ||
		  ( (SubsetCode == fOptMss) && (TowEcha >= 0) && (TowEcha < MyRootFile->MaxCrysInTow() ) ) ||
	          ( (SubsetCode == fOptMcc) && (sample  >= 0) && (sample  < MyRootFile->MaxSampADC()   ) )    )
		{
		  TMatrixD read_matrix(MatSize, MatSize);
		  
		  if ( SubsetCode == fOptMtt && opt_cov_cor == fOptMatCov )
		    {read_matrix = MyRootFile->ReadCovariancesBetweenTowersMeanOverSamplesAndChannels();}
		  if ( SubsetCode == fOptMtt && opt_cov_cor == fOptMatCor )
		    {read_matrix = MyRootFile->ReadCorrelationsBetweenTowersMeanOverSamplesAndChannels();}
		  if ( SubsetCode == fOptMss && opt_cov_cor == fOptMatCov )
		    {read_matrix = MyRootFile->ReadCovariancesBetweenSamples(SMtower_X, TowEcha);}
		  if ( SubsetCode == fOptMss && opt_cov_cor == fOptMatCor )
		    {read_matrix = MyRootFile->ReadCorrelationsBetweenSamples(SMtower_X, TowEcha);}
		  if ( SubsetCode == fOptMcc && opt_cov_cor == fOptMatCov ){
		    read_matrix = MyRootFile->ReadCovariancesBetweenCrystalsMeanOverSamples(SMtower_X, SMtower_Y);}  
		  if ( SubsetCode == fOptMcc && opt_cov_cor == fOptMatCor )
		    {read_matrix = MyRootFile->ReadCorrelationsBetweenCrystalsMeanOverSamples(SMtower_X, SMtower_Y);}
		  
		  if ( MyRootFile->DataExist() == kTRUE )
		    {
		      //......................... matrix title  (ViewMatrix)
		      char* f_in_mat_tit = new char[fgMaxCar];               fCnew++;
		      
		      if ( SubsetCode == fOptMtt && opt_cov_cor == fOptMatCov )
			{sprintf(f_in_mat_tit, "TOWER x TOWER covariance matrix (mean over crystals & samples)");}
		      if ( SubsetCode == fOptMtt && opt_cov_cor == fOptMatCor )
			{sprintf(f_in_mat_tit, "TOWER x TOWER correlation matrix (mean over crystals & samples)");}
		      if ( SubsetCode == fOptMss && opt_cov_cor == fOptMatCov )
			{sprintf(f_in_mat_tit, "SAMPLE x SAMPLE covariance matrix");}
		      if ( SubsetCode == fOptMss && opt_cov_cor == fOptMatCor )
			{sprintf(f_in_mat_tit, "SAMPLE x SAMPLE correlation matrix");}
		      if ( SubsetCode == fOptMcc && opt_cov_cor == fOptMatCov )
			{sprintf(f_in_mat_tit, "CRYSTAL x CRYSTAL covariance matrix (mean over samples)");}
		      if ( SubsetCode == fOptMcc && opt_cov_cor == fOptMatCor )
			{sprintf(f_in_mat_tit, "CRYSTAL x CRYSTAL correlation matrix (mean over samples)");}
		      
		      //................................. Axis parameters
		      TString axis_x_var_name;
		      TString axis_y_var_name;
		      
		      char* f_in_axis_x = new char[fgMaxCar];               fCnew++;
		      char* f_in_axis_y = new char[fgMaxCar];               fCnew++;
		      
		      if( SubsetCode == fOptMtt)
			{axis_x_var_name = "  Tower number  "; axis_y_var_name = "  Tower number  ";}      
		      if( SubsetCode == fOptMss)
			{axis_x_var_name = " Sample      "; axis_y_var_name = "    Sample ";}
		      if( SubsetCode == fOptMcc)
			{sprintf(f_in_axis_x, " Crystal Tower %d   ", SMtower_X);
			sprintf(f_in_axis_y, " Crystal Tower %d   ", SMtower_Y);
			axis_x_var_name = f_in_axis_x; axis_y_var_name = f_in_axis_y;}
		      
		      Int_t  nb_binx  = MatSize;
		      Int_t  nb_biny  = MatSize;
		      Axis_t xinf_bid = (Axis_t)0.;
		      Axis_t xsup_bid = (Axis_t)MatSize;
		      Axis_t yinf_bid = (Axis_t)0.;
		      Axis_t ysup_bid = (Axis_t)MatSize;   
		      
		      //..............................  histogram filling (ViewMatrix)
		      
		      TH2D* h_fbid0 = new TH2D("bidim", f_in_mat_tit,
					       nb_binx, xinf_bid, xsup_bid,
					       nb_biny, yinf_bid, ysup_bid);     fCnewRoot++;
		      
		      h_fbid0->Reset();
		      
		      //.................................... Ymin and Ymax
		      Int_t  xFlagAutoYsupMargin = 0;
		      
		      if ( opt_cov_cor == fOptMatCor )
			{
			  if (SubsetCode == fOptMss)
			    {
			      // h_fbid0->SetMinimum(GetYmin("SMEvCorssGlobal"));//same Ymin and Ymax as for SM histos
			      // h_fbid0->SetMaximum(GetYmax("SMEvCorssGlobal"));
			      //... histogram set ymin and ymax (from TCnaParameters)    
			      
			      xFlagAutoYsupMargin = HistoSetMinMax((TH1D*)h_fbid0, "SMEvCorssGlobal");
			    }
			  if ( SubsetCode == fOptMtt ||  SubsetCode == fOptMcc )
			    {
			      //h_fbid0->SetMinimum(GetYmin("SMEvCorttMatrix"));//same Ymin and Ymax as for SM histos
			      //h_fbid0->SetMaximum(GetYmax("SMEvCorttMatrix"));
			      
			      //... histogram set ymin and ymax (from TCnaParameters) 
			      xFlagAutoYsupMargin = HistoSetMinMax((TH1D*)h_fbid0, "SMEvCorttMatrix");
			    }
			  
			  // ************************** A GARDER EN RESERVE **************************
			  //............. special contour level for correlations (square root wise scale)
			  //Int_t nb_niv  = 9;
			  // Double_t* cont_niv = new Double_t[nb_niv];                  fCnew++;
			  // SqrtContourLevels(nb_niv, &cont_niv[0]);
			  //h_fbid0->SetContour(nb_niv, &cont_niv[0]);		  
			  // delete [] cont_niv;                                  fCdelete++;
			  // ******************************** (FIN RESERVE) ************************** (ViewMatrix)
			  
			}
		      
		      if ( opt_cov_cor == fOptMatCov )
			{
			  if (SubsetCode == fOptMss)
			    {
			      //h_fbid0->SetMinimum(GetYmin("SMEvCovttMatrix"));//same Ymin and Ymax as for SM histos
			      //h_fbid0->SetMaximum(GetYmax("SMEvCovttMatrix"));
			      
			      //... histogram set ymin and ymax
			      
			      InitQuantityYmin("SMEvSigGlobal");
			      InitQuantityYmax("SMEvSigGlobal");
			      xFlagAutoYsupMargin = HistoSetMinMax((TH1D*)h_fbid0, "SMEvSigGlobal");
			    }
			  if ( SubsetCode == fOptMtt ||  SubsetCode == fOptMcc )
			    {
			      //h_fbid0->SetMinimum(GetYmin("SMEvCovttMatrix"));//same Ymin and Ymax as for SM histos
			      //h_fbid0->SetMaximum(GetYmax("SMEvCovttMatrix"));
			      
			      //... histogram set ymin and ymax (from TCnaParameters) 
			      xFlagAutoYsupMargin = HistoSetMinMax((TH1D*)h_fbid0, "SMEvCovttMatrix");
			    }
			}
		      
		      
		      for(Int_t i = 0 ; i < MatSize ; i++)
			{
			  Double_t xi = (Double_t)i;
			  for(Int_t j = 0 ; j < MatSize ; j++)
			    {
			      Double_t xj      = (Double_t)j;
			      Double_t mat_val = (Double_t)read_matrix(i,j);
			      h_fbid0->Fill(xi, xj, (Stat_t)mat_val);
			    }
			}
		      
		      h_fbid0->GetXaxis()->SetTitle(axis_x_var_name);
		      h_fbid0->GetYaxis()->SetTitle(axis_y_var_name);
		      
		      // --------------------------------------- P L O T S  (ViewMatrix)
		      
		      char* f_in = new char[fgMaxCar];                            fCnew++;
		      
		      //...................... Taille/format canvas
		      
		      UInt_t canv_w = CanvasFormatW("petit");
		      UInt_t canv_h = CanvasFormatH("petit");
		      
		      //............................. options generales 
		      TString QuantityType;
		      Int_t MaxCar = fgMaxCar;
		      QuantityType.Resize(MaxCar);
		      QuantityType = "(no quantity type info)";
		      
		      if (option_plot == "COLZ"  ){QuantityType = "colz";}
		      if (option_plot == "LEGO2Z"){QuantityType = "lego";}
		      if (option_plot == "SURF1Z"){QuantityType = "surf1";}
		      if (option_plot == "SURF4" ){QuantityType = "surf4";}
		      
		      TEBNumbering* MyNumbering = new TEBNumbering();    fCnew++;
		      fFapSuMoBarrel = MyNumbering->GetSMHalfBarrel(fFapSuMoNumber);
		      PutAllPavesViewMatrix(MyRootFile, MyNumbering, SubsetCode, SMtower_X, SMtower_Y, TowEcha);
		      delete MyNumbering;                                  fCdelete++;
		      
		      //---------------------------------------- Canvas name (ViewMatrix)
		      TString  name_cov_cor;
		      MaxCar = fgMaxCar;
		      name_cov_cor.Resize(MaxCar);
		      name_cov_cor = "?";
		      if( opt_cov_cor == fOptMatCov){name_cov_cor = "cov";}
		      if( opt_cov_cor == fOptMatCor){name_cov_cor = "cor";}
		      
		      TString name_chan_samp;
		      MaxCar = fgMaxCar;
		      name_chan_samp.Resize(MaxCar);
		      name_chan_samp = "?";
		      
		      Int_t  num_element = -1;
		      if( SubsetCode == fOptMtt )
			{name_chan_samp = "tt_moc"; num_element = 0;}
		      if( SubsetCode == fOptMss )
			{name_chan_samp = "ss_c"; num_element = TowEcha;}
		      if( SubsetCode == fOptMcc )
			{name_chan_samp = "cc_mos";}
		      
		      TString name_visu;
		      MaxCar = fgMaxCar;
		      name_visu.Resize(MaxCar);
		      name_visu = "?";
		      
		      name_visu = option_plot;
		      
		      if( SubsetCode == fOptMtt ){
			sprintf(f_in, "%s_%s_%d_%s_%s_%d_%d_SM%d",
				name_visu.Data(),
				fFapAnaType.Data(), fFapRunNumber, name_cov_cor.Data(),
				name_chan_samp.Data(), fFapFirstEvt, fFapNbOfEvts, fFapSuMoNumber);}
		      
		      if( SubsetCode == fOptMss ){
			sprintf(f_in, "%s_tx%d_ty%d_%s_%d_%s_%s%d_%d_%d_SM%d",
				name_visu.Data(), SMtower_X, SMtower_Y,
				fFapAnaType.Data(), fFapRunNumber, name_cov_cor.Data(),
				name_chan_samp.Data(), num_element, fFapFirstEvt, fFapNbOfEvts, fFapSuMoNumber);}
		      
		      if( SubsetCode == fOptMcc ){
			sprintf(f_in, "%s_tx%d_ty%d_%s_%d_%s_%s_%d_%d_SM%d",
				name_visu.Data(), SMtower_X, SMtower_Y,
				fFapAnaType.Data(), fFapRunNumber, name_cov_cor.Data(),
				name_chan_samp.Data(), fFapFirstEvt, fFapNbOfEvts, fFapSuMoNumber);}
		      
		      //----------------------------------------------------------	(ViewMatrix)
		      
		      SetHistoPresentation((TH1D*)h_fbid0, QuantityType);
		      
		      TCanvas *NoiseCorrel =
			new TCanvas(f_in, f_in, canv_w , canv_h);   fCnewRoot++;
		      
		      // cout << "*TCnaViewEB::ViewMatrix(...)> Plot is displayed on canvas ----> " << f_in << endl;
		      
		      delete [] f_in; f_in = 0;                         fCdelete++;
		      
		      ftitle_g1->Draw();
		      fcom_top_left->Draw();
		      if(SubsetCode == "Crystal"){fcom_top_mid->Draw();}
		      if(SubsetCode == "Sample") {fcom_top_mid->Draw(); fcom_top_right->Draw();}
		      fcom_bot_left->Draw();
		      fcom_bot_right->Draw();
		      
		      NoiseCorrel->Divide(1, 1, 0.001 , 0.125);
		      gPad->cd(1);
		      
		      TVirtualPad *main_subpad = gPad; 
		      Int_t i_zonx = 1;
		      Int_t i_zony = 1;  
		      main_subpad->Divide(i_zonx,i_zony);
		      gPad->cd(1);  
		      main_subpad->cd(1);
		      
		      //----------------------------------------------------------	(ViewMatrix)      
		      Int_t logy = 0;  
		      gPad->SetLogy(logy);
		      
		      if( SubsetCode == fOptMtt ){gPad->SetGrid(1,1);}
		      
		      h_fbid0->DrawCopy(option_plot);
		      
		      h_fbid0->SetStats((Bool_t)1);    
		      
		      gPad->Update();
		      
		      h_fbid0->Delete();                fCdeleteRoot++;
		      
		      //  title_g1->Delete();                 fCdeleteRoot++;
		      //  com_bot_left->Delete();                  fCdeleteRoot++;
		      //  delete NoiseCorrel;                 fCdeleteRoot++;
		      
		      delete [] f_in_mat_tit; f_in_mat_tit = 0;       fCdelete++;
		      delete [] f_in_axis_x;  f_in_axis_x  = 0;       fCdelete++;
		      delete [] f_in_axis_y;  f_in_axis_y  = 0;       fCdelete++;

		    } // end of if ( MyRootFile->DataExist() == kTRUE )
		}
	      else
		{
		  if(SubsetCode == fOptMss)
		    {
		      cout << "*TCnaViewEB::ViewMatrix(...)> *ERROR* ==> Wrong channel number in tower. Value = "
			   << TowEcha << " (required range: [0, "
			   << MyRootFile->MaxCrysInTow()-1 << "] )"
			   << fTTBELL << endl;
		    }

		  if(SubsetCode == fOptMcc)
		    {
		      cout << "*TCnaViewEB::ViewMatrix(...)> *ERROR* ==> Wrong sample number. Value = "
			   << sample << " (required range: [0, "
			   << MyRootFile->MaxSampADC()-1 << "] )"
			   << fTTBELL << endl;
		    }
		}
	    }
	  else    // else of the if ( tower_X_ok ==1 && tower_Y_ok ==1 )
	    {
	      //----------------------------------------------------------	(ViewMatrix)
	      if ( tower_X_ok != 1 )
		{
		  cout << "*TCnaViewEB::ViewMatrix(...)> *ERROR* =====> "
		       << " Tower_X = " << SMtower_X << ". Tower not found."
		       << " Available numbers = ";
		  for(Int_t i = 0; i < nb_of_towers; i++){if( vtow(i) > 0 ){cout << vtow(i) << ", ";}}
		  cout << fTTBELL << endl;
		}
	      if ( tower_Y_ok != 1 )
		{
		  cout << "*TCnaViewEB::ViewMatrix(...)> *ERROR* =====> "
		       << " Tower_Y = " << SMtower_Y << ". Tower not found."
		       << " Available numbers = ";
		  for(Int_t i = 0; i < nb_of_towers; i++){if( vtow(i) > 0 ){cout << vtow(i) << ", ";}}
		  cout << fTTBELL << endl;
		}
	  
	    }

	}  // end of if ( MyRootFile->DataExist() == kTRUE )
    }
  else
    {
      cout  << "*TCnaViewEB::ViewMatrix(...)> *ERROR* =====> "
	    << " ROOT file not found" << fTTBELL << endl;
    }
  
  delete MyRootFile;                                                  fCdelete++;
  
}  // end of ViewMatrix(...)

//==========================================================================
//
//                         ViewTower   
//   
//==========================================================================

void TCnaViewEB::CorrelationsBetweenSamples(const Int_t& SMtower)
{
  Int_t   opt_cov_cor = fOptMatCor;
  ViewTower(SMtower, opt_cov_cor);
}

void TCnaViewEB::CovariancesBetweenSamples(const Int_t& SMtower)
{
  Int_t   opt_cov_cor = fOptMatCov;
  ViewTower(SMtower, opt_cov_cor);
}

//==========================================================================
//
//                         ViewTower      
//
//  SMtower ==>
//  (sample,sample) cor or cov matrices for all the crystal of SMtower              
//
//
//==========================================================================
void TCnaViewEB::ViewTower(const Int_t& SMtower, const Int_t& opt_cov_cor)
{
  
  TCnaReadEB*  MyRootFile = new TCnaReadEB();                              fCnew++; 
  MyRootFile->PrintNoComment();

  MyRootFile->GetReadyToReadRootFile(fFapAnaType,    fFapRunNumber, fFapFirstEvt, fFapNbOfEvts,
				     fFapSuMoNumber, fCfgResultsRootFilePath.Data());
  
  if ( MyRootFile->LookAtRootFile() == kTRUE )
    {
      TString fp_name_short = MyRootFile->GetRootFileNameShort(); 
      // cout << "*TCnaViewEB::ViewTower(...)> Data are analyzed from file ----> "
      //      << fp_name_short << endl;
      
      TEBParameters* MyEcal = new TEBParameters();   fCnew++;

      Int_t     nb_of_towers = MyEcal->MaxTowInSM();
      TVectorD  vtow(nb_of_towers);
      vtow = MyRootFile->ReadTowerNumbers();
      
      if ( MyRootFile->DataExist() == kTRUE )
	{
	  Int_t tower_ok = 0;
	  for (Int_t index_tow = 0; index_tow < nb_of_towers; index_tow++)
	    {
	      if ( vtow(index_tow) == SMtower ){tower_ok++;}
	    }
	  
	  if( tower_ok == 1)
	    {
	      fStartDate    = MyRootFile->GetStartDate();
	      fStopDate     = MyRootFile->GetStopDate();
	      
	      //......................... matrix title  
	      char* f_in_mat_tit = new char[fgMaxCar];               fCnew++;
	      
	      if ( opt_cov_cor == fOptMatCov )
		{sprintf(f_in_mat_tit, "Covariances between samples for each crystal. Tower view.");}
	      if ( opt_cov_cor == fOptMatCor )
		{sprintf(f_in_mat_tit, "Correlations between samples for each crystal. Tower view.");}
      
	      //................................. Bidim parameters
	      Int_t  GeoBidSize = MyEcal->MaxSampADC()*MyEcal->MaxCrysEtaInTow(); 
	      Int_t  nb_binx  = GeoBidSize;
	      Int_t  nb_biny  = GeoBidSize;
	      Axis_t xinf_bid = (Axis_t)0.;
	      Axis_t xsup_bid = (Axis_t)GeoBidSize;
	      Axis_t yinf_bid = (Axis_t)0.;
	      Axis_t ysup_bid = (Axis_t)GeoBidSize;   
      
	      //--------------------------------------------------------- (ViewTower)
	      //............. matrices reading and histogram filling
      
	      TH2D* h_geo_bid = new TH2D("geobidim", f_in_mat_tit,
					 nb_binx, xinf_bid, xsup_bid,
					 nb_biny, yinf_bid, ysup_bid);     fCnewRoot++;
	  
	      h_geo_bid->Reset();
	  
	      if ( opt_cov_cor == fOptMatCor )
		{
		  // h_geo_bid->SetMinimum(GetYmin("SMEvCorssGlobal"));
		  // same Ymin and Ymax as for SM histograms
		  // h_geo_bid->SetMaximum(GetYmax("SMEvCorssGlobal"));
	      
		  //... histogram set ymin and ymax (from TCnaParameters)
		  Int_t  xFlagAutoYsupMargin = 0;      
		  xFlagAutoYsupMargin = HistoSetMinMax((TH1D*)h_geo_bid, "SMEvCorssGlobal");

	      
		  // ************************** A GARDER EN RESERVE *******************************
		  //............. special  contour level for correlations (square root wise scale)
		  //Int_t nb_niv  = 9;
		  //Double_t* cont_niv = new Double_t[nb_niv];                  fCnew++;
		  //SqrtContourLevels(nb_niv, &cont_niv[0]);
		  //h_geo_bid->SetContour(nb_niv, &cont_niv[0]);	      
		  //delete [] cont_niv;                                  fCdelete++;
		  // ******************************** (FIN RESERVE) *******************************
		}

	      if ( opt_cov_cor == fOptMatCov )
		{
		  //h_geo_bid->SetMinimum(GetYmin("SMEvCovttMatrix"));
		  //same Ymin and Ymax as for SM Cov(tower,tower)
		  //h_geo_bid->SetMaximum(GetYmax("SMEvCovttMatrix")); 

		  //... histogram set ymin and ymax (from TCnaParameters)
		  Int_t  xFlagAutoYsupMargin = 0;      
		  xFlagAutoYsupMargin = HistoSetMinMax((TH1D*)h_geo_bid, "SMEvSigGlobal");  
		}

	      //======================================================== (ViewTower)
	  
	      //----------------------------------------------- Geographical bidim filling

	      Int_t  MatSize = MyEcal->MaxSampADC();
	      TMatrixD read_matrix(MatSize, MatSize);

	      TEBNumbering* MyNumbering = new TEBNumbering();    fCnew++;
	      fFapSuMoBarrel = MyNumbering->GetSMHalfBarrel(fFapSuMoNumber);

	      Int_t i_data_exist = 0;

	      for(Int_t n_crys = 0; n_crys < MyEcal->MaxCrysInTow(); n_crys++)
		{
		  if( opt_cov_cor == fOptMatCov )
		    {read_matrix = MyRootFile->ReadCovariancesBetweenSamples(SMtower, n_crys);}
		  if ( opt_cov_cor == fOptMatCor )
		    {read_matrix = MyRootFile->ReadCorrelationsBetweenSamples(SMtower, n_crys);}
		  
		  if( MyRootFile->DataExist() == kFALSE )
		    {
		      cout << "*TCnaViewEB::ViewTower> Exiting loop over the channels." << endl;
		      break;
		    }
		  else
		    {
		      i_data_exist++;
		      for(Int_t i_samp = 0 ; i_samp < MatSize ; i_samp++)
			{
			  Int_t i_xgeo = GetXSampInTow(MyNumbering, MyEcal, fFapSuMoNumber,
						       SMtower,     n_crys, i_samp);
			  for(Int_t j_samp = 0; j_samp < MatSize ; j_samp++)
			    {
			      Int_t j_ygeo = GetYSampInTow(MyNumbering, MyEcal, fFapSuMoNumber,
							   SMtower,     n_crys, j_samp);
			      h_geo_bid->Fill((Double_t)i_xgeo, (Double_t)j_ygeo,
					      (Stat_t)read_matrix(i_samp, j_samp));
			    }
			}
		    }
		}
	      
	      if( i_data_exist > 0 )
		{
		  // ------------------------------------------------ P L O T S   (ViewTower)
		  
		  char* f_in = new char[fgMaxCar];                           fCnew++;
		  
		  //...................... Taille/format canvas
		  
		  UInt_t canv_w = CanvasFormatW("petit");
		  UInt_t canv_h = CanvasFormatH("petit");
		  
		  //.................................................. paves commentaires (ViewTower)	  
		  PutAllPavesViewTower(MyRootFile, SMtower);
		  
		  //------------------------------------ Canvas name ----------------- (ViewTower)  
		  TString name_cov_cor;
		  Int_t MaxCar = fgMaxCar;
		  name_cov_cor.Resize(MaxCar);
		  name_cov_cor = "?";
		  if( opt_cov_cor == fOptMatCov){name_cov_cor = "cov_vtow";}
		  if( opt_cov_cor == fOptMatCor){name_cov_cor = "cor_vtow";}
		  
		  TString name_visu;
		  MaxCar = fgMaxCar;
		  name_visu.Resize(MaxCar);
		  name_visu = "colz";
		  
		  sprintf(f_in, "%s_%d_%s_%d_%s_%d_%d_SM%d",
			  name_visu.Data(), SMtower, fFapAnaType.Data(), fFapRunNumber,
			  name_cov_cor.Data(),fFapFirstEvt, fFapNbOfEvts, fFapSuMoNumber); 
		  
		  SetHistoPresentation((TH1D*)h_geo_bid, "tower");
		  
		  TCanvas *NoiseCorrel = new TCanvas(f_in, f_in, canv_w, canv_h);   fCnewRoot++;
		  
		  // cout << "*TCnaViewEB::ViewTower(...)> Plot is displayed on canvas ----> " << f_in << endl;
		  
		  delete [] f_in; f_in = 0;                                 fCdelete++;
		  
		  //------------------------ Canvas draw and update ------------ (ViewTower)  
		  ftitle_g1->Draw();
		  fcom_top_left->Draw();
		  fcom_top_mid->Draw();
		  fcom_bot_left->Draw();
		  fcom_bot_right->Draw();
		  
		  //	  Bool_t b_true = 1;  Bool_t b_false = 0;
		  
		  NoiseCorrel->Divide(1, 1, 0.001 , 0.125);
		  gPad->cd(1);
		  
		  TVirtualPad *main_subpad = gPad;
		  Int_t i_zonx = 1;
		  Int_t i_zony = 1;
		  main_subpad->Divide(i_zonx,i_zony);
		  gPad->cd(1);
		  main_subpad->cd(1);
		  
		  Int_t logy = 0;  
		  gPad->SetLogy(logy);
		  
		  h_geo_bid->DrawCopy("COLZ");
		  
		  //--------------------------------------------------------------------------- (ViewTower)       
		  Int_t size_eta    = MyEcal->MaxCrysEtaInTow();
		  Int_t size_phi    = MyEcal->MaxCrysPhiInTow();
		  
		  ViewTowerGrid(MyNumbering, fFapSuMoNumber, SMtower, MatSize, size_eta, size_phi, " ");
		  
		  gPad->Update();
		  
		  h_geo_bid->SetStats((Bool_t)1);    

		  //      delete title_g1;                 fCdeleteRoot++;
		  //      delete com_bot_left;             fCdeleteRoot++;
		  //      delete NoiseCorrel;              fCdeleteRoot++;
		  
		}
	      delete MyNumbering;                              fCdelete++;
	      delete [] f_in_mat_tit;                          fCdelete++;
		  
	      h_geo_bid->Delete();                fCdeleteRoot++;
	    }
	  else
	    {
	      cout << "!TCnaViewEB::ViewTower(...)> *ERROR* =====> "
		   << " Tower " << SMtower << " not found."
		   << " Available numbers = ";
	      for(Int_t i = 0; i < nb_of_towers; i++){if( vtow(i) > 0 ){cout << vtow(i) << ", ";}}
	      cout << fTTBELL << endl;  
	    }
	}  // end of if ( myRootFile_>DataExist() == kTRUE )
      delete MyEcal;                                   fCdelete++;
    }
  else
    {
      cout << "!TCnaViewEB::ViewTower(...)> *ERROR* =====> "
	   << " ROOT file not found" << fTTBELL << endl;
    }
  
  delete MyRootFile;                                                   fCdelete++;
  
}  // end of ViewTower(...)

//====================================================================================
//
//                         TowerCrystalNumbering
//              independent of the ROOT file => SMNumber as argument
//
//====================================================================================  

void TCnaViewEB::TowerCrystalNumbering(const Int_t& SMNumber, const Int_t& SMtower)
{
  //display the crystal numbering of one tower

  TEBNumbering* MyNumbering = new TEBNumbering();             fCnew++;
  fFapSuMoBarrel = MyNumbering->GetSMHalfBarrel(fFapSuMoNumber);

  TEBParameters* MyEcalParameters = new TEBParameters();    fCnew++; 

  Int_t MatSize   = MyEcalParameters->MaxSampADC();
  Int_t size_eta  = MyEcalParameters->MaxCrysEtaInTow();
  Int_t size_phi  = MyEcalParameters->MaxCrysPhiInTow();

  //---------------------------------- bidim

  Int_t nb_bins  = MyEcalParameters->MaxSampADC();
  Int_t nx_gbins = nb_bins*size_eta;
  Int_t ny_gbins = nb_bins*size_phi;

  Axis_t xinf_gbid = (Axis_t)0.;
  Axis_t xsup_gbid = (Axis_t)MyEcalParameters->MaxSampADC()*size_eta;
  Axis_t yinf_gbid = (Axis_t)0.;
  Axis_t ysup_gbid = (Axis_t)MyEcalParameters->MaxSampADC()*size_phi;

  char* fg_name = "M0' crystals";
  char* fg_tit  = "Crystal numbering (electronic channel + numbers in SM)"; 
 
  TH2D *h_gbid;
  h_gbid = new TH2D(fg_name,  fg_tit,
		    nx_gbins, xinf_gbid, xsup_gbid,
		    ny_gbins, yinf_gbid, ysup_gbid);    fCnewRoot++;
  h_gbid->Reset();

  //-----------------  T R A C E  D E S   P L O T S ------ (TowerCrystalNumbering)

  char* f_in = new char[fgMaxCar];                           fCnew++;
	  
  //...................... Taille/format canvas
  
  UInt_t canv_w = CanvasFormatW("petit");
  UInt_t canv_h = CanvasFormatH("petit");

  //........................................ couleurs
  //Color_t couleur_noir       = ColorDefinition("noir");
  //Color_t couleur_rouge      = ColorDefinition("rouge");
  //Color_t couleur_bleu_fonce = ColorDefinition("bleu_fonce");

  Color_t couleur_noir       = SetColorsForNumbers("crystal");
  Color_t couleur_rouge      = SetColorsForNumbers("lvrb_top");
  Color_t couleur_bleu_fonce = SetColorsForNumbers("lvrb_bottom");

  gStyle->SetPalette(1,0);          // Rainbow spectrum

  //.................................... options generales
  SetViewHistoStyle("tower");
          
  //.................................... paves commentaires (TowerCrystalNumbering)
  
  PutAllPavesViewTowerCrysNb(MyNumbering, SMNumber, SMtower);

  //---------------------------------------------- (TowerCrystalNumbering)

  //..................... Canvas name
  sprintf(f_in, "crystal_numbering_for_tower_X_%d_SM%d", SMtower, SMNumber);
  
  SetHistoPresentation((TH1D*)h_gbid, "tower");

  TCanvas *NoiseCor1 = new TCanvas(f_in, f_in, canv_w , canv_h);    fCnewRoot++;
  // cout << "*TCnaViewEB::TowerCrystalNumbering(...)> Plot is displayed on canvas ----> "
  //      << f_in << endl;
  
  NoiseCor1->Divide(1, 1, 0.001 , 0.125);

  fcom_top_left->Draw();
  fcom_top_mid->Draw();
  fcom_bot_mid->Draw();
  
  Bool_t b_true = 1; 
  Bool_t b_false = 0;
  gPad->cd(1);
  
  TVirtualPad *main_subpad = gPad;  
  Int_t i_zonx = 1;
  Int_t i_zony = 1;  
  main_subpad->Divide(i_zonx, i_zony);
  main_subpad->cd(1); 

  gStyle->SetMarkerColor(couleur_rouge);
  
  Int_t logy = 0;
  gPad->SetLogy(logy);
  
  //............................... bidim .......... (TowerCrystalNumbering)   

  h_gbid->SetStats(b_false); 

  SetViewHistoOffsets((TH1D*)h_gbid, "tower");
    
  h_gbid->DrawCopy("COLZ");
    
  //..... Ecriture des numeros de channels dans la grille..... (TowerCrystalNumbering)
  //      et des numeros SM des cristaux

  //............... prepa arguments fixes appels [TText]->DrawText()
  char* f_in_elec = new char[fgMaxCar];                                           fCnew++;
  TString TowerLvrbType = MyNumbering->GetTowerLvrbType(SMtower) ;
  TText *text_elec_num   = new TText();                                           fCnewRoot++;
  if ( TowerLvrbType == "top"    ){text_elec_num->SetTextColor(couleur_rouge);}
  if ( TowerLvrbType == "bottom" ){text_elec_num->SetTextColor(couleur_bleu_fonce);}
  text_elec_num->SetTextSize(0.06);

  char* f_in_sm = new char[fgMaxCar];                                             fCnew++;
  TText *text_sm_num = new TText();                                               fCnewRoot++;
  text_sm_num->SetTextColor(couleur_noir);
  text_sm_num->SetTextSize(0.04);

  //............... prepa arguments fixes appels GetXGeo(...) et GetYGeo(...)
  Int_t    i_samp  = 0;
  Double_t off_set = (Double_t)(MyEcalParameters->MaxSampADC()/3);

  //------------------ LOOP ON THE CRYSTAL ELECTRONIC CHANNEL NUMBER  (TowerCrystalNumbering)

  for (Int_t i_chan = 0; i_chan < MyEcalParameters->MaxCrysInTow(); i_chan++)
    {
      Int_t i_xgeo = GetXSampInTow(MyNumbering, MyEcalParameters, fFapSuMoNumber,
				   SMtower,     i_chan,           i_samp);
      Int_t i_ygeo = GetYSampInTow(MyNumbering, MyEcalParameters, fFapSuMoNumber,
				   SMtower,     i_chan,           i_samp);

      Double_t xgi    =  i_xgeo + off_set;
      Double_t ygj    =  i_ygeo + 2*off_set;

      Double_t xgi_sm =  i_xgeo + off_set;
      Double_t ygj_sm =  i_ygeo + off_set;

      Int_t i_crys_sm = MyNumbering->GetSMCrysFromSMTowAndTowEcha(SMtower, i_chan);

      //------------------------------------------------------- TowerCrystalNumbering

      sprintf(f_in_elec, "%d", i_chan);
      text_elec_num->DrawText(xgi, ygj, f_in_elec);

      sprintf(f_in_sm, "%d", i_crys_sm);
      text_sm_num->DrawText(xgi_sm, ygj_sm, f_in_sm);
    }
  text_sm_num->Delete();               fCdeleteRoot++;
  text_elec_num->Delete();             fCdeleteRoot++;

  ViewTowerGrid(MyNumbering, SMNumber, SMtower, MatSize, size_eta, size_phi, "CrystalNumbering");

  gPad->Update();
  h_gbid->SetStats(b_true);

  h_gbid->Delete();                             fCdeleteRoot++;

  delete [] f_in;       f_in      = 0;          fCdelete++; 
  delete [] f_in_sm;    f_in_sm   = 0;          fCdelete++;
  delete [] f_in_elec;  f_in_elec = 0;          fCdelete++;

  delete MyNumbering;                           fCdelete++;
  delete MyEcalParameters;                      fCdelete++;
}
//---------------->  end of TowerCrystalNumbering()

//==================================================================================
//
//                       GetXSampInTow, GetYSampInTow
//
//==================================================================================
Int_t TCnaViewEB::GetXSampInTow(TEBNumbering* MyNumbering, TEBParameters* MyEcal,
			      const Int_t&   SMNumber,    const Int_t&     SMtower,
			      const Int_t&   i_TowEcha,   const Int_t&     i_samp) 
{
//Gives the X coordinate in the geographic view of one tower

  Int_t ix_geo = -1;
  TString ctype = MyNumbering->GetTowerLvrbType(SMtower);
  TString btype = MyNumbering->GetSMHalfBarrel(SMNumber);

  if( (btype == "barrel+" && ctype == "bottom")  || (btype == "barrel-" && ctype == "top") )
    {ix_geo = ( (MyEcal->MaxCrysEtaInTow()-1)-(i_TowEcha/MyEcal->MaxCrysEtaInTow()) )
       *MyEcal->MaxSampADC() + i_samp;}
  
  if( (btype == "barrel+" &&  ctype  == "top")   || (btype == "barrel-" && ctype == "bottom") )
    {ix_geo = ( i_TowEcha/MyEcal->MaxCrysEtaInTow() )*MyEcal->MaxSampADC() + i_samp;}
  
  return ix_geo;
}
//--------------------------------------------------------------------------------------------
Int_t TCnaViewEB::GetYSampInTow(TEBNumbering* MyNumbering, TEBParameters* MyEcal,
			      const Int_t&   SMNumber,    const Int_t&     SMtower,
			      const Int_t&   i_TowEcha,   const Int_t&     j_samp)
{
//Gives the Y coordinate in the geographic view of one tower

  Int_t jy_geo = -1;
  TString ctype = MyNumbering->GetTowerLvrbType(SMtower);
  TString btype = MyNumbering->GetSMHalfBarrel(SMNumber);

  //.......................... jy_geo for the barrel+ (and beginning for the barrel-)

  if( (btype == "barrel+" && ctype == "top")    ||  (btype == "barrel-" && ctype == "bottom") )
    {
      if( i_TowEcha >=  0 && i_TowEcha <=  4 ) {jy_geo =  (i_TowEcha -  0)*MyEcal->MaxSampADC() + j_samp;}
      if( i_TowEcha >=  5 && i_TowEcha <=  9 ) {jy_geo = -(i_TowEcha -  9)*MyEcal->MaxSampADC() + j_samp;}
      if( i_TowEcha >= 10 && i_TowEcha <= 14 ) {jy_geo =  (i_TowEcha - 10)*MyEcal->MaxSampADC() + j_samp;}
      if( i_TowEcha >= 15 && i_TowEcha <= 19 ) {jy_geo = -(i_TowEcha - 19)*MyEcal->MaxSampADC() + j_samp;}
      if( i_TowEcha >= 20 && i_TowEcha <= 24 ) {jy_geo =  (i_TowEcha - 20)*MyEcal->MaxSampADC() + j_samp;}
    }

  if( (btype == "barrel+" && ctype == "bottom") ||  (btype == "barrel-" && ctype == "top") )
    {
      if( i_TowEcha >=  0 && i_TowEcha <=  4 )
	{jy_geo = ( (MyEcal->MaxCrysPhiInTow()-1) - (i_TowEcha- 0))*MyEcal->MaxSampADC() + j_samp;}

      if( i_TowEcha >=  5 && i_TowEcha <=  9 )
	{jy_geo = ( (MyEcal->MaxCrysPhiInTow()-1) + (i_TowEcha- 9))*MyEcal->MaxSampADC() + j_samp;}

      if( i_TowEcha >= 10 && i_TowEcha <= 14 )
	{jy_geo = ( (MyEcal->MaxCrysPhiInTow()-1) - (i_TowEcha-10))*MyEcal->MaxSampADC() + j_samp;}

      if( i_TowEcha >= 15 && i_TowEcha <= 19 )
	{jy_geo = ( (MyEcal->MaxCrysPhiInTow()-1) + (i_TowEcha-19))*MyEcal->MaxSampADC() + j_samp;}

      if( i_TowEcha >= 20 && i_TowEcha <= 24 )
	{jy_geo = ( (MyEcal->MaxCrysPhiInTow()-1) - (i_TowEcha-20))*MyEcal->MaxSampADC() + j_samp;}
    }

  return jy_geo;
}

//===============================================================================
//
//                           ViewTowerGrid
//              independent of the ROOT file => SMNumber as argument
//
//===============================================================================
void TCnaViewEB::ViewTowerGrid(TEBNumbering* MyNumbering, const Int_t&  SMNumber, 
			     const Int_t&   SMtower,     const Int_t&  MatSize,
			     const Int_t&   size_eta,    const Int_t&  size_phi,
			     const TString  chopt)
{
  //Grid of one tower with axis eta and phi
  //.......................... lignes verticales
  Double_t xline = 0.;
  
  Double_t yline_bot = 0.;
  Double_t yline_top = (Double_t)(MatSize*size_eta);
  
  for( Int_t i = 0 ; i < size_eta ; i++)
    {  
      xline = xline + (Double_t)MatSize;
      TLine *lin;
      lin = new TLine(xline, yline_bot, xline, yline_top); fCnewRoot++;
      lin->Draw();
      // delete lin;             fCdeleteRoot++;
    }
  //............................. lignes horizontales
  Double_t xline_left  = 0;
  Double_t xline_right = (Double_t)(MatSize*size_eta);
  
  Double_t yline = -(Double_t)MatSize;
  
  for( Int_t j = 0 ; j < size_eta+1 ; j++)
    {
      yline = yline + (Double_t)MatSize;
      TLine *lin;
      lin = new TLine(xline_left, yline, xline_right, yline); fCnewRoot++;
      lin->Draw();
      // delete lin;             fCdeleteRoot++;
    }
 
  //------------------ trace axes en eta et phi --------------- ViewTowerGrid

  //...................................................... Axe i(eta) (x bottom)  (ViewTowerGrid)

  Double_t eta_min = MyNumbering->GetIEtaMin(SMNumber, SMtower);
  Double_t eta_max = MyNumbering->GetIEtaMax(SMNumber, SMtower);

  TString  x_var_name  = GetEtaPhiAxisTitle("ietaTow");
  TString  x_direction = MyNumbering->GetXDirection(SMNumber);

  TF1 *f1 = new TF1("f1", x_direction, eta_min, eta_max);                fCnewRoot++;

  TGaxis* sup_axis_x = 0;

  if ( x_direction == "-x" )   // NEVER  IN THIS CASE: xmin->xmax <=> right->left ("-x") direction
    {sup_axis_x = new TGaxis( -(Float_t)MatSize, (Float_t)0, (Float_t)(size_eta*MatSize), (Float_t)0.,
			      "f1", size_eta, "BC" , 0.);                                fCnewRoot++;
    cout << "TCnaViewEB::ViewTowerGrid()> non foreseen case. eta with -x direction." << fTTBELL << endl;}

  if ( x_direction == "x" )    // ALWAYS IN THIS CASE: xmin->xmax <=> left->right ("x") direction
    {sup_axis_x = new TGaxis( (Float_t)0.      , (Float_t)0., (Float_t)(size_eta*MatSize), (Float_t)0.,
			      "f1", size_eta, "C" , 0.);                                fCnewRoot++;}
  
  sup_axis_x->SetTitle(x_var_name);
  sup_axis_x->SetTitleSize((Float_t)0.05);
  sup_axis_x->SetTitleOffset((Float_t)0.05);
  sup_axis_x->SetLabelSize((Float_t)0.04);
  sup_axis_x->SetLabelOffset((Float_t)0.02);
  sup_axis_x->SetTickSize((Float_t)0.0001);                  // <===== NE MARCHE PAS
  sup_axis_x->Draw("SAME");
  f1 = 0;

  //...................................................... Axe phi (y right)  (ViewTowerGrid)
  if( chopt == "CrystalNumbering" )
    {
      Double_t phi_min     = MyNumbering->GetPhiMin(SMNumber, SMtower);
      Double_t phi_max     = MyNumbering->GetPhiMax(SMNumber, SMtower);
      
      TString  y_var_name  = GetEtaPhiAxisTitle("phi");
      TString  y_direction = MyNumbering->GetYDirection(SMNumber);
      
      TF1 *f2 = new TF1("f2", y_direction, phi_min, phi_max);               fCnewRoot++;
      TGaxis* sup_axis_y = 0;
      
      if ( y_direction == "-x" )  // ALWAYS IN THIS CASE: ymin->ymax <=> top->bottom ("-x") direction
	{sup_axis_y = new TGaxis( (Float_t)(size_eta*MatSize), (Float_t)0.,
				  (Float_t)(size_eta*MatSize), (Float_t)(size_phi*MatSize),
				  "f2", size_phi, "+C", 0.);                fCnewRoot++;}
      
      if ( y_direction == "x" )   // NEVER  IN THIS CASE: ymin->ymax <=> bottom->top ("x") direction
	{sup_axis_y = new TGaxis( (Float_t)0.,  (Float_t)0., (Float_t) 0., (Float_t)(size_phi*MatSize),
				  "f2", size_phi, "BC", 0.);                fCnewRoot++;}
      
      sup_axis_y->SetTitle(y_var_name);
      sup_axis_y->SetTitleSize(0.05);
      sup_axis_y->SetTitleOffset(-0.085);
      sup_axis_y->SetLabelSize(0.04);
      sup_axis_y->SetLabelOffset(0.04);
      sup_axis_y->SetTickSize((Float_t)0.);                  // <===== NE MARCHE PAS
      sup_axis_y->Draw("SAME");
      f2 = 0;
    }
  //...................................................... Axe j(phi) (y left)  (ViewTowerGrid)

  Double_t j_phi_min     = MyNumbering->GetJPhiMin(SMNumber, SMtower);
  Double_t j_phi_max     = MyNumbering->GetJPhiMax(SMNumber, SMtower);

  TString  jy_var_name  = GetEtaPhiAxisTitle("jphiTow");
  TString  jy_direction = MyNumbering->GetJYDirection(SMNumber);

  TF1 *f3 = new TF1("f3", jy_direction, j_phi_min, j_phi_max);               fCnewRoot++;
  TGaxis* sup_axis_jy = 0;

  sup_axis_jy = new TGaxis( (Float_t)0., (Float_t)0.,
			    (Float_t)0., (Float_t)(size_phi*MatSize),
			    "f3", size_phi, "C", 0.);                fCnewRoot++;
  
  sup_axis_jy->SetTitle(jy_var_name);
  sup_axis_jy->SetTitleSize(0.05);
  sup_axis_jy->SetTitleOffset(-0.085);
  sup_axis_jy->SetLabelSize(0.04);
  sup_axis_jy->SetLabelOffset(0.04);
  sup_axis_jy->SetTickSize((Float_t)0.);                  // <===== NE MARCHE PAS
  sup_axis_jy->Draw("SAME");
  f3 = 0;

} // end of ViewTowerGrid

//=======================================================================================
//
//                                  EtaPhiSuperModule...
//
//=======================================================================================
void TCnaViewEB::EtaPhiSuperModuleFoundEvents(){ViewSuperModule("SMFoundEvtsGlobal");}
void TCnaViewEB::EtaPhiSuperModuleMeanPedestals(){ViewSuperModule("SMEvEvGlobal");}
void TCnaViewEB::EtaPhiSuperModuleMeanOfSampleSigmas(){ViewSuperModule("SMEvSigGlobal");}
void TCnaViewEB::EtaPhiSuperModuleMeanOfCorss(){ViewSuperModule("SMEvCorssGlobal");}
void TCnaViewEB::EtaPhiSuperModuleSigmaPedestals(){ViewSuperModule("SMSigEvGlobal");}
void TCnaViewEB::EtaPhiSuperModuleSigmaOfSampleSigmas(){ViewSuperModule("SMSigSigGlobal");}
void TCnaViewEB::EtaPhiSuperModuleSigmaOfCorss(){ViewSuperModule("SMSigCorssGlobal");}
//=======================================================================================
//
//                              ViewSuperModule      
//
//           (eta,phi) matrices for all the towers of a super-module             
//
//
//=======================================================================================
void TCnaViewEB::ViewSuperModule(const TString QuantityCode)
{
// (eta, phi) matrices for all the towers of a super-module

  TCnaReadEB*  MyRootFile = new TCnaReadEB();                              fCnew++; 
  MyRootFile->PrintNoComment();

  MyRootFile->GetReadyToReadRootFile(fFapAnaType,    fFapRunNumber, fFapFirstEvt, fFapNbOfEvts,
				     fFapSuMoNumber, fCfgResultsRootFilePath.Data());
  
  if ( MyRootFile->LookAtRootFile() == kTRUE )
    {
      TString QuantityType;
      Int_t MaxCar = fgMaxCar;
      QuantityType.Resize(MaxCar);
      QuantityType = "(no quantity type info)";

      TString fp_name_short = MyRootFile->GetRootFileNameShort(); 
      // cout << "*TCnaViewEB::ViewSuperModule(...)> Data are analyzed from file ----> "
      //      << fp_name_short << endl;

      fStartDate    = MyRootFile->GetStartDate();
      fStopDate     = MyRootFile->GetStopDate();
      
      //......................... matrix title  
      char* f_in_mat_tit = new char[fgMaxCar];               fCnew++;
      
      if (QuantityCode == "SMFoundEvtsGlobal") {sprintf(f_in_mat_tit, "Number of events (mean over samples)");}
      if (QuantityCode == "SMEvEvGlobal"     ) {sprintf(f_in_mat_tit, "Mean of sample means (mean pedestal)");}
      if (QuantityCode == "SMEvSigGlobal"    ) {sprintf(f_in_mat_tit, "Mean of sample sigmas");}
      if (QuantityCode == "SMEvCorssGlobal"  ) {sprintf(f_in_mat_tit, "Mean of cor(s,s)");}
      if (QuantityCode == "SMSigEvGlobal"    ) {sprintf(f_in_mat_tit, "Sigma of sample means");}
      if (QuantityCode == "SMSigSigGlobal"   ) {sprintf(f_in_mat_tit, "Sigma of sample sigmas");}
      if (QuantityCode == "SMSigCorssGlobal" ) {sprintf(f_in_mat_tit, "Sigma of cor(s,s)");}
      
      //................................. Axis parameters

      TEBParameters* MyEcal = new TEBParameters();   fCnew++;

      Int_t  GeoBidSizeEta = MyEcal->MaxTowEtaInSM()*MyEcal->MaxCrysEtaInTow();
      Int_t  GeoBidSizePhi = MyEcal->MaxTowPhiInSM()*MyEcal->MaxCrysPhiInTow();

      Int_t  nb_binx  = GeoBidSizeEta;
      Int_t  nb_biny  = GeoBidSizePhi;
      Axis_t xinf_bid = (Axis_t)0.;
      Axis_t xsup_bid = (Axis_t)GeoBidSizeEta;
      Axis_t yinf_bid = (Axis_t)0.;
      Axis_t ysup_bid = (Axis_t)GeoBidSizePhi;   
      
      TString axis_x_var_name = "  #eta  ";
      TString axis_y_var_name = "  #varphi  ";

      //--------------------------------------------------------- (ViewSuperModule)

      //............. matrices reading and histogram filling
      
      TH2D* h_geo_bid = new TH2D("geobidim_eta_phi", f_in_mat_tit,
				 nb_binx, xinf_bid,  xsup_bid,
				 nb_biny, yinf_bid,  ysup_bid);     fCnewRoot++;

      h_geo_bid->Reset();

      //... histogram set ymin and ymax (from TCnaParameters)
      Int_t  xFlagAutoYsupMargin = 0;      
      xFlagAutoYsupMargin = HistoSetMinMax((TH1D*)h_geo_bid, QuantityCode);


      // ************************** A GARDER EN RESERVE *******************************
      //............. special contour level for correlations (square root wise scale)
      //if ( QuantityCode == "SMEvCorssGlobal" )
      //{
      //  Int_t nb_niv  = 9;
      //  Double_t* cont_niv = new Double_t[nb_niv];           fCnew++;
      //  SqrtContourLevels(nb_niv, &cont_niv[0]);      
      //  h_geo_bid->SetContour(nb_niv, &cont_niv[0]);	      
      //  delete [] cont_niv;                                  fCdelete++;
      //}
      // ******************************** (FIN RESERVE) *******************************

      //======================================================== (ViewSuperModule)
	  
      Int_t nb_of_towers   = MyEcal->MaxTowInSM();
      Int_t nb_crys_in_tow = MyEcal->MaxCrysInTow();
      Int_t nb_of_samples  = MyEcal->MaxSampADC();
      Int_t nb_crys_in_sm  = MyEcal->MaxCrysInSM();
      
      TVectorD partial_histo(nb_crys_in_tow);
      TVectorD partial_histp(nb_crys_in_sm);

      TMatrixD partial_matrix(nb_crys_in_tow, nb_of_samples);
      TMatrixD read_matrix(nb_binx, nb_biny);

      if (QuantityCode == "SMEvEvGlobal" ){
	partial_histp = MyRootFile->ReadExpectationValuesOfExpectationValuesOfSamples();}
      if (QuantityCode == "SMEvSigGlobal" ){
	partial_histp = MyRootFile->ReadExpectationValuesOfSigmasOfSamples();}
      if (QuantityCode == "SMEvCorssGlobal" ){
	partial_histp = MyRootFile->ReadExpectationValuesOfCorrelationsBetweenSamples();}
      if (QuantityCode == "SMSigEvGlobal" ){
	partial_histp = MyRootFile->ReadSigmasOfExpectationValuesOfSamples();}
      if (QuantityCode == "SMSigSigGlobal" ){
	partial_histp = MyRootFile->ReadSigmasOfSigmasOfSamples();}
      if (QuantityCode == "SMSigCorssGlobal" ){
	partial_histp = MyRootFile->ReadSigmasOfCorrelationsBetweenSamples();}

      if ( MyRootFile->DataExist() == kTRUE )
	{
	  TEBNumbering* MyNumbering = new TEBNumbering();                fCnew++;
	  fFapSuMoBarrel = MyNumbering->GetSMHalfBarrel(fFapSuMoNumber);
	  
	  for(Int_t i_tow=0; i_tow<nb_of_towers; i_tow++)
	    {
	      Int_t SMtow = MyRootFile->GetSMTowFromIndex(i_tow);
	      if (SMtow != -1)
		{
		  if (QuantityCode == "SMFoundEvtsGlobal" )
		    {
		      partial_matrix = MyRootFile->ReadNumbersOfFoundEventsForSamples(SMtow);
		      
		      for(Int_t i_crys=0; i_crys<nb_crys_in_tow; i_crys++)
			{
			  //.... average value over the samples
			  partial_histo(i_crys) = 0;
			  for(Int_t i_samp=0; i_samp<nb_of_samples; i_samp++)
			    {
			      partial_histo(i_crys) = partial_histo(i_crys) + partial_matrix(i_crys, i_samp);
			    }
			  partial_histo(i_crys) = partial_histo(i_crys)/nb_of_samples;
			}
		    }
		  
		  //======================================================== (ViewSuperModule)
		  
		  //------------------ Geographical bidim filling (different from the case of a tower,
		  //                   here EB+ and EB- differences are important)
		  
		  for(Int_t i_TowEcha=0; i_TowEcha<nb_crys_in_tow; i_TowEcha++)
		    {
		      Int_t iSMEcha = (SMtow-1)*nb_crys_in_tow + i_TowEcha;
		      Int_t i_xgeo = GetXCrysInSM(MyNumbering, MyEcal, fFapSuMoNumber, SMtow, i_TowEcha);
		      Int_t i_ygeo = GetYCrysInSM(MyNumbering, MyEcal, fFapSuMoNumber, SMtow, i_TowEcha);
		      
		      if(i_xgeo >=0 && i_xgeo < nb_binx && i_ygeo >=0 && i_ygeo < nb_biny)
			{
			  if(iSMEcha >= 0 && iSMEcha < nb_of_towers*nb_crys_in_tow){
			    read_matrix(i_xgeo, i_ygeo) = partial_histo(i_TowEcha);}
			  
			  if (QuantityCode != "SMFoundEvtsGlobal" )
			    {read_matrix(i_xgeo, i_ygeo) = partial_histp(iSMEcha);}
			  
			  h_geo_bid->Fill((Double_t)i_xgeo, (Double_t)i_ygeo,
					  (Stat_t)read_matrix(i_xgeo, i_ygeo));
			}	   
		    }
		}
	    }
	  
	  // ------------------------------------------------ P L O T S   (ViewSuperModule)
	  
	  char* f_in = new char[fgMaxCar];                           fCnew++;
	  
	  //...................... Taille/format canvas
	  
	  UInt_t canv_h = CanvasFormatW("etaphiSM");
	  UInt_t canv_w = CanvasFormatH("etaphiSM");
	  
	  //............................................... paves commentaires (ViewSuperModule)
	  PutAllPavesViewSuperModule(MyRootFile);	  
	  
	  //------------------------------------ Canvas name ----------------- (ViewSuperModule)  
	  TString name_cov_cor;
	  MaxCar = fgMaxCar;
	  name_cov_cor.Resize(MaxCar);
	  name_cov_cor = "?";
	  
	  if( QuantityCode == "SMFoundEvtsGlobal"){name_cov_cor = "SMFoundEvtsGlobal";}
	  if( QuantityCode == "SMEvEvGlobal"     ){name_cov_cor = "SMEvEvGlobal";}
	  if( QuantityCode == "SMEvSigGlobal"    ){name_cov_cor = "SMEvSigGlobal";}
	  if( QuantityCode == "SMEvCorssGlobal"  ){name_cov_cor = "SMEvCorssGlobal";}
	  if( QuantityCode == "SMSigEvGlobal"    ){name_cov_cor = "SMSigEvGlobal";}
	  if( QuantityCode == "SMSigSigGlobal"   ){name_cov_cor = "SMSigSigGlobal";}
	  if( QuantityCode == "SMSigCorssGlobal" ){name_cov_cor = "SMSigCorssGlobal";}
	  
	  TString name_visu;
	  MaxCar = fgMaxCar;
	  name_visu.Resize(MaxCar);
	  name_visu = "colz";
	  
	  sprintf(f_in, "%s_EtaPhi_%s_%d_%s_%d_%d_SM%d",
		  name_visu.Data(), fFapAnaType.Data(), fFapRunNumber,
		  name_cov_cor.Data(),fFapFirstEvt, fFapNbOfEvts, fFapSuMoNumber);
	  
	  SetHistoPresentation((TH1D*)h_geo_bid, "SM2DTN");
	  
	  TCanvas *NoiseCorrel = new TCanvas(f_in, f_in, canv_w, canv_h);   fCnewRoot++;
	  
	  // cout << "*TCnaViewEB::ViewSuperModule(...)> Plot is displayed on canvas ----> " << f_in << endl;
	  
	  delete [] f_in; f_in = 0;                                 fCdelete++;
	  
	  //------------------------ Canvas draw and update ------------ (ViewSuperModule)  
	  ftitle_g1->Draw();
	  fcom_top_left->Draw();
	  fcom_bot_left->Draw();
	  fcom_bot_right->Draw();
	  
	  NoiseCorrel->Divide(1, 1, 0.001 , 0.125);
	  gPad->cd(1);
	  
	  TPad *main_subpad = (TPad*)gPad;
	  Int_t i_zonx = 1;
	  Int_t i_zony = 1;
	  main_subpad->Divide(i_zonx, i_zony, 0.001, 0.001);
	  main_subpad->cd(1);
	  
	  h_geo_bid->GetXaxis()->SetTitle(axis_x_var_name);
	  h_geo_bid->GetYaxis()->SetTitle(axis_y_var_name);
	  
	  h_geo_bid->DrawCopy("COLZ");
	  
	  // trace de la grille: un rectangle = une tour (ViewSuperModule) 
	  ViewSuperModuleGrid(MyEcal, MyNumbering, fFapSuMoNumber, " ");
	  gPad->Draw();
	  gPad->Update();
	  
#define DEUP
#ifndef DEUP
	  //-------------------- deuxieme Pad
	  sprintf(f_in_mat_tit, "SuperModule tower numbering");
	  TH2D* h_empty_bid = new TH2D("emptybidim_eta_phi", f_in_mat_tit,
				       nb_binx, xinf_bid,    xsup_bid,
				       nb_biny, yinf_bid,    ysup_bid);          fCnewRoot++; 
	  h_empty_bid->Reset();
	  
	  SetHistoPresentation((TH1D*)h_empty_bid, "SM2DTN");
	  
	  h_empty_bid->GetXaxis()->SetTitle(axis_x_var_name);
	  h_empty_bid->GetYaxis()->SetTitle(axis_y_var_name);
	  
	  main_subpad->cd(2);      
	  h_empty_bid->DrawCopy("COL");   // il faut retracer un bidim vide de meme taille que le precedent
	  // pour pouvoir tracer la grille et les axes dans la 2eme pad
	  
	  ViewSuperModuleTowerNumberingPad(MyEcal, MyNumbering, fFapSuMoNumber);
	  gPad->Draw();
	  gPad->Update();
	  
	  // h_empty_bid->Delete();              fCdeleteRoot++;    
	  
#endif
	  
	  //..................... retour aux options standard
	  Bool_t b_true = 1;
	  h_geo_bid->SetStats(b_true);    
	  h_geo_bid->Delete();                fCdeleteRoot++;
	   
	  //      delete title_g1;                 fCdeleteRoot++;
	  //      delete com_bot_left;             fCdeleteRoot++;
	  //      delete NoiseCorrel;              fCdeleteRoot++;
	  
	  delete MyNumbering;                              fCdelete++;
	}
      delete [] f_in_mat_tit;                            fCdelete++;
      delete MyEcal;                                     fCdelete++;
    }
  else
    {
      cout << "!TCnaViewEB::ViewSuperModule(...)> *ERROR* =====> "
	   << " ROOT file not found" << fTTBELL << endl;
    }
  
  delete MyRootFile;                                                   fCdelete++;
  
}  // end of ViewSuperModule(...)

//===========================================================================
//
//  EtaPhiSuperModuleCorccMeanOverSamples():
//     Geographical view of the cor(c,c) (mean over samples) for a
//     given SuperModule 
//
//===========================================================================  
void TCnaViewEB::EtaPhiSuperModuleCorccMeanOverSamples()
{
// (eta, phi) matrices for all the towers of a super-module

  TCnaReadEB*  MyRootFile = new TCnaReadEB();                              fCnew++; 
  MyRootFile->PrintNoComment();

  MyRootFile->GetReadyToReadRootFile(fFapAnaType,    fFapRunNumber, fFapFirstEvt, fFapNbOfEvts,
				     fFapSuMoNumber, fCfgResultsRootFilePath.Data());
  
  if ( MyRootFile->LookAtRootFile() == kTRUE )
    {
      TString fp_name_short = MyRootFile->GetRootFileNameShort(); 
      //cout << "*TCnaViewEB::EtaPhiSuperModuleCorccMeanOverSamples(...)> Data are analyzed from file ----> "
      //     << fp_name_short << endl;

      fStartDate    = MyRootFile->GetStartDate();
      fStopDate     = MyRootFile->GetStopDate();
      
      //......................... matrix title  
      char* f_in_mat_tit = new char[fgMaxCar];               fCnew++;
      
      sprintf(f_in_mat_tit, "Correlations between crystals (mean over samples) for each tower. SM view.");
      //................................. Axis parameters

      TEBParameters* MyEcal = new TEBParameters();   fCnew++;

      Int_t  GeoBidSizeEta = MyEcal->MaxTowEtaInSM()*MyEcal->MaxCrysInTow();
      Int_t  GeoBidSizePhi = MyEcal->MaxTowPhiInSM()*MyEcal->MaxCrysInTow();

      Int_t  nb_binx  = GeoBidSizeEta;
      Int_t  nb_biny  = GeoBidSizePhi;
      Axis_t xinf_bid = (Axis_t)0.;
      Axis_t xsup_bid = (Axis_t)GeoBidSizeEta;
      Axis_t yinf_bid = (Axis_t)0.;
      Axis_t ysup_bid = (Axis_t)GeoBidSizePhi;   
      
      TString axis_x_var_name = "  #eta  ";
      TString axis_y_var_name = "  #varphi  ";

      //--------------------------------------------------------- (EtaPhiSuperModuleCorccMeanOverSamples)

      //............. matrices reading and histogram filling
      
      TH2D* h_geo_bid = new TH2D("geobidim_eta_phi", f_in_mat_tit,
				 nb_binx, xinf_bid,  xsup_bid,
				 nb_biny, yinf_bid,  ysup_bid);     fCnewRoot++;
      h_geo_bid->Reset();

      //... histogram set ymin and ymax (from TCnaParameters)
      Int_t  xFlagAutoYsupMargin = 0;      
      xFlagAutoYsupMargin = HistoSetMinMax((TH1D*)h_geo_bid, "SMCorccInTowers");

      //======================================================== (EtaPhiSuperModuleCorccMeanOverSamples)

      Int_t nb_tow_in_sm   = MyEcal->MaxTowInSM();
      TVectorD tow_numbers(nb_tow_in_sm);
      tow_numbers = MyRootFile->ReadTowerNumbers();

      if ( MyRootFile->DataExist() == kTRUE )
	{
	  Int_t nb_crys_in_tow = MyEcal->MaxCrysInTow();
	  Int_t nb_crys_in_sm  = nb_crys_in_tow*nb_tow_in_sm;
	  TMatrixD partial_matrix(nb_crys_in_sm, nb_crys_in_sm);
	  partial_matrix = MyRootFile->ReadCorrelationsBetweenCrystalsMeanOverSamples();

	  if ( MyRootFile->DataExist() == kTRUE )
	    {	  
	      TEBNumbering* MyNumbering = new TEBNumbering();                fCnew++;
	      fFapSuMoBarrel = MyNumbering->GetSMHalfBarrel(fFapSuMoNumber);
	      
	      for(Int_t i_tow=0; i_tow<MyEcal->MaxTowInSM(); i_tow++)
		{
		  Int_t SMtow     = (Int_t)tow_numbers(i_tow);
		  Int_t offset_x = ((SMtow-1)/((Int_t)4))*nb_crys_in_tow;
		  Int_t offset_y = ((SMtow-1)%((Int_t)4))*nb_crys_in_tow;
		  
		  if (SMtow != -1)
		    {
		      //================================================= (EtaPhiSuperModuleCorccMeanOverSamples)
		      
		      //------------------ Geographical bidim filling (different from the case of a tower,
		      //                   here EB+ and EB- differences are important)
		      
		      for(Int_t i_TowEcha=0; i_TowEcha<nb_crys_in_tow; i_TowEcha++)
			{
			  for(Int_t j_TowEcha=0; j_TowEcha<nb_crys_in_tow; j_TowEcha++)
			    {
			      Int_t i_xgeo = offset_x + i_TowEcha;
			      Int_t i_ygeo = offset_y + j_TowEcha;
			      
			      if(i_xgeo >=0 && i_xgeo < nb_binx && i_ygeo >=0 && i_ygeo < nb_biny)
				{
				  Int_t iEcha = (SMtow-1)*nb_crys_in_tow + i_TowEcha;
				  Int_t jEcha = (SMtow-1)*nb_crys_in_tow + j_TowEcha;
				  
				  h_geo_bid->Fill((Double_t)i_xgeo, (Double_t)i_ygeo,
						  (Stat_t)partial_matrix(iEcha, jEcha));
				}
			    }	   
			}
		    }
		}
	      
	      // ----------------------------------- P L O T S   (EtaPhiSuperModuleCorccMeanOverSamples)
	      
	      char* f_in = new char[fgMaxCar];                           fCnew++;
	      
	      //...................... Taille/format canvas
	      
	      UInt_t canv_h = CanvasFormatW("etaphiSM");
	      UInt_t canv_w = CanvasFormatH("etaphiSM");
	      
	      //..................................... paves commentaires (EtaPhiSuperModuleCorccMeanOverSamples)
	      PutAllPavesViewSuperModule(MyRootFile);	  
	      
	      //----------------- Canvas name ------- (EtaPhiSuperModuleCorccMeanOverSamples)
	      TString name_cov_cor;
	      Int_t MaxCar = fgMaxCar;
	      name_cov_cor.Resize(MaxCar);
	      name_cov_cor = "SMCorccMeanOverSamples";
	      
	      TString name_visu;
	      MaxCar = fgMaxCar;
	      name_visu.Resize(MaxCar);
	      name_visu = "colz";
	      
	      sprintf(f_in, "%s_EtaPhi_%s_%d_%s_%d_%d_SM%d",
		      name_visu.Data(), fFapAnaType.Data(), fFapRunNumber,
		      name_cov_cor.Data(),fFapFirstEvt, fFapNbOfEvts, fFapSuMoNumber);
	      
	      SetHistoPresentation((TH1D*)h_geo_bid, "SM2DTN");
	      
	      TCanvas *NoiseCorrel = new TCanvas(f_in, f_in, canv_w, canv_h);   fCnewRoot++;
	      
	      // cout << "*TCnaViewEB::EtaPhiSuperModuleCorccMeanOverSamples(...)> Plot is displayed on canvas ----> "
	      //      << f_in << endl;
	      
	      delete [] f_in; f_in = 0;                                 fCdelete++;
	     
	      //------------ Canvas draw and update ------ (EtaPhiSuperModuleCorccMeanOverSamples)  
	      ftitle_g1->Draw();
	      fcom_top_left->Draw();
	      fcom_bot_left->Draw();
	      fcom_bot_right->Draw();
	      
	      NoiseCorrel->Divide(1, 1, 0.001 , 0.125);
	      gPad->cd(1);
	      
	      TPad *main_subpad = (TPad*)gPad;
	      Int_t i_zonx = 1;
	      Int_t i_zony = 1;
	      main_subpad->Divide(i_zonx, i_zony, 0.001, 0.001);
	      main_subpad->cd(1);
	      
	      h_geo_bid->GetXaxis()->SetTitle(axis_x_var_name);
	      h_geo_bid->GetYaxis()->SetTitle(axis_y_var_name);
	      
	      h_geo_bid->DrawCopy("COLZ");
	      
	      // trace de la grille: un rectangle = une tour (EtaPhiSuperModuleCorccMeanOverSamples) 
	      ViewSuperModuleGrid(MyEcal, MyNumbering, fFapSuMoNumber, "corcc");
	      gPad->Draw();
	      gPad->Update();
	      
	      //#define DEUP
	      //#ifndef DEUP
	      //-------------------- deuxieme Pad
	      // voir ViewSuperModule()
	      //#endif
	      
	      //..................... retour aux options standard
	      Bool_t b_true = 1;
	      h_geo_bid->SetStats(b_true);    
	      h_geo_bid->Delete();                fCdeleteRoot++;
	      	      
	      //      delete title_g1;                 fCdeleteRoot++;
	      //      delete com_bot_left;             fCdeleteRoot++;
	      //      delete NoiseCorrel;              fCdeleteRoot++;
	      
	      delete MyNumbering;                      fCdelete++;
	    }
	}
      delete [] f_in_mat_tit;                  fCdelete++;
      delete MyEcal;                                     fCdelete++;
    }
  else
    {
      cout << "!TCnaViewEB::EtaPhiSuperModuleCorccMeanOverSamples(...)> *ERROR* =====> "
	   << " ROOT file not found" << fTTBELL << endl;
    }
  
  delete MyRootFile;                                                   fCdelete++;

} // end of EtaPhiSuperModuleCorccMeanOverSamples

//==================================================================================
//
//                          GetXCrysInSM, GetYCrysInSM
//
//==================================================================================

Int_t TCnaViewEB::GetXCrysInSM(TEBNumbering* MyNumbering, TEBParameters* MyEcal, 
			     const Int_t&   SMNumber,    const Int_t&    SMTow,
			     const Int_t&   iTowEcha) 
{
//Gives the X crystal coordinate in the geographic view of one SuperModule
// (X = 0 to MaxTowEtaInSM*NbCrysEtaInTow - 1)

  TString ctype    = MyNumbering->GetSMHalfBarrel(SMNumber);
  Int_t   i_SMCrys = MyNumbering->GetSMCrysFromSMTowAndTowEcha(SMTow, iTowEcha);
  
  Int_t ix_geo = (i_SMCrys-1)/MyEcal->MaxCrysPhiInSM();  // ix_geo for barrel+

  if( ctype == "barrel-"){ix_geo = MyEcal->MaxCrysEtaInSM() - ix_geo - 1;}

  return ix_geo;
}

Int_t TCnaViewEB::GetYCrysInSM(TEBNumbering* MyNumbering, TEBParameters* MyEcal,
			     const Int_t&   SMNumber,    const Int_t&    SMTow,
			     const Int_t&   jTowEcha) 
{
//Gives the Y crystal coordinate in the geographic view of one SuperModule
// (Y = 0 to MaxTowPhiInSM*NbCrysPhiInTow - 1)


  TString ctype    = MyNumbering->GetSMHalfBarrel(SMNumber);
  Int_t   i_SMCrys = MyNumbering->GetSMCrysFromSMTowAndTowEcha(SMTow, jTowEcha);

  Int_t ix_geo = (i_SMCrys-1)/MyEcal->MaxCrysPhiInSM();           // ix_geo for barrel+
  
  Int_t iy_geo = i_SMCrys - 1 - ix_geo*MyEcal->MaxCrysPhiInSM();  // iy_geo for barrel+

  if( ctype == "barrel-"){iy_geo = MyEcal->MaxCrysPhiInSM() - iy_geo - 1;}

  return iy_geo;
}

//===========================================================================
//
//     SuperModuleTowerNumbering, ViewSuperModuleTowerNumberingPad
//
//              independent of the ROOT file => SMNumber as argument
//
//===========================================================================  
void TCnaViewEB::SuperModuleTowerNumbering(const Int_t& SMNumber)
{
//display the tower numbering of the super-module

  TEBParameters* MyEcal = new TEBParameters();   fCnew++;

  Int_t  GeoBidSizeEta = MyEcal->MaxTowEtaInSM()*MyEcal->MaxCrysEtaInTow();
  Int_t  GeoBidSizePhi = MyEcal->MaxTowPhiInSM()*MyEcal->MaxCrysPhiInTow();

  Int_t  nb_binx  = GeoBidSizeEta;
  Int_t  nb_biny  = GeoBidSizePhi;
  Axis_t xinf_bid = (Axis_t)0.;
  Axis_t xsup_bid = (Axis_t)GeoBidSizeEta;
  Axis_t yinf_bid = (Axis_t)0.;
  Axis_t ysup_bid = (Axis_t)GeoBidSizePhi;   
  
  TString axis_x_var_name = "  #eta  ";
  TString axis_y_var_name = "  #varphi  ";

  //------------------------------------------------------------------- SuperModuleTowerNumbering
  
  //............. matrices reading and histogram filling
  char* f_in_mat_tit = new char[fgMaxCar];                           fCnew++;

  sprintf(f_in_mat_tit, "SuperModule tower numbering");

  // il faut tracer un bidim vide pour pouvoir tracer la grille et les axes

  TH2D* h_empty_bid = new TH2D("grid_bidim_eta_phi", f_in_mat_tit,
			       nb_binx, xinf_bid, xsup_bid,
			       nb_biny, yinf_bid, ysup_bid);     fCnewRoot++; 
  h_empty_bid->Reset();
  
  h_empty_bid->GetXaxis()->SetTitle(axis_x_var_name);
  h_empty_bid->GetYaxis()->SetTitle(axis_y_var_name);

  // ------------------------------------------------ P L O T S   (SuperModuleTowerNumbering)
  
  char* f_in = new char[fgMaxCar];                           fCnew++;
  
  //...................... Taille/format canvas
  
  UInt_t canv_h = CanvasFormatW("etaphiSM");
  UInt_t canv_w = CanvasFormatH("etaphiSM");
  
  //............................................... options generales

  TEBNumbering* MyNumbering = new TEBNumbering();                fCnew++;
  fFapSuMoBarrel = MyNumbering->GetSMHalfBarrel(fFapSuMoNumber);

  //............................................... paves commentaires (SuperModuleTowerNumbering)
  PutAllPavesViewSuperModule();	  

  //------------------------------------ Canvas name ----------------- (SuperModuleTowerNumbering)  

  sprintf(f_in, "tower_numbering_for_SuperModule_SM%d", SMNumber);
  
  SetHistoPresentation((TH1D*)h_empty_bid,"SM2DTN");

  TCanvas *NoiseCorrel = new TCanvas(f_in, f_in, canv_w, canv_h);   fCnewRoot++;
  
  // cout << "*TCnaViewEB::ViewSuperModule(...)> Plot is displayed on canvas ----> " << f_in << endl;
  
  delete [] f_in; f_in = 0;                                 fCdelete++;

  //------------------------ Canvas draw and update ------------ (SuperModuleTowerNumbering)  
  fcom_top_left->Draw();

  NoiseCorrel->Divide(1, 1, 0.001 , 0.125);
  gPad->cd(1);
  
  TPad *main_subpad = (TPad*)gPad;
  Int_t i_zonx = 1;
  Int_t i_zony = 1;
  main_subpad->Divide(i_zonx, i_zony, 0.001, 0.001);
  main_subpad->cd(1); 

  h_empty_bid->DrawCopy("COL");   // il faut tracer un bidim vide pour pouvoir tracer la grille et les axes

  ViewSuperModuleTowerNumberingPad(MyEcal, MyNumbering, SMNumber);
  gPad->Update();
  
  //..................... retour aux options standard
  Bool_t b_true = 1;
  h_empty_bid->SetStats(b_true);    
  
  h_empty_bid->Delete();              fCdeleteRoot++;      
  
  //      delete title_g1;                 fCdeleteRoot++;
  //      delete com_bot_left;             fCdeleteRoot++;
  //      delete NoiseCorrel;              fCdeleteRoot++;
  
  delete [] f_in_mat_tit;  f_in_mat_tit = 0;         fCdelete++;

  delete MyEcal;                                     fCdelete++;
  delete MyNumbering;                                fCdelete++;
}
// end of SuperModuleTowerNumbering

//=============================================================================
//
//                   ViewSuperModuleTowerNumberingPad
//            independent of the ROOT file => SMNumber as argument
//
//=============================================================================
void TCnaViewEB::ViewSuperModuleTowerNumberingPad(TEBParameters* MyEcal, TEBNumbering* MyNumbering,
						const Int_t&     SMNumber)
{
//display the tower numbering of the super-module in a Pad

  gStyle->SetTitleW(0.4);        // taille titre histos
  gStyle->SetTitleH(0.08);

  ViewSuperModuleGrid(MyEcal, MyNumbering, SMNumber, " ");

  //Color_t couleur_bleu_fonce   = ColorDefinition("bleu_fonce");
  //Color_t couleur_rouge        = ColorDefinition("rouge");

  Color_t couleur_rouge      = SetColorsForNumbers("lvrb_top");
  Color_t couleur_bleu_fonce = SetColorsForNumbers("lvrb_bottom");

  //..... Ecriture des numeros de tours dans la grille..... (ViewSuperModuleTowerNumberingPad)

  char* f_in = new char[fgMaxCar];                           fCnew++;
  gStyle->SetTextSize(0.075);

  // x_channel, y_channel: coordinates of the text "Txx"
  Int_t y_channel = 12;
  Int_t x_channel =  0;    // => defined here after according to the Lvrb type

  Int_t max_tow_phi = MyEcal->MaxTowPhiInSM()*MyEcal->MaxCrysPhiInTow();

  //------------------ LOOP ON THE SM_TOWER NUMBER   (ViewSuperModuleTowerNumberingPad)

  TText *text_SMtow_num = new TText();        fCnewRoot++;

  for (Int_t i_SMtow = 1; i_SMtow <= MyEcal->MaxTowInSM(); i_SMtow++)
    {
      if(MyNumbering->GetTowerLvrbType(i_SMtow) == "top")
	{x_channel =  7; text_SMtow_num->SetTextColor(couleur_rouge);}
      if(MyNumbering->GetTowerLvrbType(i_SMtow) == "bottom")
	{x_channel = 17; text_SMtow_num->SetTextColor(couleur_bleu_fonce);}

      //................................ x from eta
      Double_t x_from_eta = MyNumbering->GetEta(SMNumber, i_SMtow, x_channel);
      if( MyNumbering->GetSMHalfBarrel(SMNumber) == "barrel-")
	{x_from_eta = (MyEcal->MaxTowEtaInSM()-1)*MyEcal->MaxCrysEtaInTow() + x_from_eta + (Double_t)2;}

      //................................ y from phi
      Double_t y_from_phi = max_tow_phi - 1
	- (MyNumbering->GetPhi(SMNumber, i_SMtow, y_channel) - MyNumbering->GetPhiMin(SMNumber));
      if( MyNumbering->GetSMHalfBarrel(SMNumber) == "barrel-")
	{y_from_phi = - y_from_phi + MyEcal->MaxTowPhiInSM()*MyEcal->MaxCrysPhiInTow() - (Double_t)1;}

      sprintf(f_in, "%d", i_SMtow);
      text_SMtow_num->DrawText(x_from_eta, y_from_phi, f_in);  // <=== prend du temps si on mets "T%d" dans le sprintf
    }

  // delete text_SMtow_num;             fCdeleteRoot++;

  //.................................................... legende (ViewSuperModuleTowerNumberingPad)
  Double_t offset_tow_tex_eta = (Double_t)8.;
  Double_t offset_tow_tex_phi = (Double_t)15.;

  Color_t couleur_noir = ColorDefinition("noir");
  Double_t x_legend    = (Double_t)0.;
  Double_t y_legend    = (Double_t)0.;

  Int_t ref_tower = MyEcal->MaxTowInSM();

  //.................................................  LVRB TOP (ViewSuperModuleTowerNumberingPad)
  gStyle->SetTextSize(0.075);  
  gStyle->SetTextColor(couleur_rouge);
  x_legend = MyNumbering->GetEta(SMNumber, ref_tower, x_channel);
  y_legend = MyNumbering->GetPhi(SMNumber, ref_tower, y_channel) - MyNumbering->GetPhiMin(SMNumber);

  if( MyNumbering->GetSMHalfBarrel(SMNumber) == "barrel+" )
    {
      x_legend = x_legend + offset_tow_tex_eta;
      y_legend = y_legend + offset_tow_tex_phi;
    }
  if( MyNumbering->GetSMHalfBarrel(SMNumber) == "barrel-" )
    {
      x_legend = -x_legend + offset_tow_tex_eta;
      y_legend =  y_legend + offset_tow_tex_phi;
    }

  sprintf( f_in, "xx");
  TText *text_legend_rouge = new TText(x_legend, y_legend, f_in);        fCnewRoot++;
  text_legend_rouge->Draw();
  // delete text_SMtow_num;             fCdeleteRoot++;
  gStyle->SetTextSize(0.05);
  gStyle->SetTextColor(couleur_noir);
  x_legend = x_legend - (Double_t)3.5;
  y_legend = y_legend - (Double_t)2.;
  sprintf(f_in, "       LVRB     ");
  TText *text_legend_rouge_expl = new TText(x_legend, y_legend, f_in);        fCnewRoot++;
  text_legend_rouge_expl->Draw();
  y_legend = y_legend - (Double_t)1.75;
  sprintf(f_in, "    at the top  ");
  TText *text_legend_rouge_expm = new TText(x_legend, y_legend, f_in);        fCnewRoot++;
  text_legend_rouge_expm->Draw();
  // delete text_SMtow_num;             fCdeleteRoot++;
  
  //.................................................  LVRB BOTTOM (ViewSuperModuleTowerNumberingPad)
  gStyle->SetTextSize(0.075);  
  gStyle->SetTextColor(couleur_bleu_fonce);
  x_legend = MyNumbering->GetEta(SMNumber, ref_tower, x_channel);
  y_legend = MyNumbering->GetPhi(SMNumber, ref_tower, y_channel) - MyNumbering->GetPhiMin(SMNumber); 

  if( MyNumbering->GetSMHalfBarrel(SMNumber) == "barrel+" )
    {
      x_legend = x_legend + offset_tow_tex_eta;
      y_legend = y_legend + offset_tow_tex_phi/3;
    }
  if( MyNumbering->GetSMHalfBarrel(SMNumber) == "barrel-" )
    {
      x_legend = -x_legend + offset_tow_tex_eta;
      y_legend =  y_legend + offset_tow_tex_phi/3;
    }

  sprintf(f_in, "xx");
  TText *text_legend_bleu = new TText(x_legend, y_legend, f_in);        fCnewRoot++;
  text_legend_bleu->Draw();
  // delete text_SMtow_num;             fCdeleteRoot++;
  gStyle->SetTextSize(0.05);
  gStyle->SetTextColor(couleur_noir);
  x_legend = x_legend - (Double_t)3.5;
  y_legend = y_legend - (Double_t)2.;
  sprintf( f_in, "       LVRB     ");
  TText *text_legend_bleu_expl = new TText(x_legend, y_legend, f_in);        fCnewRoot++;
  text_legend_bleu_expl->Draw();
  y_legend = y_legend - (Double_t)1.75;
  sprintf( f_in, "  at the bottom ");
  TText *text_legend_bleu_expm = new TText(x_legend, y_legend, f_in);        fCnewRoot++;
  text_legend_bleu_expm->Draw();
  // delete text_SMtow_num;             fCdeleteRoot++;
  
  delete [] f_in;                                       fCdelete++;

  gStyle->SetTextColor(couleur_noir);
}
//---------------->  end of ViewSuperModuleTowerNumberingPad()

//==========================================================================
//
//                       ViewSuperModuleGrid
//              independent of the ROOT file => SMNumber as argument
//
//==========================================================================
void TCnaViewEB::ViewSuperModuleGrid(TEBParameters* MyEcal,  TEBNumbering* MyNumbering,
				   const Int_t& SMNumber,    const TString  c_option)
{
 //Grid of one supermodule with axis eta and phi

  Int_t  GeoBidSizeEta = MyEcal->MaxTowEtaInSM()*MyEcal->MaxCrysEtaInTow();
  Int_t  GeoBidSizePhi = MyEcal->MaxTowPhiInSM()*MyEcal->MaxCrysPhiInTow();

  if ( c_option == "corcc")
    {
      GeoBidSizeEta = MyEcal->MaxTowEtaInSM()*MyEcal->MaxCrysInTow();
      GeoBidSizePhi = MyEcal->MaxTowPhiInSM()*MyEcal->MaxCrysInTow();
    }

  Int_t  nb_binx  = GeoBidSizeEta;
  Int_t  nb_biny  = GeoBidSizePhi;
  Axis_t xinf_bid = (Axis_t)0.;
  Axis_t xsup_bid = (Axis_t)GeoBidSizeEta;
  Axis_t yinf_bid = (Axis_t)0.;
  Axis_t ysup_bid = (Axis_t)GeoBidSizePhi;   
  
  //---------------- trace de la grille: un rectangle = une tour

  Int_t size_eta = MyEcal->MaxCrysEtaInTow();
  Int_t size_phi = MyEcal->MaxCrysPhiInTow();
  if ( c_option == "corcc")
    {
      size_eta = MyEcal->MaxCrysInTow();
      size_phi = MyEcal->MaxCrysInTow();
    }
  Int_t max_x    = nb_binx/size_eta;
  Int_t max_y    = nb_biny/size_phi;

  //............................. lignes horizontales
  Double_t yline = (Double_t)yinf_bid;

  Double_t xline_left  = (Double_t)xinf_bid;
  Double_t xline_right = (Double_t)xsup_bid;
  
  for( Int_t j = 0 ; j < max_y ; j++)
    {
      yline = yline + (Double_t)size_phi;
      TLine *lin;
      lin = new TLine(xline_left, yline, xline_right, yline); fCnewRoot++;
      lin->Draw();
      // delete lin;             fCdeleteRoot++;
    }

  //.......................... lignes verticales
  Double_t xline = (Double_t)xinf_bid - (Double_t)size_eta;
  
  Double_t yline_bot = (Double_t)yinf_bid;
  Double_t yline_top = (Double_t)ysup_bid;
  
  Color_t coul_surligne = ColorDefinition("noir");
  Color_t coul_textmodu = ColorDefinition("vert36");

  gStyle->SetTextColor(coul_textmodu);
  gStyle->SetTextSize(0.075);

  char* f_in = new char[fgMaxCar];                           fCnew++;

  for( Int_t i = 0 ; i < max_x ; i++)
    {  
      xline = xline + (Double_t)size_eta;
      TLine *lin;
      lin = new TLine(xline, yline_bot, xline, yline_top); fCnewRoot++;
      
      //............. Surlignage separateur des modules
      if( (MyNumbering->GetSMHalfBarrel(SMNumber) == "barrel-") && (i == 4 || i == 8 || i == 12) )
	{lin->SetLineWidth(2); lin->SetLineColor(coul_surligne);}      
      if( (MyNumbering->GetSMHalfBarrel(SMNumber) == "barrel+") && (i == 5 || i == 9 || i == 13) )
	{lin->SetLineWidth(2); lin->SetLineColor(coul_surligne);}
       
      lin->Draw();
      // delete lin;             fCdeleteRoot++;

      //............. Numeros des modules
      if( (MyNumbering->GetSMHalfBarrel(SMNumber) == "barrel-") && (i == 7 || i == 10 || i == 14)  )
	{
	  if( i ==  7 ){sprintf( f_in, "M3");}
	  if( i == 10 ){sprintf( f_in, "M2");}
	  if( i == 14 ){sprintf( f_in, "M1");}

	  TText *text_num_module = new TText(xline, yline_top + 1, f_in);        fCnewRoot++;
	  text_num_module->Draw();
	}      
      if( (MyNumbering->GetSMHalfBarrel(SMNumber) == "barrel+") && (i == 7 || i == 11 || i == 15)  )
	{
	  if( i ==  7 ){sprintf( f_in, "M2");}
	  if( i == 11 ){sprintf( f_in, "M3");}
	  if( i == 15 ){sprintf( f_in, "M4");}
	 
	  TText *text_num_module = new TText(xline, yline_top + 1, f_in);        fCnewRoot++;
	  text_num_module->Draw();
	}
    }

  delete [] f_in;                                       fCdelete++;

  //------------------ trace axes en eta et phi --------------- ViewSupermoduleGrid

  Int_t MatSize      = MyEcal->MaxCrysEtaInTow();
  if ( c_option == "corcc"){MatSize = MyEcal->MaxCrysInTow();}

  Int_t size_eta_sm  = MyEcal->MaxTowEtaInSM();
  Int_t size_phi_sm  = MyEcal->MaxTowPhiInSM();

  //...................................................... Axe i(eta) (bottom x) ViewSupermoduleGrid
  Double_t eta_min = MyNumbering->GetIEtaMin(SMNumber);
  Double_t eta_max = MyNumbering->GetIEtaMax(SMNumber);

  TString  x_var_name  = GetEtaPhiAxisTitle("ietaSM");;
  TString  x_direction = MyNumbering->GetXDirection(SMNumber);

  TF1 *f1 = new TF1("f1", x_direction, eta_min, eta_max);          fCnewRoot++;
    TGaxis* sup_axis_x = 0;

  if( x_direction == "-x" ) // NEVER  IN THIS CASE: xmin->xmax <=> right->left ("-x") direction
    {sup_axis_x = new TGaxis( (Float_t)0., (Float_t)0., (Float_t)(size_eta_sm*MatSize), (Float_t)0.,
			      "f1", size_eta_sm, "C" , 0.);   fCnewRoot++;}

  if( x_direction == "x" )  // ALWAYS IN THIS CASE: xmin->xmax <=> left->right ("x") direction    
    {sup_axis_x = new TGaxis( (Float_t)0., (Float_t)0., (Float_t)(size_eta_sm*MatSize), (Float_t)0.,
			      "f1", size_eta_sm, "C" , 0.);   fCnewRoot++;}
  
  sup_axis_x->SetTitle(x_var_name);
  sup_axis_x->SetTitleSize((Float_t)0.085);
  sup_axis_x->SetTitleOffset((Float_t)0.725);
  sup_axis_x->SetLabelSize((Float_t)0.06);
  sup_axis_x->SetLabelOffset((Float_t)0.02);
  //  gStyle->SetTickLength((Float_t)0.,"X");            // ==> NE MARCHE PAS
  sup_axis_x->Draw("SAME");

  //...................................................... Axe phi (y) ViewSupermoduleGrid
  Double_t phi_min     = MyNumbering->GetPhiMin(SMNumber);
  Double_t phi_max     = MyNumbering->GetPhiMax(SMNumber);

  TString  y_var_name  = GetEtaPhiAxisTitle("phi");
  TString  y_direction = MyNumbering->GetYDirection(SMNumber);

  TF1 *f2 = new TF1("f2", y_direction, phi_min, phi_max);           fCnewRoot++;
  TGaxis* sup_axis_y = 0;
  
  if ( y_direction == "-x" ) // ALWAYS IN THIS CASE: ymin->ymax <=> top->bottom ("-x") direction
    {sup_axis_y = new TGaxis(-(Float_t)1.5*(Float_t)size_eta, (Float_t)0.,
			     -(Float_t)1.5*(Float_t)size_eta, (Float_t)(size_phi_sm*MatSize),
			     "f2", (Int_t)size_phi_sm, "C", 0.);   fCnewRoot++;}
  
  if ( y_direction == "x" )  // NEVER  IN THIS CASE: ymin->ymax <=> bottom->top ("x") direction
    {sup_axis_y = new TGaxis(-(Float_t)1.5*(Float_t)size_eta, (Float_t)0.,
			     -(Float_t)1.5*(Float_t)size_eta, (Float_t)(size_phi_sm*MatSize),
			     "f2", (Int_t)size_phi_sm, "C", 0.);   fCnewRoot++;}
  
  sup_axis_y->SetTitle(y_var_name);
  sup_axis_y->SetTitleSize((Float_t)0.07);
  sup_axis_y->SetTitleOffset((Float_t)0.2);
  sup_axis_y->SetLabelSize((Float_t)0.06);
  sup_axis_y->SetLabelOffset((Float_t)0.015);
  sup_axis_y->SetTickSize((Float_t)0.);            // ==> NE MARCHE PAS
  sup_axis_y->Draw("SAME");

  //...................................................... Axe jphi (jy) ViewSupermoduleGrid
  Double_t jphi_min     = MyNumbering->GetJPhiMin(SMNumber);
  Double_t jphi_max     = MyNumbering->GetJPhiMax(SMNumber);

  TString  jy_var_name  = " ";
  TString  jy_direction = MyNumbering->GetJYDirection(SMNumber);

  TF1 *f3 = new TF1("f3", jy_direction, jphi_min, jphi_max);           fCnewRoot++;
  TGaxis* sup_axis_jy = 0;
  
  //............; essai
  sup_axis_jy = new TGaxis((Float_t)0., (Float_t)0.,
			   (Float_t)0., (Float_t)(size_phi_sm*MatSize),
			   "f3", (Int_t)size_phi_sm, "C", 0.);   fCnewRoot++;
  
  if ( jy_direction == "-x" ) // IN THIS CASE FOR EB+: ymin->ymax <=> top->bottom ("-x") direction
    {jy_var_name  = GetEtaPhiAxisTitle("jphiSMB+");}
  
  if ( jy_direction == "x" )  // IN THIS CASE FOR EB-: ymin->ymax <=> bottom->top ("x") direction
    {jy_var_name  = GetEtaPhiAxisTitle("jphiSMB-");}
  
  sup_axis_jy->SetTitle(jy_var_name);
  sup_axis_jy->SetTitleSize((Float_t)0.07);
  sup_axis_jy->SetTitleOffset((Float_t)0.3);
  sup_axis_jy->SetLabelSize((Float_t)0.06);
  sup_axis_jy->SetLabelOffset((Float_t)0.015);
  sup_axis_jy->SetTickSize((Float_t)0.);        // ==> NE MARCHE PAS
  sup_axis_jy->Draw("SAME");

  //--------------------------- ViewSupermoduleGrid
	  
  f1 = 0;
  f2 = 0;
  f3 = 0;

} // end of ViewSuperModuleGrid

//=================================================================================
//
//             SqrtContourLevels(const Int_t& nb_niv, Double_t* cont_niv)
//
//=================================================================================
void TCnaViewEB::SqrtContourLevels(const Int_t& nb_niv, Double_t* cont_niv)
{
//Calculation of levels in z coordinate for 3D plots. Square root scale
  
  Int_t nb_niv2 = (nb_niv+1)/2;
  
  for (Int_t num_niv = 0; num_niv < nb_niv2; num_niv++)
    {
      Int_t ind_niv = num_niv + nb_niv2 - 1;
      if ( ind_niv < 0 || ind_niv > nb_niv )
	{
	  cout << "!TCnaViewEB::ContourLevels(...)> *** ERROR *** "
	       << "wrong contour levels for correlation matrix"
	       << fTTBELL << endl;
	}
      else
	{
	  cont_niv[ind_niv] =
	    (Double_t)(num_niv*num_niv)/
	    ((Double_t)((nb_niv2-1)*(nb_niv2-1)));
	}
    }
  for (Int_t num_niv = -1; num_niv > -nb_niv2; num_niv--)
    {
      Int_t ind_niv = num_niv + nb_niv2 - 1;
      if ( ind_niv < 0 || ind_niv > nb_niv )
	{
	  cout << "!TCnaViewEB::ContourLevels(...)> *** ERROR *** "
	       << "wrong contour levels for correlation matrix"
	       << fTTBELL << endl;
	}
      else
	{
	  cont_niv[ind_niv] =
	    -(Double_t)(num_niv*num_niv)/
	    ((Double_t)((nb_niv2-1)*(nb_niv2-1)));
	}
    }
}

//==========================================================================
//
//                    GetEtaPhiAxisTitle
//
//==========================================================================
TString TCnaViewEB::GetEtaPhiAxisTitle(const TString chcode)
{
  TString xname = " ";

  if ( chcode == "ietaSM" ){xname = "i(#eta)  ";}
  if ( chcode == "ietaTow"){xname = "i(#eta)         ";}

  if ( chcode == "jphiSMB+" ){xname = "         j(#varphi)";}
  if ( chcode == "jphiSMB-" ){xname = "j(#varphi)    ";}
  if ( chcode == "jphiTow"  ){xname = "j(#varphi)         ";}
  if ( chcode == "phi"      ){xname = "#varphi    ";}

  return xname;
}

//===============================================================================
//
//                         ViewHisto...
//  
//===============================================================================
//.................. Found evts
void TCnaViewEB::HistoSuperModuleFoundEventsOfCrystals(const TString first_same_plot)
{
  Int_t zero = 0;
  ViewHisto(zero, zero, zero, "SMFoundEvtsGlobal", fOptVisLine, first_same_plot);
}

void TCnaViewEB::HistoSuperModuleFoundEventsDistribution(const TString first_same_plot)
{
  Int_t zero = 0;
  ViewHisto(zero, zero, zero, "SMFoundEvtsProj", fOptVisLine, first_same_plot);
}

//............................. ev
void TCnaViewEB::HistoSuperModuleMeanPedestalsOfCrystals(const TString first_same_plot)
{
  Int_t zero = 0;
  ViewHisto(zero, zero, zero, "SMEvEvGlobal", fOptVisLine, first_same_plot);
}

void TCnaViewEB::HistoSuperModuleMeanPedestalsDistribution(const TString first_same_plot)
{
  Int_t zero = 0;
  ViewHisto(zero, zero, zero, "SMEvEvProj", fOptVisLine, first_same_plot);
}

void TCnaViewEB::HistoSuperModuleMeanOfSampleSigmasOfCrystals(const TString first_same_plot)
{
  Int_t zero = 0;
  ViewHisto(zero, zero, zero, "SMEvSigGlobal", fOptVisLine, first_same_plot);
}

void TCnaViewEB::HistoSuperModuleMeanOfSampleSigmasDistribution(const TString first_same_plot)
{
  Int_t zero = 0;
  ViewHisto(zero, zero, zero, "SMEvSigProj", fOptVisLine, first_same_plot);
}
void TCnaViewEB::HistoSuperModuleMeanOfCorssOfCrystals(const TString first_same_plot)
{
  Int_t zero = 0;
  ViewHisto(zero, zero, zero, "SMEvCorssGlobal", fOptVisLine, first_same_plot);
}

void TCnaViewEB::HistoSuperModuleMeanOfCorssDistribution(const TString first_same_plot)
{
  Int_t zero = 0;
  ViewHisto(zero, zero, zero, "SMEvCorssProj", fOptVisLine, first_same_plot);
}

//..................................... Sigma
void TCnaViewEB::HistoSuperModuleSigmaPedestalsOfCrystals(const TString first_same_plot)
{
  Int_t zero = 0; 
  ViewHisto(zero, zero, zero, "SMSigEvGlobal", fOptVisLine, first_same_plot);
}

void TCnaViewEB::HistoSuperModuleSigmaPedestalsDistribution(const TString first_same_plot)
{
  Int_t zero = 0;
  ViewHisto(zero, zero, zero, "SMSigEvProj", fOptVisLine, first_same_plot);
}

void TCnaViewEB::HistoSuperModuleSigmaOfSampleSigmasOfCrystals(const TString first_same_plot)
{
  Int_t zero = 0;
  ViewHisto(zero, zero, zero, "SMSigSigGlobal", fOptVisLine, first_same_plot);
}

void TCnaViewEB::HistoSuperModuleSigmaOfSampleSigmasDistribution(const TString first_same_plot)
{
  Int_t zero = 0;
  //  SetHistoScaleY("LOG");
  ViewHisto(zero, zero, zero, "SMSigSigProj", fOptVisLine, first_same_plot);
}
void TCnaViewEB::HistoSuperModuleSigmaOfCorssOfCrystals(const TString first_same_plot)
{
  Int_t zero = 0;
  ViewHisto(zero, zero, zero, "SMSigCorssGlobal", fOptVisLine, first_same_plot);
}

void TCnaViewEB::HistoSuperModuleSigmaOfCorssDistribution(const TString first_same_plot)
{
  Int_t zero = 0;
  ViewHisto(zero, zero, zero, "SMSigCorssProj", fOptVisLine, first_same_plot);
}

void TCnaViewEB::HistoCrystalExpectationValuesOfSamples(const Int_t&  SMtower_X,       const Int_t& TowEcha,
							  const TString first_same_plot)
{
  Int_t zero = 0;
  ViewHisto(SMtower_X, TowEcha, zero, "Ev", fOptVisLine, first_same_plot);
}

void TCnaViewEB::HistoCrystalSigmasOfSamples(const Int_t&  SMtower_X,       const Int_t& TowEcha,
					       const TString first_same_plot)
{
  Int_t zero = 0;
  ViewHisto(SMtower_X, TowEcha, zero, "Sigma", fOptVisLine, first_same_plot);
}

void TCnaViewEB::HistoCrystalPedestalEventNumber(const Int_t&  SMtower_X,       const Int_t& TowEcha,
						   const TString first_same_plot)
{
  Int_t zero = 0;
  ViewHisto(SMtower_X, TowEcha, zero, "SampTime", fOptVisLine, first_same_plot);
}


void TCnaViewEB:: HistoSampleEventDistribution(const Int_t& SMtower_X, const Int_t&  TowEcha,
						 const Int_t& sample,    const TString first_same_plot)
{
  ViewHisto(SMtower_X, TowEcha, sample, "Evts", fOptVisLine, first_same_plot);
}

//===============================================================================
//
//                         ViewHisto
//       
//     ViewHisto( SMtower_X, TowEcha, sample,  opt_quantity,    opt_visu)
//
//     (SMtower_X, TowEcha, sample (NOT USED), fOptHisEv,       lin/log) ==> 
//     exp values of the samples for TowEcha of SMtower_X 
//
//     (SMtower_X, TowEcha, sample (NOT USED), fOptHisSigma,    lin/log) ==> 
//     sigmas of the samples for TowEcha of SMtower_X
//
//     (SMtower_X, TowEcha, sample,            fOptHisEvts,     lin/log) ==>
//     ADC event distribution for TowEcha of SMtower_X and for sample              
//
//     (SMtower_X, TowEcha, sample (NOT USED), fOptHisSampTime, lin/log) ==>
//     Pedestal as a function of event number for TowEcha of SMtower_X              
//
//===============================================================================
void TCnaViewEB::ViewHisto(const Int_t&  SMtower_X,      const Int_t&  TowEcha,
			   const Int_t&  sample,         const TString QuantityCode,
			   const Int_t&  opt_visu,
			   const TString opt_plot)
{
//Histogram of the quantities (one run)

  Int_t opt_scale = fOptScaleLiny;
  if (fFlagScaleY == "LIN" ){opt_scale = fOptScaleLiny;}
  if (fFlagScaleY == "LOG" ){opt_scale = fOptScaleLogy;}

  TString QuantityType = GetQuantityType(QuantityCode);

  TCnaReadEB*  MyRootFile = new TCnaReadEB();              fCnew++;

  MyRootFile->PrintNoComment();

  MyRootFile->GetReadyToReadRootFile(fFapAnaType,    fFapRunNumber,  fFapFirstEvt, fFapNbOfEvts,
				     fFapSuMoNumber, fCfgResultsRootFilePath.Data());
      
  if ( MyRootFile->LookAtRootFile() == kTRUE )
    {
      //---------------------------------------------------------------------------- (ViewHisto)
      TString fp_name_short = MyRootFile->GetRootFileNameShort();
      // cout << "*TCnaViewEB::ViewHisto(...)> Data are analyzed from file ----> "
      //      << fp_name_short << endl;

      Bool_t ok_view_histo = GetOkViewHisto(MyRootFile, SMtower_X, TowEcha, sample, QuantityCode);

      if( ok_view_histo == kTRUE )
	{
	  fStartDate = MyRootFile->GetStartDate();
	  fStopDate  = MyRootFile->GetStopDate();

	  Int_t   HisSize  = GetHistoSize(MyRootFile, QuantityCode);

	  TVectorD read_histo(HisSize);
	  Int_t i_data_exist = 0;
	  read_histo = GetHistoValues(MyRootFile, QuantityCode, HisSize,
				      SMtower_X, TowEcha, sample, i_data_exist);
	  if ( i_data_exist > 0)
	    {
	      //................................. Set YMin and YMax of histo (ViewHisto)
	      if((opt_plot == fOnlyOnePlot) || (opt_plot == fSeveralPlot && GetMemoFlag(QuantityCode) == "Free") )
		{
		  InitQuantityYmin(QuantityCode);
		  InitQuantityYmax(QuantityCode);
		}
	      //..............................  prepa histogram booking (ViewHisto)
	      
	      Axis_t xinf_his = SetHistoXinf(MyRootFile, QuantityCode, HisSize, SMtower_X, TowEcha, sample);
	      Axis_t xsup_his = SetHistoXsup(MyRootFile, QuantityCode, HisSize, SMtower_X, TowEcha, sample);
	      Int_t   nb_binx = SetHistoNumberOfBins(QuantityCode, HisSize);
	      
	      //..............................  histogram booking (ViewHisto)
	      
	      TH1D* h_his0 = new TH1D("histo", GetQuantityName(QuantityCode),
				      nb_binx, xinf_his, xsup_his);                   fCnewRoot++;
	      
	      //............................... histogram filling
	      
	      FillHisto(h_his0, read_histo, QuantityCode, HisSize, xinf_his, xsup_his, nb_binx);
	      
	      //... histogram set ymin and ymax (from TCnaParameters) and consequently margin at top of the plot
	      
	      Int_t  xFlagAutoYsupMargin = HistoSetMinMax(h_his0, QuantityCode);
	      
	      //............................... histogram plot 
	      Int_t  nb_of_towers  = MyRootFile->MaxTowInSM();
	      
	      TEBNumbering* MyNumbering = new TEBNumbering();    fCnew++;
	      fFapSuMoBarrel = MyNumbering->GetSMHalfBarrel(fFapSuMoNumber);
	      
	      HistoPlot(h_his0,       MyRootFile, MyNumbering,  HisSize,  QuantityCode, QuantityType,
			nb_of_towers, SMtower_X,  TowEcha,      sample,
			opt_visu,     opt_scale,  opt_plot,     xFlagAutoYsupMargin);
	      
	      delete MyNumbering;                                  fCdelete++;
	      
	      h_his0->Delete();                fCdeleteRoot++;
	    }
	}
      else
	{
	  cout << "!TCnaViewEB::ViewHisto(...)> *ERROR* =====> possibly wrong values for: "
	       << endl
	       << "                             File: " << fp_name_short
	       << ", tower = " << SMtower_X << ", TowEcha = " << TowEcha << ", sample = " << sample
	       << ", quantity code = " << QuantityCode << ", quantity type = " << QuantityType
	       << fTTBELL << endl;
	}
    }
  else
    {
      cout << "!TCnaViewEB::ViewHisto(...)> *ERROR* =====> "
	   << " ROOT file not found" << fTTBELL << endl;
    }
  delete MyRootFile;                               fCdelete++;

}  // end of ViewHisto(...)

//======================================================================================
//
//                    ViewHistime: time evolution...
//
//======================================================================================
void TCnaViewEB::HistimeCrystalMeanPedestals(const TString  run_par_file_name,
					     const Int_t&   SMtower_X,      const Int_t& towEcha,
					     const TString  first_same_plot)
{
  ViewHistime(run_par_file_name, SMtower_X, towEcha, "EvolEvEv", fOptVisPolm, first_same_plot);
}

void TCnaViewEB::HistimeCrystalMeanSigmas(const TString  run_par_file_name,
					  const Int_t&   SMtower_X,      const Int_t& towEcha,
					  const TString  first_same_plot)
{
  ViewHistime(run_par_file_name, SMtower_X, towEcha, "EvolEvSig", fOptVisPolm, first_same_plot);
}

void TCnaViewEB::HistimeCrystalMeanCorss(const TString  run_par_file_name,
					 const Int_t&   SMtower_X,      const Int_t& towEcha,
					 const TString  first_same_plot)
{
  ViewHistime(run_par_file_name, SMtower_X, towEcha, "EvolEvCorss", fOptVisPolm, first_same_plot);
}

//======================================================================================
//
//                          ViewHistime: time evolution
//
//======================================================================================
void TCnaViewEB::ViewHistime(const TString  run_par_file_name, 
			     const Int_t&   SMtower_X,         const Int_t& towEcha,
			     const TString  QuantityCode,      const Int_t& opt_visu, 
			     const TString  opt_plot)
{
//Histogram of the quantities as a function of time (several runs)
 
  TCnaReadEB*  MyRootFile = new TCnaReadEB();                        fCnew++;
  MyRootFile->PrintNoComment();

  TString QuantityType = GetQuantityType(QuantityCode);
  
  //................................. Init YMin and YMax of histo
  if((opt_plot == fOnlyOnePlot) || (opt_plot == fSeveralPlot && GetMemoFlag(QuantityCode) == "Free") )
    {
      InitQuantityYmin(QuantityCode); 
      InitQuantityYmax(QuantityCode);
    }

  //........ GetListOfRunParameters(...) : performs the allocation of the arrays fT1Dxxx at first call
  //         increments the number of read file (fNbOfListFileEvolXXX) for option SAME of TCnaDialogEB
  //         and read the values of the arrays fT1Dxxx from the file run_par_file_name

  //............... Get the run parameters

  Int_t nb_of_runs_in_list = GetListOfRunParameters(run_par_file_name.Data(), QuantityCode);

  if( nb_of_runs_in_list > 0 )
    {
      //..............................  prepa x axis: time in hours
      //Double_t sec_in_day   = (Double_t)86400.;          //===> (number of seconds in a day)
      Double_t margin_frame_xaxis = (Double_t)25.;       //===> margin in x coordinates 

      Double_t thstart_evol = (Double_t)0.;
      Double_t thstop_evol  = (Double_t)0.;

      Int_t* exist_indic  = new Int_t[nb_of_runs_in_list];

      //---------------------------------------------------------------- (ViewHistime)
      //
      //     FIRST LOOP: OVER THE RUNS IN THE LIST OF THE ASCII FILE
      //
      //----------------------------------------------------------------
      fNbOfExistingRuns = (Int_t)0;

      for (Int_t i_run = 0; i_run < nb_of_runs_in_list; i_run++)
	{
	  exist_indic[i_run] = 0;

	  SetFile(i_run); // ==> set the attributes relative to the run (fFapRunNumber, etc...)
	                  //     The arrays fT1D...[] have been obtained from the previous call
	                  //     to GetListOfRunParameters(xFileNameRunList, QuantityCode)

	  MyRootFile->GetReadyToReadRootFile(fT1DAnaType[i_run].Data(), fT1DRunNumber[i_run],
					     fT1DFirstEvt[i_run],       fT1DNbOfEvts[i_run],
					     fT1DSuMoNumber[i_run],     fT1DResultsRootFilePath[i_run].Data());
	  // MyRootFile->PrintNoComment();
	 
	  if ( MyRootFile->LookAtRootFile() == kTRUE )
	    {
	      //------ At first list file: set fStartEvol... and fStopEvol... quantities  (ViewHistime)
	      if( GetListFileNumber(QuantityCode) == 1 )
		{
		  if( fNbOfExistingRuns == 0 ) 
		    {
		      fStartEvolTime = MyRootFile->GetStartTime();  // start time of the first run of the list
		      fStartEvolDate = MyRootFile->GetStartDate();
		      fStartEvolRun  = fT1DRunNumber[i_run]; 
		    }
		  else
		    {
		      fStopEvolTime = MyRootFile->GetStartTime();    // start time of the last  run of the list
		      fStopEvolDate = MyRootFile->GetStartDate();
		      fStopEvolRun  = fT1DRunNumber[i_run];
		    }
		}
	      //---- set flag of run existence and increase number of existing run (for the present list file)
	      exist_indic[i_run] = 1;
	      fNbOfExistingRuns++;
	    }
	  else
	    {
	      cout << "!TCnaViewEB::ViewHistime(...)> *ERROR* =====> "
		   << ": ROOT file not found for run " << fT1DRunNumber[i_run]
		   << ", first event: "  << fT1DFirstEvt[i_run]
		   << ", nb of events: " << fT1DNbOfEvts[i_run] << fTTBELL << endl << endl;
	    }
	} // end of the first loop over the runs of the ascii file
          
      if( fNbOfExistingRuns > 0 )
	{
	  //-------------------- recover the arrays after removing non existing ROOT files (ViewHistime)
	  Int_t i_existing_run = (Int_t)0;

	  for( Int_t i_run = 0; i_run < nb_of_runs_in_list;  i_run++)
	    {
	      if( exist_indic[i_run] == 1 )
		{
		  fT1DAnaType[i_existing_run]              = fT1DAnaType[i_run];
		  fT1DRunNumber[i_existing_run]            = fT1DRunNumber[i_run];
		  fT1DFirstEvt[i_existing_run]             = fT1DFirstEvt[i_run];
		  fT1DNbOfEvts[i_existing_run]             = fT1DNbOfEvts[i_run];
		  fT1DSuMoNumber[i_existing_run]           = fT1DSuMoNumber[i_run];
		  fT1DResultsRootFilePath [i_existing_run] = fT1DResultsRootFilePath[i_run];
		  i_existing_run++;
		}
	    }	 

	  //---------------------- Get start and stop time values to set the axis limits (ViewHistime)
	  //thstart_evol = (Double_t)fStartEvolTime/sec_in_day;
	  //thstop_evol  = (Double_t)fStopEvolTime/sec_in_day;
	  //
	  //Axis_t xinf_his = (Axis_t)(-(thstop_evol-thstart_evol)/margin_frame_xaxis);
	  //Axis_t xsup_his = (Axis_t)(thstop_evol-thstart_evol+(thstop_evol-thstart_evol)/margin_frame_xaxis);

	  thstart_evol = (Double_t)fStartEvolTime;
 	  thstop_evol  = (Double_t)fStopEvolTime;

	  Double_t xinf_lim = thstart_evol-(thstop_evol-thstart_evol)/margin_frame_xaxis;
	  Double_t xsup_lim = thstop_evol +(thstop_evol-thstart_evol)/margin_frame_xaxis;

	  Axis_t xinf_his = (Axis_t)(xinf_lim);
	  Axis_t xsup_his = (Axis_t)(xsup_lim);

	  //............................. SMEcha, sample
	  Int_t SMEcha = MyRootFile->GetSMEcha(SMtower_X, towEcha);
	  Int_t sample = 0;
	 
	  Double_t* time_coordx  = new Double_t[fNbOfExistingRuns];
	  Double_t* hval_coordy  = new Double_t[fNbOfExistingRuns];
	 
	  //........... Set values to -1 

	  for( Int_t i_run = 0; i_run < fNbOfExistingRuns;  i_run++)
	    {
	      time_coordx[i_run] = (Double_t)(-1);
	      hval_coordy[i_run] = (Double_t)(-1);
	    }	 
 
	  //---------------------------------------------------------------- (ViewHistime)
	  //
	  //                SECOND LOOP OVER THE EXISTING RUNS
	  //
	  //----------------------------------------------------------------
	  for (Int_t i_run = 0; i_run < fNbOfExistingRuns; i_run++)
	    {
	      SetFile(i_run); // => set the attributes relative to the run (fFapRunNumber,...)

	      MyRootFile->GetReadyToReadRootFile(fT1DAnaType[i_run].Data(), fT1DRunNumber[i_run],
						 fT1DFirstEvt[i_run],       fT1DNbOfEvts[i_run],
						 fT1DSuMoNumber[i_run],     fT1DResultsRootFilePath[i_run].Data());

	      // MyRootFile->PrintNoComment();	     
	      if ( MyRootFile->LookAtRootFile() == kTRUE )
		{
		  Bool_t ok_view_histo = GetOkViewHisto(MyRootFile, SMtower_X, towEcha, sample, QuantityCode);
		 
		  if( ok_view_histo == kTRUE )
		    {
		      //......................................... graph filling (ViewHistime)

		      time_t xStartTime = MyRootFile->GetStartTime();
		      //time_t xStopTime  = MyRootFile->GetStopTime();
		     
		      //Double_t thstart = (Double_t)xStartTime/sec_in_day;
		      Double_t thstart = (Double_t)xStartTime;
		      // Double_t thstop  = (Double_t)xStopTime/sec_in_day;
		      //time_coordx[i_run] = (Double_t)((thstart+thstop)/(Double_t)2. - thstart_evol);

		      time_coordx[i_run] = (Double_t)(thstart - xinf_lim);

		      TVectorD read_histo(MyRootFile->MaxCrysInSM());	      		     

		      if(QuantityCode == "EvolEvEv")
			{read_histo = MyRootFile->ReadExpectationValuesOfExpectationValuesOfSamples();}
		      if(QuantityCode == "EvolEvSig")
			{read_histo = MyRootFile->ReadExpectationValuesOfSigmasOfSamples();}
		      if(QuantityCode == "EvolEvCorss")
			{read_histo = MyRootFile->ReadExpectationValuesOfCorrelationsBetweenSamples();}
		     
		      hval_coordy[i_run] = (Double_t)read_histo(SMEcha);
		    }
		  else
		    {
		      cout << "!TCnaViewEB::ViewHistime(...)> *ERROR* =====> possibly wrong values for: "
			   << endl
			   << "                             File: " << MyRootFile->GetRootFileNameShort()
			   << ", tower = " << SMtower_X << ", towEcha = " << towEcha << ", sample = " << sample
			   << ", quantity code = " << QuantityCode << ", quantity type = " << QuantityType
			   << fTTBELL << endl;
		    }
		}
	    }

	  //------------------------------------------------- graph (ViewHistime)
		     	 
	  TGraph* g_graph0 = new TGraph(fNbOfExistingRuns, time_coordx, hval_coordy);
	  g_graph0->SetTitle(GetQuantityName(QuantityCode));
	 
	  //............................... histogram plot (ViewHistime)
	 
	  Int_t  nb_of_towers  = MyRootFile->MaxTowInSM();
		     
	  Int_t opt_scale = fOptScaleLiny;
	  if (fFlagScaleY == "LIN" ){opt_scale = fOptScaleLiny;}
	  if (fFlagScaleY == "LOG" ){opt_scale = fOptScaleLogy;}
	 
	  //..... graph set ymin and ymax (from TCnaParameters) and consequently margin at top of the plot  
	  Int_t  xFlagAutoYsupMargin = GraphSetMinMax(g_graph0, QuantityCode);
	 

	  TEBNumbering* MyNumbering = new TEBNumbering();         fCnew++;
	  fFapSuMoBarrel = MyNumbering->GetSMHalfBarrel(fFapSuMoNumber);
	  HistimePlot(g_graph0,  xinf_his, xsup_his, MyRootFile, MyNumbering,
		      QuantityCode, QuantityType,
		      nb_of_towers, SMtower_X,  towEcha,  sample,
		      opt_visu,     opt_scale,  opt_plot, xFlagAutoYsupMargin);
	  delete MyNumbering;                                       fCdelete++;
		     
	  //  g_graph0->Delete();         fCdeleteRoot++;  // *===> NE PAS DELETER LE GRAPH SINON CA EFFACE TOUT!
	  delete [] time_coordx;          fCdelete++;   
	  delete [] hval_coordy;          fCdelete++;  
	}
    }
  else
    {
      cout << "!TCnaViewEB::ViewHistime(...)> No run of the list in file " << run_par_file_name
	   << " corresponds to an existing ROOT file."
	   << fTTBELL << endl;
    }

  delete MyRootFile;                               fCdelete++;

} // end of ViewHistime

//------------------------------------------------------------------------------------
//
//      GetListOfRunParameters(...), AllocArraysForEvol(), GetListFileNumber(...)
//
//------------------------------------------------------------------------------------

Int_t TCnaViewEB::GetListOfRunParameters(const TString run_par_file_name, const TString QuantityCode)
{
// Build the arrays of run parameters from the list-of-runs .ascii file.
// Return the list size as given by the second line of the file
// (see comment on the file syntax in the method AllocArraysForEvol() )

// *=====> run_par_file_name is the name of the ASCII file containing the list of the runs
// with their parameters (analysis name, run number, 1rst evt, nb of events, results file path)
//
// SYNTAX OF THE FILE:
//
//cna_list0_ped_3b.ascii    <- 1rst line: comment (file name for example)
//5                         <- 2nd  line: nb of lines after this one
//ped 73677 0 150 10        <- 3rd  line and others: run parameters 
//ped 73688 0 150 10           
//ped 73689 0 150 10 
//ped 73690 0 150 10 
//ped 73692 0 150 10 
//
// In option SAME (of TCnaDialogEB), several lists of runs are called and these lists can have
// DIFFERENT sizes (here the "size" is the number of runs of the list). In addition,
// some runs in some lists may not exist in reality. So, we must adopt a convention which is
// the following: the number of runs corresponds to the number of EXISTING runs
// of the FIRST read list. Let be N1 this number.
// If another list has more runs than N1 runs, we read only the first N1 runs.
// If another list has less runs than N1 runs, we read all the runs of this list. 
//
//--------------------------------------------------------------------------------------------------
//
// *====> nb_of_runs_in_list = nb of runs in the second line of the file (= -1 if file not found)
// 
  Int_t nb_of_runs_in_list = 0;

 // ===> increase the list file numbers

  if ( QuantityCode == "EvolEvEv"    ){fNbOfListFileEvolEvEv++;}
  if ( QuantityCode == "EvolEvSig"   ){fNbOfListFileEvolEvSig++;}
  if ( QuantityCode == "EvolEvCorss" ){fNbOfListFileEvolEvCorss++;}

  fFapFileRuns = run_par_file_name.Data();    // (short name)


  //........... Add the path to the file name           ( GetListOfRunParameters )
  TString xFileNameRunList = run_par_file_name.Data();
  Text_t *t_file_name = (Text_t *)xFileNameRunList.Data();

  //.............. replace the string "$HOME" by the true $HOME path
  if(fCfgListOfRunsFilePath.BeginsWith("$HOME"))
    {
      fCfgListOfRunsFilePath.Remove(0,5);
      Text_t *t_file_nohome = (Text_t *)fCfgListOfRunsFilePath.Data(); //  /scratch0/cna/...
      
      TString home_path = gSystem->Getenv("HOME");
      fCfgListOfRunsFilePath = home_path;             //  /afs/cern.ch/u/USER
      fCfgListOfRunsFilePath.Append(t_file_nohome);   //  /afs/cern.ch/u/USER/scratch0/cna/...
    }

  xFileNameRunList = fCfgListOfRunsFilePath.Data();

  xFileNameRunList.Append('/');
  xFileNameRunList.Append(t_file_name);

  fFcin_f.open(xFileNameRunList.Data());

  //.......................................           ( GetListOfRunParameters )
  if(fFcin_f.fail() == kFALSE)
    {
      fFcin_f.clear();
      string xHeadComment;
      fFcin_f >> xHeadComment;
     
      Int_t list_size_read;
      fFcin_f >> list_size_read;
      
      //....................... Set fMaxNbOfRunsXXX to -1 at first call (first read file)
      //
      //                        fNbOfListFileEvolxxx is initialized to 0 in Init()
      //                        It is incremented once in the calling method GetListOfRunParameters(...)
      //                        So, at first call fNbOfListFileEvolXXX = 1
      //                        then fMaxNbOfRunsEvolXXX = -1
      
      if( (QuantityCode == "EvolEvEv"    && fNbOfListFileEvolEvEv    == 1) || 
	  (QuantityCode == "EvolEvSig"   && fNbOfListFileEvolEvSig   == 1) ||
	  (QuantityCode == "EvolEvCorss" && fNbOfListFileEvolEvCorss == 1) ){fFapMaxNbOfRuns = -1;}
     
      //...............fFapNbOfRuns = nb of runs from the 2nd line of the first read list file
      
      fFapNbOfRuns = list_size_read;
      
      if( fFapMaxNbOfRuns == -1 ) 
	{
	  fFapMaxNbOfRuns = fFapNbOfRuns;    // first call: fFapMaxNbOfRuns = fFapNbOfRuns = xArgNbOfRunsFromList
	}
      else
	{
	  if( fFapMaxNbOfRuns < fFapNbOfRuns ){fFapNbOfRuns = fFapMaxNbOfRuns;}
	}
         
      AllocArraysForEvol(); // performs allocation and init of the arrays.      ( GetListOfRunParameters )

      string cAnaType;
      Int_t  cRunNumber;
      Int_t  cFirstEvt;
      Int_t  cNbOfEvts;
      Int_t  cSuMoNumber;
      string cRootFilePath;
     
      for (Int_t i_list = 0; i_list < list_size_read; i_list++)
	{
	  fFcin_f >> cAnaType >> cRunNumber >> cFirstEvt >> cNbOfEvts >> cSuMoNumber;
	  
	  fT1DAnaType[i_list]      = cAnaType;
	  fT1DRunNumber[i_list]    = cRunNumber;
	  fT1DFirstEvt[i_list]     = cFirstEvt;
	  fT1DNbOfEvts[i_list]     = cNbOfEvts;
	  fT1DSuMoNumber[i_list]   = cSuMoNumber;
	  fT1DResultsRootFilePath[i_list] = fCfgResultsRootFilePath.Data();
	  fT1DListOfRunsFilePath[i_list]  = fCfgListOfRunsFilePath.Data();
	}
      //........................................           ( GetListOfRunParameters )
      nb_of_runs_in_list = list_size_read;
      fFcin_f.close();
    }
  else
    {
      fFcin_f.clear();
      cout << "!TCnaViewEB::GetListOfRunParameters(...)> *** ERROR *** =====> "
	   << run_par_file_name.Data() << " : file not found."
	   << fTTBELL << endl;
      nb_of_runs_in_list = -1;
    }
  return nb_of_runs_in_list;
}
//   end of GetListOfRunParameters(...)

//------------------------------------------------------------------------------------------------

void TCnaViewEB::AllocArraysForEvol()
{
// Allocation and initialization of the arrays.


  Int_t arrays_dim = (Int_t)1000;    //=====> METTRE CA EN PARAMETRE
  //  Nb max de runs dans les listes de runs pour l'option TIME EVOLUTION (faire des tests de depassement)

  //................................. Alloc of the arrays at first call

  if(fT1DAnaType             == 0){fT1DAnaType             = new TString[arrays_dim];  fCnew++;}
  if(fT1DRunNumber           == 0){fT1DRunNumber           = new   Int_t[arrays_dim];  fCnew++;}
  if(fT1DFirstEvt            == 0){fT1DFirstEvt            = new   Int_t[arrays_dim];  fCnew++;}
  if(fT1DNbOfEvts            == 0){fT1DNbOfEvts            = new   Int_t[arrays_dim];  fCnew++;}
  if(fT1DSuMoNumber          == 0){fT1DSuMoNumber          = new   Int_t[arrays_dim];  fCnew++;}
  if(fT1DResultsRootFilePath == 0){fT1DResultsRootFilePath = new TString[arrays_dim];  fCnew++;}
  if(fT1DListOfRunsFilePath  == 0){fT1DListOfRunsFilePath  = new TString[arrays_dim];  fCnew++;}

  //................................. Init of the arrays
  Int_t MaxCar = fgMaxCar;
  for ( Int_t i_run = 0; i_run < arrays_dim; i_run++ )
    {
      fT1DAnaType[i_run].Resize(MaxCar); fT1DAnaType[i_run] = "no analysis name info";
      fT1DRunNumber[i_run]  = -1;
      fT1DFirstEvt[i_run]   = -1;
      fT1DNbOfEvts[i_run]   = -1;
      fT1DSuMoNumber[i_run] = -1;
      MaxCar = fgMaxCar;
      fT1DResultsRootFilePath[i_run].Resize(MaxCar);
      fT1DResultsRootFilePath[i_run] = "no ROOT file path info";
      MaxCar = fgMaxCar;
      fT1DListOfRunsFilePath[i_run].Resize(MaxCar);
      fT1DListOfRunsFilePath[i_run] = "no LIST OF RUN file path info";

    }
}
//........ end of AllocArraysForEvol()

Int_t TCnaViewEB::GetListFileNumber(const TString QuantityCode)
{
// Get the number of the read list file
  
  Int_t number = 0;

  if ( QuantityCode == "EvolEvEv"    ){number = fNbOfListFileEvolEvEv;}
  if ( QuantityCode == "EvolEvSig"   ){number = fNbOfListFileEvolEvSig;}
  if ( QuantityCode == "EvolEvCorss" ){number = fNbOfListFileEvolEvCorss;}

  return number;
}

//--------------------------------------------------------------------------------------------
//
//          SetFile(...)   ( 3 METHODS )
//
//--------------------------------------------------------------------------------------------

void TCnaViewEB::SetFile(const Int_t&  xArgIndexRun)
{
// Set parameters for reading the right CNA results file

  fFapAnaType      = fT1DAnaType[xArgIndexRun];
  fFapRunNumber    = fT1DRunNumber[xArgIndexRun];
  fFapFirstEvt     = fT1DFirstEvt[xArgIndexRun];
  fFapNbOfEvts     = fT1DNbOfEvts[xArgIndexRun];
  fFapSuMoNumber   = fT1DSuMoNumber[xArgIndexRun];

  fCfgResultsRootFilePath = fT1DResultsRootFilePath[xArgIndexRun];
  fCfgListOfRunsFilePath  = fT1DListOfRunsFilePath[xArgIndexRun];

  // Init parameters that will be set by reading the info which are in the CNA results file
  Int_t MaxCar = fgMaxCar;
  fStartDate.Resize(MaxCar);
  fStartDate = "(date not found)";

  MaxCar = fgMaxCar;
  fStopDate.Resize(MaxCar);
  fStopDate  = "(date not found)";

  fStartTime = (time_t)0;
  fStopTime  = (time_t)0;

  fFapChanNumber = 0;
  fFapSampNumber = 0;
  fFapTowXNumber = 0;
  fFapTowYNumber = 0;
}

//===> DON'T SUPPRESS: THIS METHOD IS CALLED BY TCnaDialogEB and can be called by any other program

void TCnaViewEB::SetFile(const TString xArgAnaType,
			 const Int_t&  xArgRunNumber, const Int_t&  xArgFirstEvt,
			 const Int_t&  xArgNbOfEvts,  const Int_t&  xArgSuMoNumber,
			 const TString xArgResultsRootFilePath,
			 const TString xArgListOfRunsFilePath)
{
// Set parameters for reading the right CNA results file

  fFapAnaType      = xArgAnaType;
  fFapRunNumber    = xArgRunNumber;
  fFapFirstEvt     = xArgFirstEvt;
  fFapNbOfEvts     = xArgNbOfEvts;
  fFapSuMoNumber   = xArgSuMoNumber;

  fCfgResultsRootFilePath = xArgResultsRootFilePath;
  fCfgListOfRunsFilePath  = xArgListOfRunsFilePath;

  // Init parameters that will be set by reading the info which are in the CNA results file
  Int_t MaxCar = fgMaxCar;
  fStartDate.Resize(MaxCar);
  fStartDate = "(date not found)";

  MaxCar = fgMaxCar;
  fStopDate.Resize(MaxCar);
  fStopDate  = "(date not found)";

  fStartTime = (time_t)0;
  fStopTime  = (time_t)0;

  fFapChanNumber = 0;
  fFapSampNumber = 0;
  fFapTowXNumber = 0;
  fFapTowYNumber = 0;
}

//===> DON'T SUPPRESS: THIS METHOD IS CALLED BY TCnaDialogEB and can be called by any other program

void TCnaViewEB::SetFile(const TString xArgAnaType,
			 const Int_t&  xArgRunNumber, const Int_t&  xArgFirstEvt,
			 const Int_t&  xArgNbOfEvts,  const Int_t&  xArgSuMoNumber)
{
// Set parameters for reading the right CNA results file

  fFapAnaType      = xArgAnaType;
  fFapRunNumber    = xArgRunNumber;
  fFapFirstEvt     = xArgFirstEvt;
  fFapNbOfEvts     = xArgNbOfEvts;
  fFapSuMoNumber   = xArgSuMoNumber;

  // Init parameters that will be set by reading the info which are in the CNA results file
  Int_t MaxCar = fgMaxCar;
  fStartDate.Resize(MaxCar);
  fStartDate = "(date not found)";

  MaxCar = fgMaxCar;
  fStopDate.Resize(MaxCar);
  fStopDate  = "(date not found)";

  fStartTime = (time_t)0;
  fStopTime  = (time_t)0;

  fFapChanNumber = 0;
  fFapSampNumber = 0;
  fFapTowXNumber = 0;
  fFapTowYNumber = 0;
}

//======================================================================================
//
//           C O M M E N T S / I N F O S     P A V E S      M E T H O D S 
//
//======================================================================================
//======================================================================================================

Bool_t TCnaViewEB::GetOkViewHisto(TCnaReadEB*   MyRootFile,
				  const Int_t&  SMtower_X, const Int_t& TowEcha, const Int_t& sample,
				  const TString QuantityCode)
{
// Check possibility to plot the histo

  Bool_t ok_view_histo = kFALSE;

  TString   QuantityType   = GetQuantityType(QuantityCode);

  Int_t     nb_of_towers   = MyRootFile->MaxTowInSM();
  Int_t     nb_of_crystals = MyRootFile->MaxCrysInTow();
  Int_t     nb_of_samples  = MyRootFile->MaxSampADC();

  TString   root_file_name = MyRootFile->GetRootFileNameShort();

  TVectorD  vtow(nb_of_towers);
  vtow = MyRootFile->ReadTowerNumbers();

  if ( MyRootFile->DataExist() == kTRUE )
    {
      Int_t tower_ok = 0;
      for (Int_t index_tow = 0; index_tow < nb_of_towers; index_tow++)
	{ if ( vtow(index_tow) == SMtower_X ){tower_ok++;}}
      
      //.............................................. ok_view   
      Int_t ok_view    = 1;
      
      if( !( QuantityType == "Global" || QuantityType == "Proj" )  )
	{if( tower_ok != 1)
	  {cout << "!TCnaViewEB::GetOkViewHisto(...)> *ERROR* =====> " << "File: " << root_file_name
		<< ", Tower " << SMtower_X << " not found. Available numbers = ";
	  for(Int_t i = 0; i < nb_of_towers; i++){ if( vtow(i) > 0 ){cout << vtow(i) << ", ";}}
	  cout << fTTBELL << endl;
	  ok_view = -1;}
	else { ok_view = 1;}}
      
      //.............................................. ok_max_elt 
      Int_t ok_max_elt  = 1;
      
      if( ( ( (QuantityType == "NotSMRun") || (QuantityType == "NotSMNoRun") )
	    && (TowEcha >= 0) && (TowEcha<nb_of_crystals) 
	    && (sample  >= 0) && (sample <nb_of_samples ) ) ||
	  !( (QuantityType == "NotSMRun") || (QuantityType == "NotSMNoRun") ) )
	{ok_max_elt = 1;} 
      else
	{
	  if( ( (QuantityType == "NotSMRun") || (QuantityType == "NotSMNoRun") )
	      && !( (TowEcha >= 0) && (TowEcha<nb_of_crystals) ) )
	    {cout << "!TCnaViewEB::GetOkViewHisto(...)> *ERROR* =====> " << "File: " << root_file_name
		  << ". Wrong TowEcha number (required range: [0, "
		  << MyRootFile->MaxCrysInTow()-1 << "] )"
		  << fTTBELL << endl;}
	  if( (QuantityCode == "Evts"  && !((sample >= 0) && (sample <nb_of_samples)) ) )
	    {cout << "!TCnaViewEB::GetOkViewHisto(...)> *ERROR* =====> " << "File: " << root_file_name
		  << ". Wrong sample number (required range: [0, "
		  << MyRootFile->MaxSampADC()-1 << "] )"
		  << fTTBELL << endl;}
	  ok_max_elt = -1;
	}
      
      if( (ok_view == 1) && (ok_max_elt == 1) )
	{
	  ok_view_histo = kTRUE;
	}
      else
	{
	  cout << "!TCnaViewEB::GetOkViewHisto(...)> At least one ERROR has been detected. ok_view = " << ok_view
	       << ", ok_max_elt = " << ok_max_elt << fTTBELL << endl;
	}
    }
  return ok_view_histo;
}
//..............................................................................................

Int_t TCnaViewEB::HistoSetMinMax(TH1D* h_his0, const TString QuantityCode)
{
// Set min and max according to QuantityCode
  
  // if Ymin = Ymax (or Ymin > Ymax): nothing done here
  // return xFlagAutoYsupMargin = 1
  //
  // if Ymin < Ymax: min and max calculated by h_his0->SetMinimum() and h_his0->SetMaximum()
  // return xFlagAutoYsupMargin = 0

  Int_t xFlagAutoYsupMargin = 1;  

  if(QuantityCode == "SMFoundEvtsGlobal"){ 
    if(fSMFoundEvtsGlobalYmin < fSMFoundEvtsGlobalYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSMFoundEvtsGlobalYmin);  h_his0->SetMaximum(fSMFoundEvtsGlobalYmax);}}
  
  if(QuantityCode == "SMFoundEvtsProj"){
    if(fSMFoundEvtsProjYmin < fSMFoundEvtsProjYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSMFoundEvtsProjYmin); h_his0->SetMaximum(fSMFoundEvtsProjYmax);}}
  
  if(QuantityCode == "SMEvEvGlobal"){
    if(fSMEvEvGlobalYmin < fSMEvEvGlobalYmax){xFlagAutoYsupMargin = 0;
	h_his0->SetMinimum(fSMEvEvGlobalYmin);  h_his0->SetMaximum(fSMEvEvGlobalYmax);}}
  
  if(QuantityCode == "SMEvEvProj"){
    if(fSMEvEvProjYmin < fSMEvEvProjYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSMEvEvProjYmin); h_his0->SetMaximum(fSMEvEvProjYmax);}}
  
  if(QuantityCode == "SMEvSigGlobal"){
    if(fSMEvSigGlobalYmin < fSMEvSigGlobalYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSMEvSigGlobalYmin); h_his0->SetMaximum(fSMEvSigGlobalYmax);}}
  
  if(QuantityCode == "SMEvSigProj"){
    if(fSMEvSigProjYmin < fSMEvSigProjYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSMEvSigProjYmin); h_his0->SetMaximum(fSMEvSigProjYmax);}}
  
  if(QuantityCode == "SMSigEvGlobal"){
    if(fSMEvSigGlobalYmin < fSMSigEvGlobalYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSMEvSigGlobalYmin); h_his0->SetMaximum(fSMSigEvGlobalYmax);}}
  
  if(QuantityCode == "SMSigEvProj"){
    if(fSMSigEvProjYmin < fSMSigEvProjYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSMSigEvProjYmin); h_his0->SetMaximum(fSMSigEvProjYmax);}}
  
  if(QuantityCode == "SMSigSigGlobal"){
    if(fSMSigSigGlobalYmin < fSMSigSigGlobalYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSMSigSigGlobalYmin); h_his0->SetMaximum(fSMSigSigGlobalYmax);}}
  
  if(QuantityCode == "SMSigSigProj"){
    if(fSMSigSigProjYmin < fSMSigSigProjYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSMSigSigProjYmin); h_his0->SetMaximum(fSMSigSigProjYmax);}}
  
  if(QuantityCode == "SMEvCorssGlobal"){
    if(fSMEvCorssGlobalYmin < fSMEvCorssGlobalYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSMEvCorssGlobalYmin); h_his0->SetMaximum(fSMEvCorssGlobalYmax);}}
  
  if(QuantityCode == "SMEvCorssProj"){
    if(fSMEvCorssProjYmin < fSMEvCorssProjYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSMEvCorssProjYmin); h_his0->SetMaximum(fSMEvCorssProjYmax);}}
  
  if(QuantityCode == "SMSigCorssGlobal"){
    if(fSMSigCorssGlobalYmin < fSMSigCorssGlobalYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSMSigCorssGlobalYmin); h_his0->SetMaximum(fSMSigCorssGlobalYmax);}}
  
  if(QuantityCode == "SMSigCorssProj"){
    if(fSMSigCorssProjYmin < fSMSigCorssProjYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSMSigCorssProjYmin); h_his0->SetMaximum(fSMSigCorssProjYmax);}}
  
  if(QuantityCode == "Ev"){
    if(fEvYmin < fEvYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fEvYmin); h_his0->SetMaximum(fEvYmax);}}
  
  if(QuantityCode == "Sigma"){
    if(fSigmaYmin < fSigmaYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSigmaYmin); h_his0->SetMaximum(fSigmaYmax);}}
  
  if(QuantityCode == "SampTime"){
    if(fSampTimeYmin < fSampTimeYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSampTimeYmin); h_his0->SetMaximum(fSampTimeYmax);}}

  if(QuantityCode == "SMCorccInTowers"){
    if(fSMCorccInTowersYmin < fSMCorccInTowersYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSMCorccInTowersYmin); h_his0->SetMaximum(fSMCorccInTowersYmax);}}

  if(QuantityCode == "SMEvCorttMatrix"){
    if(fSMEvCorttMatrixYmin < fSMEvCorttMatrixYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fSMEvCorttMatrixYmin); h_his0->SetMaximum(fSMEvCorttMatrixYmax);}}

  return xFlagAutoYsupMargin;  
} // end of  HistoSetMinMax

Int_t TCnaViewEB::GraphSetMinMax(TGraph* g_graph0, const TString QuantityCode)
{
// Set min and max according to QuantityCode
  
  Int_t xFlagAutoYsupMargin = 1;    // DEFAULT: 1 = min and max calulated by ROOT, 0 = by this code 

  if(QuantityCode == "EvolEvEv"){
    if(fEvolEvEvYmin < fEvolEvEvYmax){xFlagAutoYsupMargin = 0;
    g_graph0->SetMinimum(fEvolEvEvYmin); g_graph0->SetMaximum(fEvolEvEvYmax);}}

  if(QuantityCode == "EvolEvSig"){
    if(fEvolEvSigYmin < fEvolEvSigYmax){xFlagAutoYsupMargin = 0;
    g_graph0->SetMinimum(fEvolEvSigYmin); g_graph0->SetMaximum(fEvolEvSigYmax);}}

  if(QuantityCode == "EvolEvCorss"){
    if(fEvolEvCorssYmin < fEvolEvCorssYmax){xFlagAutoYsupMargin = 0;
    g_graph0->SetMinimum(fEvolEvCorssYmin); g_graph0->SetMaximum(fEvolEvCorssYmax);}}

  return xFlagAutoYsupMargin;  
} // end of GraphSetMinMax



//----------------------------------------------- HistoPlot

void TCnaViewEB::HistoPlot(TH1D* h_his0,
			   TCnaReadEB*   MyRootFile,   TEBNumbering*  MyNumbering,  const Int_t&   HisSize,  
			   const TString QuantityCode, const TString  QuantityType, const Int_t&   nb_of_towers,  
			   const Int_t&  SMtower_X,    const Int_t&   TowEcha,      const Int_t&   sample, 
			   const Int_t&  opt_visu,     const Int_t&   opt_scale,    const TString  opt_plot,
			   const Int_t&  xFlagAutoYsupMargin)
{
  // Plot 1D histogram

  UInt_t canv_w = SetCanvasWidth(QuantityCode);
  UInt_t canv_h = SetCanvasHeight(QuantityCode);

  //.................................................. prepa paves commentaires (HistoPlot)

  PutAllPavesViewHisto(MyRootFile, MyNumbering, QuantityCode, SMtower_X, TowEcha, sample, opt_plot);
    
  //..................................................... Canvas name (HistoPlot) 
  
  TString canvas_name = SetCanvasName(QuantityCode, opt_visu, opt_scale, opt_plot,
				      SMtower_X,    TowEcha,  sample);
	  
  //------------------------------------------------ Canvas allocation

  SetHistoPresentation(h_his0, QuantityType);

  //......................................... declarations canvas et pad  (HistoPlot)
  TCanvas*     NoiseCorrel = 0;
  TVirtualPad* main_subpad = 0;
  TPaveText*   main_pavtxt = 0;

  if(opt_plot == fOnlyOnePlot)
    {
      NoiseCorrel = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h);   fCnewRoot++;
    }

  if(opt_plot == fSeveralPlot)
    {
      if(GetMemoFlag(QuantityCode) == "Free"){CreateCanvas(QuantityCode, canvas_name, canv_w , canv_h);}
    }	 

  // cout << "*TCnaViewEB::HistoPlot(...)> Plot is displayed on canvas ----> " << canvas_name.Data() << endl;

  //.................................................. Draw titles (pad = main canvas)
  if( opt_plot == fOnlyOnePlot )
    {
      ftitle_g1->Draw();
      fcom_top_left->Draw();

      if( !( QuantityType == "Global" || QuantityType == "Proj" ) )
	{fcom_top_mid->Draw(); fcom_top_right->Draw();}

      fcom_bot_left->Draw();
      fcom_bot_right->Draw();
    }

  //.............................. Init operations on canvas at first call to option SAME  (HistoPlot)
	  
  Int_t xMemoPlotSame = 1;   // a priori ==> SAME plot 

  Int_t last_evt = fFapFirstEvt + fFapNbOfEvts - 1;
	  
  if(opt_plot == fSeveralPlot)
    {
      if(GetMemoFlag(QuantityCode) == "Free")
	{
	  fCurrentPad = gPad;
	  ftitle_g1->Draw();

	  fcom_top_left->SetTextAlign(fTextPaveAlign);
	  fcom_top_left->SetTextFont(fTextPaveFont);
	  fcom_top_left->SetTextSize(fTextPaveSize);

	  char* f_in = new char[fgMaxCar];                            fCnew++;
	  
	  if(QuantityType == "Evts"      ){sprintf(f_in, "Analysis   RUN  1stEvt LastEvt SM Tower Crystal Sample");}
	  if(QuantityType == "NotSMRun"  ){sprintf(f_in, "Analysis   RUN  1stEvt LastEvt SM Tower Crystal");}
	  if(QuantityType == "NotSMNoRun"){sprintf(f_in, "Analysis   1stEvt  LastEvt SM Tower Crystal");}
	  if(  (QuantityType == "Global")
	     ||(QuantityType == "Proj")  ){sprintf(f_in, "Analysis   RUN  1stEvt LastEvt SM");}

	  TText* ttit = fcom_top_left->AddText(f_in);
	  ttit->SetTextColor(ColorDefinition("noir"));
	  
	  if(QuantityType == "Evts"     )
	    {
	      Int_t SM_crys  = MyNumbering->GetSMCrysFromSMTowAndTowEcha(SMtower_X, TowEcha);
	      sprintf(f_in, "%-8s%6d%8d%8d%3d%6d%8d%7d",
		      fFapAnaType.Data(), fFapRunNumber, fFapFirstEvt, last_evt, fFapSuMoNumber,
		      SMtower_X, SM_crys, sample);
	    }

	   if( QuantityType == "NotSMRun" )
	    {
	      Int_t SM_crys  = MyNumbering->GetSMCrysFromSMTowAndTowEcha(SMtower_X, TowEcha);
	      sprintf(f_in, "%-8s%6d%8d%8d%3d%6d%8d",
		      fFapAnaType.Data(), fFapRunNumber, fFapFirstEvt, last_evt, fFapSuMoNumber,
		      SMtower_X, SM_crys);
	    }

	  if( QuantityType == "NotSMNoRun" )
	    {
	      Int_t SM_crys  = MyNumbering->GetSMCrysFromSMTowAndTowEcha(SMtower_X, TowEcha);
	      sprintf(f_in, "%-8s%9d%9d%3d%6d%8d",
		     fFapAnaType.Data(), fFapFirstEvt, last_evt, fFapSuMoNumber, SMtower_X, SM_crys);
	    }

	  if( (QuantityType == "Global") || (QuantityType == "Proj")  )
	    {
	      sprintf(f_in, "%-8s%6d%8d%8d%3d",
		     fFapAnaType.Data(), fFapRunNumber, fFapFirstEvt, last_evt, fFapSuMoNumber);
	    }

	  TText* tt = fcom_top_left->AddText(f_in);
	  tt->SetTextColor(GetViewHistoColor(QuantityCode));
	  
	  delete [] f_in;                                             fCdelete++;

	  SetParametersCanvas(QuantityCode); xMemoPlotSame = 0;
	}

      //............................ cases fMemoPlotxxx = 1

      if(GetMemoFlag(QuantityCode) == "Busy")
	{
	  main_pavtxt = ActivePavTxt(QuantityCode);
	  main_subpad = ActivePad(QuantityCode);
	}
    }

  if(opt_plot == fOnlyOnePlot)
    {
      fCurrentPad = gPad;
      NoiseCorrel->Divide(1, 1, 0.001 , 0.125);
      gPad->cd(1);
      main_subpad = gPad;
      xMemoPlotSame = 0;
    }
  
  if(main_subpad != 0)
    {
      if(opt_plot == fSeveralPlot)
	{
	  if(xMemoPlotSame != 0)
	    {
	      main_pavtxt->SetTextAlign(fTextPaveAlign);
	      main_pavtxt->SetTextFont(fTextPaveFont);
	      main_pavtxt->SetTextSize(fTextPaveSize);
	      
	      char* f_in = new char[fgMaxCar];                            fCnew++;

	      if(QuantityType == "Evts"     )
		{
		  Int_t SM_crys  = MyNumbering->GetSMCrysFromSMTowAndTowEcha(SMtower_X, TowEcha);
		  sprintf(f_in, "%-8s%6d%8d%8d%3d%6d%8d%7d",
			  fFapAnaType.Data(), fFapRunNumber, fFapFirstEvt, last_evt, fFapSuMoNumber,
			  SMtower_X, SM_crys, sample);
		}
	      
	      if(QuantityType == "NotSMRun"  )
		{
		  Int_t SM_crys  = MyNumbering->GetSMCrysFromSMTowAndTowEcha(SMtower_X, TowEcha);
		  sprintf(f_in, "%-8s%6d%8d%8d%3d%6d%8d",
			  fFapAnaType.Data(), fFapRunNumber, fFapFirstEvt, last_evt, fFapSuMoNumber,
			  SMtower_X, SM_crys);
		}
	      
	      if(QuantityType == "NotSMNoRun")
		{
		  Int_t SM_crys  = MyNumbering->GetSMCrysFromSMTowAndTowEcha(SMtower_X, TowEcha);
		  sprintf(f_in, "%-8s%8d%8d%3d%6d%8d",
			  fFapAnaType.Data(), fFapFirstEvt, last_evt, fFapSuMoNumber, SMtower_X, SM_crys);
		}
	      
	      if( (QuantityType == "Global") || (QuantityType == "Proj") )
		{
		  sprintf(f_in, "%-8s%6d%8d%8d%3d",
			  fFapAnaType.Data(), fFapRunNumber, fFapFirstEvt, last_evt, fFapSuMoNumber);
		}
	       
	      TText *tt = main_pavtxt->AddText(f_in);
	      tt->SetTextColor(GetViewHistoColor(QuantityCode));
	      
	      delete [] f_in;                                             fCdelete++;
	    }
	}
      
      main_subpad->cd();
    
      //............................................ Style	(HistoPlot)
      SetViewHistoColors(h_his0, QuantityCode, opt_plot);

      //................................. Set axis titles
      TString axis_x_var_name = SetHistoXAxisTitle(QuantityCode);
      TString axis_y_var_name = SetHistoYAxisTitle(QuantityCode);
      h_his0->GetXaxis()->SetTitle(axis_x_var_name);
      h_his0->GetYaxis()->SetTitle(axis_y_var_name);
	      
      Int_t liny = 0;
      Int_t logy = 1;
	      
      if(opt_plot == fOnlyOnePlot)
	{
	  if(opt_visu == fOptVisLine && opt_scale == fOptScaleLiny){gPad->SetLogy(liny); h_his0->DrawCopy();}
	  if(opt_visu == fOptVisLine && opt_scale == fOptScaleLogy){gPad->SetLogy(logy); h_his0->DrawCopy();}
	  if(opt_visu == fOptVisPolm && opt_scale == fOptScaleLiny){gPad->SetLogy(liny); h_his0->DrawCopy("P");}
	  if(opt_visu == fOptVisPolm && opt_scale == fOptScaleLogy){gPad->SetLogy(logy); h_his0->DrawCopy("P");}
	}
	      
      if(opt_plot == fSeveralPlot)
	{
	  if(xMemoPlotSame == 0)
	    {
	      if(opt_visu == fOptVisLine && opt_scale == fOptScaleLiny){gPad->SetLogy(liny); h_his0->DrawCopy();}
	      if(opt_visu == fOptVisLine && opt_scale == fOptScaleLogy){gPad->SetLogy(logy); h_his0->DrawCopy();}
	      if(opt_visu == fOptVisPolm && opt_scale == fOptScaleLiny){gPad->SetLogy(liny); h_his0->DrawCopy("P");}
	      if(opt_visu == fOptVisPolm && opt_scale == fOptScaleLogy){gPad->SetLogy(logy); h_his0->DrawCopy("P");}

	    }
	  if(xMemoPlotSame != 0)
	    {
	      if(opt_visu == fOptVisLine && opt_scale == fOptScaleLiny)
		{gPad->SetLogy(liny); h_his0->DrawCopy("SAME");}
	      if(opt_visu == fOptVisLine && opt_scale == fOptScaleLogy)
		{gPad->SetLogy(logy); h_his0->DrawCopy("SAME");}
	      if(opt_visu == fOptVisPolm && opt_scale == fOptScaleLiny)
		{gPad->SetLogy(liny); h_his0->DrawCopy("PSAME");}
	      if(opt_visu == fOptVisPolm && opt_scale == fOptScaleLogy)
		{gPad->SetLogy(logy); h_his0->DrawCopy("PSAME");}
	    }
	}
	      
      //.................... Horizontal line at y=0
      if( !(  QuantityCode == "Evts" ||	QuantityCode == "SampTime" || QuantityType == "Proj" ) )
	{  
	  TLine* lin =  new TLine(0.,0.,(Double_t)HisSize, 0.);   fCnewRoot++;
	  lin->Draw();
	}
	      
      //...................................................... Axis for the tower numbers
      if(QuantityType == "Global"){
	TopAxisForTowerNumbers(h_his0, opt_plot,    xMemoPlotSame, nb_of_towers,
			       xFlagAutoYsupMargin, HisSize);}
      
      fCurrentPad->cd();
      
      fcom_top_left->Draw();
      gPad->Update();
    }
  else    // else du if(main_subpad !=0)
    {
      cout << "*TCnaViewEB::HistoPlot(...)> WARNING ===> Canvas already removed in option SAME." << endl
	   << "                             Click again on the same menu entry"
	   <<" to restart with a new canvas."
	   << fTTBELL << endl;

      ReInitCanvas(QuantityCode);
      xMemoPlotSame = 0;
    }

  //  title_g1->Delete();                 fCdeleteRoot++;
  //  com_bot_left->Delete();             fCdeleteRoot++;
  //  delete NoiseCorrel;                 fCdeleteRoot++;

} // end of HistoPlot

//----------------------------------------------- HistimePlot

void TCnaViewEB::HistimePlot(TGraph*       g_graph0,
			     Axis_t        xinf,               Axis_t   xsup,
			     TCnaReadEB*   MyRootFile,   TEBNumbering*  MyNumbering,  
			     const TString QuantityCode, const TString  QuantityType, const Int_t&   nb_of_towers,
			     const Int_t&  SMtower_X,    const Int_t&   TowEcha,      const Int_t&   sample, 
			     const Int_t&  opt_visu,     const Int_t&   opt_scale,    const TString  opt_plot,
			     const Int_t&  xFlagAutoYsupMargin)
{
  // Plot 1D histogram for evolution in time

  UInt_t canv_w = SetCanvasWidth(QuantityCode);
  UInt_t canv_h = SetCanvasHeight(QuantityCode);

  SetGraphPresentation(g_graph0, QuantityType);    // (gStyle parameters)

  //...................................................... paves commentaires (HistimePlot)
	 
  PutAllPavesViewHisto(MyRootFile, MyNumbering, QuantityCode, SMtower_X, TowEcha, sample, opt_plot);

  //..................................................... Canvas name (HistimePlot) 

  TString canvas_name = SetCanvasName(QuantityCode, opt_visu, opt_scale, opt_plot,
				      SMtower_X,    TowEcha,  sample);
	  
  //------------------------------------------------ Canvas allocation	(HistimePlot)

  //......................................... declarations canvas et pad
  TCanvas*     NoiseCorrel = 0;
  TVirtualPad* main_subpad = 0;
  TPaveText*   main_pavtxt = 0;

  if(opt_plot == fOnlyOnePlot)
    {
      NoiseCorrel = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h);   fCnewRoot++;
    }

  if(opt_plot == fSeveralPlot)
    {
      if(GetMemoFlag(QuantityCode) == "Free"){CreateCanvas(QuantityCode, canvas_name, canv_w , canv_h);}
    }	 

  // cout << "*TCnaViewEB::HistimePlot(...)> Plot is displayed on canvas ----> " << canvas_name.Data() << endl;

  //.................................................. Draw titles (pad = main canvas)

  if( opt_plot == fOnlyOnePlot )
    {
      ftitle_g1->Draw();
      fcom_top_left->Draw();

      if( !( QuantityType == "Global"   || QuantityType == "Proj" ) )
	{
	  fcom_top_mid->Draw();
	  fcom_top_right->Draw();
	}
      fcom_bot_left->Draw();
      fcom_bot_right->Draw();
    }

  if( opt_plot == fSeveralPlot && GetMemoFlag(QuantityCode) == "Free" )
    {
      ftitle_g1->Draw();
      fcom_top_left->Draw();

      if( !( QuantityType == "Global"   || QuantityType == "Proj"   ||
	     QuantityCode == "EvolEvEv" || QuantityCode == "EvolEvSig" || QuantityCode == "EvolEvCorss") )
	{
	  fcom_top_mid->Draw();
	  fcom_top_right->Draw();
	}
      
      if( !( QuantityCode == "EvolEvEv" || QuantityCode == "EvolEvSig" || QuantityCode == "EvolEvCorss" ) )
	{
	  fcom_bot_left->Draw();
	}
      fcom_bot_right->Draw();
    }
  
  //.............................. Init operations on canvas at first call to option SAME  (HistimePlot)

  Int_t xMemoPlotSame = 1;   // a priori ==> SAME plot 

  Int_t last_evt = fFapFirstEvt + fFapNbOfEvts - 1;
  Int_t SM_crys  = MyNumbering->GetSMCrysFromSMTowAndTowEcha(SMtower_X, TowEcha);
	  
  if(opt_plot == fSeveralPlot)
    {
      if(GetMemoFlag(QuantityCode) == "Free")
	{
	  fcom_top_left->SetTextAlign(fTextPaveAlign);
	  fcom_top_left->SetTextFont(fTextPaveFont);
	  fcom_top_left->SetTextSize(fTextPaveSize);

	  char* f_in = new char[fgMaxCar];                            fCnew++;

	  sprintf(f_in, "Analysis  1stEvt LastEvt SM Tower Crystal");
	  TText* ttit = fcom_top_left->AddText(f_in);
	  ttit->SetTextColor(ColorDefinition("noir"));

	  sprintf(f_in, "%-8s%8d%8d%3d%6d%8d",
		  fFapAnaType.Data(), fFapFirstEvt, last_evt, fFapSuMoNumber, SMtower_X, SM_crys);
	  
	  TText* tt = fcom_top_left->AddText(f_in);
	  tt->SetTextColor(GetViewHistoColor(QuantityCode));

	  delete [] f_in;                                             fCdelete++;

	  SetParametersCanvas(QuantityCode); xMemoPlotSame = 0;
	}

      //............................ cases fMemoPlotxxx = 1            (HistimePlot)

      if(GetMemoFlag(QuantityCode) == "Busy")
	{
	  main_pavtxt = ActivePavTxt(QuantityCode);
	  main_subpad = ActivePad(QuantityCode);
	}
    }

  if(opt_plot == fOnlyOnePlot)
    {
      NoiseCorrel->Divide(1, 1, 0.001 , 0.125);
      gPad->cd(1);
      main_subpad = gPad;
      xMemoPlotSame = 0;
    }

  if(main_subpad != 0)
    {
      if(opt_plot == fSeveralPlot)
	{
	  if(xMemoPlotSame != 0)
	    {
	      main_pavtxt->SetTextAlign(fTextPaveAlign);
	      main_pavtxt->SetTextFont(fTextPaveFont);
	      main_pavtxt->SetTextSize(fTextPaveSize);

	      char* f_in = new char[fgMaxCar];                            fCnew++;

	      sprintf(f_in, "%-8s%8d%8d%3d%6d%8d",
		      fFapAnaType.Data(), fFapFirstEvt, last_evt, fFapSuMoNumber, SMtower_X, SM_crys);

	      TText *tt = main_pavtxt->AddText(f_in);
	      tt->SetTextColor(GetViewHistoColor(QuantityCode));

	      delete [] f_in;                                             fCdelete++;
	    }
	}
      main_subpad->cd();
      	      
      //............................................ Style	(HistimePlot)
      SetViewGraphColors(g_graph0, QuantityCode, opt_plot);

      //................................. Set axis titles
      TString axis_x_var_name = SetHistoXAxisTitle(QuantityCode);
      TString axis_y_var_name = SetHistoYAxisTitle(QuantityCode);
      g_graph0->GetXaxis()->SetTitle(axis_x_var_name);
      g_graph0->GetYaxis()->SetTitle(axis_y_var_name);

      //................................. Xaxis is a time axis
      g_graph0->GetXaxis()->SetTimeDisplay(1);
      g_graph0->GetXaxis()->SetTimeFormat("%d %b-%Hh");
 
      g_graph0->GetXaxis()->SetTimeOffset(xinf);

      Int_t nb_displayed = 20;      // max nb of run numbers displayed

      //...........................................................................	(HistimePlot)
      Int_t liny = 0;
      Int_t logy = 1;
	      
      if(opt_plot == fOnlyOnePlot)
	{
	  fXinf = (Double_t)xinf;
	  fXsup = (Double_t)xsup;
	  fYinf = (Double_t)GetYmin(QuantityCode);
	  fYsup = (Double_t)GetYmax(QuantityCode);
	  gPad->RangeAxis(fXinf, fYinf, fXsup, fYsup);

	  if(opt_visu == fOptVisLine && opt_scale == fOptScaleLiny){gPad->SetLogy(liny); g_graph0->Draw();}
	  if(opt_visu == fOptVisLine && opt_scale == fOptScaleLogy){gPad->SetLogy(logy); g_graph0->Draw();}
	  if(opt_visu == fOptVisPolm && opt_scale == fOptScaleLiny)
	    {
	      gPad->SetLogy(liny);
	      g_graph0->Draw("ALP");
	      Int_t     nb_pts  = g_graph0->GetN();
	      Double_t* coord_x = g_graph0->GetX();
	      Double_t* coord_y = g_graph0->GetY();

	      char* f_in = new char[fgMaxCar];                            fCnew++;

	      //................. display of the run numbers

	      Double_t interv_displayed = (coord_x[nb_pts-1] - coord_x[0])/(Double_t)nb_displayed;
	      Double_t last_drawn_coordx = coord_x[0] - 1.5*interv_displayed;

	      for(Int_t i_run=0; i_run<nb_pts; i_run++)
		{
		  if ( (coord_x[i_run] - last_drawn_coordx) > interv_displayed )
		    {
		      sprintf( f_in, "R%d",  fT1DRunNumber[i_run]);
		      TText *text_run_num = new TText(coord_x[i_run], fYsup, f_in);  fCnewRoot++;
		      text_run_num->SetTextAngle((Double_t)45.);
		      text_run_num->SetTextSize((Double_t)0.035);
		      text_run_num->Draw("SAME");
		      // delete text_SMtow_num;             fCdeleteRoot++;

		      TLine *jointlign;
		      jointlign = new TLine(coord_x[i_run], fYsup, coord_x[i_run], coord_y[i_run]); fCnewRoot++;
		      jointlign->SetLineWidth(1);
		      jointlign->Draw("SAME");
		      // delete jointlign;                  fCdeleteRoot++;

		      last_drawn_coordx = coord_x[i_run];
		    }
		}

	      delete [] f_in;                                               fCdelete++;

	    }
	  if(opt_visu == fOptVisPolm && opt_scale == fOptScaleLogy)
	    {
	      gPad->SetLogy(logy);
	      g_graph0->Draw("ALP");
	    }
	}

      //................................................  (HistimePlot)
      if(opt_plot == fSeveralPlot)
	{
	  if(xMemoPlotSame == 0)
	    {
	      if(opt_visu == fOptVisLine && opt_scale == fOptScaleLiny){gPad->SetLogy(liny); g_graph0->Draw();}
	      if(opt_visu == fOptVisLine && opt_scale == fOptScaleLogy){gPad->SetLogy(logy); g_graph0->Draw();}
	      if(opt_visu == fOptVisPolm && opt_scale == fOptScaleLiny){gPad->SetLogy(liny); g_graph0->Draw("AP");}
	      if(opt_visu == fOptVisPolm && opt_scale == fOptScaleLogy){gPad->SetLogy(logy); g_graph0->Draw("AP");}

	      fXinf = (Double_t)xinf;
	      fXsup = (Double_t)xsup;
	      fYinf = (Double_t)GetYmin(QuantityCode);
	      fYsup = (Double_t)GetYmax(QuantityCode); 
	      gPad->RangeAxis(fXinf, fYinf, fXsup, fYsup);

#define DIRN
#ifndef DIRN	      
	      //................. display of the run numbers

	      Int_t     nb_pts  = g_graph0->GetN();
	      Double_t* coord_x = g_graph0->GetX();
	      //Double_t* coord_y = g_graph0->GetY();
	      
	      char* f_in = new char[fgMaxCar];                            fCnew++;

	      Double_t interv_displayed = (coord_x[nb_pts-1] - coord_x[0])/(Double_t)nb_displayed;
	      Double_t last_drawn_coordx = coord_x[0] - 1.5*interv_displayed;

	      for(Int_t i_run=0; i_run<nb_pts; i_run++)
		{
		  if ( (coord_x[i_run] - last_drawn_coordx) > interv_displayed )
		    {
		      sprintf( f_in, "R%d",  fT1DRunNumber[i_run]);
		      TText *text_run_num = new TText(coord_x[i_run], fYsup, f_in);        fCnewRoot++;
		      text_run_num->SetTextAngle((Double_t)45.);
		      text_run_num->SetTextSize((Double_t)0.035);
		      text_run_num->Draw("SAME");
		      last_drawn_coordx = coord_x[i_run];
		      // delete text_SMtow_num;             fCdeleteRoot++;
		    }
		}
	      delete [] f_in;                                               fCdelete++;
#endif // DIRN

	    }
	  
	  if(xMemoPlotSame != 0)
	    {
	      if(opt_visu == fOptVisLine && opt_scale == fOptScaleLiny){gPad->SetLogy(liny); g_graph0->Draw();}
	      if(opt_visu == fOptVisLine && opt_scale == fOptScaleLogy){gPad->SetLogy(logy); g_graph0->Draw();}
	      if(opt_visu == fOptVisPolm && opt_scale == fOptScaleLiny){gPad->SetLogy(liny); g_graph0->Draw("P");}
	      if(opt_visu == fOptVisPolm && opt_scale == fOptScaleLogy){gPad->SetLogy(logy); g_graph0->Draw("P");}
	    }
	}
      gPad->Update();
    }
  else    // else du if(main_subpad !=0)
    {
      cout << "*TCnaViewEB::HistimePlot(...)> WARNING ===> Canvas already removed in option SAME." << endl
	   << "                               Click again on the same menu entry"
	   <<" to restart with a new canvas."
	   << fTTBELL << endl;

      ReInitCanvas(QuantityCode);
      xMemoPlotSame = 0;
    }


  //  title_g1->Delete();                 fCdeleteRoot++;
  //  com_bot_left->Delete();             fCdeleteRoot++;
  //  delete NoiseCorrel;                 fCdeleteRoot++;

} // end of HistimePlot

//------------------------------------------------------------------------------------------------------

 void TCnaViewEB::TopAxisForTowerNumbers(TH1D*        h_his0,              const TString opt_plot,
					 const Int_t& xMemoPlotSame,       const Int_t&  nb_of_towers,
					 const Int_t& xFlagAutoYsupMargin, const Int_t&  HisSize)
{
// Axis on top of the plot to indicate the tower numbers

  if( opt_plot == fOnlyOnePlot || ( (opt_plot == fSeveralPlot) && (xMemoPlotSame == 0) ) )
    {   
      Double_t Maxih           = (Double_t)h_his0->GetMaximum();
      Double_t Minih           = (Double_t)h_his0->GetMinimum();

      if(xFlagAutoYsupMargin == 1)
	{
	  if( Minih >= Maxih ){Minih = (Double_t)0.; Maxih += Maxih;}  // ROOT default if ymin >= ymax
	  Double_t MaxMarginFactor = (Double_t)0.05;    // frame top line = 5% above the maximum
	  Maxih += (Maxih-Minih)*MaxMarginFactor;       // ROOT default if ymin < ymax
	}      

      Double_t tow_min     = 1;
      Double_t tow_max     = nb_of_towers;

      Int_t ndiv = 50207;

      TGaxis* tow_axis_x = 0;

      tow_axis_x = new TGaxis( 0. ,Maxih, (Double_t)HisSize, Maxih,
			       tow_min, tow_max+1, ndiv, "B-", 0.);         fCnewRoot++;

      tow_axis_x->SetTickSize(0.05);
      // tow_axis_x->SetName("tow_axis_x");
      tow_axis_x->SetTitleOffset((Float_t)(1.2));
      tow_axis_x->SetLabelOffset((Float_t)(0.0));
      TString  x_var_name  = "Tower number";
      tow_axis_x->SetTitle(x_var_name);
            
      gPad->SetGrid(1,0);
      tow_axis_x->Draw("SAME");
    } 
} // end of TopAxisForTowerNumbers
 
//............................................................................................
void TCnaViewEB::PutAllPavesViewMatrix(TCnaReadEB*  MyRootFile,   TEBNumbering* MyNumbering,   const TString SubsetCode,
				     const Int_t& SMtower_X,  const Int_t&  SMtower_Y,
				     const Int_t& TowEcha)
{
// Put all the paves of a matrix view

  ftitle_g1      = PutPaveGeneralComment();
  fcom_top_left  = PutPaveSuperModule("standard");

  if(SubsetCode == "Crystal")
    {
      fcom_top_mid   = PutPaveTowersXY(SMtower_X, SMtower_Y);
    }
  if(SubsetCode == "Sample")
    {
      fcom_top_mid   = PutPaveTower(SMtower_X);
      fcom_top_right = PutPaveCrystal(MyNumbering, SMtower_X, TowEcha);
    }

  fcom_bot_left  = PutPaveAnalysisRun(MyRootFile);
  fcom_bot_right = PutPaveTakenEvents(MyRootFile);
}

void TCnaViewEB::PutAllPavesViewTower(TCnaReadEB*  MyRootFile, const Int_t& SMtower_X)
{
// Put all the paves of a tower view

  ftitle_g1      = PutPaveGeneralComment();
  fcom_top_left  = PutPaveSuperModule("standard");

  fcom_top_mid   = PutPaveTower(SMtower_X);

  fcom_bot_left  = PutPaveAnalysisRun(MyRootFile);
  fcom_bot_right = PutPaveTakenEvents(MyRootFile);
}

void TCnaViewEB::PutAllPavesViewTowerCrysNb(TEBNumbering*  MyNumbering,
					  const Int_t& SMNumber, const Int_t& SMtower_X)
{
// Put all the paves of a crystal numbering tower view

  fcom_top_left  = PutPaveSuperModule("standard");	  

  fcom_top_mid   = PutPaveTower(SMtower_X);
  fcom_bot_mid   = PutPaveLVRB(MyNumbering, SMNumber, SMtower_X);
}

void TCnaViewEB::PutAllPavesViewSuperModule()
{
// No argument: Put all the paves of a super-module for tower numbering view alone

  fcom_top_left  = PutPaveSuperModule("standSM");
}

void TCnaViewEB::PutAllPavesViewSuperModule(TCnaReadEB*  MyRootFile)
{
// RootFile as argument: Put all the paves of a super-module for standard view (quantity+tower numbering)

  ftitle_g1      = PutPaveGeneralComment();
  fcom_top_left  = PutPaveSuperModule("standSM");

  fcom_bot_left  = PutPaveAnalysisRun(MyRootFile);
  fcom_bot_right = PutPaveTakenEvents(MyRootFile);
}

void TCnaViewEB::PutAllPavesViewHisto(TCnaReadEB*     MyRootFile,   TEBNumbering*  MyNumbering,
				      const TString QuantityCode,
				      const Int_t&  SMtower_X,    const Int_t&    TowEcha,
				      const Int_t&  sample,       const TString   opt_plot)
{
// Put all the paves of a histo view according to QuantityCode

  ftitle_g1      = PutPaveGeneralComment();

  if(opt_plot == fOnlyOnePlot)
    {
      fcom_top_left  = PutPaveSuperModule("standard");
      
      if( !( QuantityCode == "SMFoundEvtsGlobal" || QuantityCode == "SMFoundEvtsProj" ||
	     QuantityCode == "SMEvEvGlobal"      || QuantityCode == "SMEvEvProj"      ||
	     QuantityCode == "SMSigEvGlobal"     || QuantityCode == "SMSigEvProj"     ||
	     QuantityCode == "SMEvSigGlobal"     || QuantityCode == "SMEvSigProj"     || 
	     QuantityCode == "SMSigSigGlobal"    || QuantityCode == "SMSigSigProj"    || 
	     QuantityCode == "SMEvCorssGlobal"   || QuantityCode == "SMEvCorssProj"   || 
	     QuantityCode == "SMSigCorssGlobal"  || QuantityCode == "SMSigCorssProj"  ) )
	{
	  fcom_top_mid = PutPaveTower(SMtower_X);
	}
      
      if( QuantityCode == "Ev" || QuantityCode == "Sigma" || QuantityCode == "SampTime" ||
	  QuantityCode == "EvolEvEv" ||  QuantityCode == "EvolEvSig" || QuantityCode == "EvolEvCorss" )
	{
	  fcom_top_right = PutPaveCrystal(MyNumbering, SMtower_X, TowEcha);
	}
      if( QuantityCode == "Evts" )
	{
	  fcom_top_right = PutPaveCrystalSample(MyRootFile, SMtower_X, TowEcha, sample);
	}
      
      if( QuantityCode == "EvolEvEv" || QuantityCode == "EvolEvSig" || QuantityCode == "EvolEvCorss" )
	{
	  fcom_bot_left  = PutPaveAnalysisRunList(MyRootFile);
	  fcom_bot_right = PutPaveTakenEventsRunList(MyRootFile);
	}
      else
	{
	  fcom_bot_left  = PutPaveAnalysisRun(MyRootFile);
	  fcom_bot_right = PutPaveTakenEvents(MyRootFile);
	}
    }

  if( opt_plot == fSeveralPlot && GetMemoFlag(QuantityCode) == "Free" )
    {
      if( QuantityCode == "EvolEvEv" || QuantityCode == "EvolEvSig" || QuantityCode == "EvolEvCorss" )
	{
	  fcom_top_left  = PutPaveSuperModule("sevevol");    // should use a specific method (PutPaveHistory?)
	  fcom_bot_right = PutPaveTakenEventsRunList(MyRootFile);
	}
      else
	{
	  fcom_top_left = PutPaveSuperModule("several");    // should use a specific method (PutPaveHistory?)
	}
      fcom_top_left_memo = fcom_top_left;
    } 

  if( opt_plot == fSeveralPlot && GetMemoFlag(QuantityCode) == "Busy" )
    {
      fcom_top_left = fcom_top_left_memo;
    } 
}

TPaveText* TCnaViewEB::PutPaveGeneralComment()
{
// General comment
  
  char* f_in = new char[fgMaxCar];                           fCnew++;
  
  Double_t pav_gen_xgauche = BoxLeftX("general_comment");
  Double_t pav_gen_xdroite = BoxRightX("general_comment");
  Double_t pav_gen_ybas    = BoxBottomY("general_comment");
  Double_t pav_gen_yhaut   = BoxTopY("general_comment");
  TPaveText  *title_g1 =
    new TPaveText(pav_gen_xgauche, pav_gen_ybas,
		  pav_gen_xdroite, pav_gen_yhaut);        fCnewRoot++;

  TString tit_gen = fParameters->PeriodOfRun(fFapRunNumber);

  sprintf( f_in, tit_gen);
  title_g1->AddText(f_in);
  
  delete [] f_in;                                           fCdelete++;
  
  return title_g1;
}

TPaveText* TCnaViewEB::PutPaveSuperModule(const TString chopt)
{
// Super-module pav. Called only once.
  
  char* f_in = new char[fgMaxCar];                           fCnew++;

  //.................................. DEFAULT OPTION: "standard"   
  Double_t pav_top_left_xgauche = BoxLeftX("top_left_box");
  Double_t pav_top_left_xdroite = BoxRightX("top_left_box");
  Double_t pav_top_left_ybas    = BoxBottomY("top_left_box");
  Double_t pav_top_left_yhaut   = BoxTopY("top_left_box");
  
  if(chopt == "standard" || chopt == "standSM" )
    {  
      pav_top_left_xgauche = BoxLeftX("top_left_box");
      pav_top_left_xdroite = BoxRightX("top_left_box");
      pav_top_left_ybas    = BoxBottomY("top_left_box");
      pav_top_left_yhaut   = BoxTopY("top_left_box");
    }

  if(chopt == "several")
    {    
      pav_top_left_xgauche = BoxLeftX("several_plots_box");
      pav_top_left_xdroite = BoxRightX("several_plots_box");
      pav_top_left_ybas    = BoxBottomY("several_plots_box");
      pav_top_left_yhaut   = BoxTopY("several_plots_box");
    }
  if(chopt == "sevevol")
    {    
      pav_top_left_xgauche = BoxLeftX("several_evol_box");
      pav_top_left_xdroite = BoxRightX("several_evol_box");
      pav_top_left_ybas    = BoxBottomY("several_evol_box");
      pav_top_left_yhaut   = BoxTopY("several_evol_box");
    }

  TPaveText *com_top_left =
    new TPaveText(pav_top_left_xgauche, pav_top_left_ybas,
		  pav_top_left_xdroite, pav_top_left_yhaut);  fCnewRoot++;

  if( chopt == "standard" )
    {
    sprintf(f_in, " SuperModule: %d ", fFapSuMoNumber);
    com_top_left->AddText(f_in);
    sprintf(f_in, " ( %s ) ", fFapSuMoBarrel.Data());
    com_top_left->AddText(f_in);
    }

  if( chopt == "standSM" )
    {
    sprintf(f_in, " SuperModule: %d   ( %s ) ", fFapSuMoNumber,  fFapSuMoBarrel.Data());
    com_top_left->AddText(f_in);
    }

  delete [] f_in;                                           fCdelete++;
  
  return com_top_left;
}

TPaveText* TCnaViewEB::PutPaveTower(const Int_t& SMtower_X)
{
// Tower comment

  char* f_in = new char[fgMaxCar];                           fCnew++;
  //...................... Pave tower/crystal(channel)/sample (top_right_box)
  Double_t pav_top_mid_xgauche = BoxLeftX("top_mid_box");
  Double_t pav_top_mid_xdroite = BoxRightX("top_mid_box");
  Double_t pav_top_mid_ybas    = BoxBottomY("top_mid_box");
  Double_t pav_top_mid_yhaut   = BoxTopY("top_mid_box");
  TPaveText *com_top_mid =
    new TPaveText(pav_top_mid_xgauche, pav_top_mid_ybas,
		  pav_top_mid_xdroite, pav_top_mid_yhaut);  fCnewRoot++;		  
  sprintf(f_in, " Tower: %d ", SMtower_X);
  com_top_mid->AddText(f_in);
  
  delete [] f_in;                                           fCdelete++;

  return com_top_mid;
}

TPaveText* TCnaViewEB::PutPaveTowersXY(const Int_t& SMtower_X, const Int_t& SMtower_Y)
{
// Towers X and Y for (TowEcha,TowEcha) cov or cor matrix

  char* f_in = new char[fgMaxCar];                           fCnew++;
  //...................... Pave tower/TowEcha(channel)/sample (top_right_box)
  Double_t pav_top_mid_xgauche = BoxLeftX("top_mid_box");
  Double_t pav_top_mid_xdroite = BoxRightX("top_mid_box");
  Double_t pav_top_mid_ybas    = BoxBottomY("top_mid_box");
  Double_t pav_top_mid_yhaut   = BoxTopY("top_mid_box");
  TPaveText *com_top_mid =
    new TPaveText(pav_top_mid_xgauche, pav_top_mid_ybas,
		  pav_top_mid_xdroite, pav_top_mid_yhaut);  fCnewRoot++;		  
  sprintf(f_in, " Tower X: %d ", SMtower_X);
  com_top_mid->AddText(f_in);
  sprintf(f_in, " Tower Y: %d ", SMtower_Y);
  com_top_mid->AddText(f_in);  

  delete [] f_in;                                           fCdelete++;

  return com_top_mid;
}

TPaveText* TCnaViewEB::PutPaveCrystal(TEBNumbering* MyNumbering, const Int_t& SMtower_X, const Int_t& TowEcha)
{
// Tower + TowEcha comment

  char* f_in = new char[fgMaxCar];                           fCnew++;
  //...................... Pave tower/TowEcha(channel)/sample (top_right_box)
  Double_t pav_top_right_xgauche = BoxLeftX("top_right_box");
  Double_t pav_top_right_xdroite = BoxRightX("top_right_box");
  Double_t pav_top_right_ybas    = BoxBottomY("top_right_box");
  Double_t pav_top_right_yhaut   = BoxTopY("top_right_box");
  TPaveText *com_top_right =
    new TPaveText(pav_top_right_xgauche, pav_top_right_ybas,
		  pav_top_right_xdroite, pav_top_right_yhaut);  fCnewRoot++;

  Int_t SM_crys = MyNumbering->GetSMCrysFromSMTowAndTowEcha(SMtower_X, TowEcha);
  sprintf(f_in, " Crystal: %d (channel %d) ", SM_crys, TowEcha);
  com_top_right->AddText(f_in);
 
  delete [] f_in;                                           fCdelete++;

  return com_top_right;	
}


TPaveText* TCnaViewEB::PutPaveCrystalSample(TCnaReadEB* MyRootFile,       const Int_t& SMtower_X,
					  const Int_t& TowEcha, const Int_t& sample)
{
// Tower + TowEcha + sample comment

  char* f_in = new char[fgMaxCar];                           fCnew++;
  //...................... Pave tower/TowEcha(channel)/sample (top_right_box)
  Double_t pav_top_right_xgauche = BoxLeftX("top_right_box");
  Double_t pav_top_right_xdroite = BoxRightX("top_right_box");
  Double_t pav_top_right_ybas    = BoxBottomY("top_right_box");
  Double_t pav_top_right_yhaut   = BoxTopY("top_right_box");
  TPaveText *com_top_right =
    new TPaveText(pav_top_right_xgauche, pav_top_right_ybas,
		  pav_top_right_xdroite, pav_top_right_yhaut);  fCnewRoot++;

  TEBNumbering* MyNumbering = new TEBNumbering();         fCnew++;
  Int_t SM_crys = MyNumbering->GetSMCrysFromSMTowAndTowEcha(SMtower_X, TowEcha);
  sprintf(f_in, " Crystal: %d (channel %d) ", SM_crys, TowEcha);
  com_top_right->AddText(f_in);
  sprintf(f_in, " Sample: %d ", sample);
  com_top_right->AddText(f_in);
  
  delete [] f_in;                                           fCdelete++;
  delete MyNumbering;                                       fCdelete++;

  return com_top_right;
}

TPaveText* TCnaViewEB::PutPaveAnalysisRun(TCnaReadEB* MyRootFile)
{
// Analysis name + run number comment

  char* f_in = new char[fgMaxCar];                           fCnew++;
  
  //...................... Pave Analysis name/run number (bottom_left_box)
  Double_t pav_bot_left_xgauche = BoxLeftX("bottom_left_box");
  Double_t pav_bot_left_xdroite = BoxRightX("bottom_left_box");
  Double_t pav_bot_left_ybas    = BoxBottomY("bottom_left_box");
  Double_t pav_bot_left_yhaut   = BoxTopY("bottom_left_box");
  TPaveText *com_bot_left =
    new TPaveText(pav_bot_left_xgauche, pav_bot_left_ybas,
		  pav_bot_left_xdroite, pav_bot_left_yhaut);  fCnewRoot++;
  sprintf(f_in, " Analysis: %s ", fFapAnaType.Data());
  com_bot_left->AddText(f_in);   
  sprintf(f_in, " Run: %d ", fFapRunNumber);  
  com_bot_left->AddText(f_in);
  
  delete [] f_in;                                           fCdelete++;
  
  return com_bot_left;
}

TPaveText* TCnaViewEB::PutPaveTakenEvents(TCnaReadEB* MyRootFile)
{
// Taken events comment

  char* f_in = new char[fgMaxCar];                           fCnew++;
  
  //...................... Pave taken event numbers/found events (bottom_right_box)

  Double_t pav_bot_right_xgauche = BoxLeftX("bottom_right_box");
  Double_t pav_bot_right_xdroite = BoxRightX("bottom_right_box");
  Double_t pav_bot_right_ybas    = BoxBottomY("bottom_right_box");
  Double_t pav_bot_right_yhaut   = BoxTopY("bottom_right_box");
  TPaveText *com_bot_right =
    new TPaveText(pav_bot_right_xgauche, pav_bot_right_ybas,
		  pav_bot_right_xdroite, pav_bot_right_yhaut);      fCnewRoot++;
  sprintf(f_in, " First taken event: %d (%s) ", fFapFirstEvt, fStartDate.Data());
  com_bot_right->AddText(f_in);      
  Int_t last_evt = fFapFirstEvt + fFapNbOfEvts - 1;
  sprintf(f_in, " Last taken event: %d (%s) ", last_evt, fStopDate.Data());
  com_bot_right->AddText(f_in);
  TMatrixD mat_of_found_evts(MyRootFile->MaxCrysInTow(),MyRootFile->MaxSampADC());

  Int_t SMtow = (Int_t)1;
  mat_of_found_evts = MyRootFile->ReadNumbersOfFoundEventsForSamples(SMtow);

#define SENB
#ifndef SENB  
  //........... search for the greatest nb of found events over the crystals (for sample 0)
  Int_t m_crys_of_max = 0;
  for(Int_t m_crys=1; m_crys<MyRootFile->MaxCrysInTow(); m_crys++)
    if( mat_of_found_evts(m_crys,0) > mat_of_found_evts(m_crys_of_max,0) ){m_crys_of_max = m_crys;} 

  Int_t nb_of_found_evts = (Int_t)mat_of_found_evts(m_crys_of_max,0); 
  sprintf(f_in, " Found in data: %d ", nb_of_found_evts);
  com_bot_right->AddText(f_in);
#endif // SENB
 
  delete [] f_in;                                           fCdelete++;
  
  return com_bot_right;
}

TPaveText* TCnaViewEB::PutPaveTakenEventsRunList(TCnaReadEB* MyRootFile)
{
// Taken events comment

  char* f_in = new char[fgMaxCar];                           fCnew++;
  
  //...................... Pave taken event numbers/found events (bottom_right_box)

  Double_t pav_bot_right_xgauche = BoxLeftX("bottom_right_box_evol");
  Double_t pav_bot_right_xdroite = BoxRightX("bottom_right_box_evol");
  Double_t pav_bot_right_ybas    = BoxBottomY("bottom_right_box_evol");
  Double_t pav_bot_right_yhaut   = BoxTopY("bottom_right_box_evol");
  TPaveText *com_bot_right =
    new TPaveText(pav_bot_right_xgauche, pav_bot_right_ybas,
		  pav_bot_right_xdroite, pav_bot_right_yhaut);      fCnewRoot++;
  sprintf(f_in, " First run: %d (%s) ", fStartEvolRun, fStartEvolDate.Data());
  com_bot_right->AddText(f_in);
  sprintf(f_in, " Last run: %d (%s) ", fStopEvolRun, fStopEvolDate.Data());
  com_bot_right->AddText(f_in);

  delete [] f_in;                                           fCdelete++;
  
  return com_bot_right;
}

TPaveText* TCnaViewEB::PutPaveAnalysisRunList(TCnaReadEB* MyRootFile)
{
// Analysis name + run number comment

  char* f_in = new char[fgMaxCar];                           fCnew++;
  
  //...................... Pave Analysis name/run number (bottom_left_box)
  Double_t pav_bot_left_xgauche = BoxLeftX("bottom_left_box");
  Double_t pav_bot_left_xdroite = BoxRightX("bottom_left_box");
  Double_t pav_bot_left_ybas    = BoxBottomY("bottom_left_box");
  Double_t pav_bot_left_yhaut   = BoxTopY("bottom_left_box");
  TPaveText *com_bot_left =
    new TPaveText(pav_bot_left_xgauche, pav_bot_left_ybas,
		  pav_bot_left_xdroite, pav_bot_left_yhaut);  fCnewRoot++;
  sprintf(f_in, " Analysis: %s", MyRootFile->GetAnalysisName().Data());
  com_bot_left->AddText(f_in);
  sprintf(f_in, " First taken event: %d", MyRootFile->GetFirstTakenEventNumber());  
  com_bot_left->AddText(f_in);
  Int_t last_taken_evt = MyRootFile->GetFirstTakenEventNumber() + MyRootFile->GetNumberOfTakenEvents() - 1;
  sprintf(f_in, " Last taken event: %d", last_taken_evt);  
  com_bot_left->AddText(f_in);
  delete [] f_in;                                           fCdelete++;
  
  return com_bot_left;
}

TPaveText* TCnaViewEB::PutPaveLVRB(TEBNumbering* MyNumbering, const Int_t& SMNumber, const Int_t& SMtower)
{
// LVRB at the top or at the bottom comment

  //....................... GRAND pave "LVRB"
  Double_t pav_bot_xgauche = BoxLeftX("bottom_left_box");
  Double_t pav_bot_xdroite = BoxRightX("bottom_right_box");
  Double_t pav_bot_ybas    = BoxBottomY("bottom_left_box");
  Double_t pav_bot_yhaut   = BoxTopY("bottom_left_box");
  TPaveText *com_bot_mid =
    new TPaveText(pav_bot_xgauche, pav_bot_ybas,
		  pav_bot_xdroite, pav_bot_yhaut);    fCnewRoot++;

  //  com_bot_mid->SetFillColor(28);

  Color_t couleur_noir       = ColorDefinition("noir");
  //Color_t couleur_bleu_fonce = ColorDefinition("bleu_fonce");
  //Color_t couleur_rouge      = ColorDefinition("rouge");

  //Color_t couleur_noir       = SetColorsForNumbers("crystal");
  Color_t couleur_rouge      = SetColorsForNumbers("lvrb_top");
  Color_t couleur_bleu_fonce = SetColorsForNumbers("lvrb_bottom");


  if(MyNumbering->GetSMHalfBarrel(SMNumber) == "barrel+")
    {
      TText *t1 = com_bot_mid->AddText("   <= top (#eta = 0)         bottom (PATCH PANEL) =>  ");
      t1->SetTextColor(couleur_noir);
    }

  if(MyNumbering->GetSMHalfBarrel(SMNumber) == "barrel-")
    {
      TText *t2 = com_bot_mid->AddText("   <= bottom (PATCH PANEL)         top (#eta = 0) =>  ");
      t2->SetTextColor(couleur_noir);
    }

  if(MyNumbering->GetTowerLvrbType(SMtower) == "top")
    {
      TText *t3 = com_bot_mid->AddText("   Tower with LVRB at the top    ");
      t3->SetTextColor(couleur_rouge);
    }
  
  if(MyNumbering->GetTowerLvrbType(SMtower) == "bottom")
    {
      TText *t4 = com_bot_mid->AddText(" Tower with LVRB at the bottom ");
      t4->SetTextColor(couleur_bleu_fonce);
    }
  return com_bot_mid;
}

TString TCnaViewEB::SetCanvasName(const TString QuantityCode,
				  const Int_t& opt_visu,   const Int_t& opt_scale, const TString opt_plot,
				  const Int_t& SMtower_X,  const Int_t& TowEcha,   const Int_t&  sample)
{
  //......... Set Canvas name
  
  TString canvas_name;
  Int_t MaxCar = fgMaxCar;
  canvas_name.Resize(MaxCar);
  canvas_name = "?";

  char* f_in = new char[fgMaxCar];               fCnew++;
  
  TString  name_opt_plot;
  MaxCar = fgMaxCar;
  name_opt_plot.Resize(MaxCar);
  name_opt_plot = "?";
  if(opt_plot == fOnlyOnePlot){name_opt_plot = "fl";}
  if(opt_plot == fSeveralPlot){name_opt_plot = "sm";}

  TString  name_quantity;
  MaxCar = fgMaxCar;
  name_quantity.Resize(MaxCar);
  name_quantity = "?";

  Int_t    num_crys      = -1;
  Int_t    num_samp      = -1;
  Int_t    name_same     = -1;

  if(QuantityCode == "SMFoundEvtsGlobal")
    {name_quantity = "SMfegb";             name_same = fCanvSameSMFoundEvtsGlobal;}
  if(QuantityCode == "SMFoundEvtsProj"  )
    {name_quantity = "SMfepr";             name_same = fCanvSameSMFoundEvtsProj; }
  if(QuantityCode == "SMEvEvGlobal"     )
    {name_quantity = "SMeegb";             name_same = fCanvSameSMEvEvGlobal;}
  if(QuantityCode == "SMEvEvProj"       )
    {name_quantity = "SMeepr";             name_same = fCanvSameSMEvEvProj;}
  if(QuantityCode == "SMEvSigGlobal"    )
    {name_quantity = "SMesgb";             name_same = fCanvSameSMEvSigGlobal;}
  if(QuantityCode == "SMEvSigProj"      )
    {name_quantity = "SMespr";             name_same = fCanvSameSMEvSigProj;}
  if(QuantityCode == "SMEvCorssGlobal"  )
    {name_quantity = "SMecgb";             name_same = fCanvSameSMEvCorssGlobal;}
  if(QuantityCode == "SMEvCorssProj"    )
    {name_quantity = "SMecpr";             name_same = fCanvSameSMEvCorssProj;}
  if(QuantityCode == "SMSigEvGlobal"    )
    {name_quantity = "SMsegb";             name_same = fCanvSameSMSigEvGlobal;}
  if(QuantityCode == "SMSigEvProj"      )
    {name_quantity = "SMsepr";             name_same = fCanvSameSMSigEvProj;}
  if(QuantityCode == "SMSigSigGlobal"   )
    {name_quantity = "SMssgb";             name_same = fCanvSameSMSigSigGlobal;}
  if(QuantityCode == "SMSigSigProj"     )
    {name_quantity = "SMsspr";             name_same = fCanvSameSMSigSigProj;}
  if(QuantityCode == "SMSigCorssGlobal" )
    {name_quantity = "SMscgb";             name_same = fCanvSameSMSigCorssGlobal;}
  if(QuantityCode == "SMSigCorssProj"   )
    {name_quantity = "SMscpr";             name_same = fCanvSameSMSigCorssProj;}
  if(QuantityCode == "Ev"               )
    {name_quantity = "ev";                 name_same = fCanvSameEv;    num_crys = TowEcha;}
  if(QuantityCode == "Sigma"            )
    {name_quantity = "var";                name_same = fCanvSameSigma; num_crys = TowEcha;}
  if(QuantityCode == "Evts"             )
    {name_quantity = "evts";               name_same = fCanvSameEvts;  num_crys = TowEcha; num_samp = sample;}
  if(QuantityCode == "SampTime"         )
    {name_quantity = "stime";              name_same = fCanvSameSampTime;    num_crys = TowEcha;}	  
  if(QuantityCode == "EvolEvEv"         )
    {name_quantity = "evolee";             name_same = fCanvSameEvolEvEv;    num_crys = TowEcha;}
  if(QuantityCode == "EvolEvSig"        )
    {name_quantity = "evoles";             name_same = fCanvSameEvolEvSig;   num_crys = TowEcha;}
  if(QuantityCode == "EvolEvCorss"      )
    {name_quantity = "evolec";             name_same = fCanvSameEvolEvCorss; num_crys = TowEcha;}

  TString name_visu;
  MaxCar = fgMaxCar;
  name_visu.Resize(MaxCar);
  name_visu = "?";
  if(opt_visu == fOptVisLine && opt_scale == fOptScaleLiny){name_visu = "l_liny";}
  if(opt_visu == fOptVisLine && opt_scale == fOptScaleLogy){name_visu = "l_logy";}
  if(opt_visu == fOptVisPolm && opt_scale == fOptScaleLiny){name_visu = "p_liny";}
  if(opt_visu == fOptVisPolm && opt_scale == fOptScaleLogy){name_visu = "p_logy";}
	  
  if(opt_plot == fSeveralPlot)
    {
      sprintf(f_in, "%s_%s_%s_%d", name_opt_plot.Data(), name_visu.Data(),
	      name_quantity.Data(), name_same);
    }
	  
  if(opt_plot == fOnlyOnePlot)
    {
      if (QuantityCode == "SMFoundEvtsGlobal" || QuantityCode == "SMFoundEvtsProj" || 
	  QuantityCode == "SMEvEvGlobal"      || QuantityCode == "SMEvEvProj"      ||
	  QuantityCode == "SMEvSigGlobal"     || QuantityCode == "SMEvSigProj"     ||
	  QuantityCode == "SMEvCorssGlobal"   || QuantityCode == "SMEvCorssProj"   ||
	  QuantityCode == "SMSigEvGlobal"     || QuantityCode == "SMSigEvProj"     || 
	  QuantityCode == "SMSigSigGlobal"    || QuantityCode == "SMSigSigProj"    ||
	  QuantityCode == "SMSigCorssGlobal"  || QuantityCode == "SMSigCorssProj"   )
	{
	  sprintf(f_in, "%s_%s_%s_%d_%s_SuMo_%d_%d_SM%d",
		  name_opt_plot.Data(), name_visu.Data(), fFapAnaType.Data(), fFapRunNumber,
		  name_quantity.Data(), fFapFirstEvt, fFapNbOfEvts, fFapSuMoNumber);
	}
	  
      if (QuantityCode == "Ev" || QuantityCode == "Sigma" || QuantityCode == "SampTime" || 
	  QuantityCode == "EvolEvEv" ||  QuantityCode == "EvolEvSig"  ||  QuantityCode == "EvolEvCorss" )
	{
	  sprintf(f_in, "%s_%s_%s_%d_%s_t%d_c%d_%d_%d_SM%d",
		  name_opt_plot.Data(), name_visu.Data(), fFapAnaType.Data(), fFapRunNumber,
		  name_quantity.Data(), SMtower_X, num_crys,
		  fFapFirstEvt, fFapNbOfEvts, fFapSuMoNumber); 
	}
   
      if (QuantityCode == "Evts")
	{
	  sprintf(f_in, "%s_%s_%s_%d_%s_t%d_c%d_s%d_%d_%d_SM%d",
		  name_opt_plot.Data(), name_visu.Data(), fFapAnaType.Data(), fFapRunNumber,
		  name_quantity.Data(),SMtower_X, num_crys, num_samp,
		  fFapFirstEvt, fFapNbOfEvts, fFapSuMoNumber);
	}
    }

  canvas_name = f_in;
  delete [] f_in;                  fCdelete++;
  return canvas_name.Data();

}  // end of CanvasName()

//-----------------------------------------------------------------------------
//
//        M I S C E L L A N E O U S     P A R A M E T E R S
//
//        C O N C E R N I N G    T H E    D I S P L A Y   
//
//-----------------------------------------------------------------------------
//===========================================================================
//
//    SetCanvasWidth, SetCanvasHeight,
//    CanvasFormatW, CanvasFormatH,
//
//===========================================================================

UInt_t TCnaViewEB::SetCanvasWidth(const TString QuantityCode)
{
//........................................ Taille/format canvas
  
  UInt_t canv_w = CanvasFormatW("petit");

  if (QuantityCode == "SampTime"          || QuantityCode == "SMFoundEvtsGlobal" ||
      QuantityCode == "SMEvEvGlobal"      || QuantityCode == "SMSigEvGlobal"     ||
      QuantityCode == "SMEvSigGlobal"     || QuantityCode == "SMSigSigGlobal"    ||
      QuantityCode == "SMEvCorssGlobal"   || QuantityCode == "SMSigCorssGlobal"  ||
      QuantityCode == "EvolEvEv"          || QuantityCode == "EvolEvSig"         ||
      QuantityCode == "EvolEvCorss" )
    {	     
      canv_w = CanvasFormatH("moyen");    // format 29.7*21 ( = 21*29.7 en paysage)
    }
  return canv_w;
}

UInt_t TCnaViewEB::SetCanvasHeight(const TString QuantityCode)
{
//........................................ Taille/format canvas
  
  UInt_t canv_h = CanvasFormatH("petit");

  if (QuantityCode == "SampTime"          || QuantityCode == "SMFoundEvtsGlobal" ||
      QuantityCode == "SMEvEvGlobal"      || QuantityCode == "SMSigEvGlobal"     ||
      QuantityCode == "SMEvSigGlobal"     || QuantityCode == "SMSigSigGlobal"    ||
      QuantityCode == "SMEvCorssGlobal"   || QuantityCode == "SMSigCorssGlobal"  ||
      QuantityCode == "EvolEvEv"          || QuantityCode == "EvolEvSig"         ||
      QuantityCode == "EvolEvCorss")
    {
      canv_h = CanvasFormatW("moyen");    // format 29.7*21 ( = 21*29.7 en paysage)
    }
  return canv_h;
}

UInt_t TCnaViewEB::CanvasFormatW(const TString chformat)
{
//Set Canvas width for format 21x29.7

  UInt_t canv_w = 375;         // default = "petit"

  if ( chformat == "microscopique" ) {canv_w = 187;}
  if ( chformat == "minuscule"     ) {canv_w = 210;}
  if ( chformat == "tres petit"    ) {canv_w = 260;}
  if ( chformat == "petit"         ) {canv_w = 375;}
  if ( chformat == "moyen"         ) {canv_w = 450;}
  if ( chformat == "grand"         ) {canv_w = 572;}
  if ( chformat == "etaphiSM"      ) {canv_w = 350;}

  return canv_w;
}
//......................................................................
UInt_t TCnaViewEB::CanvasFormatH(const TString chformat)
{
//Set Canvas height for format 21x29.7

  UInt_t canv_h = 530;         // default = "petit"

  if ( chformat == "microscopique"  ) {canv_h = 265;}
  if ( chformat == "minuscule"      ) {canv_h = 297;}
  if ( chformat == "tres petit"     ) {canv_h = 368;}
  if ( chformat == "petit"          ) {canv_h = 530;}
  if ( chformat == "moyen"          ) {canv_h = 636;}
  if ( chformat == "grand"          ) {canv_h = 810;}
  if ( chformat == "etaphiSM"       ) {canv_h = 945;}

  return canv_h;
}
//.......................................................................

TString TCnaViewEB::GetQuantityType(const TString QuantityCode)
{
// Type of the quantity as a function of the quantity code

  TString QuantityType;
  Int_t MaxCar = fgMaxCar;
  QuantityType.Resize(MaxCar);
  QuantityType = "(no quantity type info)";

  if ( QuantityCode == "SMFoundEvtsGlobal" || QuantityCode == "SMEvEvGlobal"    ||
       QuantityCode == "SMEvSigGlobal"     || QuantityCode == "SMEvCorssGlobal" ||
       QuantityCode == "SMSigEvGlobal"     || QuantityCode == "SMSigSigGlobal"  || 
       QuantityCode == "SMSigCorssGlobal" )  {QuantityType = "Global";}
  
  if ( QuantityCode == "SMFoundEvtsProj" || QuantityCode == "SMEvEvProj"    ||
       QuantityCode == "SMEvSigProj"     || QuantityCode == "SMEvCorssProj" ||
       QuantityCode == "SMSigEvProj"     || QuantityCode == "SMSigSigProj"  || 
       QuantityCode == "SMSigCorssProj" )  {QuantityType = "Proj";}
  
  if ( QuantityCode == "Ev"        || QuantityCode == "Sigma" ||
       QuantityCode == "SampTime" ) {QuantityType = "NotSMRun";}

  if ( QuantityCode == "EvolEvEv"  || QuantityCode == "EvolEvSig" ||
       QuantityCode == "EvolEvCorss" ) {QuantityType = "NotSMNoRun";}

  if ( QuantityCode == "Evts"        ) {QuantityType = "EvtsProj";}

  return QuantityType;
}

TString TCnaViewEB::GetQuantityName(const TString chqcode)
{
// Name of the quantity as a function of the quantity code

  TString chqname;
  Int_t MaxCar = fgMaxCar;
  chqname.Resize(MaxCar);
  chqname = "(no quantity name info)";

  if(chqcode == "SMFoundEvtsGlobal"){chqname = "Number of events (mean over samples)";}
  if(chqcode == "SMFoundEvtsProj"  ){chqname = "Number of events (mean over samples)";}
  if(chqcode == "SMEvEvGlobal"     ){chqname = "Mean of sample means (mean pedestal)";}
  if(chqcode == "SMEvEvProj"       ){chqname = "Mean of sample means (mean pedestal)";}
  if(chqcode == "SMEvSigGlobal"    ){chqname = "Mean of sample sigmas";}
  if(chqcode == "SMEvSigProj"      ){chqname = "Mean of sample sigmas";}
  if(chqcode == "SMEvCorssGlobal"  ){chqname = "Mean of cor(s,s)";}
  if(chqcode == "SMEvCorssProj"    ){chqname = "Mean of cor(s,s)";}
  if(chqcode == "SMSigEvGlobal"    ){chqname = "Sigma of sample means";}
  if(chqcode == "SMSigEvProj"      ){chqname = "Sigma of sample means";}
  if(chqcode == "SMSigSigGlobal"   ){chqname = "Sigma of sample sigmas";}
  if(chqcode == "SMSigSigProj"     ){chqname = "Sigma of sample sigmas";}
  if(chqcode == "SMSigCorssGlobal" ){chqname = "Sigma of cor(s,s)";}
  if(chqcode == "SMSigCorssProj"   ){chqname = "Sigma of cor(s,s)";}
  if(chqcode == "Ev"               ){chqname = "Sample means";}
  if(chqcode == "Sigma"            ){chqname = "Sample sigmas";}
  if(chqcode == "Evts"             ){chqname = "Sample ADC distribution";}
  if(chqcode == "SampTime"         ){chqname = "Pedestal as a function of event number";}
  if(chqcode == "EvolEvEv"         ){chqname = "Mean Pedestal evolution";}
  if(chqcode == "EvolEvSig"        ){chqname = "Mean sigma evolution";}
  if(chqcode == "EvolEvCorss"      ){chqname = "Mean cor(s,s) evolution";}

  return chqname;
}

Int_t TCnaViewEB::GetHistoSize(TCnaReadEB* MyRootFile, const TString chqcode)
{
// Histo size as a function of the quantity code

// VERY IMPORTANT: in some cases the number of bins must be strictly related to the parameters values
//                 (number of crystals, number of samples, etc...). See below comments "===> ONE BIN BY..."

  Int_t   HisSize  = 100;   // default value

  Int_t     nb_of_towers   = MyRootFile->MaxTowInSM();
  Int_t     nb_of_crystals = MyRootFile->MaxCrysInTow();

  if( chqcode == "Ev" ||  chqcode == "Sigma" ){
    HisSize = MyRootFile->MaxSampADC();}    // ===> ONE BIN BY SAMPLE
  if(chqcode == "Evts")
    {
      HisSize = MyRootFile->GetNumberOfBinsEventDistributions();
    }
  if(chqcode == "SampTime")
    {
      HisSize = MyRootFile->GetNumberOfBinsSampleAsFunctionOfTime();    // ===> ONE BIN BY EVENT
    }
  
  if(chqcode == "SMFoundEvtsGlobal" || chqcode == "SMFoundEvtsProj" ||
     chqcode == "SMEvEvGlobal"      || chqcode == "SMEvEvProj"      ||
     chqcode == "SMSigEvGlobal"     || chqcode == "SMSigEvProj"     ||  
     chqcode == "SMEvSigGlobal"     || chqcode == "SMEvSigProj"     || 
     chqcode == "SMSigSigGlobal"    || chqcode == "SMSigSigProj"    || 
     chqcode == "SMEvCorssGlobal"   || chqcode == "SMEvCorssProj"   || 
     chqcode == "SMSigCorssGlobal"  || chqcode == "SMSigCorssProj"    ){
    HisSize = nb_of_towers*nb_of_crystals;}     // ===> ONE BIN BY CRYSTAL
  
  // if (chqcode == "EvolEvEv" || chqcode == "EvolEvSig" || chqcode == "EvolEvCorss"){
  //  HisSize = MyRootFile->GetNumberOfBinsEvolution(); }

  return HisSize;
}

TVectorD TCnaViewEB::GetHistoValues(TCnaReadEB*    MyRootFile, const TString QuantityCode, const Int_t& HisSize,
				  const Int_t& SMtower_X,  const Int_t& TowEcha,       const Int_t& sample,
				  Int_t&       i_data_exist)
{
// Histo values in a TVectorD. i_data_exist entry value = 0. Incremented in this method.

  TVectorD read_histo(HisSize);
  Int_t     nb_of_towers   = MyRootFile->MaxTowInSM();
  Int_t     nb_of_crystals = MyRootFile->MaxCrysInTow();
  Int_t     nb_of_samples  = MyRootFile->MaxSampADC();
  
  if (QuantityCode == "Ev")
    {
      read_histo = MyRootFile->ReadExpectationValuesOfSamples(SMtower_X, TowEcha);
      if( MyRootFile->DataExist() == kTRUE ){i_data_exist++;}
    }
  
  if (QuantityCode == "Sigma")
    {
      read_histo = MyRootFile->ReadSigmasOfSamples(SMtower_X, TowEcha);
      if( MyRootFile->DataExist() == kTRUE ){i_data_exist++;}
    }
  
  if (QuantityCode == "Evts")
    {
      read_histo = MyRootFile->ReadEventDistribution(SMtower_X, TowEcha, sample);
      if( MyRootFile->DataExist() == kTRUE ){i_data_exist++;}
    }
  
  if (QuantityCode == "SampTime")
    {
      TVectorD read_histo_temp(HisSize);
      //......... Calculation of the pedestal (= mean over the samples) for each event
      for (Int_t i_samp=0; i_samp<nb_of_samples; i_samp++)
	{
	  read_histo_temp = MyRootFile->ReadSampleAsFunctionOfTime(SMtower_X, TowEcha, i_samp);
	  
	  if( MyRootFile->DataExist() == kFALSE )
	    {cout << "*TCnaViewEB::GetHistoValues> Exiting loop over the samples." << endl; break;}
	  else{i_data_exist++;}
	  
	  for (Int_t n_evt=0; n_evt<HisSize; n_evt++)
	    {
	      read_histo(n_evt) = read_histo(n_evt) + read_histo_temp(n_evt);
	    }
	}
      
      for (Int_t n_evt=0; n_evt<HisSize; n_evt++)
	{
	  read_histo(n_evt) = read_histo(n_evt)/(Double_t)nb_of_samples;
	}
    }
	  
  //...................................................... (GetHistoValues)
  TVectorD partial_histo(nb_of_crystals);
  TMatrixD partial_matrix(nb_of_crystals, nb_of_samples);

  if (QuantityCode == "SMFoundEvtsGlobal" || QuantityCode == "SMFoundEvtsProj" )
    {
      for(Int_t i_tow=0; i_tow<nb_of_towers; i_tow++)
	{
	  Int_t SMtow = MyRootFile->GetSMTowFromIndex(i_tow);
	  if(SMtow != -1)
	    {
	      partial_matrix = MyRootFile->ReadNumbersOfFoundEventsForSamples(SMtow);
	      
	      if( MyRootFile->DataExist() == kFALSE )
		{cout << "*TCnaViewEB::GetHistoValues> Exiting loop over the towers." << endl; break;}
	      else{i_data_exist++;}

	      for(Int_t i_crys=0; i_crys<nb_of_crystals; i_crys++)
		{
		  partial_histo(i_crys) = partial_matrix(i_crys,0);
		  Int_t i_chan = (SMtow-1)*nb_of_crystals + i_crys;
		  if(i_chan >= 0 && i_chan < nb_of_towers*nb_of_crystals){
		    read_histo(i_chan) = partial_histo(i_crys);}
		}
	    }
	}
    }

  if (QuantityCode == "SMEvEvGlobal" || QuantityCode == "SMEvEvProj" )
    {
      read_histo = MyRootFile->ReadExpectationValuesOfExpectationValuesOfSamples();
      if( MyRootFile->DataExist() == kTRUE ){i_data_exist++;}
    }

  //...................................................... (GetHistoValues)
  if (QuantityCode == "SMEvSigGlobal" || QuantityCode == "SMEvSigProj")
    {
      read_histo = MyRootFile->ReadExpectationValuesOfSigmasOfSamples();
      if( MyRootFile->DataExist() == kTRUE ){i_data_exist++;}
    }
  
  if (QuantityCode == "SMEvCorssGlobal" || QuantityCode == "SMEvCorssProj" )
    {
      read_histo = MyRootFile->ReadExpectationValuesOfCorrelationsBetweenSamples();
      if( MyRootFile->DataExist() == kTRUE ){i_data_exist++;}
    }

  if (QuantityCode == "SMSigEvGlobal" || QuantityCode == "SMSigEvProj" )
    {
      read_histo = MyRootFile->ReadSigmasOfExpectationValuesOfSamples();
      if( MyRootFile->DataExist() == kTRUE ){i_data_exist++;}
    }

  if (QuantityCode == "SMSigSigGlobal" || QuantityCode == "SMSigSigProj" )
    {
      read_histo = MyRootFile->ReadSigmasOfSigmasOfSamples();
      if( MyRootFile->DataExist() == kTRUE ){i_data_exist++;}
    }

  if (QuantityCode == "SMSigCorssGlobal" || QuantityCode == "SMSigCorssProj"  )
    {
      read_histo = MyRootFile->ReadSigmasOfCorrelationsBetweenSamples();
      if( MyRootFile->DataExist() == kTRUE ){i_data_exist++;}
    }

  return read_histo;
}


TString  TCnaViewEB::SetHistoXAxisTitle(const TString QuantityCode)
{
// Set histo X axis title

  TString axis_x_var_name;
  
  if(QuantityCode == "SMFoundEvtsGlobal" || QuantityCode == "SMEvEvGlobal"    ||
     QuantityCode == "SMEvSigGlobal"     || QuantityCode == "SMEvCorssGlobal" ||
     QuantityCode == "SMSigEvGlobal"     || QuantityCode == "SMSigSigGlobal"  ||
     QuantityCode == "SMSigCorssGlobal" )
    {axis_x_var_name = "Crystal (electronic channel number)";}
  
  if(QuantityCode == "SMFoundEvtsProj")
    {axis_x_var_name = "Number of events";}
  
  if(QuantityCode == "SMEvEvProj")
    {axis_x_var_name = "Exp.val. of the exp. val of the samples";}
  
  if(QuantityCode == "SMEvSigProj")
    {axis_x_var_name = "Exp. val. of the sigmas of the samples";}
  
  if(QuantityCode == "SMEvCorssProj")
    {axis_x_var_name = "Exp. val. of the cor(samp,samp)";}
  
  if(QuantityCode == "SMSigEvProj")
    {axis_x_var_name = "Sigmas of the exp. val of the samples";} 
  
  if(QuantityCode == "SMSigSigProj")
    {axis_x_var_name = "Sigmas of the sigmas of the samples";}
  
  if(QuantityCode == "SMSigCorssProj")
    {axis_x_var_name = "Sigmas of the cor(samp,samp)";}
  
  if(QuantityCode == "Ev")
    {axis_x_var_name = "Sample";}
  
  if(QuantityCode == "Sigma")
    {axis_x_var_name = "Sample";}
  
  if(QuantityCode == "Evts")
    {axis_x_var_name = "ADC";}
  
  if(QuantityCode == "SampTime")
    {axis_x_var_name = "Event number in burst";}

  if(QuantityCode == "EvolEvEv" || QuantityCode == "EvolEvSig" || QuantityCode == "EvolEvCorss")
    {axis_x_var_name = "Time";}
    
  return axis_x_var_name; 
}

TString  TCnaViewEB::SetHistoYAxisTitle(const TString QuantityCode)
{
// Set histo Y axis title

  TString axis_y_var_name;

  if(QuantityCode == "SMFoundEvtsGlobal")
    {axis_y_var_name = "Number of events (mean over samples)";}

  if(QuantityCode == "SMFoundEvtsProj")
    {axis_y_var_name = "Number of crystals";}

  if(QuantityCode == "SMEvEvGlobal")
    {axis_y_var_name = "Mean of sample means (mean pedestal)";}

  if(QuantityCode == "SMEvEvProj")
    {axis_y_var_name = "Number of crystals";}

  if(QuantityCode == "SMEvSigGlobal")
    {axis_y_var_name = "Mean of sample sigmas";}

  if(QuantityCode == "SMEvSigProj")
    {axis_y_var_name = "Number of crystals";}

  if(QuantityCode == "SMEvCorssGlobal")
    {axis_y_var_name = "Mean of cor(s,s)";}

  if(QuantityCode == "SMEvCorssProj")
    {axis_y_var_name = "Number of crystals";}

  if(QuantityCode == "SMSigEvGlobal")
    {axis_y_var_name = "Sigma of sample means";} 

  if(QuantityCode == "SMSigEvProj")
    {axis_y_var_name = "Number of crystals";} 

  if(QuantityCode == "SMSigSigGlobal")
    {axis_y_var_name = "Sigma of sample sigmas";}  
 
  if(QuantityCode == "SMSigSigProj")
    {axis_y_var_name = "Number of crystals";}
	 
  if(QuantityCode == "SMSigCorssGlobal")
    {axis_y_var_name = "Sigma of cor(s,s)";}

  if(QuantityCode == "SMSigCorssProj")
    {axis_y_var_name = "Number of crystals";}

  if(QuantityCode == "Ev")
    {axis_y_var_name = "Expectation value";}

  if(QuantityCode == "Sigma")
    {axis_y_var_name = "Sigma";}

  if(QuantityCode == "Evts")
    {axis_y_var_name = "Number of events";}

  if(QuantityCode == "SampTime")
    {axis_y_var_name = "Pedestal value";}

  if(QuantityCode == "EvolEvEv")
    {axis_y_var_name = "Mean Pedestal value";}

  if(QuantityCode == "EvolEvSig")
    {axis_y_var_name = "Mean sigma value";}

  if(QuantityCode == "EvolEvCorss")
    {axis_y_var_name = "Mean cor(s,s) value";}

  return axis_y_var_name;
}

Axis_t TCnaViewEB::SetHistoXinf(TCnaReadEB* MyRootFile,   const TString QuantityCode,  const Int_t& HisSize,
			      const Int_t& SMtower_X, const Int_t& TowEcha,        const Int_t& sample)
{
// Set histo Xinf

  Axis_t xinf_his = (Axis_t)0;

  if(QuantityCode == "SMFoundEvtsGlobal"){xinf_his = (Axis_t)0.;}
  if(QuantityCode == "SMEvEvGlobal"     ){xinf_his = (Axis_t)0.;}
  if(QuantityCode == "SMEvSigGlobal"    ){xinf_his = (Axis_t)0.;}
  if(QuantityCode == "SMEvCorssGlobal"  ){xinf_his = (Axis_t)0.;}
  if(QuantityCode == "SMSigEvGlobal"    ){xinf_his = (Axis_t)0.;} 
  if(QuantityCode == "SMSigSigGlobal"   ){xinf_his = (Axis_t)0.;}  
  if(QuantityCode == "SMSigCorssGlobal" ){xinf_his = (Axis_t)0.;}

  if(QuantityCode == "SMFoundEvtsProj"){xinf_his = (Axis_t)fSMFoundEvtsGlobalYmin;}
  if(QuantityCode == "SMEvEvProj"     ){xinf_his = (Axis_t)fSMEvEvGlobalYmin;}
  if(QuantityCode == "SMEvSigProj"    ){xinf_his = (Axis_t)fSMEvSigGlobalYmin;}
  if(QuantityCode == "SMEvCorssProj"  ){xinf_his = (Axis_t)fSMEvCorssGlobalYmin;}
  if(QuantityCode == "SMSigEvProj"    ){xinf_his = (Axis_t)fSMSigEvGlobalYmin;}
  if(QuantityCode == "SMSigSigProj"   ){xinf_his = (Axis_t)fSMSigSigGlobalYmin;}
  if(QuantityCode == "SMSigCorssProj" ){xinf_his = (Axis_t)fSMSigCorssGlobalYmin;}

  //........................ Histo already in ROOT file => no possibility to change xinf and xsup
  if(QuantityCode == "Evts")
    {xinf_his = (Axis_t)MyRootFile->ReadEventDistributionXmin(SMtower_X, TowEcha, sample);}

  if(QuantityCode == "Ev"         ){xinf_his = (Axis_t)0.;}
  if(QuantityCode == "Sigma"      ){xinf_his = (Axis_t)0.;}
  if(QuantityCode == "SampTime"   ){xinf_his = (Axis_t)0.;}

  if(QuantityCode == "EvolEvEv"   ){xinf_his = (Axis_t)0.;}
  if(QuantityCode == "EvolEvSig"  ){xinf_his = (Axis_t)0.;}
  if(QuantityCode == "EvolEvCorss"){xinf_his = (Axis_t)0.;}

  return xinf_his;
}

Axis_t TCnaViewEB::SetHistoXsup(TCnaReadEB*    MyRootFile, const TString QuantityCode, const Int_t& HisSize,
			      const Int_t& SMtower_X,  const Int_t&  TowEcha,      const Int_t& sample)
{
// Set histo Xsup

  Axis_t xsup_his = (Axis_t)0; 

  if(QuantityCode == "SMFoundEvtsGlobal"){xsup_his = (Axis_t)HisSize;}
  if(QuantityCode == "SMEvEvGlobal"     ){xsup_his = (Axis_t)HisSize;}
  if(QuantityCode == "SMEvSigGlobal"    ){xsup_his = (Axis_t)HisSize;}
  if(QuantityCode == "SMEvCorssGlobal"  ){xsup_his = (Axis_t)HisSize;}
  if(QuantityCode == "SMSigEvGlobal"    ){xsup_his = (Axis_t)HisSize;} 
  if(QuantityCode == "SMSigSigGlobal"   ){xsup_his = (Axis_t)HisSize;}  
  if(QuantityCode == "SMSigCorssGlobal" ){xsup_his = (Axis_t)HisSize;}

  if(QuantityCode == "SMFoundEvtsProj"){xsup_his = (Axis_t)fSMFoundEvtsGlobalYmax;}
  if(QuantityCode == "SMEvEvProj"     ){xsup_his = (Axis_t)fSMEvEvGlobalYmax;}
  if(QuantityCode == "SMEvSigProj"    ){xsup_his = (Axis_t)fSMEvSigGlobalYmax;}
  if(QuantityCode == "SMEvCorssProj"  ){xsup_his = (Axis_t)fSMEvCorssGlobalYmax;}
  if(QuantityCode == "SMSigEvProj"    ){xsup_his = (Axis_t)fSMSigEvGlobalYmax;}
  if(QuantityCode == "SMSigSigProj"   ){xsup_his = (Axis_t)fSMSigSigGlobalYmax;}
  if(QuantityCode == "SMSigCorssProj" ){xsup_his = (Axis_t)fSMSigCorssGlobalYmax;}

  //........................ Histo already in ROOT file => no possibility to change xinf and xsup
  if(QuantityCode == "Evts")
    {xsup_his = (Axis_t)MyRootFile->ReadEventDistributionXmax(SMtower_X, TowEcha, sample);}

  if(QuantityCode == "Ev"   ){xsup_his = (Axis_t)HisSize;}
  if(QuantityCode == "Sigma"){xsup_his = (Axis_t)HisSize;}
  
  if(QuantityCode == "SampTime")
    {xsup_his = (Axis_t)MyRootFile->GetNumberOfBinsSampleAsFunctionOfTime();}

  if(QuantityCode == "EvolEvEv"   ){xsup_his = (Axis_t)0.;}
  if(QuantityCode == "EvolEvSig"  ){xsup_his = (Axis_t)0.;}
  if(QuantityCode == "EvolEvCorss"){xsup_his = (Axis_t)0.;}


  return xsup_his;
}

Int_t TCnaViewEB::SetHistoNumberOfBins(const TString QuantityCode,  const Int_t& HisSize)
{
// Set histo number of bins

  Int_t nb_binx = HisSize;

  if(QuantityCode == "SMFoundEvtsGlobal"){nb_binx = HisSize;}
  if(QuantityCode == "SMFoundEvtsProj"  ){nb_binx = fNbBinsProj;}
  if(QuantityCode == "SMEvEvGlobal"     ){nb_binx = HisSize;}
  if(QuantityCode == "SMEvEvProj"       ){nb_binx = fNbBinsProj;}
  if(QuantityCode == "SMEvSigGlobal"    ){nb_binx = HisSize;}
  if(QuantityCode == "SMEvSigProj"      ){nb_binx = fNbBinsProj;}
  if(QuantityCode == "SMEvCorssGlobal"  ){nb_binx = HisSize;}
  if(QuantityCode == "SMEvCorssProj"    ){nb_binx = fNbBinsProj;}
  if(QuantityCode == "SMSigEvGlobal"    ){nb_binx = HisSize;}
  if(QuantityCode == "SMSigEvProj"      ){nb_binx = fNbBinsProj;} 
  if(QuantityCode == "SMSigSigGlobal"   ){nb_binx = HisSize;} 
  if(QuantityCode == "SMSigSigProj"     ){nb_binx = fNbBinsProj;}
  if(QuantityCode == "SMSigCorssGlobal" ){nb_binx = HisSize;}
  if(QuantityCode == "SMSigCorssProj"   ){nb_binx = fNbBinsProj;}
  if(QuantityCode == "Ev"               ){nb_binx = HisSize;}
  if(QuantityCode == "Sigma"            ){nb_binx = HisSize;}
  if(QuantityCode == "Evts"             ){nb_binx = HisSize;}
  if(QuantityCode == "SampTime"         ){nb_binx = HisSize;}

  return nb_binx;
}

void TCnaViewEB::FillHisto(      TH1D*   h_his0,       const TVectorD read_histo, 
			 const TString QuantityCode, const Int_t&   HisSize,
			 const Axis_t  xinf_his,     const Axis_t xsup_his,     const Int_t& nb_binx)
{
// Fill histo

  h_his0->Reset();

  for(Int_t i = 0 ; i < HisSize; i++)
    {
      Double_t his_val = (Double_t)0;
      Double_t xi      = (Double_t)0;
      
      if (QuantityCode == "Ev"              || QuantityCode == "Sigma"             ||
	  QuantityCode == "SampTime"        || QuantityCode == "SMFoundEvtsGlobal" || 
	  QuantityCode == "SMEvEvGlobal"    || QuantityCode == "SMSigEvGlobal"     ||
	  QuantityCode == "SMEvSigGlobal"   || QuantityCode == "SMSigSigGlobal"    ||
	  QuantityCode == "SMEvCorssGlobal" || QuantityCode == "SMSigCorssGlobal" )
	{
	  xi = (Double_t)i;
	  his_val = (Double_t)read_histo[i]; 
	  h_his0->Fill(xi, (Stat_t)his_val);
	}
      if (QuantityCode == "SMFoundEvtsProj" ||
	  QuantityCode == "SMEvEvProj"      || QuantityCode == "SMSigEvProj"   ||
	  QuantityCode == "SMEvSigProj"     || QuantityCode == "SMSigSigProj"  ||
	  QuantityCode == "SMEvCorssProj"   || QuantityCode == "SMSigCorssProj" )
	{
	  his_val = (Double_t)read_histo[i]; 
	  h_his0->Fill(his_val, (Stat_t)1);
	}

      if(QuantityCode == "Evts")
	{
	  xi=(Double_t)(((Double_t)i+(Double_t)0.5)/(Double_t)nb_binx
			*(xsup_his-xinf_his)+xinf_his);
	  his_val = (Double_t)read_histo[i];
	  for (Int_t ih = 0; ih < (Int_t)his_val; ih++)
	    {
	      h_his0->Fill(xi,(Stat_t)1);
	    }
	}
    }
}

//----------------------------------------------------------------------------
//
//              PutYmin(...), PutYmax(...)
//
//----------------------------------------------------------------------------

void TCnaViewEB::PutYmin(const TString QuantityCode, const Double_t& value)
{
   if( QuantityCode == "SMFoundEvtsGlobal"){fSMFoundEvtsGlobalYmin = value;}
   if( QuantityCode == "SMFoundEvtsProj"  ){fSMFoundEvtsProjYmin   = value;}  
   if( QuantityCode == "SMEvEvGlobal"     ){fSMEvEvGlobalYmin      = value;} 
   if( QuantityCode == "SMEvEvProj"       ){fSMEvEvProjYmin        = value;} 
   if( QuantityCode == "SMEvSigGlobal"    ){fSMEvSigGlobalYmin     = value;}
   if( QuantityCode == "SMEvSigProj"      ){fSMEvSigProjYmin       = value;} 
   if( QuantityCode == "SMEvCorssGlobal"  ){fSMEvCorssGlobalYmin   = value;} 
   if( QuantityCode == "SMEvCorssProj"    ){fSMEvCorssProjYmin     = value;} 
   if( QuantityCode == "SMSigEvGlobal"    ){fSMSigEvGlobalYmin     = value;} 
   if( QuantityCode == "SMSigEvProj"      ){fSMSigEvProjYmin       = value;}
   if( QuantityCode == "SMSigSigGlobal"   ){fSMSigSigGlobalYmin    = value;} 
   if( QuantityCode == "SMSigSigProj"     ){fSMSigSigProjYmin      = value;} 
   if( QuantityCode == "SMSigCorssGlobal" ){fSMSigCorssGlobalYmin  = value;}
   if( QuantityCode == "SMSigCorssProj"   ){fSMSigCorssProjYmin    = value;}
   if( QuantityCode == "SMEvCorttMatrix"  ){fSMEvCorttMatrixYmin   = value;}
   if( QuantityCode == "SMEvCovttMatrix"  ){fSMEvCovttMatrixYmin   = value;}
   if( QuantityCode == "SMCorccInTowers"  ){fSMCorccInTowersYmin   = value;}
   if( QuantityCode == "Ev"               ){fSMEvEvGlobalYmin      = value;}
   if( QuantityCode == "Sigma"            ){fSMEvSigGlobalYmin     = value;}
   if( QuantityCode == "Evts"             ){fEvtsYmin              = value;}
   if( QuantityCode == "SampTime"         ){fSMEvEvGlobalYmin      = value;}
   if( QuantityCode == "EvolEvEv"         ){fSMEvEvGlobalYmin      = value;}
   if( QuantityCode == "EvolEvSig"        ){fSMEvSigGlobalYmin     = value;}
   if( QuantityCode == "EvolEvCorss"      ){fSMEvCorssGlobalYmin   = value;}
}

void TCnaViewEB::PutYmax(const TString QuantityCode, const Double_t& value)
{
   if( QuantityCode == "SMFoundEvtsGlobal"){fSMFoundEvtsGlobalYmax = value;}
   if( QuantityCode == "SMFoundEvtsProj"  ){fSMFoundEvtsProjYmax   = value;}  
   if( QuantityCode == "SMEvEvGlobal"     ){fSMEvEvGlobalYmax      = value;} 
   if( QuantityCode == "SMEvEvProj"       ){fSMEvEvProjYmax        = value;} 
   if( QuantityCode == "SMEvSigGlobal"    ){fSMEvSigGlobalYmax     = value;}
   if( QuantityCode == "SMEvSigProj"      ){fSMEvSigProjYmax       = value;} 
   if( QuantityCode == "SMEvCorssGlobal"  ){fSMEvCorssGlobalYmax   = value;} 
   if( QuantityCode == "SMEvCorssProj"    ){fSMEvCorssProjYmax     = value;} 
   if( QuantityCode == "SMSigEvGlobal"    ){fSMSigEvGlobalYmax     = value;} 
   if( QuantityCode == "SMSigEvProj"      ){fSMSigEvProjYmax       = value;}
   if( QuantityCode == "SMSigSigGlobal"   ){fSMSigSigGlobalYmax    = value;} 
   if( QuantityCode == "SMSigSigProj"     ){fSMSigSigProjYmax      = value;} 
   if( QuantityCode == "SMSigCorssGlobal" ){fSMSigCorssGlobalYmax  = value;}
   if( QuantityCode == "SMSigCorssProj"   ){fSMSigCorssProjYmax    = value;}
   if( QuantityCode == "SMEvCorttMatrix"  ){fSMEvCorttMatrixYmax   = value;}
   if( QuantityCode == "SMEvCovttMatrix"  ){fSMEvCovttMatrixYmax   = value;}
   if( QuantityCode == "SMCorccInTowers"  ){fSMCorccInTowersYmax   = value;}
   if( QuantityCode == "Ev"               ){fSMEvEvGlobalYmax      = value;}
   if( QuantityCode == "Sigma"            ){fSMEvSigGlobalYmax     = value;}
   if( QuantityCode == "Evts"             ){fEvtsYmax              = value;}
   if( QuantityCode == "SampTime"         ){fSMEvEvGlobalYmax      = value;}
   if( QuantityCode == "EvolEvEv"         ){fSMEvEvGlobalYmax      = value;}
   if( QuantityCode == "EvolEvSig"        ){fSMEvSigGlobalYmax     = value;}
   if( QuantityCode == "EvolEvCorss"      ){fSMEvCorssGlobalYmax   = value;}
}
//===========================================================================
//
//               GetYmin(...), GetYmax(...)
//
//===========================================================================
Double_t TCnaViewEB::GetYmin(const TString QuantityCode)
{
  Double_t val_min = (Double_t)0.;

   if( QuantityCode == "SMFoundEvtsGlobal"){val_min = fSMFoundEvtsGlobalYmin;}
   if( QuantityCode == "SMFoundEvtsProj"  ){val_min = (Double_t)0.5;}  
   if( QuantityCode == "SMEvEvGlobal"     ){val_min = fSMEvEvGlobalYmin;} 
   if( QuantityCode == "SMEvEvProj"       ){val_min = (Double_t)0.5;} 
   if( QuantityCode == "SMEvSigGlobal"    ){val_min = fSMEvSigGlobalYmin;}
   if( QuantityCode == "SMEvSigProj"      ){val_min = (Double_t)0.5;} 
   if( QuantityCode == "SMEvCorssGlobal"  ){val_min = fSMEvCorssGlobalYmin;} 
   if( QuantityCode == "SMEvCorssProj"    ){val_min = (Double_t)0.5;} 
   if( QuantityCode == "SMSigEvGlobal"    ){val_min = fSMSigEvGlobalYmin;} 
   if( QuantityCode == "SMSigEvProj"      ){val_min = (Double_t)0.5;}
   if( QuantityCode == "SMSigSigGlobal"   ){val_min = fSMSigSigGlobalYmin;} 
   if( QuantityCode == "SMSigSigProj"     ){val_min = (Double_t)0.5;} 
   if( QuantityCode == "SMSigCorssGlobal" ){val_min = fSMSigCorssGlobalYmin;}
   if( QuantityCode == "SMSigCorssProj"   ){val_min = (Double_t)0.5;}
   if( QuantityCode == "SMEvCorttMatrix"  ){val_min = fSMEvCorttMatrixYmin;}
   if( QuantityCode == "SMEvCovttMatrix"  ){val_min = fSMEvCovttMatrixYmin;}
   if( QuantityCode == "SMCorccInTowers"  ){val_min = fSMCorccInTowersYmin;}
   if( QuantityCode == "Ev"               ){val_min = fSMEvEvGlobalYmin;}
   if( QuantityCode == "Sigma"            ){val_min = fSMEvSigGlobalYmin;}
   if( QuantityCode == "Evts"             ){val_min = (Double_t)0.5;}
   if( QuantityCode == "SampTime"         ){val_min = fSMEvEvGlobalYmin;}
   if( QuantityCode == "EvolEvEv"         ){val_min = fSMEvEvGlobalYmin;}
   if( QuantityCode == "EvolEvSig"        ){val_min = fSMEvSigGlobalYmin;}
   if( QuantityCode == "EvolEvCorss"      ){val_min = fSMEvCorssGlobalYmin;}

  return val_min;
}

Double_t TCnaViewEB::GetYmax(const TString QuantityCode)
{
  Double_t val_max = (Double_t)0.;

   if( QuantityCode == "SMFoundEvtsGlobal"){val_max = fSMFoundEvtsGlobalYmax;}
   if( QuantityCode == "SMFoundEvtsProj"  ){val_max = (Double_t)1000.;} 
   if( QuantityCode == "SMEvEvGlobal"     ){val_max = fSMEvEvGlobalYmax;} 
   if( QuantityCode == "SMEvEvProj"       ){val_max = (Double_t)1000.;}  
   if( QuantityCode == "SMEvSigGlobal"    ){val_max = fSMEvSigGlobalYmax;}   
   if( QuantityCode == "SMEvSigProj"      ){val_max = (Double_t)1000.;} 
   if( QuantityCode == "SMEvCorssGlobal"  ){val_max = fSMEvCorssGlobalYmax;}
   if( QuantityCode == "SMEvCorssProj"    ){val_max = (Double_t)1000.;} 
   if( QuantityCode == "SMSigEvGlobal"    ){val_max = fSMSigEvGlobalYmax;} 
   if( QuantityCode == "SMSigEvProj"      ){val_max = (Double_t)1000.;} 
   if( QuantityCode == "SMSigSigGlobal"   ){val_max = fSMSigSigGlobalYmax;}  
   if( QuantityCode == "SMSigSigProj"     ){val_max = (Double_t)1000.;} 
   if( QuantityCode == "SMSigCorssGlobal" ){val_max = fSMSigCorssGlobalYmax;} 
   if( QuantityCode == "SMSigCorssProj"   ){val_max = (Double_t)1000.;}
   if( QuantityCode == "SMEvCorttMatrix"  ){val_max = fSMEvCorttMatrixYmax;}
   if( QuantityCode == "SMEvCovttMatrix"  ){val_max = fSMEvCovttMatrixYmax;}
   if( QuantityCode == "SMCorccInTowers"  ){val_max = fSMCorccInTowersYmax;}
   if( QuantityCode == "Ev"               ){val_max = fSMEvEvGlobalYmax;}
   if( QuantityCode == "Sigma"            ){val_max = fSMEvSigGlobalYmax;}
   if( QuantityCode == "Evts"             ){val_max = (Double_t)1000.;}
   if( QuantityCode == "SampTime"         ){val_max = fSMEvEvGlobalYmax;}
   if( QuantityCode == "EvolEvEv"         ){val_max = fSMEvEvGlobalYmax;}
   if( QuantityCode == "EvolEvSig"        ){val_max = fSMEvSigGlobalYmax;}
   if( QuantityCode == "EvolEvCorss"      ){val_max = fSMSigCorssGlobalYmax;}

  return val_max;
}


void TCnaViewEB::InitQuantityYmin(const TString  QuantityCode)
{
// InitQuantity Ymin

  if(QuantityCode == "SMFoundEvtsGlobal"){fSMFoundEvtsGlobalYmin = GetYmin("SMFoundEvtsGlobal");}
  if(QuantityCode == "SMFoundEvtsProj"  ){fSMFoundEvtsProjYmin   = GetYmin("SMFoundEvtsProj");}
  if(QuantityCode == "SMEvEvGlobal"     ){fSMEvEvGlobalYmin      = GetYmin("SMEvEvGlobal");}
  if(QuantityCode == "SMEvEvProj"       ){fSMEvEvProjYmin        = GetYmin("SMEvEvProj");}
  if(QuantityCode == "SMEvSigGlobal"    ){fSMEvSigGlobalYmin     = GetYmin("SMEvSigGlobal");}
  if(QuantityCode == "SMEvSigProj"      ){fSMEvSigProjYmin       = GetYmin("SMEvSigProj");}
  if(QuantityCode == "SMEvCorssGlobal"  ){fSMEvCorssGlobalYmin   = GetYmin("SMEvCorssGlobal");}
  if(QuantityCode == "SMEvCorssProj"    ){fSMEvCorssProjYmin     = GetYmin("SMEvCorssProj");}
  if(QuantityCode == "SMSigEvGlobal"    ){fSMSigEvGlobalYmin     = GetYmin("SMSigEvGlobal");}
  if(QuantityCode == "SMSigEvProj"      ){fSMSigEvProjYmin       = GetYmin("SMSigEvProj");}
  if(QuantityCode == "SMSigSigGlobal"   ){fSMSigSigGlobalYmin    = GetYmin("SMSigSigGlobal");}
  if(QuantityCode == "SMSigSigProj"     ){fSMSigSigProjYmin      = GetYmin("SMSigSigProj");}
  if(QuantityCode == "SMSigCorssGlobal" ){fSMSigCorssGlobalYmin  = GetYmin("SMSigCorssGlobal");}
  if(QuantityCode == "SMSigCorssProj"   ){fSMSigCorssProjYmin    = GetYmin("SMSigCorssProj");}
  if(QuantityCode == "SMEvCorttMatrix"  ){fSMEvCorttMatrixYmin   = GetYmin("SMEvCorttMatrix");}
  if(QuantityCode == "SMEvCovttMatrix"  ){fSMEvCovttMatrixYmin   = GetYmin("SMEvCovttMatrix");}
  if(QuantityCode == "SMCorccInTowers"  ){fSMCorccInTowersYmin   = GetYmin("SMCorccInTowers");}
  if(QuantityCode == "Ev"               ){fEvYmin                = GetYmin("Ev");}
  if(QuantityCode == "Sigma"            ){fSigmaYmin             = GetYmin("Sigma");}
  if(QuantityCode == "Evts"             ){fEvtsYmin              = GetYmin("Evts");}
  if(QuantityCode == "SampTime"         ){fSampTimeYmin          = GetYmin("SampTime");}
  if(QuantityCode == "EvolEvEv"         ){fEvolEvEvYmin          = GetYmin("EvolEvEv");}
  if(QuantityCode == "EvolEvSig"        ){fEvolEvSigYmin         = GetYmin("EvolEvSig");}
  if(QuantityCode == "EvolEvCorss"      ){fEvolEvCorssYmin       = GetYmin("EvolEvCorss");}
}

void TCnaViewEB::InitQuantityYmax(const TString  QuantityCode)
{
// InitQuantity Ymax

  if(QuantityCode == "SMFoundEvtsGlobal"){fSMFoundEvtsGlobalYmax = GetYmax("SMFoundEvtsGlobal");}
  if(QuantityCode == "SMFoundEvtsProj"  ){fSMFoundEvtsProjYmax   = GetYmax("SMFoundEvtsProj");}
  if(QuantityCode == "SMEvEvGlobal"     ){fSMEvEvGlobalYmax      = GetYmax("SMEvEvGlobal");}
  if(QuantityCode == "SMEvEvProj"       ){fSMEvEvProjYmax        = GetYmax("SMEvEvProj");}
  if(QuantityCode == "SMEvSigGlobal"    ){fSMEvSigGlobalYmax     = GetYmax("SMEvSigGlobal");}
  if(QuantityCode == "SMEvSigProj"      ){fSMEvSigProjYmax       = GetYmax("SMEvSigProj");}
  if(QuantityCode == "SMEvCorssGlobal"  ){fSMEvCorssGlobalYmax   = GetYmax("SMEvCorssGlobal");}
  if(QuantityCode == "SMEvCorssProj"    ){fSMEvCorssProjYmax     = GetYmax("SMEvCorssProj");}
  if(QuantityCode == "SMSigEvGlobal"    ){fSMSigEvGlobalYmax     = GetYmax("SMSigEvGlobal");}
  if(QuantityCode == "SMSigEvProj"      ){fSMSigEvProjYmax       = GetYmax("SMSigEvProj");}
  if(QuantityCode == "SMSigSigGlobal"   ){fSMSigSigGlobalYmax    = GetYmax("SMSigSigGlobal");}
  if(QuantityCode == "SMSigSigProj"     ){fSMSigSigProjYmax      = GetYmax("SMSigSigProj");}
  if(QuantityCode == "SMSigCorssGlobal" ){fSMSigCorssGlobalYmax  = GetYmax("SMSigCorssGlobal");}
  if(QuantityCode == "SMSigCorssProj"   ){fSMSigCorssProjYmax    = GetYmax("SMSigCorssProj");}
  if(QuantityCode == "SMEvCorttMatrix"  ){fSMEvCorttMatrixYmax   = GetYmax("SMEvCorttMatrix");}
  if(QuantityCode == "SMEvCovttMatrix"  ){fSMEvCovttMatrixYmax   = GetYmax("SMEvCovttMatrix");}
  if(QuantityCode == "SMCorccInTowers"  ){fSMCorccInTowersYmax   = GetYmax("SMCorccInTowers");}
  if(QuantityCode == "Ev"               ){fEvYmax                = GetYmax("Ev");}
  if(QuantityCode == "Sigma"            ){fSigmaYmax             = GetYmax("Sigma");}
  if(QuantityCode == "Evts"             ){fEvtsYmax              = GetYmax("Evts");}
  if(QuantityCode == "SampTime"         ){fSampTimeYmax          = GetYmax("SampTime");}
  if(QuantityCode == "EvolEvEv"         ){fEvolEvEvYmax          = GetYmax("EvolEvEv");}
  if(QuantityCode == "EvolEvSig"        ){fEvolEvSigYmax         = GetYmax("EvolEvSig");}
  if(QuantityCode == "EvolEvCorss"      ){fEvolEvCorssYmax       = GetYmax("EvolEvCorss");}
}


TString TCnaViewEB::GetMemoFlag(const TString  QuantityCode)
{
// Get Memo Flag

  TString memo_flag;
  Int_t MaxCar = fgMaxCar;
  memo_flag.Resize(MaxCar);
  memo_flag = "(no memo_flag info)";

  Int_t memo_flag_number = -1;

  if(QuantityCode == "SMFoundEvtsGlobal"){memo_flag_number = fMemoPlotSMFoundEvtsGlobal;}
  if(QuantityCode == "SMFoundEvtsProj"  ){memo_flag_number = fMemoPlotSMFoundEvtsProj;}
  if(QuantityCode == "SMEvEvGlobal"     ){memo_flag_number = fMemoPlotSMEvEvGlobal;}
  if(QuantityCode == "SMEvEvProj"       ){memo_flag_number = fMemoPlotSMEvEvProj;}
  if(QuantityCode == "SMEvSigGlobal"    ){memo_flag_number = fMemoPlotSMEvSigGlobal;}
  if(QuantityCode == "SMEvSigProj"      ){memo_flag_number = fMemoPlotSMEvSigProj;}
  if(QuantityCode == "SMEvCorssGlobal"  ){memo_flag_number = fMemoPlotSMEvCorssGlobal;}
  if(QuantityCode == "SMEvCorssProj"    ){memo_flag_number = fMemoPlotSMEvCorssProj;}
  if(QuantityCode == "SMSigEvGlobal"    ){memo_flag_number = fMemoPlotSMSigEvGlobal;}
  if(QuantityCode == "SMSigEvProj"      ){memo_flag_number = fMemoPlotSMSigEvProj;} 
  if(QuantityCode == "SMSigSigGlobal"   ){memo_flag_number = fMemoPlotSMSigSigGlobal;} 
  if(QuantityCode == "SMSigSigProj"     ){memo_flag_number = fMemoPlotSMSigSigProj;}
  if(QuantityCode == "SMSigCorssGlobal" ){memo_flag_number = fMemoPlotSMSigCorssGlobal;}
  if(QuantityCode == "SMSigCorssProj"   ){memo_flag_number = fMemoPlotSMSigCorssProj;}
  if(QuantityCode == "Ev"               ){memo_flag_number = fMemoPlotEv;}
  if(QuantityCode == "Sigma"            ){memo_flag_number = fMemoPlotSigma;}
  if(QuantityCode == "Evts"             ){memo_flag_number = fMemoPlotEvts;}
  if(QuantityCode == "SampTime"         ){memo_flag_number = fMemoPlotSampTime;}
  if(QuantityCode == "EvolEvEv"         ){memo_flag_number = fMemoPlotEvolEvEv;}
  if(QuantityCode == "EvolEvSig"        ){memo_flag_number = fMemoPlotEvolEvSig;}
  if(QuantityCode == "EvolEvCorss"      ){memo_flag_number = fMemoPlotEvolEvCorss;}

  if(memo_flag_number == 0){memo_flag = "Free";}
  if(memo_flag_number == 1){memo_flag = "Busy";}

  return memo_flag;
}

void TCnaViewEB::SetMemoFlagFree(const TString  QuantityCode)
{
// Set Memo Flag to FREE

  if(QuantityCode == "SMFoundEvtsGlobal"){fMemoPlotSMFoundEvtsGlobal = 0;}
  if(QuantityCode == "SMFoundEvtsProj"  ){fMemoPlotSMFoundEvtsProj   = 0;}
  if(QuantityCode == "SMEvEvGlobal"     ){fMemoPlotSMEvEvGlobal      = 0;}
  if(QuantityCode == "SMEvEvProj"       ){fMemoPlotSMEvEvProj        = 0;}
  if(QuantityCode == "SMEvSigGlobal"    ){fMemoPlotSMEvSigGlobal     = 0;}
  if(QuantityCode == "SMEvSigProj"      ){fMemoPlotSMEvSigProj       = 0;}
  if(QuantityCode == "SMEvCorssGlobal"  ){fMemoPlotSMEvCorssGlobal   = 0;}
  if(QuantityCode == "SMEvCorssProj"    ){fMemoPlotSMEvCorssProj     = 0;}
  if(QuantityCode == "SMSigEvGlobal"    ){fMemoPlotSMSigEvGlobal     = 0;}
  if(QuantityCode == "SMSigEvProj"      ){fMemoPlotSMSigEvProj       = 0;} 
  if(QuantityCode == "SMSigSigGlobal"   ){fMemoPlotSMSigSigGlobal    = 0;} 
  if(QuantityCode == "SMSigSigProj"     ){fMemoPlotSMSigSigProj      = 0;}
  if(QuantityCode == "SMSigCorssGlobal" ){fMemoPlotSMSigCorssGlobal  = 0;}
  if(QuantityCode == "SMSigCorssProj"   ){fMemoPlotSMSigCorssProj    = 0;}
  if(QuantityCode == "Ev"               ){fMemoPlotEv                = 0;}
  if(QuantityCode == "Sigma"            ){fMemoPlotSigma             = 0;}
  if(QuantityCode == "Evts"             ){fMemoPlotEvts              = 0;}
  if(QuantityCode == "SampTime"         ){fMemoPlotSampTime          = 0;}
  if(QuantityCode == "EvolEvEv"         ){fMemoPlotEvolEvEv          = 0;}
  if(QuantityCode == "EvolEvSig"        ){fMemoPlotEvolEvSig         = 0;}
  if(QuantityCode == "EvolEvCorss"      ){fMemoPlotEvolEvCorss       = 0;}
}

void TCnaViewEB::SetMemoFlagBusy(const TString  QuantityCode)
{
// Set Memo Flag to BUSY
  if(QuantityCode == "SMFoundEvtsGlobal"){fMemoPlotSMFoundEvtsGlobal = 1;}
  if(QuantityCode == "SMFoundEvtsProj"  ){fMemoPlotSMFoundEvtsProj   = 1;}
  if(QuantityCode == "SMEvEvGlobal"     ){fMemoPlotSMEvEvGlobal      = 1;}
  if(QuantityCode == "SMEvEvProj"       ){fMemoPlotSMEvEvProj        = 1;}
  if(QuantityCode == "SMEvSigGlobal"    ){fMemoPlotSMEvSigGlobal     = 1;}
  if(QuantityCode == "SMEvSigProj"      ){fMemoPlotSMEvSigProj       = 1;}
  if(QuantityCode == "SMEvCorssGlobal"  ){fMemoPlotSMEvCorssGlobal   = 1;}
  if(QuantityCode == "SMEvCorssProj"    ){fMemoPlotSMEvCorssProj     = 1;}
  if(QuantityCode == "SMSigEvGlobal"    ){fMemoPlotSMSigEvGlobal     = 1;}
  if(QuantityCode == "SMSigEvProj"      ){fMemoPlotSMSigEvProj       = 1;} 
  if(QuantityCode == "SMSigSigGlobal"   ){fMemoPlotSMSigSigGlobal    = 1;} 
  if(QuantityCode == "SMSigSigProj"     ){fMemoPlotSMSigSigProj      = 1;}
  if(QuantityCode == "SMSigCorssGlobal" ){fMemoPlotSMSigCorssGlobal  = 1;}
  if(QuantityCode == "SMSigCorssProj"   ){fMemoPlotSMSigCorssProj    = 1;}
  if(QuantityCode == "Ev"               ){fMemoPlotEv                = 1;}
  if(QuantityCode == "Sigma"            ){fMemoPlotSigma             = 1;}
  if(QuantityCode == "Evts"             ){fMemoPlotEvts              = 1;}
  if(QuantityCode == "SampTime"         ){fMemoPlotSampTime          = 1;}
  if(QuantityCode == "EvolEvEv"         ){fMemoPlotEvolEvEv          = 1;}
  if(QuantityCode == "EvolEvSig"        ){fMemoPlotEvolEvSig         = 1;}
  if(QuantityCode == "EvolEvCorss"      ){fMemoPlotEvolEvCorss       = 1;}
}

void TCnaViewEB::CreateCanvas(const TString QuantityCode, const TString canvas_name, UInt_t canv_w,  UInt_t canv_h)
{
// Create canvas according to QuantityCode

  if(QuantityCode == "SMFoundEvtsGlobal"){
    fCanvSMFoundEvtsGlobal = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "SMFoundEvtsProj"  ){
    fCanvSMFoundEvtsProj   = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "SMEvEvGlobal"     ){
    fCanvSMEvEvGlobal      = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "SMEvEvProj"       ){
    fCanvSMEvEvProj        = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "SMEvSigGlobal"    ){
    fCanvSMEvSigGlobal     = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "SMEvSigProj"      ){
    fCanvSMEvSigProj       = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "SMEvCorssGlobal"  ){
    fCanvSMEvCorssGlobal   = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "SMEvCorssProj"    ){
    fCanvSMEvCorssProj     = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "SMSigEvGlobal"    ){
    fCanvSMSigEvGlobal     = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "SMSigEvProj"      ){
    fCanvSMSigEvProj       = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "SMSigSigGlobal"   ){
    fCanvSMSigSigGlobal    = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "SMSigSigProj"     ){
    fCanvSMSigSigProj      = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "SMSigCorssGlobal" ){
    fCanvSMSigCorssGlobal  = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "SMSigCorssProj"   ){
    fCanvSMSigCorssProj    = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "Ev"               ){
    fCanvEv                = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode =="Sigma"             ){
    fCanvSigma             = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "Evts"             ){
    fCanvEvts              = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "SampTime"         ){
    fCanvSampTime          = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "EvolEvEv"         ){
    fCanvEvolEvEv          = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "EvolEvSig"         ){
    fCanvEvolEvSig         = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
  if(QuantityCode == "EvolEvCorss"         ){
    fCanvEvolEvCorss       = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h); fCnewRoot++;}
}

void TCnaViewEB::SetParametersCanvas(const TString QuantityCode)
{
// Set parameters canvas according to QuantityCode

  Double_t x_margin_factor = 0.001;
  Double_t y_margin_factor = 0.125;

  if(QuantityCode == "SMFoundEvtsGlobal")
    {
      fImpSMFoundEvtsGlobal = (TRootCanvas*)fCanvSMFoundEvtsGlobal->GetCanvasImp();
      fPavTxtSMFoundEvtsGlobal = fcom_top_left;
      fCanvSMFoundEvtsGlobal->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadSMFoundEvtsGlobal = gPad;
      fMemoPlotSMFoundEvtsGlobal = 1; fMemoColorSMFoundEvtsGlobal = 0;
    }

  if(QuantityCode == "SMFoundEvtsProj")
    {
      fImpSMFoundEvtsProj = (TRootCanvas*)fCanvSMFoundEvtsProj->GetCanvasImp();
      fPavTxtSMFoundEvtsProj = fcom_top_left;
      fCanvSMFoundEvtsProj->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadSMFoundEvtsProj = gPad;
      fMemoPlotSMFoundEvtsProj = 1; fMemoColorSMFoundEvtsProj = 0;
    }

  if(QuantityCode == "SMEvEvGlobal")
    {
      fImpSMEvEvGlobal = (TRootCanvas*)fCanvSMEvEvGlobal->GetCanvasImp();
      fPavTxtSMEvEvGlobal = fcom_top_left;
      fCanvSMEvEvGlobal->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadSMEvEvGlobal = gPad;
      fMemoPlotSMEvEvGlobal = 1; fMemoColorSMEvEvGlobal = 0;
    }

  if(QuantityCode == "SMEvEvProj")
    {
      fImpSMEvEvProj = (TRootCanvas*)fCanvSMEvEvProj->GetCanvasImp();
      fPavTxtSMEvEvProj = fcom_top_left;
      fCanvSMEvEvProj->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadSMEvEvProj = gPad;
      fMemoPlotSMEvEvProj = 1; fMemoColorSMEvEvProj = 0;
    }

  if(QuantityCode == "SMEvSigGlobal")
    {
      fImpSMEvSigGlobal = (TRootCanvas*)fCanvSMEvSigGlobal->GetCanvasImp();
      fPavTxtSMEvSigGlobal = fcom_top_left;
      fCanvSMEvSigGlobal->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadSMEvSigGlobal = gPad; 
      fMemoPlotSMEvSigGlobal = 1; fMemoColorSMEvSigGlobal = 0;
    }

  if(QuantityCode == "SMEvSigProj")
    {
      fImpSMEvSigProj = (TRootCanvas*)fCanvSMEvSigProj->GetCanvasImp();
      fPavTxtSMEvSigProj = fcom_top_left;
      fCanvSMEvSigProj->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadSMEvSigProj = gPad;
      fMemoPlotSMEvSigProj = 1; fMemoColorSMEvSigProj = 0;
    }

  if(QuantityCode == "SMEvCorssGlobal")
    {
      fImpSMEvCorssGlobal = (TRootCanvas*)fCanvSMEvCorssGlobal->GetCanvasImp();
      fPavTxtSMEvCorssGlobal = fcom_top_left;
      fCanvSMEvCorssGlobal->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadSMEvCorssGlobal = gPad;
      fMemoPlotSMEvCorssGlobal = 1; fMemoColorSMEvCorssGlobal = 0;
    }

  if(QuantityCode == "SMEvCorssProj")
    {
      fImpSMEvCorssProj = (TRootCanvas*)fCanvSMEvCorssProj->GetCanvasImp();
      fPavTxtSMEvCorssProj = fcom_top_left;
      fCanvSMEvCorssProj->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadSMEvCorssProj = gPad;
      fMemoPlotSMEvCorssProj = 1; fMemoColorSMEvCorssProj = 0;
    }

  if(QuantityCode == "SMSigEvGlobal")
    {
      fImpSMSigEvGlobal = (TRootCanvas*)fCanvSMSigEvGlobal->GetCanvasImp();
      fPavTxtSMSigEvGlobal = fcom_top_left;
      fCanvSMSigEvGlobal->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadSMSigEvGlobal = gPad;
      fMemoPlotSMSigEvGlobal = 1; fMemoColorSMSigEvGlobal = 0;
    }

  if(QuantityCode == "SMSigEvProj")
    {
      fImpSMSigEvProj = (TRootCanvas*)fCanvSMSigEvProj->GetCanvasImp();
      fPavTxtSMSigEvProj = fcom_top_left;
      fCanvSMSigEvProj->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadSMSigEvProj = gPad;
      fMemoPlotSMSigEvProj = 1; fMemoColorSMSigEvProj = 0;
    }

  if(QuantityCode == "SMSigSigGlobal")
    {
      fImpSMSigSigGlobal = (TRootCanvas*)fCanvSMSigSigGlobal->GetCanvasImp();
      fPavTxtSMSigSigGlobal = fcom_top_left;
      fCanvSMSigSigGlobal->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadSMSigSigGlobal = gPad;
      fMemoPlotSMSigSigGlobal = 1; fMemoColorSMSigSigGlobal = 0;
    }

  if(QuantityCode == "SMSigSigProj")
    {
      fImpSMSigSigProj = (TRootCanvas*)fCanvSMSigSigProj->GetCanvasImp();
      fPavTxtSMSigSigProj = fcom_top_left;
      fCanvSMSigSigProj->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadSMSigSigProj = gPad;
      fMemoPlotSMSigSigProj = 1; fMemoColorSMSigSigProj = 0;
    }

  if(QuantityCode == "SMSigCorssGlobal")
    {
      fImpSMSigCorssGlobal = (TRootCanvas*)fCanvSMSigCorssGlobal->GetCanvasImp();
      fPavTxtSMSigCorssGlobal = fcom_top_left;
      fCanvSMSigCorssGlobal->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadSMSigCorssGlobal = gPad;
      fMemoPlotSMSigCorssGlobal = 1; fMemoColorSMSigCorssGlobal = 0;
    }

  if(QuantityCode == "SMSigCorssProj")
    {
      fImpSMSigCorssProj = (TRootCanvas*)fCanvSMSigCorssProj->GetCanvasImp();
      fPavTxtSMSigCorssProj = fcom_top_left;
      fCanvSMSigCorssProj->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadSMSigCorssProj = gPad;
      fMemoPlotSMSigCorssProj = 1; fMemoColorSMSigCorssProj = 0;
    }

  if(QuantityCode == "Ev")
    {
      fImpEv = (TRootCanvas*)fCanvEv->GetCanvasImp();
      fPavTxtEv = fcom_top_left;
      fCanvEv->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadEv = gPad;
      fMemoPlotEv = 1; fMemoColorEv = 0;
    }

  if(QuantityCode == "Sigma")
    {
      fImpSigma = (TRootCanvas*)fCanvSigma->GetCanvasImp();
      fPavTxtSigma = fcom_top_left;
      fCanvSigma->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadSigma = gPad;
      fMemoPlotSigma = 1; fMemoColorSigma = 0;
    }

  if(QuantityCode == "Evts")
    {
      fImpEvts = (TRootCanvas*)fCanvEvts->GetCanvasImp();
      fPavTxtEvts = fcom_top_left;
      fCanvEvts->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadEvts = gPad;
      fMemoPlotEvts = 1; fMemoColorEvts = 0;		  
    }

  if(QuantityCode == "SampTime")
    {
      fImpSampTime = (TRootCanvas*)fCanvSampTime->GetCanvasImp();
      fPavTxtSampTime = fcom_top_left;
      fCanvSampTime->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadSampTime = gPad;
      fMemoPlotSampTime = 1; fMemoColorSampTime = 0;
    }

  if(QuantityCode == "EvolEvEv")
    {
      fImpEvolEvEv = (TRootCanvas*)fCanvEvolEvEv->GetCanvasImp();
      fPavTxtEvolEvEv = fcom_top_left;
      fCanvEvolEvEv->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadEvolEvEv = gPad;
      fMemoPlotEvolEvEv = 1; fMemoColorEvolEvEv = 0;
    }
  if(QuantityCode == "EvolEvSig")
    {
      fImpEvolEvSig = (TRootCanvas*)fCanvEvolEvSig->GetCanvasImp();
      fPavTxtEvolEvSig = fcom_top_left;
      fCanvEvolEvSig->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadEvolEvSig = gPad;
      fMemoPlotEvolEvSig = 1; fMemoColorEvolEvSig = 0;
    }
  if(QuantityCode == "EvolEvCorss")
    {
      fImpEvolEvCorss = (TRootCanvas*)fCanvEvolEvCorss->GetCanvasImp();
      fPavTxtEvolEvCorss = fcom_top_left;
      fCanvEvolEvCorss->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadEvolEvCorss = gPad;
      fMemoPlotEvolEvCorss = 1; fMemoColorEvolEvCorss = 0;
    }
}

TVirtualPad* TCnaViewEB::ActivePad(const TString QuantityCode)
{
// Active Pad for Same plot option

  TVirtualPad* main_subpad = 0;

  if(QuantityCode == "SMFoundEvtsGlobal"){
    if( (TRootCanvas*)fCanvSMFoundEvtsGlobal->GetCanvasImp() == fImpSMFoundEvtsGlobal ){
      main_subpad = fPadSMFoundEvtsGlobal;}}

  if(QuantityCode == "SMFoundEvtsProj"  ){
    if( (TRootCanvas*)fCanvSMFoundEvtsProj->GetCanvasImp() == fImpSMFoundEvtsProj ){
      main_subpad = fPadSMFoundEvtsProj;}}

  if(QuantityCode == "SMEvEvGlobal"     ){
    if( (TRootCanvas*)fCanvSMEvEvGlobal->GetCanvasImp() == fImpSMEvEvGlobal ){
      main_subpad = fPadSMEvEvGlobal;}}

  if(QuantityCode == "SMEvEvProj"       ){
    if( (TRootCanvas*)fCanvSMEvEvProj->GetCanvasImp() == fImpSMEvEvProj ){
      main_subpad = fPadSMEvEvProj;}}
	      
  if(QuantityCode == "SMEvSigGlobal"    ){
    if((TRootCanvas*)fCanvSMEvSigGlobal->GetCanvasImp() == fImpSMEvSigGlobal){
      main_subpad = fPadSMEvSigGlobal;}}
	      
  if(QuantityCode == "SMEvSigProj"      ){
    if( (TRootCanvas*)fCanvSMEvSigProj->GetCanvasImp() == fImpSMEvSigProj ){
      main_subpad = fPadSMEvSigProj;}}

  if(QuantityCode == "SMEvCorssGlobal"  ){
    if( (TRootCanvas*)fCanvSMEvCorssGlobal->GetCanvasImp() == fImpSMEvCorssGlobal ){
      main_subpad = fPadSMEvCorssGlobal;}}

  if(QuantityCode == "SMEvCorssProj"    ){
    if( (TRootCanvas*)fCanvSMEvCorssProj->GetCanvasImp() == fImpSMEvCorssProj ){
      main_subpad = fPadSMEvCorssProj;}}

  if(QuantityCode == "SMSigEvGlobal"    ){
    if( (TRootCanvas*)fCanvSMSigEvGlobal->GetCanvasImp() == fImpSMSigEvGlobal ){
      main_subpad = fPadSMSigEvGlobal;}}

  if(QuantityCode == "SMSigEvProj"      ){
    if( (TRootCanvas*)fCanvSMSigEvProj->GetCanvasImp() == fImpSMSigEvProj ){
      main_subpad = fPadSMSigEvProj;}}

  if(QuantityCode == "SMSigSigGlobal"   ){
    if( (TRootCanvas*)fCanvSMSigSigGlobal->GetCanvasImp() == fImpSMSigSigGlobal ){
      main_subpad = fPadSMSigSigGlobal;}}

  if(QuantityCode == "SMSigSigProj"     ){
    if( (TRootCanvas*)fCanvSMSigSigProj->GetCanvasImp() == fImpSMSigSigProj ){
      main_subpad = fPadSMSigSigProj;}}

  if(QuantityCode == "SMSigCorssGlobal" ){
    if( (TRootCanvas*)fCanvSMSigCorssGlobal->GetCanvasImp() == fImpSMSigCorssGlobal ){
      main_subpad = fPadSMSigCorssGlobal;}}

  if(QuantityCode == "SMSigCorssProj"   ){
    if( (TRootCanvas*)fCanvSMSigCorssProj->GetCanvasImp() == fImpSMSigCorssProj ){
      main_subpad = fPadSMSigCorssProj;}}

  if(QuantityCode == "Ev"               ){
    if( (TRootCanvas*)fCanvEv->GetCanvasImp() == fImpEv ){
      main_subpad = fPadEv;}}

  if(QuantityCode == "Sigma"            ){
    if( (TRootCanvas*)fCanvSigma->GetCanvasImp() == fImpSigma ){
      main_subpad = fPadSigma;}}

  if(QuantityCode == "Evts"             ){
    if( (TRootCanvas*)fCanvEvts->GetCanvasImp() == fImpEvts ){
      main_subpad = fPadEvts;}}

  if(QuantityCode == "SampTime"         ){
    if( (TRootCanvas*)fCanvSampTime->GetCanvasImp() == fImpSampTime ){
      main_subpad = fPadSampTime;}}

  if(QuantityCode == "EvolEvEv"        ){
    if( (TRootCanvas*)fCanvEvolEvEv->GetCanvasImp() == fImpEvolEvEv ){
      main_subpad = fPadEvolEvEv;}}

  if(QuantityCode == "EvolEvSig"       ){
    if( (TRootCanvas*)fCanvEvolEvSig->GetCanvasImp() == fImpEvolEvSig ){
      main_subpad = fPadEvolEvSig;}}

  if(QuantityCode == "EvolEvCorss"     ){
    if( (TRootCanvas*)fCanvEvolEvCorss->GetCanvasImp() == fImpEvolEvCorss ){
      main_subpad = fPadEvolEvCorss;}}
	      	      
  return main_subpad;
}

//--------------------------------------------------------------------------
TPaveText* TCnaViewEB::ActivePavTxt(const TString QuantityCode)
{
// Active Pad for Same plot option

  TPaveText* main_pavtxt = 0;

  if(QuantityCode == "SMFoundEvtsGlobal"){
    if( (TRootCanvas*)fCanvSMFoundEvtsGlobal->GetCanvasImp() == fImpSMFoundEvtsGlobal ){
      main_pavtxt = fPavTxtSMFoundEvtsGlobal;}}

  if(QuantityCode == "SMFoundEvtsProj"  ){
    if( (TRootCanvas*)fCanvSMFoundEvtsProj->GetCanvasImp() == fImpSMFoundEvtsProj ){
      main_pavtxt = fPavTxtSMFoundEvtsProj;}}

  if(QuantityCode == "SMEvEvGlobal"     ){
    if( (TRootCanvas*)fCanvSMEvEvGlobal->GetCanvasImp() == fImpSMEvEvGlobal ){
      main_pavtxt = fPavTxtSMEvEvGlobal;}}

  if(QuantityCode == "SMEvEvProj"       ){
    if( (TRootCanvas*)fCanvSMEvEvProj->GetCanvasImp() == fImpSMEvEvProj ){
      main_pavtxt = fPavTxtSMEvEvProj;}}
	      
  if(QuantityCode == "SMEvSigGlobal"    ){
    if((TRootCanvas*)fCanvSMEvSigGlobal->GetCanvasImp() == fImpSMEvSigGlobal){
      main_pavtxt = fPavTxtSMEvSigGlobal;}}
	      
  if(QuantityCode == "SMEvSigProj"      ){
    if( (TRootCanvas*)fCanvSMEvSigProj->GetCanvasImp() == fImpSMEvSigProj ){
      main_pavtxt = fPavTxtSMEvSigProj;}}

  if(QuantityCode == "SMEvCorssGlobal"  ){
    if( (TRootCanvas*)fCanvSMEvCorssGlobal->GetCanvasImp() == fImpSMEvCorssGlobal ){
      main_pavtxt = fPavTxtSMEvCorssGlobal;}}

  if(QuantityCode == "SMEvCorssProj"    ){
    if( (TRootCanvas*)fCanvSMEvCorssProj->GetCanvasImp() == fImpSMEvCorssProj ){
      main_pavtxt = fPavTxtSMEvCorssProj;}}

  if(QuantityCode == "SMSigEvGlobal"    ){
    if( (TRootCanvas*)fCanvSMSigEvGlobal->GetCanvasImp() == fImpSMSigEvGlobal ){
      main_pavtxt = fPavTxtSMSigEvGlobal;}}

  if(QuantityCode == "SMSigEvProj"      ){
    if( (TRootCanvas*)fCanvSMSigEvProj->GetCanvasImp() == fImpSMSigEvProj ){
      main_pavtxt = fPavTxtSMSigEvProj;}}

  if(QuantityCode == "SMSigSigGlobal"   ){
    if( (TRootCanvas*)fCanvSMSigSigGlobal->GetCanvasImp() == fImpSMSigSigGlobal ){
      main_pavtxt = fPavTxtSMSigSigGlobal;}}

  if(QuantityCode == "SMSigSigProj"     ){
    if( (TRootCanvas*)fCanvSMSigSigProj->GetCanvasImp() == fImpSMSigSigProj ){
      main_pavtxt = fPavTxtSMSigSigProj;}}

  if(QuantityCode == "SMSigCorssGlobal" ){
    if( (TRootCanvas*)fCanvSMSigCorssGlobal->GetCanvasImp() == fImpSMSigCorssGlobal ){
      main_pavtxt = fPavTxtSMSigCorssGlobal;}}

  if(QuantityCode == "SMSigCorssProj"   ){
    if( (TRootCanvas*)fCanvSMSigCorssProj->GetCanvasImp() == fImpSMSigCorssProj ){
      main_pavtxt = fPavTxtSMSigCorssProj;}}

  if(QuantityCode == "Ev"               ){
    if( (TRootCanvas*)fCanvEv->GetCanvasImp() == fImpEv ){
      main_pavtxt = fPavTxtEv;}}

  if(QuantityCode == "Sigma"            ){
    if( (TRootCanvas*)fCanvSigma->GetCanvasImp() == fImpSigma ){
      main_pavtxt = fPavTxtSigma;}}

  if(QuantityCode == "Evts"             ){
    if( (TRootCanvas*)fCanvEvts->GetCanvasImp() == fImpEvts ){
      main_pavtxt = fPavTxtEvts;}}

  if(QuantityCode == "SampTime"         ){
    if( (TRootCanvas*)fCanvSampTime->GetCanvasImp() == fImpSampTime ){
      main_pavtxt = fPavTxtSampTime;}}

  if(QuantityCode == "EvolEvEv"        ){
    if( (TRootCanvas*)fCanvEvolEvEv->GetCanvasImp() == fImpEvolEvEv ){
      main_pavtxt = fPavTxtEvolEvEv;}}

  if(QuantityCode == "EvolEvSig"       ){
    if( (TRootCanvas*)fCanvEvolEvSig->GetCanvasImp() == fImpEvolEvSig ){
      main_pavtxt = fPavTxtEvolEvSig;}}

  if(QuantityCode == "EvolEvCorss"     ){
    if( (TRootCanvas*)fCanvEvolEvCorss->GetCanvasImp() == fImpEvolEvCorss ){
      main_pavtxt = fPavTxtEvolEvCorss;}}
	      	      
  return main_pavtxt;
}


void TCnaViewEB::SetViewHistoColors(TH1D* h_his0, const TString QuantityCode, const TString opt_plot)
{
// Set colors for histo view
  Int_t MaxNbOfColors = fMaxNbColLine;
	      
  if(QuantityCode == "SMFoundEvtsGlobal")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("gris15"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorSMFoundEvtsGlobal));
      fMemoColorSMFoundEvtsGlobal++;
      if(fMemoColorSMFoundEvtsGlobal>MaxNbOfColors){fMemoColorSMFoundEvtsGlobal = 0;}}
    }
  if(QuantityCode == "SMFoundEvtsProj")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("gris15"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorSMFoundEvtsProj));
      fMemoColorSMFoundEvtsProj++;
      if(fMemoColorSMFoundEvtsProj>MaxNbOfColors){fMemoColorSMFoundEvtsProj = 0;}}
    }
  if(QuantityCode == "SMEvEvGlobal")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("bleu38"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorSMEvEvGlobal));
      fMemoColorSMEvEvGlobal++;
      if(fMemoColorSMEvEvGlobal>MaxNbOfColors){fMemoColorSMEvEvGlobal = 0;}}
    }
  if(QuantityCode == "SMEvEvProj")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("bleu38"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorSMEvEvProj));
      fMemoColorSMEvEvProj++;
      if(fMemoColorSMEvEvProj>MaxNbOfColors){fMemoColorSMEvEvProj = 0;}}
    }
  if(QuantityCode == "SMEvSigGlobal")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("rouge48"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorSMEvSigGlobal));
      fMemoColorSMEvSigGlobal++;
      if(fMemoColorSMEvSigGlobal>MaxNbOfColors){fMemoColorSMEvSigGlobal = 0;}}
    }
  if(QuantityCode == "SMEvSigProj")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("rouge48"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorSMEvSigProj));
      fMemoColorSMEvSigProj++;
      if(fMemoColorSMEvSigProj>MaxNbOfColors){fMemoColorSMEvSigProj = 0;}}
    }
	      
  if(QuantityCode == "SMEvCorssGlobal")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("vert31"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorSMEvCorssGlobal));
      fMemoColorSMEvCorssGlobal++;
      if(fMemoColorSMEvCorssGlobal>MaxNbOfColors){fMemoColorSMEvCorssGlobal = 0;}}
    }
  if(QuantityCode == "SMEvCorssProj")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("vert31"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorSMEvCorssProj));
      fMemoColorSMEvCorssProj++;
      if(fMemoColorSMEvCorssProj>MaxNbOfColors){fMemoColorSMEvCorssProj = 0;}}
    }
  if(QuantityCode == "SMSigEvGlobal")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("rouge44"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorSMSigEvGlobal));
      fMemoColorSMSigEvGlobal++;
      if(fMemoColorSMSigEvGlobal>MaxNbOfColors){fMemoColorSMSigEvGlobal = 0;}}
    }
  if(QuantityCode == "SMSigEvProj")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("rouge44"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorSMSigEvProj));
      fMemoColorSMSigEvProj++;
      if(fMemoColorSMSigEvProj>MaxNbOfColors){fMemoColorSMSigEvProj = 0;}}
    }
  if(QuantityCode == "SMSigSigGlobal")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("rouge50"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorSMSigSigGlobal));
      fMemoColorSMSigSigGlobal++;
      if(fMemoColorSMSigSigGlobal>MaxNbOfColors){fMemoColorSMSigSigGlobal = 0;}}
    }
  if(QuantityCode == "SMSigSigProj")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("rouge50"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorSMSigSigProj));
      fMemoColorSMSigSigProj++;
      if(fMemoColorSMSigSigProj>MaxNbOfColors){fMemoColorSMSigSigProj = 0;}}
    }

  if(QuantityCode == "SMSigCorssGlobal")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("marron23"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorSMSigCorssGlobal));
      fMemoColorSMSigCorssGlobal++;
      if(fMemoColorSMSigCorssGlobal>MaxNbOfColors){fMemoColorSMSigCorssGlobal = 0;}}
    }
  if(QuantityCode == "SMSigCorssProj")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("marron23"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorSMSigCorssProj));
      fMemoColorSMSigCorssProj++;
      if(fMemoColorSMSigCorssProj>MaxNbOfColors){fMemoColorSMSigCorssProj = 0;}}
    }
	      
  if(QuantityCode == "Ev")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("bleu38"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorEv));
      fMemoColorEv++;
      if(fMemoColorEv>MaxNbOfColors){fMemoColorEv = 0;}}
    }
	      
  if(QuantityCode == "Sigma")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("rouge50"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorSigma));
      fMemoColorSigma++;
      if(fMemoColorSigma>MaxNbOfColors){fMemoColorSigma = 0;}}
    }
	      
  if(QuantityCode == "Evts")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("jaune"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorEvts));
      fMemoColorEvts++;
      if(fMemoColorEvts>MaxNbOfColors){fMemoColorEvts = 0;}}
    }
  if(QuantityCode == "SampTime")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(ColorDefinition("orange42"));}
      if(opt_plot == fSeveralPlot){h_his0->SetLineColor(ColorTab(fMemoColorSampTime));
      fMemoColorSampTime++;
      if(fMemoColorSampTime>MaxNbOfColors){fMemoColorSampTime = 0;}}
      gPad->SetGrid(1,0);
    }

  if(QuantityCode == "EvolEvEv")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetMarkerColor(ColorDefinition("bleu38"));}
      if(opt_plot == fSeveralPlot){h_his0->SetMarkerColor(ColorTab(fMemoColorEvolEvEv));
      fMemoColorEvolEvEv++;
      if(fMemoColorEvolEvEv>MaxNbOfColors){fMemoColorEvolEvEv = 0;}}
      gPad->SetGrid(1,1);
    }

  if(QuantityCode == "EvolEvSig")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetMarkerColor(ColorDefinition("rouge48"));}
      if(opt_plot == fSeveralPlot){h_his0->SetMarkerColor(ColorTab(fMemoColorEvolEvSig));
      fMemoColorEvolEvSig++;
      if(fMemoColorEvolEvSig>MaxNbOfColors){fMemoColorEvolEvSig = 0;}}
      gPad->SetGrid(1,1);
    }

  if(QuantityCode == "EvolEvCorss")
    {
      if(opt_plot == fOnlyOnePlot){h_his0->SetMarkerColor(ColorDefinition("vert31"));}
      if(opt_plot == fSeveralPlot){h_his0->SetMarkerColor(ColorTab(fMemoColorEvolEvCorss));
      fMemoColorEvolEvCorss++;
      if(fMemoColorEvolEvCorss>MaxNbOfColors){fMemoColorEvolEvCorss = 0;}}
      gPad->SetGrid(1,1);
    }

  if(opt_plot == fSeveralPlot){h_his0->SetLineWidth(2);}
}

void TCnaViewEB::SetViewGraphColors(TGraph* g_graph0, const TString QuantityCode, const TString opt_plot)
{
// Set colors for histo view

  Int_t MaxNbOfColors = fMaxNbColLine;

  if(QuantityCode == "EvolEvEv")
    {
      if(opt_plot == fOnlyOnePlot){g_graph0->SetMarkerColor(ColorDefinition("bleu38"));}
      if(opt_plot == fSeveralPlot){g_graph0->SetMarkerColor(ColorTab(fMemoColorEvolEvEv));
      fMemoColorEvolEvEv++;
      if(fMemoColorEvolEvEv>MaxNbOfColors){fMemoColorEvolEvEv = 0;}}
      gPad->SetGrid(1,1);
    }

  if(QuantityCode == "EvolEvSig")
    {
      if(opt_plot == fOnlyOnePlot){g_graph0->SetMarkerColor(ColorDefinition("rouge48"));}
      if(opt_plot == fSeveralPlot){g_graph0->SetMarkerColor(ColorTab(fMemoColorEvolEvSig));
      fMemoColorEvolEvSig++;
      if(fMemoColorEvolEvSig>MaxNbOfColors){fMemoColorEvolEvSig = 0;}}
      gPad->SetGrid(1,1);
    }

  if(QuantityCode == "EvolEvCorss")
    {
      if(opt_plot == fOnlyOnePlot){g_graph0->SetMarkerColor(ColorDefinition("vert31"));}
      if(opt_plot == fSeveralPlot){g_graph0->SetMarkerColor(ColorTab(fMemoColorEvolEvCorss));
      fMemoColorEvolEvCorss++;
      if(fMemoColorEvolEvCorss>MaxNbOfColors){fMemoColorEvolEvCorss = 0;}}
      gPad->SetGrid(1,1);
    }

  if(opt_plot == fSeveralPlot){g_graph0->SetLineWidth(2);}
}

Color_t TCnaViewEB::GetViewHistoColor(const TString QuantityCode)
{
  Color_t couleur = ColorDefinition("noir");        // a priori = "noir"

  if(QuantityCode == "SMFoundEvtsGlobal"){couleur = ColorTab(fMemoColorSMFoundEvtsGlobal) ;}
  if(QuantityCode == "SMFoundEvtsProj"  ){couleur = ColorTab(fMemoColorSMFoundEvtsProj)   ;}
  if(QuantityCode == "SMEvEvGlobal"     ){couleur = ColorTab(fMemoColorSMEvEvGlobal)      ;}
  if(QuantityCode == "SMEvEvProj"       ){couleur = ColorTab(fMemoColorSMEvEvProj)        ;}
  if(QuantityCode == "SMEvSigGlobal"    ){couleur = ColorTab(fMemoColorSMEvSigGlobal)     ;}
  if(QuantityCode == "SMEvSigProj"      ){couleur = ColorTab(fMemoColorSMEvSigProj)       ;}
  if(QuantityCode == "SMEvCorssGlobal"  ){couleur = ColorTab(fMemoColorSMEvCorssGlobal)   ;}
  if(QuantityCode == "SMEvCorssProj"    ){couleur = ColorTab(fMemoColorSMEvCorssProj)     ;}
  if(QuantityCode == "SMSigEvGlobal"    ){couleur = ColorTab(fMemoColorSMSigEvGlobal)     ;}
  if(QuantityCode == "SMSigEvProj"      ){couleur = ColorTab(fMemoColorSMSigEvProj)       ;} 
  if(QuantityCode == "SMSigSigGlobal"   ){couleur = ColorTab(fMemoColorSMSigSigGlobal)    ;} 
  if(QuantityCode == "SMSigSigProj"     ){couleur = ColorTab(fMemoColorSMSigSigProj)      ;}
  if(QuantityCode == "SMSigCorssGlobal" ){couleur = ColorTab(fMemoColorSMSigCorssGlobal)  ;}
  if(QuantityCode == "SMSigCorssProj"   ){couleur = ColorTab(fMemoColorSMSigCorssProj)    ;}
  if(QuantityCode == "Ev"               ){couleur = ColorTab(fMemoColorEv)                ;}
  if(QuantityCode == "Sigma"            ){couleur = ColorTab(fMemoColorSigma)             ;}
  if(QuantityCode == "Evts"             ){couleur = ColorTab(fMemoColorEvts)              ;}
  if(QuantityCode == "SampTime"         ){couleur = ColorTab(fMemoColorSampTime)          ;}
  if(QuantityCode == "EvolEvEv"         ){couleur = ColorTab(fMemoColorEvolEvEv)          ;}
  if(QuantityCode == "EvolEvSig"        ){couleur = ColorTab(fMemoColorEvolEvSig)         ;}
  if(QuantityCode == "EvolEvCorss"      ){couleur = ColorTab(fMemoColorEvolEvCorss)       ;}

  return couleur;
}

Color_t TCnaViewEB::ColorTab(const Int_t& user_color_number)
{
  //Set color from user color number

  //=========> Color definition: see ROOT User's guide p.151

  TColor* my_color = new TColor();

  Color_t couleur = ColorDefinition("noir");        // default = "noir"

  //................... Rainbow colors
  if(user_color_number == 0){couleur = (Color_t)my_color->GetColor("#EE0000");}  //    rouge
  if(user_color_number == 1){couleur = (Color_t)my_color->GetColor("#FF6611");}  //    orange
  if(user_color_number == 2){couleur = (Color_t)my_color->GetColor("#FFCC00");}  //    jaune
  if(user_color_number == 3){couleur = (Color_t)my_color->GetColor("#009900");}  //    vert
  if(user_color_number == 4){couleur = (Color_t)my_color->GetColor("#0044EE");}  //    bleu
  if(user_color_number == 5){couleur = (Color_t)my_color->GetColor("#6633BB");}  //    indigo
  if(user_color_number == 6){couleur = (Color_t)my_color->GetColor("#9900BB");}  //    violet

  if( user_color_number < 0 || user_color_number > fMaxNbColLine ){couleur = 0;}
  return couleur;
}

Color_t TCnaViewEB::SetColorsForNumbers(const TString chtype_number)
{
 //Set color of the numbers for SuperModule- or Tower-  numbering plots

  Color_t couleur = ColorDefinition("noir");        // default = "noir"

  if ( chtype_number == "crystal"     ){couleur = ColorDefinition("noir");}
  if ( chtype_number == "lvrb_top"    ){couleur = ColorDefinition("rouge");}
  if ( chtype_number == "lvrb_bottom" ){couleur = ColorDefinition("bleu_fonce");}

  return couleur;
}

Color_t TCnaViewEB::ColorDefinition(const TString chcolor)
{
  //Set color from color name

  //=========> Color definition: see ROOT User's guide p.151

  Color_t couleur = 1;        // default = "noir"

  if ( chcolor == "noir"       ) {couleur =  1;}
  if ( chcolor == "rouge"      ) {couleur =  2;}
  if ( chcolor == "vert_fonce" ) {couleur =  3;}
  if ( chcolor == "bleu_fonce" ) {couleur =  4;}
  if ( chcolor == "jaune"      ) {couleur =  5;}
  if ( chcolor == "rose"       ) {couleur =  6;}
  if ( chcolor == "bleu_clair" ) {couleur =  7;}
  if ( chcolor == "vert"       ) {couleur =  8;}
  if ( chcolor == "bleu"       ) {couleur =  9;}
  if ( chcolor == "blanc"      ) {couleur = 10;}

  if ( chcolor == "marron23"   ) {couleur = 23;}
  if ( chcolor == "marron24"   ) {couleur = 24;}
  if ( chcolor == "marron27"   ) {couleur = 27;}
  if ( chcolor == "marron28"   ) {couleur = 28;}

  if ( chcolor == "bleu33"     ) {couleur = 33;}
  if ( chcolor == "bleu36"     ) {couleur = 36;}
  if ( chcolor == "bleu38"     ) {couleur = 38;}
  if ( chcolor == "bleu39"     ) {couleur = 39;}

  if ( chcolor == "orange41"   ) {couleur = 41;}
  if ( chcolor == "orange42"   ) {couleur = 42;}

  if ( chcolor == "rouge44"    ) {couleur = 44;}
  if ( chcolor == "rouge46"    ) {couleur = 46;}
  if ( chcolor == "rouge47"    ) {couleur = 47;}
  if ( chcolor == "rouge48"    ) {couleur = 48;}
  if ( chcolor == "rouge50"    ) {couleur = 50;}

  if ( chcolor == "vert31"     ) {couleur = 31;}
  if ( chcolor == "vert32"     ) {couleur = 32;}
  if ( chcolor == "vert36"     ) {couleur = 36;}

  if ( chcolor == "violet"     ) {couleur = 49;}

  if ( chcolor == "turquoise29") {couleur = 29;}

  if ( chcolor == "gris15"     ) {couleur = 15;}

  return couleur;
}

void TCnaViewEB::SetHistoPresentation(TH1D* histo, const TString QuantityType)
{
// Set presentation (axis title offsets, title size, label size, etc... 

  SetViewHistoStyle(QuantityType);
  SetViewHistoPadMargins(QuantityType);

  SetViewHistoOffsets(histo, QuantityType);
  SetViewHistoStats(histo, QuantityType);
  
  //............................... Marker
  histo->SetMarkerStyle(1);

  if( (QuantityType == "NotSMRun") || (QuantityType == "NotSMNoRun") )
    {
      histo->SetMarkerStyle(20);
    }
}

void TCnaViewEB::SetGraphPresentation(TGraph* graph, const TString QuantityType)
{
// Set presentation (axis title offsets, title size, label size, etc... 

  SetViewHistoStyle(QuantityType);
  SetViewHistoPadMargins(QuantityType);

  SetViewGraphOffsets(graph, QuantityType);
  
  //............................... Marker
  graph->SetMarkerStyle(1);

  if( (QuantityType == "NotSMRun") || (QuantityType == "NotSMNoRun") )
    {
      graph->SetMarkerStyle(20);
    }
}

void TCnaViewEB::SetViewHistoStyle(const TString QuantityType)
{
// Set style parameters for histo view

  gStyle->SetPalette(1,0);          // Rainbow spectrum

  //...................... "standard" type (no QuantityType needed)

  //............................... Histo title size
  gStyle->SetTitleW(0.35);
  gStyle->SetTitleH(0.075);
  
  //............................... Date & Statistics box
  gStyle->SetOptDate(3);
  gStyle->SetStatW(0.55);  gStyle->SetStatH(0.2);
  gStyle->SetStatY(1);

//............................ specific types

  if(QuantityType == "colz"  || QuantityType == "lego"  ||
     QuantityType == "surf1" || QuantityType == "surf4" ||
     QuantityType == "tower" )
    {
      //............................... Histo title size
      gStyle->SetTitleW(0.8);
      gStyle->SetTitleH(0.075);
      
      //............................... Date & Statistics box
      gStyle->SetOptDate(3);
      gStyle->SetStatW(0.55);  gStyle->SetStatH(0.2);
      gStyle->SetStatY(1);
    }

  if(QuantityType == "Global" || QuantityType == "Proj" || QuantityType == "EvtsProj"  )
    {
      //............................... Histo title size
      gStyle->SetTitleW(0.5);
      gStyle->SetTitleH(0.08);

      //............................... Date & Statistics box  
      gStyle->SetOptDate(3);
      gStyle->SetOptStat(1110);
      gStyle->SetStatW(0.375);  gStyle->SetStatH(0.180);
      gStyle->SetStatY(0.9875);
    }
   
  if( (QuantityType == "NotSMRun") || (QuantityType == "NotSMNoRun"))
    {
      //............................... Histo title size
      gStyle->SetTitleW(0.5);
      gStyle->SetTitleH(0.075);

      //............................... Date & Statistics box  	  
      gStyle->SetOptDate(3);
      gStyle->SetOptStat(1110);
      gStyle->SetStatW(0.375);   gStyle->SetStatH(0.180);
      gStyle->SetStatY(0.9875);
    }
}

void TCnaViewEB::SetViewHistoStats(TH1D* histo, const TString QuantityType)
{
  // Set stats for histo view

  Bool_t b_true  = 1;
  Bool_t b_false = 0;

  histo->SetStats(b_false);
	      
  if(QuantityType == "Global"){histo->SetStats(b_false);}
  if(QuantityType == "Proj" || QuantityType == "EvtsProj" ){histo->SetStats(b_true);}
}

void TCnaViewEB::SetViewHistoOffsets(TH1D* histo, const TString QuantityType)
{
// Set offsets of labels, title axis, etc... for histo view

  if(QuantityType == "Global")
    {
      //....................... x axis
      histo->GetXaxis()->SetTitleOffset((Float_t)1.05);
      histo->GetXaxis()->SetTitleSize((Float_t)0.04);

      histo->GetXaxis()->SetLabelOffset((Float_t)0.006);
      histo->GetXaxis()->SetLabelSize((Float_t)0.04);

      histo->GetXaxis()->SetTickLength((Float_t)0.03);
      histo->GetXaxis()->SetNdivisions((Int_t)510);

      //....................... y axis
      histo->GetYaxis()->SetTitleOffset((Float_t)1.25);
      histo->GetYaxis()->SetTitleSize((Float_t)0.04);

      histo->GetYaxis()->SetLabelOffset((Float_t)0.01);
      histo->GetYaxis()->SetLabelSize((Float_t)0.04);

      histo->GetYaxis()->SetTickLength((Float_t)0.03);
      histo->GetYaxis()->SetNdivisions((Int_t)510);
    }
 
  if(QuantityType == "Proj" || QuantityType == "EvtsProj" )
    {
      //....................... x axis
      histo->GetXaxis()->SetTitleOffset((Float_t)1.05);
      histo->GetXaxis()->SetTitleSize((Float_t)0.04);

      histo->GetXaxis()->SetLabelOffset((Float_t)0.006);
      histo->GetXaxis()->SetLabelSize((Float_t)0.04);

      histo->GetXaxis()->SetTickLength((Float_t)0.03);
      histo->GetXaxis()->SetNdivisions((Int_t)510);

      //....................... y axis
      histo->GetYaxis()->SetTitleOffset((Float_t)1.5);
      histo->GetYaxis()->SetTitleSize((Float_t)0.04);

      histo->GetYaxis()->SetLabelOffset((Float_t)0.01);
      histo->GetYaxis()->SetLabelSize((Float_t)0.04);

      histo->GetYaxis()->SetTickLength((Float_t)0.03);
      histo->GetYaxis()->SetNdivisions((Int_t)510);
    }

  if( (QuantityType == "NotSMRun") || (QuantityType == "NotSMNoRun") )
    {
      //....................... x axis
      histo->GetXaxis()->SetTitleOffset((Float_t)1.25);
      histo->GetXaxis()->SetTitleSize((Float_t)0.04);

      histo->GetXaxis()->SetLabelOffset((Float_t)0.005);
      histo->GetXaxis()->SetLabelSize((Float_t)0.04);

      histo->GetXaxis()->SetTickLength((Float_t)0.03);
      histo->GetXaxis()->SetNdivisions((Int_t)510);

      //....................... y axis
      histo->GetYaxis()->SetTitleOffset((Float_t)1.75);
      histo->GetYaxis()->SetTitleSize((Float_t)0.04);

      histo->GetYaxis()->SetLabelOffset((Float_t)0.01);
      histo->GetYaxis()->SetLabelSize((Float_t)0.04);

      histo->GetYaxis()->SetTickLength((Float_t)0.03);
      histo->GetYaxis()->SetNdivisions((Int_t)510);
    }

  if(QuantityType == "colz")
    {
      //....................... x axis
      histo->GetXaxis()->SetTitleOffset((Float_t)1.25);
      histo->GetXaxis()->SetTitleSize((Float_t)0.04);

      histo->GetXaxis()->SetLabelOffset((Float_t)0.005);
      histo->GetXaxis()->SetLabelSize((Float_t)0.04);

      histo->GetXaxis()->SetTickLength((Float_t)0.03);
      histo->GetXaxis()->SetNdivisions((Int_t)510);

      //....................... y axis
      histo->GetYaxis()->SetTitleOffset((Float_t)1.3);
      histo->GetYaxis()->SetTitleSize((Float_t)0.04);

      histo->GetYaxis()->SetLabelOffset((Float_t)0.01);
      histo->GetYaxis()->SetLabelSize((Float_t)0.04);

      histo->GetYaxis()->SetTickLength((Float_t)0.03);
      histo->GetYaxis()->SetNdivisions((Int_t)510);
    }

  if(QuantityType == "lego" || QuantityType == "surf1" || QuantityType == "surf4" )
    {
      //....................... x axis
      histo->GetXaxis()->SetTitleOffset((Float_t)1.7);
      histo->GetXaxis()->SetTitleSize((Float_t)0.04);

      histo->GetXaxis()->SetLabelOffset((Float_t)0.005);
      histo->GetXaxis()->SetLabelSize((Float_t)0.04);

      histo->GetXaxis()->SetTickLength((Float_t)0.03);
      histo->GetXaxis()->SetNdivisions((Int_t)510);

      //....................... y axis
      histo->GetYaxis()->SetTitleOffset((Float_t)1.85);
      histo->GetYaxis()->SetTitleSize((Float_t)0.04);

      histo->GetYaxis()->SetLabelOffset((Float_t)0.01);
      histo->GetYaxis()->SetLabelSize((Float_t)0.04);

      histo->GetYaxis()->SetTickLength((Float_t)0.03);
      histo->GetYaxis()->SetNdivisions((Int_t)510);
    }

  if(QuantityType == "tower")
    {
      //.......... x axis (remove labels and ticks)
      histo->GetXaxis()->SetTitleOffset((Float_t)9999.);
      histo->GetXaxis()->SetTitleSize((Float_t)0.05); 
 
      histo->GetXaxis()->SetLabelOffset((Float_t)9999.);
      histo->GetXaxis()->SetLabelSize((Float_t)0.);

      histo->GetXaxis()->SetNdivisions((Int_t)1);
      histo->GetXaxis()->SetTickLength((Float_t)0.);

      //.......... y axis (remove labels and ticks)
      histo->GetYaxis()->SetTitleOffset((Float_t)9999.);
      histo->GetYaxis()->SetTitleSize((Float_t)0.05);
     
      histo->GetYaxis()->SetLabelOffset((Float_t)9999.);
      histo->GetYaxis()->SetLabelSize((Float_t)0.);

      histo->GetYaxis()->SetNdivisions((Int_t)1);
      histo->GetYaxis()->SetTickLength((Float_t)0.);
    }

  if( QuantityType == "SM2D" || QuantityType == "SM2DTN")
    {
      //.......... x axis (remove labels and ticks)
      histo->GetXaxis()->SetTitleOffset((Float_t)9999.);
      histo->GetXaxis()->SetTitleSize((Float_t)0.075); 
 
      histo->GetXaxis()->SetLabelOffset((Float_t)9999.);
      histo->GetXaxis()->SetLabelSize((Float_t)0.);

      histo->GetXaxis()->SetNdivisions((Int_t)1);
      histo->GetXaxis()->SetTickLength((Float_t)0.);

      //.......... y axis (remove labels and ticks)
      histo->GetYaxis()->SetTitleOffset((Float_t)9999.);
      histo->GetYaxis()->SetTitleSize((Float_t)0.075);
     
      histo->GetYaxis()->SetLabelOffset((Float_t)9999.);
      histo->GetYaxis()->SetLabelSize((Float_t)0.);

      histo->GetYaxis()->SetNdivisions((Int_t)1);
      histo->GetYaxis()->SetTickLength((Float_t)0.);
    }
}

void TCnaViewEB::SetViewHistoPadMargins(const TString QuantityType)
{
// Set active pad margins for histo view

//.......... default
  gStyle->SetPadBottomMargin(0.125);	  	  
  gStyle->SetPadTopMargin(0.125);
  gStyle->SetPadLeftMargin(0.125);
  gStyle->SetPadRightMargin(0.125); 
  
  if(QuantityType == "colz"  || QuantityType == "lego"  ||
     QuantityType == "surf1" || QuantityType == "surf4" ||
     QuantityType == "tower")  
    {
      gStyle->SetPadBottomMargin(0.125);	  	  
      gStyle->SetPadTopMargin(0.135);
      gStyle->SetPadLeftMargin(0.125);
      gStyle->SetPadRightMargin(0.125); 
    }

  if(QuantityType == "SM2D")
    {
      gStyle->SetPadBottomMargin(0.145);	  	  
      gStyle->SetPadTopMargin(0.135);
      gStyle->SetPadLeftMargin(0.115);
      gStyle->SetPadRightMargin(0.115); 
    }

  if(QuantityType == "SM2DTN")
    {
      gStyle->SetPadBottomMargin(0.145);	  	  
      gStyle->SetPadTopMargin(0.135);
      gStyle->SetPadLeftMargin(0.115);
      gStyle->SetPadRightMargin(0.115); 
    }

  if(QuantityType == "Global")
    {
      gStyle->SetPadBottomMargin((Float_t)0.125);	  	  
      gStyle->SetPadTopMargin((Float_t)0.175);
      gStyle->SetPadLeftMargin((Float_t)0.115);
      gStyle->SetPadRightMargin((Float_t)0.015);
    }

  if(QuantityType == "Proj" || QuantityType == "EvtsProj" )
    {
      gStyle->SetPadBottomMargin(0.115);
      gStyle->SetPadTopMargin(0.155);
      gStyle->SetPadLeftMargin(0.15);
      gStyle->SetPadRightMargin(0.05);
    }

  if( QuantityType == "NotSMRun" )
    {
      gStyle->SetPadBottomMargin(0.1275);	  	  
      gStyle->SetPadTopMargin(0.165);
      gStyle->SetPadLeftMargin(0.15);
      gStyle->SetPadRightMargin(0.05);
    }

  if( QuantityType == "NotSMNoRun" )
    {
      gStyle->SetPadBottomMargin(0.110);	  	  
      gStyle->SetPadTopMargin(0.185);
      gStyle->SetPadLeftMargin(0.15);
      gStyle->SetPadRightMargin(0.025);
    }
}

void TCnaViewEB::SetViewGraphOffsets(TGraph* graph, const TString QuantityType)
{
// Set offsets of labels, title axis, etc... for histo view

  if( (QuantityType == "NotSMRun") || (QuantityType == "NotSMNoRun")  )
    {
      //....................... x axis
      graph->GetXaxis()->SetTitleOffset((Float_t)1.15);
      graph->GetXaxis()->SetTitleSize((Float_t)0.04);

      graph->GetXaxis()->SetLabelOffset((Float_t)0.005);
      graph->GetXaxis()->SetLabelSize((Float_t)0.04);

      graph->GetXaxis()->SetTickLength((Float_t)0.03);
      graph->GetXaxis()->SetNdivisions((Int_t)510);

      //....................... y axis
      graph->GetYaxis()->SetTitleOffset((Float_t)1.75);
      graph->GetYaxis()->SetTitleSize((Float_t)0.04);

      graph->GetYaxis()->SetLabelOffset((Float_t)0.01);
      graph->GetYaxis()->SetLabelSize((Float_t)0.04);

      graph->GetYaxis()->SetTickLength((Float_t)0.03);
      graph->GetYaxis()->SetNdivisions((Int_t)510);
    }
}

void TCnaViewEB::ReInitCanvas(const TString  QuantityCode)
{
// ReInit canvas in option same plot
 
  if(QuantityCode == "SMFoundEvtsGlobal")
    {	      
      fImpSMFoundEvtsGlobal = 0;       fCanvSMFoundEvtsGlobal = 0;
      fPadSMFoundEvtsGlobal = 0;       fMemoPlotSMFoundEvtsGlobal = 0;
      fMemoColorSMFoundEvtsGlobal = 0; fCanvSameSMFoundEvtsGlobal++;
      fPavTxtSMFoundEvtsGlobal = 0; 
    }

  if(QuantityCode == "SMFoundEvtsProj")
    {	      
      fImpSMFoundEvtsProj = 0;        fCanvSMFoundEvtsProj = 0;
      fPadSMFoundEvtsProj = 0;        fMemoPlotSMFoundEvtsProj = 0;
      fMemoColorSMFoundEvtsProj = 0;  fCanvSameSMFoundEvtsProj++;
      fPavTxtSMFoundEvtsProj = 0;
    }

  if(QuantityCode == "SMEvEvGlobal")
    {	      
      fImpSMEvEvGlobal = 0;           fCanvSMEvEvGlobal = 0;
      fPadSMEvEvGlobal = 0;           fMemoPlotSMEvEvGlobal = 0;
      fMemoColorSMEvEvGlobal = 0;     fCanvSameSMEvEvGlobal++;
      fPavTxtSMEvEvGlobal = 0;   
    }

  if(QuantityCode == "SMEvEvProj")
    {	      
      fImpSMEvEvProj = 0;             fCanvSMEvEvProj = 0;
      fPadSMEvEvProj = 0;             fMemoPlotSMEvEvProj = 0;
      fMemoColorSMEvEvProj = 0;       fCanvSameSMEvEvProj++;
      fPavTxtSMEvEvProj = 0; 
    }

  if(QuantityCode == "SMEvSigGlobal")
    {	      
      fImpSMEvSigGlobal = 0;          fCanvSMEvSigGlobal = 0;
      fPadSMEvSigGlobal = 0;          fMemoPlotSMEvSigGlobal = 0;
      fMemoColorSMEvSigGlobal = 0;    fCanvSameSMEvSigGlobal++;
      fPavTxtSMEvSigGlobal = 0; 
    }

  if(QuantityCode == "SMEvSigProj")
    {	      
      fImpSMEvSigProj = 0;            fCanvSMEvSigProj = 0;
      fPadSMEvSigProj = 0;            fMemoPlotSMEvSigProj = 0;
      fMemoColorSMEvSigProj = 0;      fCanvSameSMEvSigProj++;
      fPavTxtSMEvSigProj = 0;
    }

  if(QuantityCode == "SMEvCorssGlobal")
    {	      
      fImpSMEvCorssGlobal = 0;        fCanvSMEvCorssGlobal = 0;
      fPadSMEvCorssGlobal = 0;        fMemoPlotSMEvCorssGlobal = 0;
      fMemoColorSMEvCorssGlobal = 0;  fCanvSameSMEvCorssGlobal++;
      fPavTxtSMEvCorssGlobal = 0;
    }

  if(QuantityCode == "SMEvCorssProj")
    {	      
      fImpSMEvCorssProj = 0;          fCanvSMEvCorssProj = 0;
      fPadSMEvCorssProj = 0;          fMemoPlotSMEvCorssProj = 0;
      fMemoColorSMEvCorssProj = 0;    fCanvSameSMEvCorssProj++;
      fPavTxtSMEvCorssProj = 0;
    }

  if(QuantityCode == "SMSigEvGlobal")
    {	      
      fImpSMSigEvGlobal = 0;          fCanvSMSigEvGlobal = 0;
      fPadSMSigEvGlobal = 0;          fMemoPlotSMSigEvGlobal = 0;
      fMemoColorSMSigEvGlobal = 0;    fCanvSameSMSigEvGlobal++;
      fPavTxtSMSigEvGlobal = 0;
    }

  if(QuantityCode == "SMSigEvProj")
    {	      
      fImpSMSigEvProj = 0;            fCanvSMSigEvProj = 0;
      fPadSMSigEvProj= 0;             fMemoPlotSMSigEvProj = 0;
      fMemoColorSMSigEvProj = 0;      fCanvSameSMSigEvProj++;
      fPavTxtSMSigEvProj= 0;
    }

  if(QuantityCode == "SMSigSigGlobal")
    {	      
      fImpSMSigSigGlobal = 0;         fCanvSMSigSigGlobal = 0;
      fPadSMSigSigGlobal = 0;         fMemoPlotSMSigSigGlobal = 0;
      fMemoColorSMSigSigGlobal = 0;   fCanvSameSMSigSigGlobal++;
      fPavTxtSMSigSigGlobal = 0;
    }

  if(QuantityCode == "SMSigSigProj")
    {	      
      fImpSMSigSigProj = 0;           fCanvSMSigSigProj = 0;
      fPadSMSigSigProj = 0;           fMemoPlotSMSigSigProj = 0;
      fMemoColorSMSigSigProj = 0;     fCanvSameSMSigSigProj++;
      fPavTxtSMSigSigProj = 0;
    }

  if(QuantityCode == "SMSigCorssGlobal")
    {	      
      fImpSMSigCorssGlobal = 0;       fCanvSMSigCorssGlobal = 0;
      fPadSMSigCorssGlobal = 0;       fMemoPlotSMSigCorssGlobal = 0;
      fMemoColorSMSigCorssGlobal = 0; fCanvSameSMSigCorssGlobal++;
      fPavTxtSMSigCorssGlobal = 0;
    }

  if(QuantityCode == "SMSigCorssProj")
    {	      
      fImpSMSigCorssProj = 0;        fCanvSMSigCorssProj = 0;
      fPadSMSigCorssProj = 0;        fMemoPlotSMSigCorssProj = 0;
      fMemoColorSMSigCorssProj = 0;  fCanvSameSMSigCorssProj++;
      fPavTxtSMSigCorssProj = 0;
    }

  if(QuantityCode == "Ev")
    {	      
      fImpEv = 0;             fCanvEv = 0;
      fPadEv = 0;             fMemoPlotEv = 0; 
      fMemoColorEv = 0;       fCanvSameEv++;
      fPavTxtEv = 0;
    }

  if(QuantityCode == "Sigma")
    {	      
      fImpSigma = 0;          fCanvSigma = 0;
      fPadSigma = 0;          fMemoPlotSigma= 0;
      fMemoColorSigma = 0;    fCanvSameSigma++;
      fPavTxtSigma = 0;
    }

  if(QuantityCode == "Evts")
    {	      
      fImpEvts = 0;          fCanvEvts = 0;
      fPadEvts = 0;          fMemoPlotEvts = 0;
      fMemoColorEvts = 0;    fCanvSameEvts++;
      fPavTxtEvts = 0;
    }

  if(QuantityCode == "SampTime")
    {	      
      fImpSampTime = 0;       fCanvSampTime = 0;
      fPadSampTime = 0;       fMemoPlotSampTime = 0;
      fMemoColorSampTime = 0; fCanvSameSampTime++;
      fPavTxtSampTime = 0;
    }

  if(QuantityCode == "EvolEvEv")
    {	      
      fImpEvolEvEv = 0;       fCanvEvolEvEv = 0;
      fPadEvolEvEv = 0;       fMemoPlotEvolEvEv = 0;
      fMemoColorEvolEvEv = 0; fCanvSameEvolEvEv++;
      fNbOfListFileEvolEvEv = 0;
    }

  if(QuantityCode == "EvolEvSig")
    {	      
      fImpEvolEvSig = 0;       fCanvEvolEvSig = 0;
      fPadEvolEvSig = 0;       fMemoPlotEvolEvSig = 0;
      fMemoColorEvolEvSig = 0; fCanvSameEvolEvSig++;
      fNbOfListFileEvolEvSig = 0;
    }

  if(QuantityCode == "EvolEvCorss")
    {	      
      fImpEvolEvCorss = 0;       fCanvEvolEvCorss = 0;
      fPadEvolEvCorss = 0;       fMemoPlotEvolEvCorss = 0;
      fMemoColorEvolEvCorss = 0; fCanvSameEvolEvCorss++;
      fNbOfListFileEvolEvCorss = 0;
    }
}

//===========================================================================
//
//          BoxLeftX, BoxRightX, BoxBottomY, BoxTopY
//
//===========================================================================

Double_t TCnaViewEB::BoxLeftX(const TString chtype)
{
//Set the x left coordinate of the box

  Double_t value = 0.;

   if ( chtype == "general_comment"      ) {value = 0.015;}
   if ( chtype == "top_left_box"         ) {value = 0.015;}
   if ( chtype == "top_mid_box"          ) {value = 0.335;}
   if ( chtype == "top_right_box"        ) {value = 0.540;}
   if ( chtype == "bottom_left_box"      ) {value = 0.015;}
   if ( chtype == "bottom_right_box"     ) {value = 0.325;}
   if ( chtype == "bottom_right_box_evol") {value = 0.615;}
   if ( chtype == "several_plots_box"    ) {value = 0.015;}
   if ( chtype == "several_evol_box"     ) {value = 0.015;}

 return value;
}
//.................................................................
Double_t TCnaViewEB::BoxRightX(const TString chtype)
{
//Set the x right coordinate of the box

  Double_t value = 1.0;

   if ( chtype == "general_comment"      ) {value = 0.680;}
   if ( chtype == "top_left_box"         ) {value = 0.334;}
   if ( chtype == "top_mid_box"          ) {value = 0.539;}
   if ( chtype == "top_right_box"        ) {value = 0.985;}
   if ( chtype == "bottom_left_box"      ) {value = 0.315;}
   if ( chtype == "bottom_right_box"     ) {value = 0.985;}
   if ( chtype == "bottom_right_box_evol") {value = 0.985;}
   if ( chtype == "several_plots_box"    ) {value = 0.985;}
   if ( chtype == "several_evol_box"     ) {value = 0.600;}

 return value;
}
//.................................................................
Double_t TCnaViewEB::BoxBottomY(const TString chtype)
{
//Set the y bottom coordinate of the box

  Double_t value = 0.8;

   if ( chtype == "general_comment"      ) {value = 0.960;}
   if ( chtype == "top_left_box"         ) {value = 0.880;}
   if ( chtype == "top_mid_box"          ) {value = 0.880;}
   if ( chtype == "top_right_box"        ) {value = 0.880;}
   if ( chtype == "bottom_left_box"      ) {value = 0.010;}
   if ( chtype == "bottom_right_box"     ) {value = 0.010;}
   if ( chtype == "bottom_right_box_evol") {value = 0.010;}
   if ( chtype == "several_plots_box"    ) {value = 0.015;}
   if ( chtype == "several_evol_box"     ) {value = 0.015;}

 return value;
}
//.................................................................
Double_t TCnaViewEB::BoxTopY(const TString chtype)
{
//Set the y top coordinate of the box

  Double_t value = 0.9;

   if ( chtype == "general_comment"      ) {value = 0.999;}
   if ( chtype == "top_left_box"         ) {value = 0.955;}
   if ( chtype == "top_mid_box"          ) {value = 0.955;}
   if ( chtype == "top_right_box"        ) {value = 0.955;}
   if ( chtype == "bottom_left_box"      ) {value = 0.120;}
   if ( chtype == "bottom_right_box"     ) {value = 0.120;}
   if ( chtype == "bottom_right_box_evol") {value = 0.120;}
   if ( chtype == "several_plots_box"    ) {value = 0.120;}
   if ( chtype == "several_evol_box"     ) {value = 0.120;}

 return value;
}

void TCnaViewEB::GetPathForResultsRootFiles()
{
  GetPathForResultsRootFiles("");
}

void TCnaViewEB::GetPathForListOfRunFiles()
{
  GetPathForListOfRunFiles("");
}


void TCnaViewEB::GetPathForResultsRootFiles(const TString argFileName)
{
  // Init fCfgResultsRootFilePath and get it from the file named argFileName
  // argFileName = complete name of the file (/afs/cern.ch/...)
  // if string is empty, file name = "cna_results_root.cfg" and file located in $HOME user's directory (default)

  Int_t MaxCar = fgMaxCar;
  fCfgResultsRootFilePath.Resize(MaxCar);
  fCfgResultsRootFilePath            = "?";

  //..... put the name of the ascii file (containing the paths for CNA treatements)
  //      in the string cFileNameForCnaPaths and in class attribute fFileForResultsRootFilePath

  if ( argFileName == "" )
    {
      string cFileNameForCnaPaths = "cna_results_root.cfg";     // config file name
      TString s_file_name = cFileNameForCnaPaths.c_str();
      Text_t *t_file_name = (Text_t *)s_file_name.Data();
      
      TString s_path_name = gSystem->Getenv("HOME");       // get user's home directory path
      
      fFileForResultsRootFilePath = s_path_name;
      fFileForResultsRootFilePath.Append('/');
      fFileForResultsRootFilePath.Append(t_file_name);
    }
  else
    {
      fFileForResultsRootFilePath = argFileName.Data();
    }

  //........ Reading of the paths in the file named fFileForResultsRootFilePath.Data()
  //

  fFcin_rr.open(fFileForResultsRootFilePath.Data());
  if(fFcin_rr.fail() == kFALSE)
    {
      fFcin_rr.clear();
      string xResultsFileP;
      fFcin_rr >> xResultsFileP;
      fCfgResultsRootFilePath = xResultsFileP.c_str();

      //fCnaCommand++;
      //cout << "   *CNA [" << fCnaCommand
      //   << "]> Automatic registration of cna paths -> " << endl
      //   << "                      Results files :     " << fCfgResultsRootFilePath.Data() << endl;
      fFcin_rr.close();
    }
  else
    {
      fFcin_rr.clear();
      fCnaError++;
      cout << fTTBELL << endl
	   << " ***************************************************************************** " << endl;
      cout << "   !CNA(TCnaViewEB) (" << fCnaError << ") *** ERROR *** " << endl << endl
	   << "    "
	   << fFileForResultsRootFilePath.Data() << ": file not found. " << endl
	   << "    "
	   << endl << endl
	   << "    "
           << " The file " << fFileForResultsRootFilePath.Data()
	   << " is a configuration file for the CNA and"
	   << " must contain one line with the following syntax:" << endl << endl
	   << "    "
	   << "   path of the results .root files ($HOME/etc...) " << endl
	   << "    "
	   << "          (without slash at the end of line)" << endl
	   << endl << endl
	   << "    "
	   << " EXAMPLE:" << endl << endl
	   << "    "
	   << "  $HOME/scratch0/cna/results_files" << endl << endl
	   << " ***************************************************************************** "
	   << fTTBELL << endl;

      fFcin_rr.close();
    }
}

void TCnaViewEB::GetPathForListOfRunFiles(const TString argFileName)
{
  // Init fCfgListOfRunsFilePath and get it from the file named argFileName
  // argFileName = complete name of the file (/afs/cern.ch/...)
  // if string is empty, file name = "cna_stability.cfg" and file located in $HOME user's directory (default)

  Int_t MaxCar = fgMaxCar;
  fCfgListOfRunsFilePath.Resize(MaxCar);
  fCfgListOfRunsFilePath          = "?";

  //..... put the name of the ascii file (containing the paths for CNA treatements)
  //      in the string cFileNameForCnaPaths and in class attribute fFileForListOfRunFilePath

  if ( argFileName == "" )
    {
      string cFileNameForCnaPaths = "cna_stability.cfg";     // config file name
      TString s_file_name = cFileNameForCnaPaths.c_str();
      Text_t *t_file_name = (Text_t *)s_file_name.Data();
      
      TString s_path_name = gSystem->Getenv("HOME");       // get user's home directory path
      
      fFileForListOfRunFilePath = s_path_name;
      fFileForListOfRunFilePath.Append('/');
      fFileForListOfRunFilePath.Append(t_file_name);
    }
  else
    {
      fFileForListOfRunFilePath = argFileName.Data();
    }

  //........ Reading of the paths in the file named fFileForListOfRunFilePath.Data()
  //

  fFcin_lor.open(fFileForListOfRunFilePath.Data());
  if(fFcin_lor.fail() == kFALSE)
    {
      fFcin_lor.clear();
      string xListOfRunsP;
      fFcin_lor >> xListOfRunsP;
      fCfgListOfRunsFilePath = xListOfRunsP.c_str();

      //fCnaCommand++;
      //cout << "   *CNA [" << fCnaCommand
      //   << "]> Automatic registration of cna paths -> " << endl
      //   << "                  List-of-runs files:     " << fCfgListOfRunsFilePath.Data() << endl;
      fFcin_lor.close();
    }
  else
    {
      fFcin_lor.clear();
      fCnaError++;
      cout << fTTBELL << endl
	   << " ***************************************************************************** " << endl;
      cout << "   !CNA(TCnaViewEB) (" << fCnaError << ") *** ERROR *** " << endl << endl
	   << "    "
	   << fFileForListOfRunFilePath.Data() << " : file not found. " << endl
	   << "    "
	   << " Please create this file in your HOME directory and then restart."
	   << endl << endl
	   << "    "
           << " The file " << fFileForListOfRunFilePath.Data()
	   << " is a configuration file for the CNA and"
	   << " must contain one line with the following syntax:" << endl << endl
	   << "    "
	   << "   path of the list-of-runs files for time evolution plots ($HOME/etc...) " << endl
	   << "    "
	   << "          (without slash at the end of line)" << endl
	   << endl << endl
	   << "    "
	   << " EXAMPLE:" << endl << endl
	   << "    "
	   << "  $HOME/scratch0/cna/list_runs_time_evol" << endl << endl
	   << " ***************************************************************************** "
	   << fTTBELL << endl;

      fFcin_lor.close();
    }
}
