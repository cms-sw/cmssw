#ifndef ZTR_TEcnaGui
#define ZTR_TEcnaGui

#include "TObject.h"
#include "TSystem.h"

#include "TROOT.h"

#include "TApplication.h"
#include "TGClient.h"
#include "TRint.h"

#include "TString.h"

#include "TGButton.h"
#include "TGWidget.h"
#include "TGToolTip.h"
#include "TGResourcePool.h"
#include "TGCanvas.h"
#include "TGWindow.h"
#include "TGMenu.h"
#include "TGFrame.h"
#include "TGTextEntry.h"
#include "TGLayout.h"
#include "TGFont.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaObject.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParEcal.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaNumbering.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaHistos.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParHistos.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRun.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaWrite.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParPaths.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParCout.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRead.h"


///-----------------------------------------------------------
///   TEcnaGui.h
///   Update: 14/02/2011
///   Author:    B.Fabbro (bernard.fabbro@cea.fr)
///              DSM/IRFU/SPP CEA-Saclay
///   Copyright: Those valid for CEA sofware
///
///   ECNA web page:
///     http://cms-fabbro.web.cern.ch/cms-fabbro/
///     cna_new/Correlated_Noise_Analysis/ECNA_cna_1.htm
///-----------------------------------------------------------
///
/// This class provides a dialog box for ECNA (Ecal Correlated Noise Analysis)
/// in the framework of ROOT Graphical User Interface (GUI)
///
///   In the following, "Stin", "Stex", "Stas" means:
///
///                 "Stin" = "Tower"  if the subdetector is "EB"  
///                        = "SC"     if the subdetector is "EE"
///
///                 "Stex" = "SM"     if the subdetector is "EB"
///                        = "Dee"    if the subdetector is "EE"
///
///                 "Stas" = "EB"     if the subdetector is "EB"
///                        = "EE"     if the subdetector is "EE"  
///
///
///==================== GUI DIALOG BOX PRESENTATION ==================
/// 
/// line# 
///   
///
///   1      Analysis                      (button + input widget)
///          First requested event number  (button + input widget)
///          Run number                    (button + input widget)
///
///   2      Number of samples             (button + input widget)
///          Last requested event number   (button + input widget)
///          Clean                         (menu)
///          Submit                        (menu)
///
///   3      Stex number                   (button + input widget)
///          Requested number of events    (button + input widget)
///
///   4      Stex Stin numbering           (button)
///          Nb of events for calculations (button + input widget)
///          Calculations                  (menu)
///
///........................................................................
///
///   5      Number of events                  (menu)
///   6      Pedestals                         (menu) 
///   7      Total noise                       (menu) 
///   8      Low  frequency noise              (menu)
///   9      High frequency noise              (menu)
///  10      Mean cor(s,s')                    (menu)
///  11      Sigma of cor(s,s')                (menu)
///  12      GeoView LF,HF Cor(c,c') (expert)  (menu)
///  13      Mean LF |Cor(c,c')| in (tow,tow') (menu)
///  14      Mean LH |Cor(c,c')| in (tow,tow') (menu)
///
///........................................................................
///
///  15      Stin                            (button + input widget)
///          Stin'                           (button + input widget)
///  16      Stin Xtal Numbering             (button) 
///  17      GeoView Cor(s,s') (expert)      (menu)
///
///  18      Low  Frequency Cor(Xtal Stin, Xtal Stin')   (menu)
///  19      High Frequency Cor(Xtal Stin, Xtal Stin')   (menu)
///
///...........................................................................
///
///  20      Channel number in Stin        (button + input widget)
///          Sample number                 (button + input widget)
///
///
///  21      Correlations between samples        (menu)
///  22      Covariances between samples         (menu)
///  23      Sample means                        (menu)
///  24      Sample sigmas                       (menu)
///
///  25      ADC sample values for (Xtal,Sample) (menu)
///
///............................................................................
///
///  26      List of run file name for history plots     (button + input widget)
///
///  27      Menu for history plots                      (menu)         
///............................................................................
///
///  28      LOG X          (check button: OFF: LIN scale / ON: LOG scale) 
///          LOG Y          (check button: OFF: LIN scale / ON: LOG scale) 
///          Y projection   (check button: OFF: X = variable
///                                             Y = quantity
///                                        ON : X = quantity
///                                             Y = distribution of the variable)
///............................................................................
///
///  29      General Title for Plots  (button + input widget)
///
///  30      Colors         (check button: ON = Rainbow,   OFF = ECNAColor )
///          Exit                 (button)
///
///  31      Clone Last Canvas    (button)
///          ROOT version         (button)
///          Help                 (button)
///
///===============================================================================
///     
///            Example of main program using the class TEcnaGui:
///
///%~%~%~%~%~%~%~%~%~%~%~%~%~~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~
///
///     #include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaGui.h"
///     #include <cstdlib>
///     
///     #include "Riostream.h"
///     #include "TROOT.h"
///     #include "TGApplication.h"
///     #include "TGClient.h"
///     #include "TRint.h"
///     
///     #include <stdlib.h>
///     #include <string>
///     #include "TSystem.h"
///     #include "TObject.h"
///     #include "TGWindow.h"
///     
///     #include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParPaths.h"
///     
///     extern void InitGui();
///     VoidFuncPtr_t initfuncs[] = { InitGui, 0 };
///     TROOT root("GUI","GUI test environnement", initfuncs);
///     
///     using namespace std;
///     
///     int main(int argc, char **argv)
///     {
///       TEcnaObject* MyEcnaObjectManager = new TEcnaObject();
///       TEcnaParPaths* pCnaParPaths = new TEcnaParPaths(MyEcnaObjectManager);
///       if( pCnaParPaths->GetPaths() == kTRUE )
///         {
///           cout << "*EcnaGuiEB> Starting ROOT session" << endl;
///           TRint theApp("App", &argc, argv);
///           
///           cout << "*EcnaGuiEB> Starting ECNA session" << endl;
///           TEcnaGui* mainWin = new TEcnaGui(MyEcnaObjectManager, "EB", gClient->GetRoot(), 395, 710);
///           mainWin->DialogBox();
///           Bool_t retVal = kTRUE;
///           theApp.Run(retVal);
///           cout << "*EcnaGuiEB> End of ECNA session." << endl;
///           delete mainWin;
///     
///           cout << "*EcnaGuiEB> End of ROOT session." << endl;
///           theApp.Terminate(0);
///           cout << "*EcnaGuiEB> Exiting main program." << endl;
///           exit(0);
///         }
///     }
///     
///
///%~%~%~%~%~%~%~%~%~%~%~%~%~~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~
///
///...........................................................................
///
///   Location of the ECNA web page:
///
///   http://cms-fabbro.web.cern.ch/cms-fabbro/cna_new/Correlated_Noise_Analysis/ECNA_cna_1.htm
///
///   For questions or comments, please send e-mail to: bernard.fabbro@cea.fr 
///
///---------------------------------------------------------------------------------

class TEcnaGui : public TGMainFrame {

 private:

  //..... Attributes

  Int_t              fgMaxCar;   // Max nb of caracters for char*

  Int_t              fCnew,        fCdelete;
  Int_t              fCnewRoot,    fCdeleteRoot;

  TString            fTTBELL;

  Int_t              fCnaCommand,  fCnaError;

  Int_t              fConfirmSubmit;
  Int_t              fConfirmRunNumber;
  TString            fConfirmRunNumberString;
  Int_t              fConfirmCalcScc;

  //==============================================. GUI box: menus, buttons,..

  TGWindow*          fCnaP;
  UInt_t             fCnaW,    fCnaH;
  TString            fSubDet;
  TString            fStexName, fStinName;

  //------------------------------------------------------------------------------------
  TEcnaObject*        fObjectManager;
  TEcnaHistos*        fHistos;
  TEcnaParHistos*     fCnaParHistos;
  TEcnaParPaths*      fCnaParPaths;
  TEcnaParCout*       fCnaParCout;
  TEcnaParEcal*       fEcal;
  TEcnaNumbering*     fEcalNumbering;
  TEcnaWrite*         fCnaWrite;
  //TEcnaRead*          fMyRootFile;

  //------------------- General frame, void frame, standard layout
  TGLayoutHints      *fLayoutGeneral,     *fLayoutBottLeft,     *fLayoutBottRight;
  TGLayoutHints      *fLayoutTopLeft,     *fLayoutTopRight;  
  TGLayoutHints      *fLayoutCenterYLeft, *fLayoutCenterYRight, *fLayoutCenterXTop;      
 
  TGCompositeFrame   *fVoidFrame;

  //===================== 1rst PART: SUBMIT, CALCULATIONS =======================================

  //++++++++++++++++++++++++++++++ Horizontal frame Analysis + First requested evt number + Run number
  TGCompositeFrame   *fAnaNorsRunFrame;
  TGLayoutHints      *fLayoutAnaNorsRunFrame;
  //--------------------------------- Sub-Frame Analysis Name (Button+Entry Field)
  TGCompositeFrame   *fAnaFrame;
  TGTextButton       *fAnaBut;
  Int_t               fAnaButC;
  TGLayoutHints      *fLayoutAnaBut;
  TGTextEntry        *fAnaText;
  TGTextBuffer       *fEntryAnaNumber;
  TGLayoutHints      *fLayoutAnaField;
  //--------------------------------- Sub-Frame First requested event number (Button+Entry Field)
  TGCompositeFrame   *fFevFrame;
  TGTextButton       *fFevBut;
  TGLayoutHints      *fLayoutFevBut;
  TGTextEntry        *fFevText;
  TGTextBuffer       *fEntryFevNumber;
  TGLayoutHints      *fLayoutFevFieldText;
  TGLayoutHints      *fLayoutFevFieldFrame;
  //-------------------------------- Sub-Frame Run number (Button+Entry Field)  
  TGCompositeFrame   *fRunFrame;  
  TGTextButton       *fRunBut;
  Int_t               fRunButC;
  TGLayoutHints      *fLayoutRunBut;
  TGTextEntry        *fRunText;
  TGTextBuffer       *fEntryRunNumber;
  TGLayoutHints      *fLayoutRunField;

  //++++++++++++++++++++++++ Horizontal frame Nb Of Samples + last requested events + Clean + Submit
  TGCompositeFrame   *fFevLevStexFrame;
  TGLayoutHints      *fLayoutFevLevStexFrame;
  //--------------------------------- Sub-Frame Number Of Requested Samples (Button+Entry Field)
  TGCompositeFrame   *fNorsFrame;
  TGTextButton       *fNorsBut;
  Int_t               fNorsButC;
  TGLayoutHints      *fLayoutNorsBut;
  TGTextEntry        *fNorsText;
  TGTextBuffer       *fEntryNorsNumber;
  TGLayoutHints      *fLayoutNorsField;

  //--------------------------------- Sub-Frame Last requested event (Button+Entry Field)
  TGCompositeFrame   *fLevFrame;
  TGTextButton       *fLevBut;
  TGLayoutHints      *fLayoutLevBut;
  TGTextEntry        *fLevText;
  TGTextBuffer       *fEntryLevNumber;
  TGLayoutHints      *fLayoutLevFieldText;
  TGLayoutHints      *fLayoutLevFieldFrame;
  //------------------------------------------- Clean Menu
  TGPopupMenu        *fMenuClean;
  TGMenuBar          *fMenuBarClean;
  Int_t               fMenuCleanSubC, fMenuCleanJobC, fMenuCleanPythC, fMenuCleanAllC;
  TGLayoutHints      *fLayoutRunCleanFrame;
  //------------------------------------------- Submit Menu
  TGPopupMenu        *fMenuSubmit;
  TGMenuBar          *fMenuBarSubmit;
  Int_t               fMenuSubmit8nmC, fMenuSubmit1nhC, fMenuSubmit8nhC, fMenuSubmit1ndC, fMenuSubmit1nwC;

 //++++++++++++++++++++++++++++++  Horizontal Frame:Stex number + NbOfReqEvts
  TGCompositeFrame   *fCompStRqFrame;
  TGLayoutHints      *fLayoutCompStRqFrame;
  //--------------------------------- Sub-Frame Stex Number
  TGCompositeFrame   *fStexFrame;
  TGTextButton       *fStexBut;
  TGLayoutHints      *fLayoutStexBut;
  TGTextEntry        *fStexText;
  TGTextBuffer       *fEntryStexNumber;
  TGLayoutHints      *fLayoutStexFieldText;
  TGLayoutHints      *fLayoutStexFieldFrame;

  //--------------------------------- Sub-Frame Number of requested Events (Button+Entry Field)
  TGCompositeFrame   *fRevFrame;
  TGTextButton       *fRevBut;
  TGLayoutHints      *fLayoutRevBut;
  TGTextEntry        *fRevText;
  TGTextBuffer       *fEntryRevNumber;
  TGLayoutHints      *fLayoutRevFieldText;
  TGLayoutHints      *fLayoutRevFieldFrame;

  //++++++++++++++++++++++++++++++  Horizontal StexStin numbering + Calculations
  TGCompositeFrame   *fCompStnbFrame;
  TGLayoutHints      *fLayoutCompStnbFrame;

  //------------------------------------------- Stex Stin Numbering view (Button)
  TGTextButton       *fButStexNb;
  Int_t               fButStexNbC;
  TGLayoutHints      *fLayoutStexNbBut;
  //--------------------------------- Sub-Frame NbSamp for calculation
  TGCompositeFrame   *fNbSampForCalcFrame;
  TGTextButton       *fNbSampForCalcBut;
  TGLayoutHints      *fLayoutNbSampForCalcBut;
  TGTextEntry        *fNbSampForCalcText;
  TGTextBuffer       *fEntryNbSampForCalcNumber;
  TGLayoutHints      *fLayoutNbSampForCalcFieldText;
  TGLayoutHints      *fLayoutNbSampForCalcFieldFrame;
  //------------------------------------------- CALCULATION (comput) Menu
  TGPopupMenu        *fMenuComput;
  TGMenuBar          *fMenuBarComput;
  Int_t               fMenuComputStdC, fMenuComputSccC, fMenuComputSttC;
  TGLayoutHints      *fLayoutRunComputFrame;

  //=========================== 2nd PART: STEXs ================================================

  //+++++++++++++++++++++++++++++++++++++++++++ Frame for quantities relative to the Stex
  TGCompositeFrame   *fStexUpFrame;

  //................................ Menus+Ymin+Ymax for the Stex ............................

  //...................................... Found evts in the data

  TGCompositeFrame   *fVmmD_NOE_ChNbFrame;

  TGCompositeFrame   *fVmaxD_NOE_ChNbFrame;
  TGTextButton       *fVmaxD_NOE_ChNbBut;
  TGLayoutHints      *fLayoutVmaxD_NOE_ChNbBut;
  TGTextBuffer       *fEntryVmaxD_NOE_ChNbNumber;
  TGTextEntry        *fVmaxD_NOE_ChNbText;
  TGLayoutHints      *fLayoutVmaxD_NOE_ChNbFieldText;
  TGLayoutHints      *fLayoutVmaxD_NOE_ChNbFrame;
 
  TGCompositeFrame   *fVminD_NOE_ChNbFrame;
  TGTextButton       *fVminD_NOE_ChNbBut;
  TGLayoutHints      *fLayoutVminD_NOE_ChNbBut;
  TGTextBuffer       *fEntryVminD_NOE_ChNbNumber;
  TGTextEntry        *fVminD_NOE_ChNbText;
  TGLayoutHints      *fLayoutVminD_NOE_ChNbFieldText;
  TGLayoutHints      *fLayoutVminD_NOE_ChNbFrame;

  TGPopupMenu        *fMenuD_NOE_ChNb;
  TGMenuBar          *fMenuBarD_NOE_ChNb;
  TGLayoutHints      *fLayoutMenuBarD_NOE_ChNb;
  Int_t               fMenuD_NOE_ChNbFullC;
  Int_t               fMenuD_NOE_ChNbSameC;
  Int_t               fMenuD_NOE_ChNbHocoVecoC;
  Int_t               fMenuD_NOE_ChNbAsciiFileC;

  TGLayoutHints      *fLayoutVmmD_NOE_ChNbFrame;

  //................................... Horizontal frame Pedestals, noises, cor(s,s)
  TGCompositeFrame   *fStexHozFrame;

  //------------------------------------------------------------- (PEDESTALS)
  TGCompositeFrame   *fVmmD_Ped_ChNbFrame;

  TGCompositeFrame   *fVmaxD_Ped_ChNbFrame;
  TGTextButton       *fVmaxD_Ped_ChNbBut;
  TGLayoutHints      *fLayoutVmaxD_Ped_ChNbBut;
  TGTextEntry        *fVmaxD_Ped_ChNbText;
  TGTextBuffer       *fEntryVmaxD_Ped_ChNbNumber;
  TGLayoutHints      *fLayoutVmaxD_Ped_ChNbFieldText;
  TGLayoutHints      *fLayoutVmaxD_Ped_ChNbFrame;
 
  TGCompositeFrame   *fVminD_Ped_ChNbFrame;
  TGTextButton       *fVminD_Ped_ChNbBut;
  TGLayoutHints      *fLayoutVminD_Ped_ChNbBut;
  TGTextEntry        *fVminD_Ped_ChNbText;
  TGTextBuffer       *fEntryVminD_Ped_ChNbNumber;
  TGLayoutHints      *fLayoutVminD_Ped_ChNbFieldText;
  TGLayoutHints      *fLayoutVminD_Ped_ChNbFrame;

  TGPopupMenu        *fMenuD_Ped_ChNb;
  TGMenuBar          *fMenuBarD_Ped_ChNb;
  TGLayoutHints      *fLayoutMenuBarD_Ped_ChNb;
  Int_t               fMenuD_Ped_ChNbFullC;
  Int_t               fMenuD_Ped_ChNbSameC;
  Int_t               fMenuD_Ped_ChNbHocoVecoC;
  Int_t               fMenuD_Ped_ChNbAsciiFileC;

  TGLayoutHints      *fLayoutVmmD_Ped_ChNbFrame;

  //---------------------------------------------------- (TOTAL NOISE)
  TGCompositeFrame   *fVmmD_TNo_ChNbFrame;

  TGCompositeFrame   *fVmaxD_TNo_ChNbFrame;
  TGTextButton       *fVmaxD_TNo_ChNbBut;
  TGLayoutHints      *fLayoutVmaxD_TNo_ChNbBut;
  TGTextBuffer       *fEntryVmaxD_TNo_ChNbNumber;
  TGTextEntry        *fVmaxD_TNo_ChNbText;
  TGLayoutHints      *fLayoutVmaxD_TNo_ChNbFieldText;
  TGLayoutHints      *fLayoutVmaxD_TNo_ChNbFrame;

  TGCompositeFrame   *fVminD_TNo_ChNbFrame;
  TGTextButton       *fVminD_TNo_ChNbBut;
  TGLayoutHints      *fLayoutVminD_TNo_ChNbBut;
  TGTextBuffer       *fEntryVminD_TNo_ChNbNumber;
  TGTextEntry        *fVminD_TNo_ChNbText;
  TGLayoutHints      *fLayoutVminD_TNo_ChNbFieldText;
  TGLayoutHints      *fLayoutVminD_TNo_ChNbFrame;

  TGPopupMenu        *fMenuD_TNo_ChNb;
  TGMenuBar          *fMenuBarD_TNo_ChNb;
  TGLayoutHints      *fLayoutMenuBarD_TNo_ChNb; 
  Int_t               fMenuD_TNo_ChNbFullC;
  Int_t               fMenuD_TNo_ChNbSameC;
  Int_t               fMenuD_TNo_ChNbSamePC;
  Int_t               fMenuD_TNo_ChNbHocoVecoC;
  Int_t               fMenuD_TNo_ChNbAsciiFileC;

  TGLayoutHints      *fLayoutVmmD_TNo_ChNbFrame;

  //--------------------------------------------------------- (LOW FREQUENCY NOISE)
  TGCompositeFrame   *fVmmD_LFN_ChNbFrame;

  TGCompositeFrame   *fVmaxD_LFN_ChNbFrame;
  TGTextButton       *fVmaxD_LFN_ChNbBut;
  TGLayoutHints      *fLayoutVmaxD_LFN_ChNbBut;
  TGTextEntry        *fVmaxD_LFN_ChNbText;
  TGTextBuffer       *fEntryVmaxD_LFN_ChNbNumber;
  TGLayoutHints      *fLayoutVmaxD_LFN_ChNbFieldText;
  TGLayoutHints      *fLayoutVmaxD_LFN_ChNbFrame;

  TGCompositeFrame   *fVminD_LFN_ChNbFrame;
  TGTextButton       *fVminD_LFN_ChNbBut;
  TGLayoutHints      *fLayoutVminD_LFN_ChNbBut;
  TGTextBuffer       *fEntryVminD_LFN_ChNbNumber;
  TGTextEntry        *fVminD_LFN_ChNbText;
  TGLayoutHints      *fLayoutVminD_LFN_ChNbFieldText;
  TGLayoutHints      *fLayoutVminD_LFN_ChNbFrame;

  TGPopupMenu        *fMenuD_LFN_ChNb;
  TGMenuBar          *fMenuBarD_LFN_ChNb;
  TGLayoutHints      *fLayoutMenuBarD_LFN_ChNb;
  Int_t               fMenuD_LFN_ChNbFullC;
  Int_t               fMenuD_LFN_ChNbSameC;
  Int_t               fMenuD_LFN_ChNbSamePC;
  Int_t               fMenuD_LFN_ChNbHocoVecoC;
  Int_t               fMenuD_LFN_ChNbAsciiFileC;

  TGLayoutHints      *fLayoutVmmD_LFN_ChNbFrame;

  //---------------------------------------------- (HIGH FREQUENCY NOISE)
  TGCompositeFrame   *fVmmD_HFN_ChNbFrame;

  TGCompositeFrame   *fVmaxD_HFN_ChNbFrame;
  TGTextButton       *fVmaxD_HFN_ChNbBut;
  TGLayoutHints      *fLayoutVmaxD_HFN_ChNbBut;
  TGTextEntry        *fVmaxD_HFN_ChNbText;
  TGTextBuffer       *fEntryVmaxD_HFN_ChNbNumber;
  TGLayoutHints      *fLayoutVmaxD_HFN_ChNbFieldText;
  TGLayoutHints      *fLayoutVmaxD_HFN_ChNbFrame;

  TGCompositeFrame   *fVminD_HFN_ChNbFrame;
  TGTextButton       *fVminD_HFN_ChNbBut;
  TGLayoutHints      *fLayoutVminD_HFN_ChNbBut;
  TGTextBuffer       *fEntryVminD_HFN_ChNbNumber;
  TGTextEntry        *fVminD_HFN_ChNbText;
  TGLayoutHints      *fLayoutVminD_HFN_ChNbFieldText;
  TGLayoutHints      *fLayoutVminD_HFN_ChNbFrame;

  TGPopupMenu        *fMenuD_HFN_ChNb;
  TGMenuBar          *fMenuBarD_HFN_ChNb;
  TGLayoutHints      *fLayoutMenuBarD_HFN_ChNb;
  Int_t               fMenuD_HFN_ChNbFullC;
  Int_t               fMenuD_HFN_ChNbSameC;
  Int_t               fMenuD_HFN_ChNbSamePC;
  Int_t               fMenuD_HFN_ChNbHocoVecoC;
  Int_t               fMenuD_HFN_ChNbAsciiFileC;

  TGLayoutHints      *fLayoutVmmD_HFN_ChNbFrame;

  //--------------------------------------------------- (MEAN CORSS)
  TGCompositeFrame   *fVmmD_MCs_ChNbFrame;

  TGCompositeFrame   *fVmaxD_MCs_ChNbFrame;
  TGTextButton       *fVmaxD_MCs_ChNbBut;
  TGLayoutHints      *fLayoutVmaxD_MCs_ChNbBut;
  TGTextEntry        *fVmaxD_MCs_ChNbText;
  TGTextBuffer       *fEntryVmaxD_MCs_ChNbNumber;
  TGLayoutHints      *fLayoutVmaxD_MCs_ChNbFieldText;
  TGLayoutHints      *fLayoutVmaxD_MCs_ChNbFrame;

  TGCompositeFrame   *fVminD_MCs_ChNbFrame;
  TGTextButton       *fVminD_MCs_ChNbBut;
  TGLayoutHints      *fLayoutVminD_MCs_ChNbBut;
  TGTextBuffer       *fEntryVminD_MCs_ChNbNumber;
  TGTextEntry        *fVminD_MCs_ChNbText;
  TGLayoutHints      *fLayoutVminD_MCs_ChNbFieldText;
  TGLayoutHints      *fLayoutVminD_MCs_ChNbFrame;

  TGPopupMenu        *fMenuD_MCs_ChNb;
  TGMenuBar          *fMenuBarD_MCs_ChNb;
  TGLayoutHints      *fLayoutMenuBarD_MCs_ChNb;
  Int_t               fMenuD_MCs_ChNbFullC;
  Int_t               fMenuD_MCs_ChNbSameC;
  Int_t               fMenuD_MCs_ChNbSamePC;
  Int_t               fMenuD_MCs_ChNbHocoVecoC;
  Int_t               fMenuD_MCs_ChNbAsciiFileC;

  TGLayoutHints      *fLayoutVmmD_MCs_ChNbFrame;

  //---------------------------------------------- (SIGMA OF COR(S,S))
  TGCompositeFrame   *fVmmD_SCs_ChNbFrame;

  TGCompositeFrame   *fVmaxD_SCs_ChNbFrame;
  TGTextButton       *fVmaxD_SCs_ChNbBut;
  TGLayoutHints      *fLayoutVmaxD_SCs_ChNbBut;
  TGTextEntry        *fVmaxD_SCs_ChNbText;
  TGTextBuffer       *fEntryVmaxD_SCs_ChNbNumber;
  TGLayoutHints      *fLayoutVmaxD_SCs_ChNbFieldText;
  TGLayoutHints      *fLayoutVmaxD_SCs_ChNbFrame;

  TGCompositeFrame   *fVminD_SCs_ChNbFrame;
  TGTextButton       *fVminD_SCs_ChNbBut;
  TGLayoutHints      *fLayoutVminD_SCs_ChNbBut;
  TGTextEntry        *fVminD_SCs_ChNbText;
  TGTextBuffer       *fEntryVminD_SCs_ChNbNumber;
  TGLayoutHints      *fLayoutVminD_SCs_ChNbFieldText;
  TGLayoutHints      *fLayoutVminD_SCs_ChNbFrame;

  TGPopupMenu        *fMenuD_SCs_ChNb;
  TGMenuBar          *fMenuBarD_SCs_ChNb;
  TGLayoutHints      *fLayoutMenuBarD_SCs_ChNb;
  Int_t               fMenuD_SCs_ChNbFullC;
  Int_t               fMenuD_SCs_ChNbSameC;
  Int_t               fMenuD_SCs_ChNbSamePC;
  Int_t               fMenuD_SCs_ChNbHocoVecoC; 
  Int_t               fMenuD_SCs_ChNbAsciiFileC;

  TGLayoutHints      *fLayoutVmmD_SCs_ChNbFrame;

  //----------------------------------------------------------------------------------

  TGLayoutHints      *fLayoutStexHozFrame;

  //...................................... LF and HF Correlations between channels
  TGCompositeFrame   *fVmmLHFccFrame;

  TGCompositeFrame   *fVmaxLHFccFrame;
  TGTextButton       *fVmaxLHFccBut;
  TGLayoutHints      *fLayoutVmaxLHFccBut;
  TGTextEntry        *fVmaxLHFccText;
  TGTextBuffer       *fEntryVmaxLHFccNumber;
  TGLayoutHints      *fLayoutVmaxLHFccFieldText;
  TGLayoutHints      *fLayoutVmaxLHFccFrame;

  TGCompositeFrame   *fVminLHFccFrame;
  TGTextButton       *fVminLHFccBut;
  TGLayoutHints      *fLayoutVminLHFccBut;
  TGTextBuffer       *fEntryVminLHFccNumber;
  TGTextEntry        *fVminLHFccText;
  TGLayoutHints      *fLayoutVminLHFccFieldText;
  TGLayoutHints      *fLayoutVminLHFccFrame;

  TGPopupMenu        *fMenuLHFcc;
  TGMenuBar          *fMenuBarLHFcc;
  TGLayoutHints      *fLayoutMenuBarLHFcc;
  Int_t               fMenuLFccColzC, fMenuLFccLegoC, fMenuHFccColzC, fMenuHFccLegoC;

  TGLayoutHints      *fLayoutVmmLHFccFrame;

  //...................................... Low Freq Mean Cor(c,c) for each pair of Stins
  TGCompositeFrame   *fVmmLFccMosFrame;

  TGCompositeFrame   *fVmaxLFccMosFrame;
  TGTextButton       *fVmaxLFccMosBut;
  TGLayoutHints      *fLayoutVmaxLFccMosBut;
  TGTextEntry        *fVmaxLFccMosText;
  TGTextBuffer       *fEntryVmaxLFccMosNumber;
  TGLayoutHints      *fLayoutVmaxLFccMosFieldText;
  TGLayoutHints      *fLayoutVmaxLFccMosFrame;

  TGCompositeFrame   *fVminLFccMosFrame;
  TGTextButton       *fVminLFccMosBut;
  TGLayoutHints      *fLayoutVminLFccMosBut;
  TGTextEntry        *fVminLFccMosText;
  TGTextBuffer       *fEntryVminLFccMosNumber;
  TGLayoutHints      *fLayoutVminLFccMosFieldText;
  TGLayoutHints      *fLayoutVminLFccMosFrame;

  TGPopupMenu        *fMenuLFccMos;
  TGMenuBar          *fMenuBarLFccMos;
  TGLayoutHints      *fLayoutMenuBarLFccMos;
  Int_t               fMenuLFccMosColzC, fMenuLFccMosLegoC;

  TGLayoutHints      *fLayoutVmmLFccMosFrame;

  //...................................... High Freq Mean Cor(c,c) for each pair of Stins
  TGCompositeFrame   *fVmmHFccMosFrame;

  TGCompositeFrame   *fVmaxHFccMosFrame;
  TGTextButton       *fVmaxHFccMosBut;
  TGLayoutHints      *fLayoutVmaxHFccMosBut;
  TGTextEntry        *fVmaxHFccMosText;
  TGTextBuffer       *fEntryVmaxHFccMosNumber;
  TGLayoutHints      *fLayoutVmaxHFccMosFieldText;
  TGLayoutHints      *fLayoutVmaxHFccMosFrame;

  TGCompositeFrame   *fVminHFccMosFrame;
  TGTextButton       *fVminHFccMosBut;
  TGLayoutHints      *fLayoutVminHFccMosBut;
  TGTextEntry        *fVminHFccMosText;
  TGTextBuffer       *fEntryVminHFccMosNumber;
  TGLayoutHints      *fLayoutVminHFccMosFieldText;
  TGLayoutHints      *fLayoutVminHFccMosFrame;

  TGPopupMenu        *fMenuHFccMos;
  TGMenuBar          *fMenuBarHFccMos;
  TGLayoutHints      *fLayoutMenuBarHFccMos;
  Int_t               fMenuHFccMosColzC, fMenuHFccMosLegoC;

  TGLayoutHints      *fLayoutVmmHFccMosFrame;


  TGLayoutHints      *fLayoutStexUpFrame;

 //================================= 3rd PART: STINs ================================================

  //+++++++++++++++++++++++++++++++++++++++++ Horizontal frame Stin_A + Stin_B
  TGCompositeFrame   *fStinSpFrame; 

  //----------------------------------- SubFrame Stin_A (Button + EntryField)
  TGCompositeFrame   *fTxSubFrame;

  TGCompositeFrame   *fStinAFrame;          
  TGTextButton       *fStinABut;
  Int_t               fStinAButC;
  TGLayoutHints      *fLayoutStinABut; 
  TGTextBuffer       *fEntryStinANumber; 
  TGTextEntry        *fStinAText;
  TGLayoutHints      *fLayoutStinAField;
  
  //............................ Stin Crystal Numbering view (Button)
  TGTextButton       *fButChNb;
  Int_t               fButChNbC;
  TGLayoutHints      *fLayoutChNbBut;

  //............................ Menus Stin_A
  TGPopupMenu        *fMenuCorssAll;
  TGMenuBar          *fMenuBarCorssAll;
  Int_t               fMenuCorssAllColzC, fMenuCovssAllColzC;

  //TGPopupMenu        *fMenuCovssAll;
  //TGMenuBar          *fMenuBarCovssAll;
  //Int_t               fMenuCovssAllColzC;

  TGLayoutHints      *fLayoutTxSubFrame;

  //----------------------------------- SubFrame Stin_B (Button + EntryField)
  TGCompositeFrame   *fTySubFrame;

  TGCompositeFrame   *fStinBFrame;
  TGTextButton       *fStinBBut;
  Int_t               fStinBButC;
  TGLayoutHints      *fLayoutStinBBut;
  TGTextBuffer       *fEntryStinBNumber;
  TGTextEntry        *fStinBText;
  TGLayoutHints      *fLayoutStinBField;

  TGLayoutHints      *fLayoutTySubFrame;

  TGLayoutHints      *fLayoutStinSpFrame;

  //.................................. Menus for Horizontal frame (Stin_A + Stin_B)
  TGPopupMenu        *fMenuLFCorcc; 
  TGMenuBar          *fMenuBarLFCorcc;   
  Int_t               fMenuLFCorccColzC, fMenuLFCorccLegoC; 

  TGPopupMenu        *fMenuHFCorcc;
  TGMenuBar          *fMenuBarHFCorcc;   
  Int_t               fMenuHFCorccColzC, fMenuHFCorccLegoC;    

 //======================== 4th PART:CHANNEL, SAMPLE ================================================

  //++++++++++++++++++++++++ Horizontal frame channel number (Stin_A crystal number) + sample number
  TGCompositeFrame   *fChSpFrame;

  //------------------------------------- SubFrame Channel (Button + EntryField)
  TGCompositeFrame   *fChSubFrame;

  TGCompositeFrame   *fChanFrame;
  TGTextButton       *fChanBut;
  Int_t               fChanButC;
  TGLayoutHints      *fLayoutChanBut;
  TGTextBuffer       *fEntryChanNumber;
  TGTextEntry        *fChanText;
  TGLayoutHints      *fLayoutChanField;

  //................................ Menus Stin_A crystal number
  TGPopupMenu        *fMenuCorss;
  TGMenuBar          *fMenuBarCorss;
  Int_t               fMenuCorssColzC,  fMenuCorssBoxC,   fMenuCorssTextC;
  Int_t               fMenuCorssContzC, fMenuCorssLegoC;
  Int_t               fMenuCorssSurf1C, fMenuCorssSurf2C, fMenuCorssSurf3C, fMenuCorssSurf4C;
  Int_t               fMenuCorssAsciiFileC;

  TGPopupMenu        *fMenuCovss;
  TGMenuBar          *fMenuBarCovss;
  Int_t               fMenuCovssColzC,  fMenuCovssBoxC,   fMenuCovssTextC;
  Int_t               fMenuCovssContzC, fMenuCovssLegoC;
  Int_t               fMenuCovssSurf1C, fMenuCovssSurf2C, fMenuCovssSurf3C, fMenuCovssSurf4C;
  Int_t               fMenuCovssAsciiFileC;

  TGPopupMenu        *fMenuD_MSp_SpNb;
  TGMenuBar          *fMenuBarD_MSp_SpNb;
  Int_t               fMenuD_MSp_SpNbLineFullC,    fMenuD_MSp_SpNbLineSameC,
                      fMenuD_MSp_SpNbLineAllStinC;

  TGPopupMenu        *fMenuD_MSp_SpDs;
  TGMenuBar          *fMenuBarD_MSp_SpDs;
  Int_t               fMenuD_MSp_SpDsLineFullC,    fMenuD_MSp_SpDsLineSameC,
                      fMenuD_MSp_SpDsLineAllStinC;

  TGPopupMenu        *fMenuD_SSp_SpNb;
  TGMenuBar          *fMenuBarD_SSp_SpNb; 
  Int_t               fMenuD_SSp_SpNbLineFullC, fMenuD_SSp_SpNbLineSameC,
                      fMenuD_SSp_SpNbLineAllStinC;

  TGPopupMenu        *fMenuD_SSp_SpDs;
  TGMenuBar          *fMenuBarD_SSp_SpDs; 
  Int_t               fMenuD_SSp_SpDsLineFullC, fMenuD_SSp_SpDsLineSameC,
                      fMenuD_SSp_SpDsLineAllStinC;

  TGLayoutHints      *fLayoutChSubFrame;

  //------------------------------------ SubFrame Sample (Button + EntryField)
  TGCompositeFrame   *fSpSubFrame;

  TGCompositeFrame   *fSampFrame;
  TGTextButton       *fSampBut;
  TGLayoutHints      *fLayoutSampBut;
  Int_t               fSampButC;
  TGTextEntry        *fSampText;
  TGTextBuffer       *fEntrySampNumber;
  TGLayoutHints      *fLayoutSampField;

  TGLayoutHints      *fLayoutSpSubFrame;

  //................................ Menus Sample number

  //     (no menu in this SubFrame)

  TGLayoutHints      *fLayoutChSpFrame;

  //++++++++++++++++++++++++++++++++++++ Menu Adc count Distribution
  TGPopupMenu        *fMenuAdcProj;
  TGMenuBar          *fMenuBarAdcProj;
  TGLayoutHints      *fLayoutMenuBarAdcProj;
  Int_t               fMenuAdcProjLineLinyFullC, fMenuAdcProjLineLinySameC;
  Int_t               fMenuAdcProjSampLineFullC, fMenuAdcProjSampLineSameC;

 //========================= 5th PART: HISTORY PLOTS ================================================

  //++++++++++++++++++++++++++++++++++++ Frame: Run List (Rul) (Button + EntryField)
  TGCompositeFrame   *fRulFrame;
  TGTextButton       *fRulBut;
  TGLayoutHints      *fLayoutRulBut;
  TGTextEntry        *fRulText;
  TGTextBuffer       *fEntryRulNumber;
  TGLayoutHints      *fLayoutRulFieldText;
  TGLayoutHints      *fLayoutRulFieldFrame;

  //................................ Menus for history plots
  TGPopupMenu        *fMenuHistory;
  TGMenuBar          *fMenuBarHistory;
  Int_t               fMenuH_Ped_DatePolmFullC, fMenuH_Ped_DatePolmSameC;
  Int_t               fMenuH_TNo_DatePolmFullC, fMenuH_TNo_DatePolmSameC, fMenuH_TNo_DatePolmSamePC;
  Int_t               fMenuH_LFN_DatePolmFullC, fMenuH_LFN_DatePolmSameC, fMenuH_LFN_DatePolmSamePC;
  Int_t               fMenuH_HFN_DatePolmFullC, fMenuH_HFN_DatePolmSameC, fMenuH_HFN_DatePolmSamePC;
  Int_t               fMenuH_MCs_DatePolmFullC, fMenuH_MCs_DatePolmSameC, fMenuH_MCs_DatePolmSamePC;
  Int_t               fMenuH_SCs_DatePolmFullC, fMenuH_SCs_DatePolmSameC, fMenuH_SCs_DatePolmSamePC;

 //========================= 6th PART: LAST BUTTONS ================================================

  //++++++++++++++++++++++++++++++++++++ Lin/Log X +  Lin/Log Y + Projection along Y axis Frame
  TGCompositeFrame   *fLinLogFrame;

  //---------------------------------- Lin/Log X
  TGCheckButton      *fButLogx;
  Int_t               fButLogxC; 
  TGLayoutHints      *fLayoutLogxBut;
  //---------------------------------- Lin/Log Y
  TGCheckButton      *fButLogy;
  Int_t               fButLogyC; 
  TGLayoutHints      *fLayoutLogyBut;

  //---------------------------------- Projection on Y axis
  TGCheckButton      *fButProjy;
  Int_t               fButProjyC; 
  TGLayoutHints      *fLayoutProjyBut;

  //++++++++++++++++++++++++++++++++++++ Frame: General title (Gent) (Button + EntryField)
  TGCompositeFrame   *fGentFrame;
  TGTextButton       *fGentBut;
  TGLayoutHints      *fLayoutGentBut;
  TGTextEntry        *fGentText;
  TGTextBuffer       *fEntryGentNumber;
  TGLayoutHints      *fLayoutGentFieldText;
  TGLayoutHints      *fLayoutGentFieldFrame;

  //++++++++++++++++++++++++++++++++++++ Color palette + EXIT BUTTON frame
  TGCompositeFrame   *fColorExitFrame;
  TGLayoutHints      *fLayoutColorExitFrame;

  //---------------------------------- Color palette
  TGCheckButton      *fButColPal;
  Int_t               fButColPalC; 
  TGLayoutHints      *fLayoutColPalBut;
  //---------------------------------- Exit
  TGTextButton       *fButExit;
  Int_t               fButExitC;      
  TGLayoutHints      *fLayoutExitBut;

  //++++++++++++++++++++++++++++++++++++ Last Frame
  TGCompositeFrame   *fLastFrame;

  //--------------------------------- Clone Last Canvas (Button)
  TGTextButton       *fButClone;
  Int_t               fButCloneC;
  TGLayoutHints      *fLayoutCloneBut;
  //--------------------------------- Root version (Button)
  TGTextButton       *fButRoot;
  Int_t               fButRootC;
  TGLayoutHints      *fLayoutRootBut;
  //--------------------------------- Help (Button)
  TGTextButton       *fButHelp;
  Int_t               fButHelpC;
  TGLayoutHints      *fLayoutHelpBut;

  //==================================================== Miscellaneous parameters

  //ofstream fFcout_f;

  TString  fKeyAnaType;           // Type of analysis

  Int_t    fKeyNbOfSamples;       // Nb of required samples (file)
  TString  fKeyNbOfSamplesString; // Nb of required samples (file) in TString
  Int_t    fKeyRunNumber;               // Run number
  TString  fKeyRunNumberString;         // Run number characters in TString
  Int_t    fKeyFirstReqEvtNumber;       // First requested event number
  TString  fKeyFirstReqEvtNumberString; // First requested event number in TString
  Int_t    fKeyLastReqEvtNumber;        // Last requested event number
  TString  fKeyLastReqEvtNumberString;  // Last requested event number in TString 
  Int_t    fKeyReqNbOfEvts;             // Requested number of events
  TString  fKeyReqNbOfEvtsString;       // Requested number of events in TString 

  Int_t    fKeyStexNumber;              // Stex number
  TString  fKeyStexNumberString;        // Stex number in TString 
  Int_t    fKeyNbOfSampForCalc;         // Nb of required samples (calculation)
  TString  fKeyNbOfSampForCalcString;   // Nb of required samples (calculation) in TString

  TString  fKeyFileNameRunList;   // Name of the file containing the run parameters list
  TString  fKeyRunListInitCode;
  //TString  fKeyPyf;               //  Name of file containing the data file names
  //                                // which are in the "source" sector of the python file

  TString  fPythonFileName;       //  python file name (for cmsRun)

  TString  fKeyScaleX;
  TString  fKeyScaleY;
  TString  fKeyProjY;
  TString  fKeyColPal;
  TString  fKeyGeneralTitle;    // General title for the plots 

  //................... VISUALIZATION PARAMETERS  

  Int_t    fKeyStinANumber; // Stin X number
  Int_t    fKeyStinBNumber; // Stin Y number
  Int_t    fKeyChanNumber;  // Channel number
  Int_t    fKeySampNumber;  // Sample number

  //................... ymin and ymax values

  Double_t fKeyVminD_NOE_ChNb; 
  Double_t fKeyVmaxD_NOE_ChNb;
 
  Double_t fKeyVminD_Ped_ChNb; 
  Double_t fKeyVmaxD_Ped_ChNb; 

  Double_t fKeyVminD_TNo_ChNb; 
  Double_t fKeyVmaxD_TNo_ChNb; 

  Double_t fKeyVminD_MCs_ChNb; 
  Double_t fKeyVmaxD_MCs_ChNb;

  Double_t fKeyVminD_LFN_ChNb; 
  Double_t fKeyVmaxD_LFN_ChNb; 

  Double_t fKeyVminD_HFN_ChNb; 
  Double_t fKeyVmaxD_HFN_ChNb; 

  Double_t fKeyVminD_SCs_ChNb; 
  Double_t fKeyVmaxD_SCs_ChNb; 

  Double_t fKeyVminLFccMos; 
  Double_t fKeyVmaxLFccMos;
  Double_t fKeyVminHFccMos; 
  Double_t fKeyVmaxHFccMos;

  Double_t fKeyVminLHFcc; 
  Double_t fKeyVmaxLHFcc;

  //................... plot parameters (for onlyone,same options)

  TString  fMemoScaleX;
  TString  fMemoScaleY;
  TString  fMemoProjY;

  TString  fMemoColPal;

  TString  fOptPlotFull;
  TString  fOptPlotSame;
  TString  fOptPlotSameP;
  TString  fOptPlotSameInStin;
  TString  fOptAscii;

  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

 public:
  TEcnaGui();
  TEcnaGui(TEcnaObject*, const TString&, const TGWindow *, UInt_t, UInt_t);

  // TEcnaGui(const TString&, const TGWindow *, UInt_t, UInt_t);
  virtual  ~TEcnaGui();

  void Init();
  void DialogBox();

  void InitKeys();

  void DisplayInEntryField(TGTextEntry*, Int_t&);
  void DisplayInEntryField(TGTextEntry*, Double_t&);
  void DisplayInEntryField(TGTextEntry*, const TString&);

  void DoButtonAna();

  void DoButtonNors();
  void DoButtonRun();


  void DoButtonFev();
  void DoButtonLev();
  void DoButtonRev();
  void DoButtonStex();
  void DoButtonNbSampForCalc();
  void DoButtonStexNb();

  //########################################
  void DoButtonVminD_NOE_ChNb();
  void DoButtonVmaxD_NOE_ChNb();

  void DoButtonVminD_Ped_ChNb();
  void DoButtonVmaxD_Ped_ChNb();

  void DoButtonVminD_TNo_ChNb();
  void DoButtonVmaxD_TNo_ChNb();

  void DoButtonVminD_LFN_ChNb();
  void DoButtonVmaxD_LFN_ChNb();

  void DoButtonVminD_HFN_ChNb();
  void DoButtonVmaxD_HFN_ChNb();

  void DoButtonVminD_MCs_ChNb();
  void DoButtonVmaxD_MCs_ChNb();

  void DoButtonVminD_SCs_ChNb();
  void DoButtonVmaxD_SCs_ChNb();

  void DoButtonVminLFccMos();
  void DoButtonVmaxLFccMos();
  void DoButtonVminHFccMos();
  void DoButtonVmaxHFccMos();

  void DoButtonVminLHFcc();
  void DoButtonVmaxLHFcc();

  //########################################

  void DoButtonStinA();
  void DoButtonStinB();

  void DoButtonChNb();
  void DoButtonChan();
  void DoButtonSamp();

  void DoButtonRul();

  void DoButtonLogx();
  void DoButtonLogy();
  void DoButtonProjy();

  void DoButtonGent();

  void DoButtonColPal();
  void DoButtonExit();

  void DoButtonClone();
  void DoButtonRoot();
  void DoButtonHelp();

  void HandleMenu(Int_t);

  void SubmitOnBatchSystem(const TString&);
  void CleanBatchFiles(const TString&);
  void Calculations(const TString&);

  //------------------- VISUALIZATION METHODS

  // void ViewMatrixCorrelationStins(const TString&);  // (RESERVE)
  // void ViewMatrixCovarianceStins(const TString&);   // (RESERVE)

  void ViewMatrixLowFrequencyMeanCorrelationsBetweenStins(const TString&);  
  void ViewMatrixHighFrequencyMeanCorrelationsBetweenStins(const TString&);   

  void ViewMatrixLowFrequencyCorrelationsBetweenChannels(const Int_t&, const Int_t&, const TString&);
  void ViewMatrixHighFrequencyCorrelationsBetweenChannels(const Int_t&, const Int_t&, const TString&);

  void ViewMatrixCorrelationSamples(const Int_t&, const Int_t&, const TString&);
  void ViewMatrixCovarianceSamples(const Int_t&, const Int_t&, const TString&);

  void ViewSorSNumberOfEvents();             // SorS = Stas or Stex
  void ViewSorSPedestals();
  void ViewSorSTotalNoise();
  void ViewSorSMeanCorss();
  void ViewSorSLowFrequencyNoise();
  void ViewSorSHighFrequencyNoise();
  void ViewSorSSigmaOfCorss();

  void ViewStexLowFrequencyCorcc();
  void ViewStexHighFrequencyCorcc();

  void ViewStinCorrelationSamples(const Int_t&);
  void ViewStinCovarianceSamples(const Int_t&);
  void ViewStinCrystalNumbering(const Int_t&);
  void ViewStexStinNumbering();

  void ViewHistoSorSNumberOfEventsOfCrystals(const TString&);    // SorS = Stas or Stex
  void ViewHistoSorSNumberOfEventsDistribution(const TString&);
  void ViewHistoSorSPedestalsOfCrystals(const TString&);
  void ViewHistoSorSPedestalsDistribution(const TString&);
  void ViewHistoSorSTotalNoiseOfCrystals(const TString&);
  void ViewHistoSorSTotalNoiseDistribution(const TString&);
  void ViewHistoSorSMeanCorssOfCrystals(const TString&);
  void ViewHistoSorSMeanCorssDistribution(const TString&);
  void ViewHistoSorSLowFrequencyNoiseOfCrystals(const TString&);
  void ViewHistoSorSLowFrequencyNoiseDistribution(const TString&);
  void ViewHistoSorSHighFrequencyNoiseOfCrystals(const TString&);
  void ViewHistoSorSHighFrequencyNoiseDistribution(const TString&);
  void ViewHistoSorSSigmaOfCorssOfCrystals(const TString&);
  void ViewHistoSorSSigmaOfCorssDistribution(const TString&);

  void ViewHistoCrystalSampleMeans(const Int_t&, const Int_t&, const TString&);
  void ViewHistoCrystalSampleMeansDistribution(const Int_t&, const Int_t&, const TString&);
  void ViewHistoCrystalSampleSigmas(const Int_t&, const Int_t&, const TString&);
  void ViewHistoCrystalSampleSigmasDistribution(const Int_t&, const Int_t&, const TString&);

  void ViewHistoCrystalSampleValues(const Int_t&, const Int_t&, const Int_t&, const TString&);
  void ViewHistoSampleEventDistribution(const Int_t&, const Int_t&, const Int_t&, const TString&);

  void ViewHistimeCrystalPedestals(const TString&, const Int_t&, const Int_t&, const TString&);
  void ViewHistimeCrystalTotalNoise(const TString&, const Int_t&, const Int_t&, const TString&);
  void ViewHistimeCrystalLowFrequencyNoise(const TString&, const Int_t&, const Int_t&, const TString&);
  void ViewHistimeCrystalHighFrequencyNoise(const TString&, const Int_t&, const Int_t&, const TString&);
  void ViewHistimeCrystalMeanCorss(const TString&, const Int_t&, const Int_t&, const TString&);
  void ViewHistimeCrystalSigmaOfCorss(const TString&, const Int_t&, const Int_t&, const TString&);

  void ViewHistimeCrystalPedestalsRuns(const TString&, const Int_t&, const Int_t&, const TString&);
  void ViewHistimeCrystalTotalNoiseRuns(const TString&, const Int_t&, const Int_t&, const TString&);
  void ViewHistimeCrystalLowFrequencyNoiseRuns(const TString&, const Int_t&, const Int_t&, const TString&);
  void ViewHistimeCrystalHighFrequencyNoiseRuns(const TString&, const Int_t&, const Int_t&, const TString&);
  void ViewHistimeCrystalMeanCorssRuns(const TString&, const Int_t&, const Int_t&, const TString&);
  void ViewHistimeCrystalSigmaOfCorssRuns(const TString&, const Int_t&, const Int_t&, const TString&);

  void MessageCnaCommandReplyA(const TString&);
  void MessageCnaCommandReplyB(const TString&);

ClassDef(TEcnaGui,1)// Dialog box with GUI + methods for CNA (Correlated Noises Analysis)

};

#endif   //    ZTR_TEcnaGui
