//----------Author's Name: B.Fabbro DSM/DAPNIA/SPP CEA-Saclay
//----------Copyright: Those valid for CEA sofware
//----------Modified: 11/07/2007


#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEBParameters.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaDialogEB.h"
#include <cstdlib>

ClassImp(TCnaDialogEB)
//______________________________________________________________________________
//
// TCnaDialogEB.
//
// This class provides a dialog box for CNA (Correlated Noise Analysis)
// of the Ecal Barrel (EB) in the framework of ROOT and GUI (Graphical User Interface)
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//          *--------------------------------*
//          |                                |
//          |    |  I M P O R T A N T  |     |
//          |    |                     |     |
//          |    v                     v     |
//          |                                |
//          *--------------------------------*
//
//  A "cna-configuration" file named "cna_results_root.cfg"
//  must be present in your HOME directory.
//
//  This file must have one line which must contain the path of the directory 
//  (without slash at the end) where are the .root result files
//
//  EXAMPLE:
//
//  $HOME/scratch0/cna/results_root
//
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// ***************** PRESENTATION OF THE DIALOG BOX ********************
// 
//      * GENERAL BUTTONS:
//
//          Analysis name          (button + input widget)
//          Run number             (button + input widget)
//          First taken event      (button + input widget)
//          Number of taken events (button + input widget)
//          Super-Module number    (button + input widget)
//          Super-Module numbering (button)
//
//........................................................................
//
//      * UNDER THE SUPER-MODULE NUMBER BUTTON: plots concerning the super-module
//
//        Number of events found in the data (menu: 4 options)
//
//        Mean of samle means (mean pedestals)   (menu: 4 options) 
//        Mean of sample sigmas                  (menu: 4 options) 
//        Mean of (sample,sample) correlations   (menu: 4 options)
//
//        Sigma of sample means                  (menu: 4 options)
//        Sigma of sample sigmas                 (menu: 4 options)
//        Sigma of (sample,sample) correlations  (menu: 4 options)
//
//    * explanation> "Mean of" and "Sigma of" = mean and sigma over the samples
//                   "sample means" and "sample sigmas" = means and sigmas over the events 
//
//        Correlations and covariances between towers (averaged over crystals and samples)
//
//........................................................................
//
//      * TOWER X and TOWER Y BUTTONS (button + input widget)
//
//
//      * UNDER THE TOWER X BUTTON: plots concerning the Tower X:
//
//         Channel Numbering                                       (button)
//         Global view of the correlation matrices between samples (menu: 1 option)
//         Global view of the covariance  matrices between samples (menu: 1 option)
//
//      * UNDER BOTH TOWER X AND TOWER Y BUTTONS:
//         Covariances and correlations between crystals (averaged over samples)
//         (menu: 2 options)
//
//...........................................................................
//
//      * TOWER X CRYSTAL NUMBER BUTTON (button + input widget)
//
//      * UNDER THE TOWER X CRYSTAL NUMBER BUTTON: 
//        Quantities relative to the crystal:
//
//        (sample,sample) correlation matrix      (menu: 4 options)
//        (sample,sample) covariance  matrix      (menu: 4 options)
//        Expectation values of the samples       (menu: 2 options)
//        Sigmas of the samples                   (menu: 2 options)
//
//..........................................................................
//
//      * SAMPLE NUMBER BUTTON (button + input widget)
//      
//...........................................................................
//
//      * UNDER THE TOWER X CRYSTAL NUMBER BUTTON AND THE SAMPLE NUMBER BUTTON:
//
//        Event distribution:
//             histogram of the ADC event distribution (menu: 2 options)
//
//............................................................................
//
//      * TIME EVOLUTION
//
//         File name for time evolution plots     (button + input widget)
//
//         time evolution menu                    (menu: 8 options)         
//............................................................................
//
//      * SCALE 
//
//        LOG X                  (check button: ON = LOG scale, OFF = LIN scale) 
//        LOG Y                  (check button: ON = LOG scale, OFF = LIN scale) 
//
//............................................................................
//
//      * EXIT                   (button)
//
//      * ROOT version           (button)
//      * Help                   (button)
//
//............................................................................
//
// Example of main program using the class TCnaDialogEB:
// 
//%~%~%~%~%~%~%~%~%~%~%~%~%~~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~
// #include "Riostream.h"
// #include "TObject.h"
// #include "TROOT.h"
// #include "TGApplication.h"
// #include "TGClient.h"
// #include "TRint.h"
// #include "TGWindow.h"
// 
// // CMSSW include files
// 
// #include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaDialogEB.h"
// 
// extern void InitGui();
// VoidFuncPtr_t initfuncs[] = { InitGui, 0 };
// TROOT root("GUI","GUI test environnement", initfuncs);
// 
// int main(int argc, char **argv)
// {
//   cout << "*EcalCorrelatedNoiseInteractiveAnalysis> Starting ROOT session" << endl;
//   TRint theApp("App", &argc, argv);
//   cout << "*EcalCorrelatedNoiseInteractiveAnalysis> Starting new CNA session" << endl;
//   TCnaDialogEB* mainWin = new TCnaDialogEB(gClient->GetRoot(), 420, 800);
//   cout << "*EcalCorrelatedNoiseInteractiveAnalysis> CNA session: preparing to run application." << endl;
//   theApp.Run(kTRUE);
//   cout << "*EcalCorrelatedNoiseInteractiveAnalysis> End of CNA session: preparing to delete." << endl;
//   delete mainWin;
//   cout << "*EcalCorrelatedNoiseInteractiveAnalysis> delete done." << endl;  
//   exit(0);
// }
//%~%~%~%~%~%~%~%~%~%~%~%~%~~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~
//
//...........................................................................
//
//   For more details on the classes of the CNA package:
//
//              http://www.cern.ch/cms-fabbro/cna
//

//---------------------------- TCnaDialogEB.cxx -------------------------
//
//   Dialog box + methods for CNA (Correlated Noise Analysis)
//
//   For questions or comments, please send e-mail to Bernard Fabbro:
//             
//   fabbro@hep.saclay.cea.fr 
//
//----------------------------------------------------------------------

  TCnaDialogEB::~TCnaDialogEB()
{
//destructor

  // cout << "TCnaDialogEB> Entering destructor" << endl;
  // cout << "            fCnew = " << fCnew << ", fCdelete = " << fCdelete << endl;
 
  //.... general variables

  if ( fParameters         != 0 ) {delete fParameters;         fCdelete++;}     // 001
  if ( fView               != 0 ) {delete fView;               fCdelete++;}

  //.... general frames

  if ( fLayoutGeneral      != 0 ) {delete fLayoutGeneral;      fCdelete++;}
  if ( fLayoutBottLeft     != 0 ) {delete fLayoutBottLeft;     fCdelete++;}
  if ( fLayoutBottRight    != 0 ) {delete fLayoutBottRight;    fCdelete++;}
  if ( fLayoutTopLeft      != 0 ) {delete fLayoutTopLeft;      fCdelete++;}
  if ( fLayoutTopRight     != 0 ) {delete fLayoutTopRight;     fCdelete++;}
  if ( fLayoutCenterYLeft  != 0 ) {delete fLayoutCenterYLeft;  fCdelete++;}
  if ( fLayoutCenterYRight != 0 ) {delete fLayoutCenterYRight; fCdelete++;}

  if ( fVoidFrame          != 0 ) {delete fVoidFrame;          fCdelete++;}     // 010

  //..... specific frames + buttons + menus

  //++++++++++++++++++++++++++++++++++++++++++ Horizontal frame Ana + Run 
  if ( fAnaRunFrame        != 0 ) {delete fAnaRunFrame;        fCdelete++;}

  //--------------------------------- Sub-Frame Analysis Name (Button+Entry Field)
  if ( fAnaFrame           != 0 ) {delete fAnaFrame;           fCdelete++;}
  if ( fAnaBut             != 0 ) {delete fAnaBut;             fCdelete++;}
  if ( fLayoutAnaBut       != 0 ) {delete fLayoutAnaBut;       fCdelete++;}
  if ( fEntryAnaNumber     != 0 ) {delete fEntryAnaNumber;     fCdelete++;}
  if ( fAnaText            != 0 ) {fAnaText->Delete();         fCdelete++;}  
  if ( fLayoutAnaField     != 0 ) {delete fLayoutAnaField;     fCdelete++;}

  //-------------------------------- Sub-Frame Run number (Button+Entry Field)  
  if ( fRunFrame           != 0 ) {delete fRunFrame;           fCdelete++;}
  if ( fRunBut             != 0 ) {delete fRunBut;             fCdelete++;}
  if ( fLayoutRunBut       != 0 ) {delete fLayoutRunBut;       fCdelete++;}
  if ( fEntryRunNumber     != 0 ) {delete fEntryRunNumber;     fCdelete++;}
  if ( fRunText            != 0 ) {fRunText->Delete();         fCdelete++;}
  if ( fLayoutRunField     != 0 ) {delete fLayoutRunField;     fCdelete++;}

  if ( fLayoutAnaRunFrame  != 0 ) {delete fLayoutAnaRunFrame;  fCdelete++;}

  //+++++++++++++++++++++++++++++++++++++++++++ Frames: first taken evt + nb of events + super-module
  if ( fFevFrame            != 0 ) {delete fFevFrame;            fCdelete++;}
  if ( fFevBut              != 0 ) {delete fFevBut;              fCdelete++;}
  if ( fLayoutFevBut        != 0 ) {delete fLayoutFevBut;        fCdelete++;}
  if ( fEntryFevNumber      != 0 ) {delete fEntryFevNumber;      fCdelete++;}
  if ( fFevText             != 0 ) {fFevText->Delete();          fCdelete++;}  
  if ( fLayoutFevFieldText  != 0 ) {delete fLayoutFevFieldText;  fCdelete++;}     // 030
  if ( fLayoutFevFieldFrame != 0 ) {delete fLayoutFevFieldFrame; fCdelete++;}

  if ( fNoeFrame            != 0 ) {delete fNoeFrame;            fCdelete++;}
  if ( fNoeBut              != 0 ) {delete fNoeBut;              fCdelete++;}
  if ( fLayoutNoeBut        != 0 ) {delete fLayoutNoeBut;        fCdelete++;}
  if ( fEntryNoeNumber      != 0 ) {delete fEntryNoeNumber;      fCdelete++;}
  if ( fNoeText             != 0 ) {fNoeText->Delete();          fCdelete++;}
  if ( fLayoutNoeFieldText  != 0 ) {delete fLayoutNoeFieldText;  fCdelete++;}
  if ( fLayoutNoeFieldFrame != 0 ) {delete fLayoutNoeFieldFrame; fCdelete++;}

  if ( fSuMoFrame            != 0 ) {delete fSuMoFrame;            fCdelete++;}
  if ( fSuMoBut              != 0 ) {delete fSuMoBut;              fCdelete++;}
  if ( fLayoutSuMoBut        != 0 ) {delete fLayoutSuMoBut;        fCdelete++;}
  if ( fEntrySuMoNumber      != 0 ) {delete fEntrySuMoNumber;      fCdelete++;}
  if ( fSuMoText             != 0 ) {fSuMoText->Delete();          fCdelete++;}  
  if ( fLayoutSuMoFieldText  != 0 ) {delete fLayoutSuMoFieldText;  fCdelete++;}
  if ( fLayoutSuMoFieldFrame != 0 ) {delete fLayoutSuMoFieldFrame; fCdelete++;}

  //............................ SuperModule Tower Numbering view (Button)
  if ( fButSMNb              != 0 ) {delete fButSMNb;              fCdelete++;}
  if ( fLayoutSMNbBut        != 0 ) {delete fLayoutSMNbBut;        fCdelete++;}

  //+++++++++++++++++++++++++++++++++++++++++++ Frame for quantities relative to the Super-Module 
  if ( fSuMoUpFrame          != 0 ) {delete fSuMoUpFrame;          fCdelete++;}

  //................................ Menus+Ymin+Ymax for the Super-Module ............................

  //...................................... Found evts in the data

  if ( fVmmFoundEvtsFrame            != 0 ) {delete fVmmFoundEvtsFrame;            fCdelete++;}

  if ( fVmaxFoundEvtsFrame           != 0 ) {delete fVmaxFoundEvtsFrame;           fCdelete++;}     // 050
  if ( fVmaxFoundEvtsBut             != 0 ) {delete fVmaxFoundEvtsBut;             fCdelete++;}
  if ( fLayoutVmaxFoundEvtsBut       != 0 ) {delete fLayoutVmaxFoundEvtsBut;       fCdelete++;}
  if ( fEntryVmaxFoundEvtsNumber     != 0 ) {delete fEntryVmaxFoundEvtsNumber;     fCdelete++;}
  if ( fVmaxFoundEvtsText            != 0 ) {fVmaxFoundEvtsText->Delete();         fCdelete++;}
  if ( fLayoutVmaxFoundEvtsFieldText != 0 ) {delete fLayoutVmaxFoundEvtsFieldText; fCdelete++;}
  if ( fLayoutVmaxFoundEvtsFrame     != 0 ) {delete fLayoutVmaxFoundEvtsFrame;     fCdelete++;}

  if ( fVminFoundEvtsFrame           != 0 ) {delete fVminFoundEvtsFrame;           fCdelete++;}
  if ( fVminFoundEvtsBut             != 0 ) {delete fVminFoundEvtsBut;             fCdelete++;}
  if ( fLayoutVminFoundEvtsBut       != 0 ) {delete fLayoutVminFoundEvtsBut;       fCdelete++;}
  if ( fEntryVminFoundEvtsNumber     != 0 ) {delete fEntryVminFoundEvtsNumber;     fCdelete++;}     // 060
  if ( fVminFoundEvtsText            != 0 ) {fVminFoundEvtsText->Delete();         fCdelete++;}
  if ( fLayoutVminFoundEvtsFieldText != 0 ) {delete fLayoutVminFoundEvtsFieldText; fCdelete++;}
  if ( fLayoutVminFoundEvtsFrame     != 0 ) {delete fLayoutVminFoundEvtsFrame;     fCdelete++;}

  if ( fMenuFoundEvts                != 0 ) {delete fMenuFoundEvts;                fCdelete++;}
  if ( fMenuBarFoundEvts             != 0 ) {fMenuBarFoundEvts->Delete();          fCdelete++;}
  if ( fVminFoundEvtsText            != 0 ) {fVminFoundEvtsText->Delete();         fCdelete++;}

  if ( fLayoutVmmFoundEvtsFrame      != 0 ) {delete fLayoutVmmFoundEvtsFrame;      fCdelete++;}

  //................................... Horizontal frame ev + sig
  if ( fSuMoHozFrame                 != 0 ) {delete fSuMoHozFrame;                 fCdelete++;}

  //............................................... Frame Ev + Menus Ev
  if ( fSuMoHozSubEvFrame            != 0 ) {delete fSuMoHozSubEvFrame;            fCdelete++;}

  //---------------------------------------------------
  if ( fVmmEvEvFrame            != 0 ) {delete fVmmEvEvFrame;            fCdelete++;}    // 070

  if ( fVmaxEvEvFrame           != 0 ) {delete fVmaxEvEvFrame;           fCdelete++;}
  if ( fVmaxEvEvBut             != 0 ) {delete fVmaxEvEvBut;             fCdelete++;}
  if ( fLayoutVmaxEvEvBut       != 0 ) {delete fLayoutVmaxEvEvBut;       fCdelete++;}
  if ( fVmaxEvEvText            != 0 ) {fVmaxEvEvText->Delete();         fCdelete++;}
  if ( fEntryVmaxEvEvNumber     != 0 ) {delete fEntryVmaxEvEvNumber;     fCdelete++;}
  if ( fLayoutVmaxEvEvFieldText != 0 ) {delete fLayoutVmaxEvEvFieldText; fCdelete++;}
  if ( fLayoutVmaxEvEvFrame     != 0 ) {delete fLayoutVmaxEvEvFrame;     fCdelete++;}

  if ( fVminEvEvFrame           != 0 ) {delete fVminEvEvFrame;           fCdelete++;}
  if ( fVminEvEvBut             != 0 ) {delete fVminEvEvBut;             fCdelete++;}
  if ( fLayoutVminEvEvBut       != 0 ) {delete fLayoutVminEvEvBut;       fCdelete++;}
  if ( fVminEvEvText            != 0 ) {fVminEvEvText->Delete();         fCdelete++;}
  if ( fEntryVminEvEvNumber     != 0 ) {delete fEntryVminEvEvNumber;     fCdelete++;}
  if ( fLayoutVminEvEvFieldText != 0 ) {delete fLayoutVminEvEvFieldText; fCdelete++;}
  if ( fLayoutVminEvEvFrame     != 0 ) {delete fLayoutVminEvEvFrame;     fCdelete++;}

  if ( fMenuEvEv                != 0 ) {delete fMenuEvEv;                fCdelete++;}
  if ( fMenuBarEvEv             != 0 ) {fMenuBarEvEv->Delete();          fCdelete++;}
  if ( fLayoutMenuBarEvEv       != 0 ) {delete fLayoutMenuBarEvEv;       fCdelete++;}

  if ( fLayoutVmmEvEvFrame      != 0 ) {delete fLayoutVmmEvEvFrame;      fCdelete++;}

  //----------------------------------------------------
  if ( fVmmEvSigFrame            != 0 ) {delete fVmmEvSigFrame;            fCdelete++;}

  if ( fVmaxEvSigFrame           != 0 ) {delete fVmaxEvSigFrame;           fCdelete++;}   // 090
  if ( fVmaxEvSigBut             != 0 ) {delete fVmaxEvSigBut;             fCdelete++;}
  if ( fLayoutVmaxEvSigBut       != 0 ) {delete fLayoutVmaxEvSigBut;       fCdelete++;}
  if ( fVmaxEvSigText            != 0 ) {fVmaxEvSigText->Delete();         fCdelete++;}
  if ( fEntryVmaxEvSigNumber     != 0 ) {delete fEntryVmaxEvSigNumber;     fCdelete++;}
  if ( fLayoutVmaxEvSigFieldText != 0 ) {delete fLayoutVmaxEvSigFieldText; fCdelete++;}
  if ( fLayoutVmaxEvSigFrame     != 0 ) {delete fLayoutVmaxEvSigFrame;     fCdelete++;}
 
  if ( fVminEvSigFrame           != 0 ) {delete fVminEvSigFrame;           fCdelete++;}
  if ( fVminEvSigBut             != 0 ) {delete fVminEvSigBut;             fCdelete++;}
  if ( fLayoutVminEvSigBut       != 0 ) {delete fLayoutVminEvSigBut;       fCdelete++;}
  if ( fVminEvSigText            != 0 ) {fVminEvSigText->Delete();         fCdelete++;}    // 100
  if ( fEntryVminEvSigNumber     != 0 ) {delete fEntryVminEvSigNumber;     fCdelete++;}
  if ( fLayoutVminEvSigFieldText != 0 ) {delete fLayoutVminEvSigFieldText; fCdelete++;}
  if ( fLayoutVminEvSigFrame     != 0 ) {delete fLayoutVminEvSigFrame;     fCdelete++;}
 
  if ( fMenuEvSig                != 0 ) {delete fMenuEvSig;                fCdelete++;}
  if ( fMenuBarEvSig             != 0 ) {fMenuBarEvSig->Delete();          fCdelete++;}
  if ( fLayoutMenuBarEvSig       != 0 ) {delete fLayoutMenuBarEvSig;       fCdelete++;}

  if ( fLayoutVmmEvSigFrame      != 0 ) {delete fLayoutVmmEvSigFrame;      fCdelete++;}

  //-----------------------------------------------------------
  if ( fVmmEvCorssFrame            != 0 ) {delete fVmmEvCorssFrame;            fCdelete++;}

  if ( fVmaxEvCorssFrame           != 0 ) {delete fVmaxEvCorssFrame;           fCdelete++;}
  if ( fVmaxEvCorssBut             != 0 ) {delete fVmaxEvCorssBut;             fCdelete++;}   // 110
  if ( fLayoutVmaxEvCorssBut       != 0 ) {delete fLayoutVmaxEvCorssBut;       fCdelete++;}
  if ( fVmaxEvCorssText            != 0 ) {fVmaxEvCorssText->Delete();         fCdelete++;}
  if ( fEntryVmaxEvCorssNumber     != 0 ) {delete fEntryVmaxEvCorssNumber;     fCdelete++;}
  if ( fLayoutVmaxEvCorssFieldText != 0 ) {delete fLayoutVmaxEvCorssFieldText; fCdelete++;}
  if ( fLayoutVmaxEvCorssFrame     != 0 ) {delete fLayoutVmaxEvCorssFrame;     fCdelete++;}

  if ( fVminEvCorssFrame           != 0 ) {delete fVminEvCorssFrame;           fCdelete++;}
  if ( fVminEvCorssBut             != 0 ) {delete fVminEvCorssBut;             fCdelete++;}
  if ( fLayoutVminEvCorssBut       != 0 ) {delete fLayoutVminEvCorssBut;       fCdelete++;}
  if ( fVminEvCorssText            != 0 ) {fVminEvCorssText->Delete();         fCdelete++;}
  if ( fEntryVminEvCorssNumber     != 0 ) {delete fEntryVminEvCorssNumber;     fCdelete++;}   // 120
  if ( fLayoutVminEvCorssFieldText != 0 ) {delete fLayoutVminEvCorssFieldText; fCdelete++;}
  if ( fLayoutVminEvCorssFrame     != 0 ) {delete fLayoutVminEvCorssFrame;     fCdelete++;}

  if ( fMenuEvCorss                != 0 ) {delete fMenuEvCorss;                fCdelete++;}
  if ( fMenuBarEvCorss             != 0 ) {fMenuBarEvCorss->Delete();          fCdelete++;}
  if ( fLayoutMenuBarEvCorss       != 0 ) {delete fLayoutMenuBarEvCorss;       fCdelete++;}

  if ( fLayoutVmmEvCorssFrame      != 0 ) {delete fLayoutVmmEvCorssFrame;      fCdelete++;}
  
  if ( fLayoutSuMoHozSubEvFrame    != 0 ) {delete fLayoutSuMoHozSubEvFrame;    fCdelete++;}

  //............................................... Frame Sig + Menus Sig
  if ( fSuMoHozSubSigFrame         != 0 ) {delete fSuMoHozSubSigFrame;         fCdelete++;}

  //------------------------------------------------------------- 
  if ( fVmmSigEvFrame            != 0 ) {delete fVmmSigEvFrame;            fCdelete++;}

  if ( fVmaxSigEvFrame           != 0 ) {delete fVmaxSigEvFrame;           fCdelete++;}   // 130
  if ( fVmaxSigEvBut             != 0 ) {delete fVmaxSigEvBut;             fCdelete++;}
  if ( fLayoutVmaxSigEvBut       != 0 ) {delete fLayoutVmaxSigEvBut;       fCdelete++;}
  if ( fVmaxSigEvText            != 0 ) {fVmaxSigEvText->Delete();         fCdelete++;}
  if ( fEntryVmaxSigEvNumber     != 0 ) {delete fEntryVmaxSigEvNumber;     fCdelete++;}
  if ( fLayoutVmaxSigEvFieldText != 0 ) {delete fLayoutVmaxSigEvFieldText; fCdelete++;}
  if ( fLayoutVmaxSigEvFrame     != 0 ) {delete fLayoutVmaxSigEvFrame;     fCdelete++;}

  if ( fVminSigEvFrame           != 0 ) {delete fVminSigEvFrame;           fCdelete++;}
  if ( fVminSigEvBut             != 0 ) {delete fVminSigEvBut;             fCdelete++;}
  if ( fLayoutVminSigEvBut       != 0 ) {delete fLayoutVminSigEvBut;       fCdelete++;}
  if ( fVminSigEvText            != 0 ) {fVminSigEvText->Delete();         fCdelete++;}
  if ( fEntryVminSigEvNumber     != 0 ) {delete fEntryVminSigEvNumber;     fCdelete++;}
  if ( fLayoutVminSigEvFieldText != 0 ) {delete fLayoutVminSigEvFieldText; fCdelete++;}
  if ( fLayoutVminSigEvFrame     != 0 ) {delete fLayoutVminSigEvFrame;     fCdelete++;}

  if ( fMenuSigEv                != 0 ) {delete fMenuSigEv;                fCdelete++;}
  if ( fMenuBarSigEv             != 0 ) {fMenuBarSigEv->Delete();          fCdelete++;}
  if ( fLayoutMenuBarSigEv       != 0 ) {delete fLayoutMenuBarSigEv;       fCdelete++;}

  if ( fLayoutVmmSigEvFrame      != 0 ) {delete fLayoutVmmSigEvFrame;      fCdelete++;}

  //-------------------------------------------------------------
  if ( fVmmSigSigFrame            != 0 ) {delete fVmmSigSigFrame;            fCdelete++;}

  if ( fVmaxSigSigFrame           != 0 ) {delete fVmaxSigSigFrame;           fCdelete++;}
  if ( fVmaxSigSigBut             != 0 ) {delete fVmaxSigSigBut;             fCdelete++;}   // 150
  if ( fLayoutVmaxSigSigBut       != 0 ) {delete fLayoutVmaxSigSigBut;       fCdelete++;}
  if ( fVmaxSigSigText            != 0 ) {fVmaxSigSigText->Delete();         fCdelete++;}
  if ( fEntryVmaxSigSigNumber     != 0 ) {delete fEntryVmaxSigSigNumber;     fCdelete++;}
  if ( fLayoutVmaxSigSigFieldText != 0 ) {delete fLayoutVmaxSigSigFieldText; fCdelete++;}
  if ( fLayoutVmaxSigSigFrame     != 0 ) {delete fLayoutVmaxSigSigFrame;     fCdelete++;}

  if ( fVminSigSigFrame           != 0 ) {delete fVminSigSigFrame;           fCdelete++;}
  if ( fVminSigSigBut             != 0 ) {delete fVminSigSigBut;             fCdelete++;}
  if ( fLayoutVminSigSigBut       != 0 ) {delete fLayoutVminSigSigBut;       fCdelete++;}
  if ( fVminSigSigText            != 0 ) {fVminSigSigText->Delete();         fCdelete++;}
  if ( fEntryVminSigSigNumber     != 0 ) {delete fEntryVminSigSigNumber;     fCdelete++;}
  if ( fLayoutVminSigSigFieldText != 0 ) {delete fLayoutVminSigSigFieldText; fCdelete++;}
  if ( fLayoutVminSigSigFrame     != 0 ) {delete fLayoutVminSigSigFrame;     fCdelete++;}

  if ( fMenuSigSig                != 0 ) {delete fMenuSigSig;                fCdelete++;}
  if ( fMenuBarSigSig             != 0 ) {fMenuBarSigSig->Delete();          fCdelete++;}
  if ( fLayoutMenuBarSigSig       != 0 ) {delete fLayoutMenuBarSigSig;       fCdelete++;}

  if ( fLayoutVmmSigSigFrame      != 0 ) {delete fLayoutVmmSigSigFrame;      fCdelete++;}

  //-------------------------------------------------------------
  if ( fVmmSigCorssFrame            != 0 ) {delete fVmmSigCorssFrame;            fCdelete++;}

  if ( fVmaxSigCorssFrame           != 0 ) {delete fVmaxSigCorssFrame;           fCdelete++;}
  if ( fVmaxSigCorssBut             != 0 ) {delete fVmaxSigCorssBut;             fCdelete++;}
  if ( fLayoutVmaxSigCorssBut       != 0 ) {delete fLayoutVmaxSigCorssBut;       fCdelete++;}    // 170
  if ( fVmaxSigCorssText            != 0 ) {fVmaxSigCorssText->Delete();         fCdelete++;}
  if ( fEntryVmaxSigCorssNumber     != 0 ) {delete fEntryVmaxSigCorssNumber;     fCdelete++;}
  if ( fLayoutVmaxSigCorssFieldText != 0 ) {delete fLayoutVmaxSigCorssFieldText; fCdelete++;}
  if ( fLayoutVmaxSigCorssFrame     != 0 ) {delete fLayoutVmaxSigCorssFrame;     fCdelete++;}

  if ( fVminSigCorssFrame           != 0 ) {delete fVminSigCorssFrame;           fCdelete++;}
  if ( fVminSigCorssBut             != 0 ) {delete fVminSigCorssBut;             fCdelete++;}
  if ( fLayoutVminSigCorssBut       != 0 ) {delete fLayoutVminSigCorssBut;       fCdelete++;}
  if ( fVminSigCorssText            != 0 ) {fVminSigCorssText->Delete();         fCdelete++;}
  if ( fEntryVminSigCorssNumber     != 0 ) {delete fEntryVminSigCorssNumber;     fCdelete++;}
  if ( fLayoutVminSigCorssFieldText != 0 ) {delete fLayoutVminSigCorssFieldText; fCdelete++;}
  if ( fLayoutVminSigCorssFrame     != 0 ) {delete fLayoutVminSigCorssFrame;     fCdelete++;}

  if ( fMenuSigCorss                != 0 ) {delete fMenuSigCorss;                fCdelete++;}
  if ( fMenuBarSigCorss             != 0 ) {fMenuBarSigCorss->Delete();          fCdelete++;}
  if ( fLayoutMenuBarSigCorss       != 0 ) {delete fLayoutMenuBarSigCorss;       fCdelete++;}

  if ( fLayoutVmmSigCorssFrame      != 0 ) {delete fLayoutVmmSigCorssFrame;      fCdelete++;}   // 185
  //-------------------------------------------------------------

  if ( fLayoutSuMoHozSubSigFrame    != 0 ) {delete fLayoutSuMoHozSubSigFrame;    fCdelete++;}

  if ( fLayoutSuMoHozFrame          != 0 ) {delete fLayoutSuMoHozFrame;          fCdelete++;}

  //----------------------------------------------------------------------------------------------

  //...................................... Covariances between towers
  if ( fVmmEvCovttFrame            != 0 ) {delete fVmmEvCovttFrame;            fCdelete++;}

  if ( fVmaxEvCovttFrame           != 0 ) {delete fVmaxEvCovttFrame;           fCdelete++;}
  if ( fVmaxEvCovttBut             != 0 ) {delete fVmaxEvCovttBut;             fCdelete++;}
  if ( fLayoutVmaxEvCovttBut       != 0 ) {delete fLayoutVmaxEvCovttBut;       fCdelete++;}
  if ( fVmaxEvCovttText            != 0 ) {fVmaxEvCovttText->Delete();         fCdelete++;}
  if ( fEntryVmaxEvCovttNumber     != 0 ) {delete fEntryVmaxEvCovttNumber;     fCdelete++;}
  if ( fLayoutVmaxEvCovttFieldText != 0 ) {delete fLayoutVmaxEvCovttFieldText; fCdelete++;}
  if ( fLayoutVmaxEvCovttFrame     != 0 ) {delete fLayoutVmaxEvCovttFrame;     fCdelete++;}

  if ( fVminEvCovttFrame           != 0 ) {delete fVminEvCovttFrame;           fCdelete++;}
  if ( fVminEvCovttBut             != 0 ) {delete fVminEvCovttBut;             fCdelete++;}
  if ( fLayoutVminEvCovttBut       != 0 ) {delete fLayoutVminEvCovttBut;       fCdelete++;}
  if ( fVminEvCovttText            != 0 ) {fVminEvCovttText->Delete();         fCdelete++;}
  if ( fEntryVminEvCovttNumber     != 0 ) {delete fEntryVminEvCovttNumber;     fCdelete++;}  // 200
  if ( fLayoutVminEvCovttFieldText != 0 ) {delete fLayoutVminEvCovttFieldText; fCdelete++;}
  if ( fLayoutVminEvCovttFrame     != 0 ) {delete fLayoutVminEvCovttFrame;     fCdelete++;}

  if ( fMenuCovtt                  != 0 ) {delete fMenuCovtt;                  fCdelete++;}
  if ( fMenuBarCovtt               != 0 ) {fMenuBarCovtt->Delete();            fCdelete++;}
  if ( fLayoutMenuBarCovtt         != 0 ) {delete fLayoutMenuBarCovtt;         fCdelete++;}

  if ( fLayoutVmmEvCovttFrame      != 0 ) {delete fLayoutVmmEvCovttFrame;      fCdelete++;}

  //...................................... Correlations between towers  
  if ( fVmmEvCorttFrame            != 0 ) {delete fVmmEvCorttFrame;            fCdelete++;}

  if ( fVmaxEvCorttFrame           != 0 ) {delete fVmaxEvCorttFrame;           fCdelete++;}
  if ( fVmaxEvCorttBut             != 0 ) {delete fVmaxEvCorttBut;             fCdelete++;}
  if ( fLayoutVmaxEvCorttBut       != 0 ) {delete fLayoutVmaxEvCorttBut;       fCdelete++;}   // 210
  if ( fVmaxEvCorttText            != 0 ) {fVmaxEvCorttText->Delete();         fCdelete++;}
  if ( fEntryVmaxEvCorttNumber     != 0 ) {delete fEntryVmaxEvCorttNumber;     fCdelete++;}
  if ( fLayoutVmaxEvCorttFieldText != 0 ) {delete fLayoutVmaxEvCorttFieldText; fCdelete++;}
  if ( fLayoutVmaxEvCorttFrame     != 0 ) {delete fLayoutVmaxEvCorttFrame;     fCdelete++;}

  if ( fVminEvCorttFrame           != 0 ) {delete fVminEvCorttFrame;           fCdelete++;}
  if ( fVminEvCorttBut             != 0 ) {delete fVminEvCorttBut;             fCdelete++;}
  if ( fLayoutVminEvCorttBut       != 0 ) {delete fLayoutVminEvCorttBut;       fCdelete++;}
  if ( fVminEvCorttText            != 0 ) {fVminEvCorttText->Delete();         fCdelete++;}
  if ( fEntryVminEvCorttNumber     != 0 ) {delete fEntryVminEvCorttNumber;     fCdelete++;}
  if ( fLayoutVminEvCorttFieldText != 0 ) {delete fLayoutVminEvCorttFieldText; fCdelete++;}
  if ( fLayoutVminEvCorttFrame     != 0 ) {delete fLayoutVminEvCorttFrame;     fCdelete++;}

  if ( fMenuCortt                  != 0 ) {delete fMenuCortt;                  fCdelete++;}
  if ( fMenuBarCortt               != 0 ) {fMenuBarCortt->Delete();            fCdelete++;}
  if ( fLayoutMenuBarCortt         != 0 ) {delete fLayoutMenuBarCortt;         fCdelete++;}

  if ( fLayoutVmmEvCorttFrame      != 0 ) {delete fLayoutVmmEvCorttFrame;      fCdelete++;}

  if ( fLayoutSuMoUpFrame          != 0 ) {delete fLayoutSuMoUpFrame;          fCdelete++;}

  //+++++++++++++++++++++++++++++++++++++++++ Horizontal frame Tower_X + Tower_Y
  if ( fTowSpFrame       != 0 ) {delete fTowSpFrame;         fCdelete++;}
  
  //----------------------------------- SubFrame Tower_X (Button + EntryField)
  if ( fTxSubFrame       != 0 ) {delete fTxSubFrame;         fCdelete++;}

  if ( fTowXFrame        != 0 ) {delete fTowXFrame;          fCdelete++;}
  if ( fTowXBut          != 0 ) {delete fTowXBut;            fCdelete++;}      // 230
  if ( fLayoutTowXBut    != 0 ) {delete fLayoutTowXBut;      fCdelete++;} 
  if ( fEntryTowXNumber  != 0 ) {delete fEntryTowXNumber;    fCdelete++;}
  if ( fTowXText         != 0 ) {fTowXText->Delete();        fCdelete++;} 
  if ( fLayoutTowXField  != 0 ) {delete fLayoutTowXField;    fCdelete++;} 

  //............................ Tower Crystal Numbering view (Button)
  if ( fButChNb          != 0 ) {delete fButChNb;            fCdelete++;}
  if ( fLayoutChNbBut    != 0 ) {delete fLayoutChNbBut;      fCdelete++;} 

  //............................ Menus Tower_X
  if ( fMenuCorssAll     != 0 ) {delete fMenuCorssAll;       fCdelete++;}
  if ( fMenuBarCorssAll  != 0 ) {fMenuBarCorssAll->Delete(); fCdelete++;}

  if ( fMenuCovssAll     != 0 ) {delete fMenuCovssAll;       fCdelete++;}
  if ( fMenuBarCovssAll  != 0 ) {fMenuBarCovssAll->Delete(); fCdelete++;}      // 240

  if ( fLayoutTxSubFrame != 0 ) {delete fLayoutTxSubFrame;   fCdelete++;}

  //----------------------------------- SubFrame Tower_Y (Button + EntryField)

  if ( fTySubFrame       != 0 ) {delete fTySubFrame;        fCdelete++;}

  if ( fTowYFrame        != 0 ) {delete fTowYFrame;         fCdelete++;}
  if ( fTowYBut          != 0 ) {delete fTowYBut;           fCdelete++;}
  if ( fLayoutTowYBut    != 0 ) {delete fLayoutTowYBut;     fCdelete++;}
  if ( fEntryTowYNumber  != 0 ) {delete fEntryTowYNumber;   fCdelete++;}
  if ( fTowYText         != 0 ) {fTowYText->Delete();       fCdelete++;}
  if ( fLayoutTowYField  != 0 ) {delete fLayoutTowYField;   fCdelete++;}

  if ( fLayoutTySubFrame != 0 ) {delete fLayoutTySubFrame;  fCdelete++;}

  if ( fLayoutTowSpFrame != 0 ) {delete fLayoutTowSpFrame;  fCdelete++;}

  //.................................. Menus for Horizontal frame (Tower_X + Tower_Y)

  if ( fMenuCorcc        != 0 ) {delete fMenuCorcc;         fCdelete++;}
  if ( fMenuBarCorcc     != 0 ) {fMenuBarCorcc->Delete();   fCdelete++;}

  if ( fMenuCovcc        != 0 ) {delete fMenuCovcc;         fCdelete++;}
  if ( fMenuBarCovcc     != 0 ) {fMenuBarCovcc->Delete();   fCdelete++;}

  //++++++++++++++++++++++++ Horizontal frame channel number (tower_X crystal number) + sample number
  if ( fChSpFrame        != 0 ) {delete fChSpFrame;         fCdelete++;}

  //------------------------------------- SubFrame Channel (Button + EntryField)

  if ( fChSubFrame       != 0 ) {delete fChSubFrame;        fCdelete++;}

  if ( fChanFrame        != 0 ) {delete fChanFrame;         fCdelete++;}
  if ( fChanBut          != 0 ) {delete fChanBut;           fCdelete++;}
  if ( fLayoutChanBut    != 0 ) {delete fLayoutChanBut;     fCdelete++;}
  if ( fEntryChanNumber  != 0 ) {delete fEntryChanNumber;   fCdelete++;}    // 260
  if ( fChanText         != 0 ) {fChanText->Delete();       fCdelete++;}
  if ( fLayoutChanField  != 0 ) {delete fLayoutChanField;   fCdelete++;}

  //................................ Menus Tower_X crystal number
  if ( fMenuCorss        != 0 ) {delete fMenuCorss;         fCdelete++;}
  if ( fMenuBarCorss     != 0 ) {fMenuBarCorss->Delete();   fCdelete++;}

  if ( fMenuCovss        != 0 ) {delete fMenuCovss;         fCdelete++;}
  if ( fMenuBarCovss     != 0 ) {fMenuBarCovss->Delete();   fCdelete++;}

  if ( fMenuEv           != 0 ) {delete fMenuEv;            fCdelete++;}
  if ( fMenuBarEv        != 0 ) {fMenuBarEv->Delete();      fCdelete++;}

  if ( fMenuVar          != 0 ) {delete fMenuVar;           fCdelete++;}
  if ( fMenuBarVar       != 0 ) {fMenuBarVar->Delete();     fCdelete++;}    // 270

  if ( fLayoutChSubFrame != 0 ) {delete fLayoutChSubFrame;  fCdelete++;}

  //------------------------------------ SubFrame Sample (Button + EntryField)
  if ( fSpSubFrame       != 0 ) {delete fSpSubFrame;        fCdelete++;}
  if ( fSampFrame        != 0 ) {delete fSampFrame;         fCdelete++;}
  if ( fSampBut          != 0 ) {delete fSampBut;           fCdelete++;}
  if ( fLayoutSampBut    != 0 ) {delete fLayoutSampBut;     fCdelete++;}
  if ( fEntrySampNumber  != 0 ) {delete fEntrySampNumber;   fCdelete++;}
  if ( fSampText         != 0 ) {fSampText->Delete();       fCdelete++;}
  if ( fLayoutSampField  != 0 ) {delete fLayoutSampField;   fCdelete++;}
  if ( fLayoutSpSubFrame != 0 ) {delete fLayoutSpSubFrame;  fCdelete++;}

  //................................ Menus Sample number

  //     (no menu in this SubFrame)

  if ( fLayoutChSpFrame  != 0 ) {delete fLayoutChSpFrame;   fCdelete++;}    // 280


  //++++++++++++++++++++++++++++++++++++ Menu Event Distribution
  if ( fMenuEvts            != 0 ) {delete fMenuEvts;            fCdelete++;}
  if ( fMenuBarEvts         != 0 ) {fMenuBarEvts->Delete();      fCdelete++;}
  if ( fLayoutMenuBarEvts   != 0 ) {delete fLayoutMenuBarEvts;   fCdelete++;}

  //++++++++++++++++++++++++++++++++++++ Frame: Run List (Rul) (Button + EntryField)

  if ( fRulFrame            != 0 ) {delete fRulFrame;            fCdelete++;}
  if ( fRulBut              != 0 ) {delete fRulBut;              fCdelete++;}
  if ( fLayoutRulBut        != 0 ) {delete fLayoutRulBut;        fCdelete++;}
  if ( fEntryRulNumber      != 0 ) {delete fEntryRulNumber;      fCdelete++;}
  if ( fRulText             != 0 ) {fRulText->Delete();          fCdelete++;}
  if ( fLayoutRulFieldText  != 0 ) {delete fLayoutRulFieldText;  fCdelete++;}
  if ( fLayoutRulFieldFrame != 0 ) {delete fLayoutRulFieldFrame; fCdelete++;}     // 290

  //................................ Menus for time evolution
  if ( fMenuEvol            != 0 ) {delete fMenuEvol;            fCdelete++;}
  if ( fMenuBarEvol         != 0 ) {fMenuBarEvol->Delete();      fCdelete++;}


  //++++++++++++++++++++++++++++++++++++ LinLog Frame
  if ( fLinLogFrame   != 0 ) {delete fLinLogFrame;   fCdelete++;}

  //---------------------------------- Lin/Log X
  if ( fButLogx       != 0 ) {delete fButLogx;       fCdelete++;}
  if ( fLayoutLogxBut != 0 ) {delete fLayoutLogxBut; fCdelete++;}
  //---------------------------------- Lin/Log Y
  if ( fButLogy       != 0 ) {delete fButLogy;       fCdelete++;}
  if ( fLayoutLogyBut != 0 ) {delete fLayoutLogyBut; fCdelete++;} 

  //++++++++++++++++++++++++++++++++++++ EXIT BUTTON
  if ( fButExit       != 0 ) {delete fButExit;       fCdelete++;}
  if ( fLayoutExitBut != 0 ) {delete fLayoutExitBut; fCdelete++;}
 
  //++++++++++++++++++++++++++++++++++++ Last Frame
  if ( fLastFrame     != 0 ) {delete fLastFrame;     fCdelete++;}     // 300

  //--------------------------------- Root version (Button)
  if ( fButRoot       != 0 ) {delete fButRoot;       fCdelete++;}
  if ( fLayoutRootBut != 0 ) {delete fLayoutRootBut; fCdelete++;}

  //--------------------------------- Help (Button)
  if ( fButHelp       != 0 ) {delete fButHelp;       fCdelete++;}
  if ( fLayoutHelpBut != 0 ) {delete fLayoutHelpBut; fCdelete++;}     // 304 

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#define MMOA
#ifndef MMOA
  if ( fCnew != fCdelete )
    {
      cout << "*TCnaDialogEB> WRONG MANAGEMENT OF ALLOCATIONS: fCnew = "
	   << fCnew << ", fCdelete = " << fCdelete << endl;
    }
  else
    {
      cout << "*TCnaDialogEB> BRAVO! GOOD MANAGEMENT OF ALLOCATIONS: fCnew = "
	   << fCnew << ", fCdelete = " << fCdelete << endl;
    }


  if ( fCnewRoot != fCdeleteRoot )
    {
      cout << "*TCnaDialogEB> WRONG MANAGEMENT OF ROOT ALLOCATIONS: fCnewRoot = "
	   << fCnewRoot << ", fCdeleteRoot = " << fCdeleteRoot << endl;
    }
  else
    {
      cout << "*TCnaDialogEB> BRAVO! GOOD MANAGEMENT OF ROOT ALLOCATIONS:"
	   << " fCnewRoot = " << fCnewRoot <<", fCdeleteRoot = "
	   << fCdeleteRoot << endl;
    }
#endif // MMOA

  // cout << "TCnaDialogEB> Leaving destructor" << endl;
  // cout << "            fCnew = " << fCnew << ", fCdelete = " << fCdelete << endl;

}
//   end of destructor

//===================================================================
//
//                   Constructor with arguments
//
//===================================================================
TCnaDialogEB::TCnaDialogEB(const TGWindow *p, UInt_t w, UInt_t h):
  TGMainFrame(p, w, h) 
{
//Constructor with arguments. Dialog box making

  // cout << "TCnaDialogEB> Entering constructor with arguments" << endl;
  // cout << "            fCnew = " << fCnew << ", fCdelete = " << fCdelete << endl;

//========================= GENERAL INITIALISATION 

  fCnew        = 0;
  fCdelete     = 0;
  fCnewRoot    = 0;
  fCdeleteRoot = 0;

  fCnaP = (TGWindow *)p;
  fCnaW = w;
  fCnaH = h;

  fgMaxCar = (Int_t)512;

  //------------------------------ initialisations ----------------------

  fTTBELL = '\007';

  //........................ init CNA parameters

  fParameters = new TCnaParameters();  fCnew++;
  fParameters->SetPeriodTitles();               // define the titles of the different periods of run

  fView = new TCnaViewEB();              fCnew++;

  //................ Init CNA Command and error numbering
  
  fCnaCommand = 0;
  fCnaError   = 0;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Init GUI DIALOG BOX pointers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

  fLayoutGeneral      = 0;
  fLayoutBottLeft     = 0;
  fLayoutBottRight    = 0;
  fLayoutTopLeft      = 0;
  fLayoutTopRight     = 0;  
  fLayoutCenterYLeft  = 0;
  fLayoutCenterYRight = 0; 

  fVoidFrame = 0;
  
  //++++++++++++++++++++++++++++++++++++++++++ Horizontal frame Ana + Run 
  fAnaRunFrame       = 0;
  fLayoutAnaRunFrame = 0;

  //--------------------------------- Sub-Frame Analysis Name (Button+Entry Field)
  fAnaFrame       = 0;
  fAnaBut         = 0;
  fLayoutAnaBut   = 0;
  fAnaText        = 0;
  fEntryAnaNumber = 0;
  fLayoutAnaField = 0;

  //-------------------------------- Sub-Frame Run number (Button+Entry Field)  
  fRunFrame       = 0;  
  fRunBut         = 0;
  fLayoutRunBut   = 0;
  fRunText        = 0;
  fEntryRunNumber = 0;
  fLayoutRunField = 0;

  //+++++++++++++++++++++++++++++++++++++++++++ Frames: first taken evt + nb of events + super-module
  fFevFrame            = 0;
  fFevBut              = 0;
  fLayoutFevBut        = 0;
  fFevText             = 0;
  fEntryFevNumber      = 0;
  fLayoutFevFieldText  = 0;
  fLayoutFevFieldFrame = 0;

  fNoeFrame            = 0;
  fNoeBut              = 0;
  fLayoutNoeBut        = 0;
  fNoeText             = 0;
  fEntryNoeNumber      = 0;
  fLayoutNoeFieldText  = 0;
  fLayoutNoeFieldFrame = 0;
 
  fSuMoFrame            = 0;
  fSuMoBut              = 0;
  fLayoutSuMoBut        = 0;
  fSuMoText             = 0;
  fEntrySuMoNumber      = 0;
  fLayoutSuMoFieldText  = 0;
  fLayoutSuMoFieldFrame = 0;

  //............................ SuperModule Tower Numbering view (Button)
  fButSMNb       = 0;
  fLayoutSMNbBut = 0;

  //+++++++++++++++++++++++++++++++++++++++++++ Frame for quantities relative to the Super-Module
  fSuMoUpFrame       = 0; 

  //................................ Menus+Ymin+Ymax for the Super-Module ............................

  //...................................... Found evts in the data

  fVmmFoundEvtsFrame       = 0;

  fVmaxFoundEvtsFrame           = 0;
  fVmaxFoundEvtsBut             = 0;
  fLayoutVmaxFoundEvtsBut       = 0;
  fVmaxFoundEvtsText            = 0;
  fEntryVmaxFoundEvtsNumber     = 0;
  fLayoutVmaxFoundEvtsFieldText = 0;
  fLayoutVmaxFoundEvtsFrame     = 0;

  fVminFoundEvtsFrame           = 0;
  fVminFoundEvtsBut             = 0;
  fLayoutVminFoundEvtsBut       = 0;
  fVminFoundEvtsText            = 0;
  fEntryVminFoundEvtsNumber     = 0;
  fLayoutVminFoundEvtsFieldText = 0;
  fLayoutVminFoundEvtsFrame     = 0;

  fMenuFoundEvts           = 0;
  fMenuBarFoundEvts        = 0;
  fLayoutMenuBarFoundEvts  = 0;

  fLayoutVmmFoundEvtsFrame = 0;

  //................................... Horizontal frame ev + sig
  fSuMoHozFrame       = 0; 
  fLayoutSuMoHozFrame = 0;

  //............................................... Frame Ev + Menus Ev
  fSuMoHozSubEvFrame       = 0; 
  fLayoutSuMoHozSubEvFrame = 0;

  //-------------------------------------------------------------
  fVmmEvEvFrame       = 0;

  fVmaxEvEvFrame           = 0;
  fVmaxEvEvBut             = 0;
  fLayoutVmaxEvEvBut       = 0;
  fVmaxEvEvText            = 0;
  fEntryVmaxEvEvNumber     = 0;
  fLayoutVmaxEvEvFieldText = 0;
  fLayoutVmaxEvEvFrame     = 0;

  fVminEvEvFrame           = 0;
  fVminEvEvBut             = 0;
  fLayoutVminEvEvBut       = 0;
  fVminEvEvText            = 0;
  fEntryVminEvEvNumber     = 0;
  fLayoutVminEvEvFieldText = 0;
  fLayoutVminEvEvFrame     = 0;

  fMenuEvEv          = 0;
  fMenuBarEvEv       = 0;
  fLayoutMenuBarEvEv = 0;

  fLayoutVmmEvEvFrame = 0;

  //-------------------------------------------------------------
  fVmmEvSigFrame       = 0;

  fVmaxEvSigFrame           = 0;
  fVmaxEvSigBut             = 0;
  fLayoutVmaxEvSigBut       = 0;
  fVmaxEvSigText            = 0;
  fEntryVmaxEvSigNumber     = 0;
  fLayoutVmaxEvSigFieldText = 0;

  fVminEvSigFrame           = 0;
  fVminEvSigBut             = 0;
  fLayoutVminEvSigBut       = 0;
  fVminEvSigText            = 0;
  fEntryVminEvSigNumber     = 0;
  fLayoutVminEvSigFieldText = 0;
  fLayoutVminEvSigFrame     = 0;

  fMenuEvSig          = 0;
  fMenuBarEvSig       = 0;
  fLayoutMenuBarEvSig = 0;
  fLayoutVmaxEvSigFrame     = 0;

  fLayoutVmmEvSigFrame = 0;

  //-------------------------------------------------------------
  fVmmEvCorssFrame       = 0;

  fVmaxEvCorssFrame           = 0;
  fVmaxEvCorssBut             = 0;
  fLayoutVmaxEvCorssBut       = 0;
  fVmaxEvCorssText            = 0;
  fEntryVmaxEvCorssNumber     = 0;
  fLayoutVmaxEvCorssFieldText = 0;
  fLayoutVmaxEvCorssFrame     = 0;

  fVminEvCorssFrame           = 0;
  fVminEvCorssBut             = 0;
  fLayoutVminEvCorssBut       = 0;
  fVminEvCorssText            = 0;
  fEntryVminEvCorssNumber     = 0;
  fLayoutVminEvCorssFieldText = 0;
  fLayoutVminEvCorssFrame     = 0;

  fMenuEvCorss          = 0;
  fMenuBarEvCorss       = 0;
  fLayoutMenuBarEvCorss = 0;
  fLayoutVmmEvCorssFrame = 0;

  //............................................... Frame Sig + Menus Sig
  fSuMoHozSubSigFrame       = 0; 

  //-------------------------------------------------------------
  fVmmSigEvFrame       = 0;

  fVmaxSigEvFrame           = 0;
  fVmaxSigEvBut             = 0;
  fLayoutVmaxSigEvBut       = 0;
  fVmaxSigEvText            = 0;
  fEntryVmaxSigEvNumber     = 0;
  fLayoutVmaxSigEvFieldText = 0;
  fLayoutVmaxSigEvFrame     = 0;

  fVminSigEvFrame           = 0;
  fVminSigEvBut             = 0;
  fLayoutVminSigEvBut       = 0;
  fVminSigEvText            = 0;
  fEntryVminSigEvNumber     = 0;
  fLayoutVminSigEvFieldText = 0;
  fLayoutVminSigEvFrame     = 0;

  fMenuSigEv          = 0;
  fMenuBarSigEv       = 0;
  fLayoutMenuBarSigEv = 0;

  fLayoutVmmSigEvFrame = 0;

  //-------------------------------------------------------------
  fVmmSigSigFrame       = 0;

  fVmaxSigSigFrame           = 0;
  fVmaxSigSigBut             = 0;
  fLayoutVmaxSigSigBut       = 0;
  fVmaxSigSigText            = 0;
  fEntryVmaxSigSigNumber     = 0;
  fLayoutVmaxSigSigFieldText = 0;
  fLayoutVmaxSigSigFrame     = 0;

  fVminSigSigFrame           = 0;
  fVminSigSigBut             = 0;
  fLayoutVminSigSigBut       = 0;
  fVminSigSigText            = 0;
  fEntryVminSigSigNumber     = 0;
  fLayoutVminSigSigFieldText = 0;
  fLayoutVminSigSigFrame     = 0;

  fMenuSigSig          = 0; 
  fMenuBarSigSig       = 0;
  fLayoutMenuBarSigSig = 0;

  fLayoutVmmSigSigFrame = 0;

  //-------------------------------------------------------------
  fVmmSigCorssFrame       = 0;

  fVmaxSigCorssFrame           = 0;
  fVmaxSigCorssBut             = 0;
  fLayoutVmaxSigCorssBut       = 0;
  fVmaxSigCorssText            = 0;
  fEntryVmaxSigCorssNumber     = 0;
  fLayoutVmaxSigCorssFieldText = 0;
  fLayoutVmaxSigCorssFrame     = 0;

  fVminSigCorssFrame           = 0;
  fVminSigCorssBut             = 0;
  fLayoutVminSigCorssBut       = 0;
  fVminSigCorssText            = 0;
  fEntryVminSigCorssNumber     = 0;
  fLayoutVminSigCorssFieldText = 0;
  fLayoutVminSigCorssFrame     = 0;

  fMenuSigCorss          = 0;
  fMenuBarSigCorss       = 0;
  fLayoutMenuBarSigCorss = 0;

  fLayoutVmmSigCorssFrame = 0;

  //----------------------------------------------------------------------------------

  //...................................... Correlations between towers
  fVmmEvCorttFrame       = 0;

  fVmaxEvCorttFrame           = 0;
  fVmaxEvCorttBut             = 0;
  fLayoutVmaxEvCorttBut       = 0;
  fVmaxEvCorttText            = 0;
  fEntryVmaxEvCorttNumber     = 0;
  fLayoutVmaxEvCorttFieldText = 0;
  fLayoutVmaxEvCorttFrame     = 0;

  fVminEvCorttFrame           = 0;
  fVminEvCorttBut             = 0;
  fLayoutVminEvCorttBut       = 0;
  fVminEvCorttText            = 0;
  fEntryVminEvCorttNumber     = 0;
  fLayoutVminEvCorttFieldText = 0;
  fLayoutVminEvCorttFrame     = 0;

  fMenuCortt          = 0;
  fMenuBarCortt       = 0;
  fLayoutMenuBarCortt = 0;

  fLayoutVmmEvCorttFrame = 0;

  //...................................... Covariances between towers
  fVmmEvCovttFrame       = 0;

  fVmaxEvCovttFrame           = 0;
  fVmaxEvCovttBut             = 0;
  fLayoutVmaxEvCovttBut       = 0;
  fVmaxEvCovttText            = 0;
  fEntryVmaxEvCovttNumber     = 0;
  fLayoutVmaxEvCovttFieldText = 0;
  fLayoutVmaxEvCovttFrame     = 0;

  fVminEvCovttFrame           = 0;
  fVminEvCovttBut             = 0;
  fLayoutVminEvCovttBut       = 0;
  fVminEvCovttText            = 0;
  fEntryVminEvCovttNumber     = 0;
  fLayoutVminEvCovttFieldText = 0;
  fLayoutVminEvCovttFrame     = 0;

  fMenuCovtt          = 0;
  fMenuBarCovtt       = 0;
  fLayoutMenuBarCovtt = 0;

  fLayoutVmmEvCovttFrame = 0;

  fLayoutSuMoHozSubSigFrame = 0;

  fLayoutSuMoUpFrame = 0;

  //+++++++++++++++++++++++++++++++++++++++++ Horizontal frame Tower_X + Tower_Y
  fTowSpFrame      = 0;
  fLayoutTowSpFrame = 0;
  
  //----------------------------------- SubFrame Tower_X (Button + EntryField)

  fTxSubFrame       = 0; 
  fLayoutTxSubFrame = 0;

  fTowXFrame     = 0;
  fTowXBut       = 0;
  fLayoutTowXBut = 0; 

  fTowXText        = 0;
  fEntryTowXNumber = 0; 
  fLayoutTowXField = 0;
  
  //............................ Tower Crystal Numbering view (Button)
  fButChNb       = 0;
  fLayoutChNbBut = 0;

  //............................ Menus Tower_X
  fMenuCorssAll    = 0;
  fMenuBarCorssAll = 0;

  fMenuCovssAll    = 0;
  fMenuBarCovssAll = 0;

  //----------------------------------- SubFrame Tower_Y (Button + EntryField)
  fTySubFrame       = 0;
  fLayoutTySubFrame = 0;

  fTowYFrame     = 0;
  fTowYBut       = 0;
  fLayoutTowYBut = 0;

  fTowYText        = 0;  
  fEntryTowYNumber = 0;
  fLayoutTowYField = 0;

  //.................................. Menus for Horizontal frame (Tower_X + Tower_Y)
  fMenuBarCorcc = 0;
  fMenuCorcc    = 0; 

  fMenuBarCovcc = 0;  
  fMenuCovcc    = 0;

  //++++++++++++++++++++++++ Horizontal frame channel number (tower_X crystal number) + sample number
  fChSpFrame       = 0;
  fLayoutChSpFrame = 0;

  //------------------------------------- SubFrame Channel (Button + EntryField)
  fChanFrame       = 0;
  fChanBut         = 0;
  fChanText        = 0;
  fEntryChanNumber = 0;
  fLayoutChanBut   = 0;
  fLayoutChanField = 0;

  fChSubFrame       = 0;
  fLayoutChSubFrame = 0;

  //................................ Menus Tower_X crystal number
  fMenuCorss    = 0;
  fMenuBarCorss = 0;

  fMenuCovss    = 0; 
  fMenuBarCovss = 0;

  fMenuEv    = 0;
  fMenuBarEv = 0;

  fMenuVar    = 0;
  fMenuBarVar = 0;

  //------------------------------------ SubFrame Sample (Button + EntryField)
  fSampFrame = 0;
  fSampBut   = 0;

  fSampText        = 0;  
  fEntrySampNumber = 0;
  fLayoutSampBut   = 0;
  fLayoutSampField = 0;

  fSpSubFrame       = 0;
  fLayoutSpSubFrame = 0;

  //................................ Menus Sample number

  //     (no menu in this SubFrame)

  //++++++++++++++++++++++++++++++++++++ Frame: Run List (Rul) (Button + EntryField)

  fRulFrame            = 0;
  fRulBut              = 0;
  fLayoutRulBut        = 0;
  fRulText             = 0;
  fEntryRulNumber      = 0;
  fLayoutRulFieldText  = 0;
  fLayoutRulFieldFrame = 0;

  //................................ Menus for time evolution
  fMenuEvol    = 0;
  fMenuBarEvol = 0;

  //++++++++++++++++++++++++++++++++++++ Menu Event Distribution
  fMenuEvts          = 0;
  fMenuBarEvts       = 0;
  fLayoutMenuBarEvts = 0;

  //++++++++++++++++++++++++++++++++++++ LinLog Frame
  fLinLogFrame = 0;  

  //---------------------------------- Lin/Log X
  fButLogx       = 0;
  fLayoutLogxBut = 0;
  //---------------------------------- Lin/Log Y
  fButLogy       = 0;
  fLayoutLogyBut = 0;

  //++++++++++++++++++++++++++++++++++++ EXIT BUTTON
  fButExit       = 0;   
  fLayoutExitBut = 0;

  //++++++++++++++++++++++++++++++++++++ Last Frame
  fLastFrame = 0;   

  //--------------------------------- Root version (Button)
  fButRoot       = 0;
  fLayoutRootBut = 0;

  //--------------------------------- Help (Button)
  fButHelp       = 0;
  fLayoutHelpBut = 0;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of init GUI DIALOG BOX pointers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  //........ Init Buttons codes with input widgets:
  //         run, channel, sample

  fAnaButC  = 1;
  fRunButC  = 2;

   //.................. Init codes Menu bars

  fMenuFoundEvtsGlobalFullC = 600101;
  fMenuFoundEvtsGlobalSameC = 600102;

  fMenuFoundEvtsProjFullC   = 600111;
  fMenuFoundEvtsProjSameC   = 600112;

  fMenuEvEvGlobalFullC      = 123051;
  fMenuEvEvGlobalSameC      = 123052;

  fMenuEvSigGlobalFullC     = 123061;
  fMenuEvSigGlobalSameC     = 123062;

  fMenuEvCorssGlobalFullC   = 123071;
  fMenuEvCorssGlobalSameC   = 123072;

  fMenuEvEvProjFullC        = 124051;
  fMenuEvEvProjSameC        = 124052;

  fMenuEvSigProjFullC       = 124061;
  fMenuEvSigProjSameC       = 124062;

  fMenuEvCorssProjFullC     = 124071;
  fMenuEvCorssProjSameC     = 124072;

  fMenuSigEvGlobalFullC     = 800051;
  fMenuSigEvGlobalSameC     = 800052;

  fMenuSigSigGlobalFullC    = 800061;
  fMenuSigSigGlobalSameC    = 800062;

  fMenuSigCorssGlobalFullC  = 800071;
  fMenuSigCorssGlobalSameC  = 800072;

  fMenuSigEvProjFullC       = 810051;
  fMenuSigEvProjSameC       = 810052;

  fMenuSigSigProjFullC      = 810061;
  fMenuSigSigProjSameC      = 810062;

  fMenuSigCorssProjFullC    = 810071;
  fMenuSigCorssProjSameC    = 810072;

  fMenuCovttColzC           = 70010;
  fMenuCovttLegoC           = 70011;

  fMenuCorttColzC           = 70110;
  fMenuCorttLegoC           = 70111;

  fMenuFoundEvtsEtaPhiC     = 524051;
  fMenuEvEvEtaPhiC          = 524052;
  fMenuEvSigEtaPhiC         = 524053;
  fMenuEvCorssEtaPhiC       = 524054;
  fMenuSigEvEtaPhiC         = 524055;
  fMenuSigSigEtaPhiC        = 524056;
  fMenuSigCorssEtaPhiC      = 524057;

  fTowXButC = 90009; 
  fTowYButC = 90010;

  fChanButC = 6;
  fSampButC = 7;

  fMenuCorssAllColzC = 10;
  fMenuCovssAllColzC = 11;
 
  fMenuCorssColzC    = 21;
  fMenuCorssLegoC    = 22;
  fMenuCorssSurf1C   = 23;
  fMenuCorssSurf4C   = 24;
  
  fMenuCovssColzC    = 31;
  fMenuCovssLegoC    = 32;
  fMenuCovssSurf1C   = 33;
  fMenuCovssSurf4C   = 34;
  
  fMenuEvLineFullC   = 411;
  fMenuEvLineSameC   = 412;

  fMenuVarLineFullC  = 421;
  fMenuVarLineSameC  = 422;
  
  fMenuCorccColzC    = 51;
  fMenuCorccLegoC    = 52;

  fMenuCovccColzC    = 61;
  fMenuCovccLegoC    = 62;
  
  fMenuEvtsLineLinyFullC = 711;
  fMenuEvtsLineLinySameC = 712;

  fMenuEvolEvEvPolmFullC    = 811;
  fMenuEvolEvEvPolmSameC    = 812;
  fMenuEvolEvSigPolmFullC   = 821;
  fMenuEvolEvSigPolmSameC   = 822;
  fMenuEvolEvCorssPolmFullC = 831;
  fMenuEvolEvCorssPolmSameC = 832;
  fMenuEvolSampLineFullC    = 841;
  fMenuEvolSampLineSameC    = 842;

 //...................... Init Button codes: Root version, Help, Exit

  fButSMNbC     = 90;
  fButChNbC     = 91;
  fButRootC     = 92;
  fButHelpC     = 93;
  fButExitC     = 94;
  
  //................. Init Keys

  InitKeys();

  GetPathForResultsRootFiles();
  GetPathForListOfRunFiles();

  //=================================== LIN/LOG flags
  Int_t MaxCar = fgMaxCar;
  fMemoScaleX.Resize(MaxCar);   
  fMemoScaleX = "LIN";
  MaxCar = fgMaxCar;
  fMemoScaleY.Resize(MaxCar); 
  fMemoScaleY = "LIN";

  //=================================== Init option codes =================================

  MaxCar = fgMaxCar;
  fOptPlotFull.Resize(MaxCar);
  fOptPlotFull = "ONLYONE";

  MaxCar = fgMaxCar;
  fOptPlotSame.Resize(MaxCar);
  fOptPlotSame = "SEVERAL";

  //================================================================================================

  //-------------------------------------------------------------------------
  //
  //
  //                      B O X     M A K I N G
  //
  //
  //-------------------------------------------------------------------------

  // enum ELayoutHints {
  //   kLHintsNoHints = 0,
  //   kLHintsLeft    = BIT(0),
  //   kLHintsCenterX = BIT(1),
  //   kLHintsRight   = BIT(2),
  //   kLHintsTop     = BIT(3),
  //   kLHintsCenterY = BIT(4),
  //   kLHintsBottom  = BIT(5),
  //   kLHintsExpandX = BIT(6),
  //   kLHintsExpandY = BIT(7),
  //   kLHintsNormal  = (kLHintsLeft | kLHintsTop)
  //   bits 8-11 used by ETableLayoutHints
  // };

  fLayoutGeneral      = new TGLayoutHints (kLHintsCenterX | kLHintsCenterY);   fCnew++;  // 003
                                                                                         // (after fParameters
                                                                                         //  and fCView)
  fLayoutBottLeft     = new TGLayoutHints (kLHintsLeft    | kLHintsBottom);    fCnew++;
  fLayoutTopLeft      = new TGLayoutHints (kLHintsLeft    | kLHintsTop);       fCnew++;
  fLayoutBottRight    = new TGLayoutHints (kLHintsRight   | kLHintsBottom);    fCnew++;
  fLayoutTopRight     = new TGLayoutHints (kLHintsRight   | kLHintsTop);       fCnew++;
  fLayoutCenterYLeft  = new TGLayoutHints (kLHintsLeft    | kLHintsCenterY);   fCnew++;  
  fLayoutCenterYRight = new TGLayoutHints (kLHintsRight   | kLHintsCenterY);   fCnew++;  

  fVoidFrame = new TGCompositeFrame(this,60,20, kVerticalFrame, kSunkenFrame);  fCnew++;   // 010
  AddFrame(fVoidFrame, fLayoutGeneral);

  //......................... Pads border width

  Int_t xB1 = 1;

  //=============================================== Button Texts
  TString xAnaButText      = " Analysis name-> ";
  TString xRunButText      = "   Run number ->   ";

  //  TString xPathButText     = " Path (optional) ";

  //...................................... lenghts of the input widgets
  Int_t run_buf_lenght        = 55;
  Int_t first_evt_buf_lenght  = 60;
  Int_t nb_of_evts_buf_lenght = 60;
  Int_t typ_of_ana_buf_lenght = 120;

  //==================== Ana + Run FRAME ====================================

  fAnaRunFrame = new TGCompositeFrame(this,60,20,kHorizontalFrame,
				      GetDefaultFrameBackground());   fCnew++;

 //=================================== ANALYSIS NAME (type of analysis)
  
  fAnaFrame =  new TGCompositeFrame(fAnaRunFrame,60,20, kHorizontalFrame,
				    kSunkenFrame);                    fCnew++;
  
  //..................... Button  
  fAnaBut = new TGTextButton(fAnaFrame, xAnaButText, fAnaButC);       fCnew++;
  fAnaBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonAna()");
  fAnaBut->Resize(typ_of_ana_buf_lenght, fAnaBut->GetDefaultHeight());
  fAnaBut->SetToolTipText("Click here to register the analysis name written on the right");
  fLayoutAnaBut =
    new TGLayoutHints(kLHintsLeft | kLHintsTop, xB1,xB1,xB1,xB1);     fCnew++;
  fAnaFrame->AddFrame(fAnaBut,  fLayoutAnaBut);

  //...................... Entry field
  fEntryAnaNumber = new TGTextBuffer();                               fCnew++;
  fAnaText = new TGTextEntry(fAnaFrame, fEntryAnaNumber);             fCnew++;
  fAnaText->SetToolTipText
    ("Click and enter the analysis name (code for type of analysis)");
  fAnaText->Resize(typ_of_ana_buf_lenght, fAnaText->GetDefaultHeight());

  DisplayInEntryField(fAnaText,fKeyAnaType);

  fAnaText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonAna()");
  fLayoutAnaField =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);     fCnew++;
  fAnaFrame->AddFrame(fAnaText, fLayoutAnaField);

 //=================================== RUN

  fRunFrame = new TGCompositeFrame(fAnaRunFrame,0,0,
				   kHorizontalFrame, kSunkenFrame);   fCnew++;
  fRunBut = new TGTextButton(fRunFrame, xRunButText, fRunButC);       fCnew++;
  fRunBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonRun()");
  fRunBut->SetToolTipText("Click here to register the run number");
  fLayoutRunBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);  fCnew++;   //  020
  fRunFrame->AddFrame(fRunBut,  fLayoutRunBut);
  fEntryRunNumber = new TGTextBuffer();                               fCnew++;
  fRunText = new TGTextEntry(fRunFrame, fEntryRunNumber);             fCnew++;
  fRunText->SetToolTipText
    ("Click and enter the run number of the name of the file containing the run list");
  fRunText->Resize(run_buf_lenght, fRunText->GetDefaultHeight());

  DisplayInEntryField(fRunText,fKeyRunNumber);

  fRunText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonRun()");
  fLayoutRunField =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);    fCnew++;
  fRunFrame->AddFrame(fRunText, fLayoutRunField);

  //-------------------------- display frame Ana + Run 
  fAnaRunFrame->AddFrame(fAnaFrame, fLayoutTopLeft);
  fAnaRunFrame->AddFrame(fRunFrame, fLayoutTopRight);
 
  fLayoutAnaRunFrame =  new TGLayoutHints(kLHintsTop | kLHintsExpandX,
					  xB1, xB1, xB1, xB1);        fCnew++;

  AddFrame(fAnaRunFrame, fLayoutAnaRunFrame);
  AddFrame(fVoidFrame, fLayoutGeneral);

  //=============================================== Button Texts
  TString xFirstEvtButText = "      First taken event    ( < 0 = 0 )      ";
  TString xNbOfEvtsButText = " Number of taken events ( <= 0 = all ) ";
  TString xSumoButText     = "           Super-Module number->       ";

  //=================================== FIRST EVENT NUMBER

  fFevFrame = new TGCompositeFrame(this,60,20, kHorizontalFrame, kSunkenFrame); fCnew++;
  
  fFevBut= new TGTextButton(fFevFrame, xFirstEvtButText);                       fCnew++;
  fFevBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonFev()");
  fFevBut->SetToolTipText
    ("Click here to register the number of the first event to be analyzed (written on the right)");
  fLayoutFevBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);     fCnew++;
  fFevFrame->AddFrame(fFevBut,  fLayoutFevBut);

  fEntryFevNumber = new TGTextBuffer();                               fCnew++;
  fFevText = new TGTextEntry(fFevFrame, fEntryFevNumber);             fCnew++;
  fFevText->SetToolTipText("Click and enter the first event number");
  fFevText->Resize(first_evt_buf_lenght, fFevText->GetDefaultHeight());
  DisplayInEntryField(fFevText,fKeyFirstEvt);
  fFevText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonFev()");
  fLayoutFevFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsLeft, xB1,xB1,xB1,xB1);  fCnew++;     // 030
  fFevFrame->AddFrame(fFevText, fLayoutFevFieldText);

  fLayoutFevFieldFrame =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);     fCnew++;
  AddFrame(fFevFrame, fLayoutFevFieldFrame);

  //=================================== NUMBER OF EVENTS
  fNoeFrame = new TGCompositeFrame(this,60,20, kHorizontalFrame, kSunkenFrame); fCnew++;
  
  fNoeBut = new TGTextButton(fNoeFrame, xNbOfEvtsButText);                      fCnew++;
  fNoeBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonNoe()");
  fNoeBut->SetToolTipText("Click here to register the number of events written on the right");
  fLayoutNoeBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);               fCnew++;
  fNoeFrame->AddFrame(fNoeBut,  fLayoutNoeBut);

  fEntryNoeNumber = new TGTextBuffer();                                         fCnew++;
  fNoeText = new TGTextEntry(fNoeFrame, fEntryNoeNumber);                       fCnew++;
  fNoeText->SetToolTipText("Click and enter the number of events");
  fNoeText->Resize(nb_of_evts_buf_lenght, fNoeText->GetDefaultHeight());
  DisplayInEntryField(fNoeText,fKeyNbOfEvts);
  fNoeText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonNoe()");
  fLayoutNoeFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);           fCnew++;
  fNoeFrame->AddFrame(fNoeText, fLayoutNoeFieldText);

  fLayoutNoeFieldFrame =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);               fCnew++;
  AddFrame(fNoeFrame, fLayoutNoeFieldFrame);

  //=================================== SUPER-MODULE

  fSuMoFrame = new TGCompositeFrame(this,60,20, kHorizontalFrame, kSunkenFrame); fCnew++;

  fSuMoBut = new TGTextButton(fSuMoFrame, xSumoButText);                         fCnew++;   // 040
  fSuMoBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonSuMo()");
  fSuMoBut->SetToolTipText("Click here to register the Super-Module number written on the right");
  fLayoutSuMoBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1); fCnew++;
  fSuMoFrame->AddFrame(fSuMoBut,  fLayoutSuMoBut);

  fEntrySuMoNumber = new TGTextBuffer();                                     fCnew++;
  fSuMoText = new TGTextEntry(fSuMoFrame, fEntrySuMoNumber);                 fCnew++;
  fSuMoText->SetToolTipText("Click and enter the Super-Module number");
  fSuMoText->Resize(nb_of_evts_buf_lenght, fSuMoText->GetDefaultHeight());
  DisplayInEntryField(fSuMoText, fKeySuMoNumber);
  fSuMoText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonSuMo()");

  fLayoutSuMoFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);        fCnew++;
  fSuMoFrame->AddFrame(fSuMoText, fLayoutSuMoFieldText);

  fLayoutSuMoFieldFrame =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);            fCnew++;
  AddFrame(fSuMoFrame, fLayoutSuMoFieldFrame);
  //========================== SUPER-MODULE TOWER NUMBERING VIEW BUTTON
  TString xSMNbButText     = "  Super-Module Tower Numbering  ";
  fButSMNb = new TGTextButton(this, xSMNbButText, fButSMNbC);                fCnew++;
  fButSMNb->Connect("Clicked()","TCnaDialogEB", this, "DoButtonSMNb()");
  fLayoutSMNbBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1); 
  AddFrame(fButSMNb, fLayoutSMNbBut);                                        fCnew++;

  AddFrame(fVoidFrame, fLayoutGeneral);

  //=================================== QUANTITIES RELATIVES TO THE SUPER-MODULE
  fSuMoUpFrame = new TGCompositeFrame
    (this,60,20,kVerticalFrame, GetDefaultFrameBackground());                fCnew++;

  TString xYminButText = " Ymin ";
  TString xYmaxButText = " Ymax ";
  //########################################### Composite frame number of events found in the data
  fVmmFoundEvtsFrame = new TGCompositeFrame
    (fSuMoUpFrame,60,20, kHorizontalFrame, kSunkenFrame);                      fCnew++;

  //...................................... Menu number of events found in the data

  //...................................... Frame for Ymax
  fVmaxFoundEvtsFrame = new TGCompositeFrame
    (fVmmFoundEvtsFrame,60,20, kHorizontalFrame, kSunkenFrame);                fCnew++;    // 050
  //...................................... Button Max + Entry field 
  fVmaxFoundEvtsBut = new TGTextButton(fVmaxFoundEvtsFrame, xYmaxButText);     fCnew++;
  fVmaxFoundEvtsBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVmaxFoundEvts()");
  fVmaxFoundEvtsBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fLayoutVmaxFoundEvtsBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);              fCnew++;
  fVmaxFoundEvtsFrame->AddFrame(fVmaxFoundEvtsBut,  fLayoutVmaxFoundEvtsBut);
  fEntryVmaxFoundEvtsNumber = new TGTextBuffer();                              fCnew++;
  fVmaxFoundEvtsText =
    new TGTextEntry(fVmaxFoundEvtsFrame, fEntryVmaxFoundEvtsNumber);           fCnew++;
  fVmaxFoundEvtsText->SetToolTipText("Click and enter ymax");
  fVmaxFoundEvtsText->Resize(nb_of_evts_buf_lenght, fVmaxFoundEvtsText->GetDefaultHeight());
  DisplayInEntryField(fVmaxFoundEvtsText, fKeyVmaxFoundEvts);
  fVmaxFoundEvtsText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVmaxFoundEvts()");

  fLayoutVmaxFoundEvtsFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);          fCnew++;
  fVmaxFoundEvtsFrame->AddFrame(fVmaxFoundEvtsText, fLayoutVmaxFoundEvtsFieldText);
  fLayoutVmaxFoundEvtsFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);             fCnew++;
  fVmmFoundEvtsFrame->AddFrame(fVmaxFoundEvtsFrame, fLayoutVmaxFoundEvtsFrame);

  //...................................... Frame for Ymin
  fVminFoundEvtsFrame = new TGCompositeFrame       
    (fVmmFoundEvtsFrame,60,20, kHorizontalFrame, kSunkenFrame);                fCnew++;
  //...................................... Button Min + Entry field 
  fVminFoundEvtsBut = new TGTextButton(fVminFoundEvtsFrame, xYminButText);     fCnew++;
  fVminFoundEvtsBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVminFoundEvts()");
  fVminFoundEvtsBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fLayoutVminFoundEvtsBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);              fCnew++;
  fVminFoundEvtsFrame->AddFrame(fVminFoundEvtsBut,  fLayoutVminFoundEvtsBut);
  fEntryVminFoundEvtsNumber = new TGTextBuffer();                              fCnew++;    // 060
  fVminFoundEvtsText =
    new TGTextEntry(fVminFoundEvtsFrame, fEntryVminFoundEvtsNumber);           fCnew++;
  fVminFoundEvtsText->SetToolTipText("Click and enter ymin");
  fVminFoundEvtsText->Resize(nb_of_evts_buf_lenght, fVminFoundEvtsText->GetDefaultHeight());
  DisplayInEntryField(fVminFoundEvtsText,fKeyVminFoundEvts);
  fVminFoundEvtsText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVminFoundEvts()");
  fLayoutVminFoundEvtsFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);          fCnew++;
  fVminFoundEvtsFrame->AddFrame(fVminFoundEvtsText, fLayoutVminFoundEvtsFieldText);
  fLayoutVminFoundEvtsFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);             fCnew++;
  fVmmFoundEvtsFrame->AddFrame(fVminFoundEvtsFrame, fLayoutVminFoundEvtsFrame);

  //...................................... Frame for text
  TString xMenuFoundEvts = " Numbers of events  ";
  fMenuFoundEvts = new TGPopupMenu(gClient->GetRoot());                                    fCnew++;
  fMenuFoundEvts->AddEntry("1D, Histo SM channels",fMenuFoundEvtsGlobalFullC);
  fMenuFoundEvts->AddEntry("1D, Histo SM channels SAME",fMenuFoundEvtsGlobalSameC);
  fMenuFoundEvts->AddSeparator();
  fMenuFoundEvts->AddEntry("1D, Histo Projection" ,fMenuFoundEvtsProjFullC);
  fMenuFoundEvts->AddEntry("1D, Histo Projection  SAME",fMenuFoundEvtsProjSameC);
  fMenuFoundEvts->AddSeparator();
  fMenuFoundEvts->AddEntry("2D, (eta,phi) view SM ",fMenuFoundEvtsEtaPhiC);
  fMenuFoundEvts->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarFoundEvts = new TGMenuBar(fVmmFoundEvtsFrame, 1, 1, kHorizontalFrame);           fCnew++;
  fMenuBarFoundEvts->AddPopup(xMenuFoundEvts, fMenuFoundEvts, fLayoutGeneral);
  fLayoutMenuBarFoundEvts = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);              fCnew++;
  fVmmFoundEvtsFrame->AddFrame(fMenuBarFoundEvts, fLayoutMenuBarFoundEvts);
  fLayoutVmmFoundEvtsFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                         fCnew++;
  fSuMoUpFrame->AddFrame(fVmmFoundEvtsFrame, fLayoutVmmFoundEvtsFrame);

  //............................. Ev + Sig Vertical frame
  fSuMoHozFrame =
    new TGCompositeFrame(fSuMoUpFrame,60,20,kVerticalFrame,
			 GetDefaultFrameBackground());                   fCnew++;

  //............................... Ev Vertical subframe
  fSuMoHozSubEvFrame = new TGCompositeFrame
    (fSuMoHozFrame,60,20,kVerticalFrame, GetDefaultFrameBackground());   fCnew++;

  //########################################### Composite frame ev of ev (mean pedestals)
  fVmmEvEvFrame = new TGCompositeFrame
    (fSuMoHozSubEvFrame,60,20, kHorizontalFrame, kSunkenFrame);          fCnew++;        // 070

  //...................................... Menu ev of ev

  //...................................... Frame for Ymax
  fVmaxEvEvFrame = new TGCompositeFrame
    (fVmmEvEvFrame,60,20, kHorizontalFrame, kSunkenFrame);                           fCnew++;
  //...................................... Button Max + Entry field 
  fVmaxEvEvBut = new TGTextButton(fVmaxEvEvFrame, xYmaxButText);                     fCnew++;
  fVmaxEvEvBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVmaxEvEv()");
  fVmaxEvEvBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fLayoutVmaxEvEvBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1); fCnew++;
  fVmaxEvEvFrame->AddFrame(fVmaxEvEvBut,  fLayoutVmaxEvEvBut);
  fEntryVmaxEvEvNumber = new TGTextBuffer();                                         fCnew++;
  fVmaxEvEvText = new TGTextEntry(fVmaxEvEvFrame, fEntryVmaxEvEvNumber);             fCnew++;
  fVmaxEvEvText->SetToolTipText("Click and enter ymax");
  fVmaxEvEvText->Resize(nb_of_evts_buf_lenght, fVmaxEvEvText->GetDefaultHeight());
  DisplayInEntryField(fVmaxEvEvText,fKeyVmaxEvEv);
  fVmaxEvEvText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVmaxEvEv()");
  fLayoutVmaxEvEvFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVmaxEvEvFrame->AddFrame(fVmaxEvEvText, fLayoutVmaxEvEvFieldText);
  fLayoutVmaxEvEvFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmEvEvFrame->AddFrame(fVmaxEvEvFrame, fLayoutVmaxEvEvFrame);

  //...................................... Frame for Ymin
  fVminEvEvFrame = new TGCompositeFrame
    (fVmmEvEvFrame,60,20, kHorizontalFrame, kSunkenFrame);                           fCnew++;
  //...................................... Button Min + Entry field 
  fVminEvEvBut = new TGTextButton(fVminEvEvFrame, xYminButText);                     fCnew++;
  fVminEvEvBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVminEvEv()");
  fVminEvEvBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fLayoutVminEvEvBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;  // 080
  fVminEvEvFrame->AddFrame(fVminEvEvBut,  fLayoutVminEvEvBut);

  fEntryVminEvEvNumber = new TGTextBuffer();                                         fCnew++;
  fVminEvEvText = new TGTextEntry(fVminEvEvFrame, fEntryVminEvEvNumber);             fCnew++;
  fVminEvEvText->SetToolTipText("Click and enter ymin");
  fVminEvEvText->Resize(nb_of_evts_buf_lenght, fVminEvEvText->GetDefaultHeight());
  DisplayInEntryField(fVminEvEvText,fKeyVminEvEv);
  fVminEvEvText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVminEvEv()");
  fLayoutVminEvEvFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVminEvEvFrame->AddFrame(fVminEvEvText, fLayoutVminEvEvFieldText);
  fLayoutVminEvEvFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmEvEvFrame->AddFrame(fVminEvEvFrame, fLayoutVminEvEvFrame);

  //...................................... Frame for text
  TString xMenuEvEv =      "      Mean pedestals    ";
  fMenuEvEv = new TGPopupMenu(gClient->GetRoot());                                   fCnew++;
  fMenuEvEv->AddEntry("1D, Histo SM channels",fMenuEvEvGlobalFullC);
  fMenuEvEv->AddEntry("1D, Histo SM channels SAME",fMenuEvEvGlobalSameC);
  fMenuEvEv->AddSeparator();
  fMenuEvEv->AddEntry("1D, Histo Projection" ,fMenuEvEvProjFullC);
  fMenuEvEv->AddEntry("1D, Histo Projection  SAME",fMenuEvEvProjSameC);
  fMenuEvEv->AddSeparator();
  fMenuEvEv->AddEntry("2D, (eta,phi) view SM ",fMenuEvEvEtaPhiC);
  fMenuEvEv->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarEvEv = new TGMenuBar(fVmmEvEvFrame, 1, 1, kHorizontalFrame);               fCnew++;
  fMenuBarEvEv->AddPopup(xMenuEvEv, fMenuEvEv, fLayoutGeneral);
  fLayoutMenuBarEvEv = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);             fCnew++;
  fVmmEvEvFrame->AddFrame(fMenuBarEvEv, fLayoutMenuBarEvEv);  

  fLayoutVmmEvEvFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fSuMoHozSubEvFrame->AddFrame(fVmmEvEvFrame, fLayoutVmmEvEvFrame);

  //########################################### Composite frame ev of sig (noise)
  fVmmEvSigFrame = new TGCompositeFrame
    (fSuMoHozSubEvFrame,60,20, kHorizontalFrame, kSunkenFrame);                      fCnew++;

  //...................................... Menu ev of sig 
  //...................................... Frame for Ymax
  fVmaxEvSigFrame = new TGCompositeFrame
    (fVmmEvSigFrame,60,20, kHorizontalFrame, kSunkenFrame);                          fCnew++;   // 090
  //...................................... Button Max + Entry field 
  fVmaxEvSigBut = new TGTextButton(fVmaxEvSigFrame, xYmaxButText);                   fCnew++;
  fVmaxEvSigBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVmaxEvSig()");
  fVmaxEvSigBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fLayoutVmaxEvSigBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVmaxEvSigFrame->AddFrame(fVmaxEvSigBut,  fLayoutVmaxEvSigBut);
  fEntryVmaxEvSigNumber = new TGTextBuffer();                                        fCnew++;
  fVmaxEvSigText = new TGTextEntry(fVmaxEvSigFrame, fEntryVmaxEvSigNumber);          fCnew++;
  fVmaxEvSigText->SetToolTipText("Click and enter ymax");
  fVmaxEvSigText->Resize(nb_of_evts_buf_lenght, fVmaxEvSigText->GetDefaultHeight());
  DisplayInEntryField(fVmaxEvSigText,fKeyVmaxEvSig);
  fVmaxEvSigText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVmaxEvSig()");
  fLayoutVmaxEvSigFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVmaxEvSigFrame->AddFrame(fVmaxEvSigText, fLayoutVmaxEvSigFieldText);
  fLayoutVmaxEvSigFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmEvSigFrame->AddFrame(fVmaxEvSigFrame, fLayoutVmaxEvSigFrame);

  //...................................... Frame for Ymin
  fVminEvSigFrame = new TGCompositeFrame
    (fVmmEvSigFrame,60,20, kHorizontalFrame, kSunkenFrame);                          fCnew++;
  //...................................... Button Min + Entry field 
  fVminEvSigBut = new TGTextButton(fVminEvSigFrame, xYminButText);                   fCnew++;
  fVminEvSigBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVminEvSig()");
  fVminEvSigBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fLayoutVminEvSigBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVminEvSigFrame->AddFrame(fVminEvSigBut,  fLayoutVminEvSigBut);

  fEntryVminEvSigNumber = new TGTextBuffer();                                        fCnew++;   // 100
  fVminEvSigText = new TGTextEntry(fVminEvSigFrame, fEntryVminEvSigNumber);          fCnew++;
  fVminEvSigText->SetToolTipText("Click and enter ymin");
  fVminEvSigText->Resize(nb_of_evts_buf_lenght, fVminEvSigText->GetDefaultHeight());
  DisplayInEntryField(fVminEvSigText,fKeyVminEvSig);
  fVminEvSigText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVminEvSig()");
  fLayoutVminEvSigFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVminEvSigFrame->AddFrame(fVminEvSigText, fLayoutVminEvSigFieldText);
  fLayoutVminEvSigFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmEvSigFrame->AddFrame(fVminEvSigFrame, fLayoutVminEvSigFrame);

  //...................................... Frame for text
  TString xMenuEvSig =  "Mean of sample sigmas";
  fMenuEvSig = new TGPopupMenu(gClient->GetRoot());                                  fCnew++;
  fMenuEvSig->AddEntry("1D, Histo SM channels",fMenuEvSigGlobalFullC);
  fMenuEvSig->AddEntry("1D, Histo SM channels SAME",fMenuEvSigGlobalSameC);
  fMenuEvSig->AddSeparator();
  fMenuEvSig->AddEntry("1D, Histo Projection" ,fMenuEvSigProjFullC);
  fMenuEvSig->AddEntry("1D, Histo Projection  SAME",fMenuEvSigProjSameC);
  fMenuEvSig->AddSeparator();
  fMenuEvSig->AddEntry("2D, (eta,phi) view SM ",fMenuEvSigEtaPhiC);
  fMenuEvSig->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarEvSig = new TGMenuBar(fVmmEvSigFrame, 1, 1, kHorizontalFrame);             fCnew++;
  fMenuBarEvSig->AddPopup(xMenuEvSig, fMenuEvSig, fLayoutGeneral);
  fLayoutMenuBarEvSig = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);            fCnew++;
  fVmmEvSigFrame->AddFrame(fMenuBarEvSig, fLayoutMenuBarEvSig);

  fLayoutVmmEvSigFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fSuMoHozSubEvFrame->AddFrame(fVmmEvSigFrame, fLayoutVmmEvSigFrame);

  //########################################### Composite frame ev of corss
  fVmmEvCorssFrame = new TGCompositeFrame
    (fSuMoHozSubEvFrame,60,20, kHorizontalFrame, kSunkenFrame);                      fCnew++;

  //...................................... Menu ev of Corss

  //...................................... Frame for Ymax
  fVmaxEvCorssFrame = new TGCompositeFrame
    (fVmmEvCorssFrame,60,20, kHorizontalFrame, kSunkenFrame);                        fCnew++;
  //...................................... Button Max + Entry field 
  fVmaxEvCorssBut = new TGTextButton(fVmaxEvCorssFrame, xYmaxButText);               fCnew++;    // 110
  fVmaxEvCorssBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVmaxEvCorss()");
  fVmaxEvCorssBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fLayoutVmaxEvCorssBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVmaxEvCorssFrame->AddFrame(fVmaxEvCorssBut,  fLayoutVmaxEvCorssBut);
  fEntryVmaxEvCorssNumber = new TGTextBuffer();                                      fCnew++;
  fVmaxEvCorssText = new TGTextEntry(fVmaxEvCorssFrame, fEntryVmaxEvCorssNumber);    fCnew++;
  fVmaxEvCorssText->SetToolTipText("Click and enter ymax");
  fVmaxEvCorssText->Resize(nb_of_evts_buf_lenght, fVmaxEvCorssText->GetDefaultHeight());
  DisplayInEntryField(fVmaxEvCorssText, fKeyVmaxEvCorss);
  fVmaxEvCorssText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVmaxEvCorss()");
  fLayoutVmaxEvCorssFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVmaxEvCorssFrame->AddFrame(fVmaxEvCorssText, fLayoutVmaxEvCorssFieldText);
  fLayoutVmaxEvCorssFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmEvCorssFrame->AddFrame(fVmaxEvCorssFrame, fLayoutVmaxEvCorssFrame);

  //...................................... Frame for Ymin
  fVminEvCorssFrame = new TGCompositeFrame
    (fVmmEvCorssFrame,60,20, kHorizontalFrame, kSunkenFrame);                        fCnew++;
  //...................................... Button Min + Entry field 
  fVminEvCorssBut = new TGTextButton(fVminEvCorssFrame, xYminButText);               fCnew++;
  fVminEvCorssBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVminEvCorss()");
  fVminEvCorssBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fLayoutVminEvCorssBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVminEvCorssFrame->AddFrame(fVminEvCorssBut,  fLayoutVminEvCorssBut);
  fEntryVminEvCorssNumber = new TGTextBuffer();                                      fCnew++;
  fVminEvCorssText = new TGTextEntry(fVminEvCorssFrame, fEntryVminEvCorssNumber);    fCnew++;   // 120
  fVminEvCorssText->SetToolTipText("Click and enter ymin");
  fVminEvCorssText->Resize(nb_of_evts_buf_lenght, fVminEvCorssText->GetDefaultHeight());
  DisplayInEntryField(fVminEvCorssText,fKeyVminEvCorss);
  fVminEvCorssText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVminEvCorss()");
  fLayoutVminEvCorssFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVminEvCorssFrame->AddFrame(fVminEvCorssText, fLayoutVminEvCorssFieldText);
  fLayoutVminEvCorssFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmEvCorssFrame->AddFrame(fVminEvCorssFrame, fLayoutVminEvCorssFrame);

  //...................................... Frame for text
  TString xMenuEvCorss = "     Mean of cor(s,s)    ";
  fMenuEvCorss = new TGPopupMenu(gClient->GetRoot());                                fCnew++;
  fMenuEvCorss->AddEntry("1D, Histo SM channels",fMenuEvCorssGlobalFullC);
  fMenuEvCorss->AddEntry("1D, Histo SM channels SAME",fMenuEvCorssGlobalSameC);
  fMenuEvCorss->AddSeparator();
  fMenuEvCorss->AddEntry("1D, Histo Projection" ,fMenuEvCorssProjFullC);
  fMenuEvCorss->AddEntry("1D, Histo Projection  SAME",fMenuEvCorssProjSameC);
  fMenuEvCorss->AddSeparator();
  fMenuEvCorss->AddEntry("2D, (eta,phi) view SM ",fMenuEvCorssEtaPhiC);
  fMenuEvCorss->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarEvCorss = new TGMenuBar(fVmmEvCorssFrame, 1, 1, kHorizontalFrame);         fCnew++;
  fMenuBarEvCorss->AddPopup(xMenuEvCorss, fMenuEvCorss, fLayoutGeneral);
  fLayoutMenuBarEvCorss = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);          fCnew++;
  fVmmEvCorssFrame->AddFrame(fMenuBarEvCorss, fLayoutMenuBarEvCorss);

  fLayoutVmmEvCorssFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fSuMoHozSubEvFrame->AddFrame(fVmmEvCorssFrame, fLayoutVmmEvCorssFrame);

  //------------------------------------------------------------------------------------------

  fLayoutSuMoHozSubEvFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1, xB1, xB1, xB1);                fCnew++;
  fSuMoHozFrame->AddFrame(fSuMoHozSubEvFrame, fLayoutSuMoHozSubEvFrame);

  //............................... Sig Vertical subframe
  fSuMoHozSubSigFrame = new TGCompositeFrame
    (fSuMoHozFrame,60,20,kVerticalFrame, GetDefaultFrameBackground());               fCnew++;

  //########################################### Composite frame sig of ev
  fVmmSigEvFrame = new TGCompositeFrame
    (fSuMoHozSubSigFrame,60,20, kHorizontalFrame, kSunkenFrame);                     fCnew++;

  //...................................... Menu sig of ev
  //...................................... Frame for Ymax
  fVmaxSigEvFrame = new TGCompositeFrame
    (fVmmSigEvFrame,60,20, kHorizontalFrame, kSunkenFrame);                          fCnew++;   // 130
  //...................................... Button Max + Entry field 
  fVmaxSigEvBut = new TGTextButton(fVmaxSigEvFrame, xYmaxButText);                   fCnew++;
  fVmaxSigEvBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVmaxSigEv()");
  fVmaxSigEvBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fLayoutVmaxSigEvBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVmaxSigEvFrame->AddFrame(fVmaxSigEvBut,  fLayoutVmaxSigEvBut);
  fEntryVmaxSigEvNumber = new TGTextBuffer();                                        fCnew++;
  fVmaxSigEvText = new TGTextEntry(fVmaxSigEvFrame, fEntryVmaxSigEvNumber);          fCnew++;
  fVmaxSigEvText->SetToolTipText("Click and enter ymax");
  fVmaxSigEvText->Resize(nb_of_evts_buf_lenght, fVmaxSigEvText->GetDefaultHeight());
  DisplayInEntryField(fVmaxSigEvText,fKeyVmaxSigEv);
  fVmaxSigEvText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVmaxSigEv()");
  fLayoutVmaxSigEvFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVmaxSigEvFrame->AddFrame(fVmaxSigEvText, fLayoutVmaxSigEvFieldText);
  fLayoutVmaxSigEvFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmSigEvFrame->AddFrame(fVmaxSigEvFrame, fLayoutVmaxSigEvFrame);

  //...................................... Frame for Ymin
  fVminSigEvFrame = new TGCompositeFrame
    (fVmmSigEvFrame,60,20, kHorizontalFrame, kSunkenFrame);                          fCnew++;
  //...................................... Button Min + Entry field 
  fVminSigEvBut = new TGTextButton(fVminSigEvFrame, xYminButText);                   fCnew++;
  fVminSigEvBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVminSigEv()");
  fVminSigEvBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fLayoutVminSigEvBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVminSigEvFrame->AddFrame(fVminSigEvBut,  fLayoutVminSigEvBut);
  fEntryVminSigEvNumber = new TGTextBuffer();                                        fCnew++;      // 140
  fVminSigEvText = new TGTextEntry(fVminSigEvFrame, fEntryVminSigEvNumber);          fCnew++;
  fVminSigEvText->SetToolTipText("Click and enter ymin");
  fVminSigEvText->Resize(nb_of_evts_buf_lenght, fVminSigEvText->GetDefaultHeight());
  DisplayInEntryField(fVminSigEvText,fKeyVminSigEv);
  fVminSigEvText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVminSigEv()");
  fLayoutVminSigEvFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVminSigEvFrame->AddFrame(fVminSigEvText, fLayoutVminSigEvFieldText);
  fLayoutVminSigEvFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmSigEvFrame->AddFrame(fVminSigEvFrame, fLayoutVminSigEvFrame);

  //...................................... Frame for text
  TString xMenuSigEv =      " Sigma of sample means ";
  fMenuSigEv = new TGPopupMenu(gClient->GetRoot());                                  fCnew++;
  fMenuSigEv->AddEntry("1D, Histo SM channels",fMenuSigEvGlobalFullC);
  fMenuSigEv->AddEntry("1D, Histo SM channels SAME",fMenuSigEvGlobalSameC);
  fMenuSigEv->AddSeparator();
  fMenuSigEv->AddEntry("1D, Histo Projection" ,fMenuSigEvProjFullC);
  fMenuSigEv->AddEntry("1D, Histo Projection  SAME",fMenuSigEvProjSameC);
  fMenuSigEv->AddSeparator();
  fMenuSigEv->AddEntry("2D, (eta,phi) view SM ",fMenuSigEvEtaPhiC);
  fMenuSigEv->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarSigEv = new TGMenuBar(fVmmSigEvFrame, 1, 1, kHorizontalFrame);             fCnew++;
  fMenuBarSigEv->AddPopup(xMenuSigEv, fMenuSigEv, fLayoutGeneral);
  fLayoutMenuBarSigEv = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);            fCnew++;
  fVmmSigEvFrame->AddFrame(fMenuBarSigEv, fLayoutMenuBarSigEv);
  fLayoutVmmSigEvFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fSuMoHozSubSigFrame->AddFrame(fVmmSigEvFrame, fLayoutVmmSigEvFrame);

  //########################################### Composite frame sig of sig
  fVmmSigSigFrame = new TGCompositeFrame
    (fSuMoHozSubSigFrame,60,20, kHorizontalFrame, kSunkenFrame);                     fCnew++;

  //...................................... Menu sig of sig 
  //...................................... Frame for Ymax
  fVmaxSigSigFrame = new TGCompositeFrame
    (fVmmSigSigFrame,60,20, kHorizontalFrame, kSunkenFrame);                         fCnew++;
  //...................................... Button Max + Entry field 
  fVmaxSigSigBut = new TGTextButton(fVmaxSigSigFrame, xYmaxButText);                 fCnew++;      // 150
  fVmaxSigSigBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVmaxSigSig()");
  fVmaxSigSigBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fLayoutVmaxSigSigBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVmaxSigSigFrame->AddFrame(fVmaxSigSigBut,  fLayoutVmaxSigSigBut);
  fEntryVmaxSigSigNumber = new TGTextBuffer();                                       fCnew++;
  fVmaxSigSigText = new TGTextEntry(fVmaxSigSigFrame, fEntryVmaxSigSigNumber);       fCnew++;
  fVmaxSigSigText->SetToolTipText("Click and enter ymax");
  fVmaxSigSigText->Resize(nb_of_evts_buf_lenght, fVmaxSigSigText->GetDefaultHeight());
  DisplayInEntryField(fVmaxSigSigText,fKeyVmaxSigSig);
  fVmaxSigSigText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVmaxSigSig()");
  fLayoutVmaxSigSigFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVmaxSigSigFrame->AddFrame(fVmaxSigSigText, fLayoutVmaxSigSigFieldText);
  fLayoutVmaxSigSigFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmSigSigFrame->AddFrame(fVmaxSigSigFrame, fLayoutVmaxSigSigFrame);

  //...................................... Frame for Ymin
  fVminSigSigFrame = new TGCompositeFrame
    (fVmmSigSigFrame,60,20, kHorizontalFrame, kSunkenFrame);                         fCnew++;
  //...................................... Button Min + Entry field 
  fVminSigSigBut = new TGTextButton(fVminSigSigFrame, xYminButText);                 fCnew++;
  fVminSigSigBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVminSigSig()");
  fVminSigSigBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fLayoutVminSigSigBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVminSigSigFrame->AddFrame(fVminSigSigBut,  fLayoutVminSigSigBut);
  fEntryVminSigSigNumber = new TGTextBuffer();                                       fCnew++;
  fVminSigSigText = new TGTextEntry(fVminSigSigFrame, fEntryVminSigSigNumber);       fCnew++;   // 160
  fVminSigSigText->SetToolTipText("Click and enter ymin");
  fVminSigSigText->Resize(nb_of_evts_buf_lenght, fVminSigSigText->GetDefaultHeight());
  DisplayInEntryField(fVminSigSigText,fKeyVminSigSig);
  fVminSigSigText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVminSigSig()");
  fLayoutVminSigSigFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVminSigSigFrame->AddFrame(fVminSigSigText, fLayoutVminSigSigFieldText);
  fLayoutVminSigSigFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmSigSigFrame->AddFrame(fVminSigSigFrame, fLayoutVminSigSigFrame);

  //...................................... Frame for text
  TString xMenuSigSig =  " Sigma of sample sigmas ";
  fMenuSigSig = new TGPopupMenu(gClient->GetRoot());                                 fCnew++;
  fMenuSigSig->AddEntry("1D, Histo SM channels",fMenuSigSigGlobalFullC);
  fMenuSigSig->AddEntry("1D, Histo SM channels SAME",fMenuSigSigGlobalSameC);
  fMenuSigSig->AddSeparator();
  fMenuSigSig->AddEntry("1D, Histo Projection" ,fMenuSigSigProjFullC);
  fMenuSigSig->AddEntry("1D, Histo Projection  SAME",fMenuSigSigProjSameC);
  fMenuSigSig->AddSeparator();
  fMenuSigSig->AddEntry("2D, (eta,phi) view SM ",fMenuSigSigEtaPhiC);
  fMenuSigSig->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarSigSig = new TGMenuBar(fVmmSigSigFrame, 1, 1, kHorizontalFrame);           fCnew++;
  fMenuBarSigSig->AddPopup(xMenuSigSig, fMenuSigSig, fLayoutGeneral);
  fLayoutMenuBarSigSig = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);           fCnew++;
  fVmmSigSigFrame->AddFrame(fMenuBarSigSig, fLayoutMenuBarSigSig);

  fLayoutVmmSigSigFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fSuMoHozSubSigFrame->AddFrame(fVmmSigSigFrame, fLayoutVmmSigSigFrame);


  //########################################### Composite frame sig of corss
  fVmmSigCorssFrame = new TGCompositeFrame
    (fSuMoHozSubSigFrame,60,20, kHorizontalFrame, kSunkenFrame);                     fCnew++;

  //...................................... Menu sig of Corss
  //...................................... Frame for Ymax
  fVmaxSigCorssFrame = new TGCompositeFrame
    (fVmmSigCorssFrame,60,20, kHorizontalFrame, kSunkenFrame);                       fCnew++;
  //...................................... Button Max + Entry field 
  fVmaxSigCorssBut = new TGTextButton(fVmaxSigCorssFrame, xYmaxButText);             fCnew++;
  fVmaxSigCorssBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVmaxSigCorss()");
  fVmaxSigCorssBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fLayoutVmaxSigCorssBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;    // 170
  fVmaxSigCorssFrame->AddFrame(fVmaxSigCorssBut,  fLayoutVmaxSigCorssBut);
  fEntryVmaxSigCorssNumber = new TGTextBuffer();                                     fCnew++;
  fVmaxSigCorssText = new TGTextEntry(fVmaxSigCorssFrame, fEntryVmaxSigCorssNumber); fCnew++;
  fVmaxSigCorssText->SetToolTipText("Click and enter ymax");
  fVmaxSigCorssText->Resize(nb_of_evts_buf_lenght, fVmaxSigCorssText->GetDefaultHeight());
  DisplayInEntryField(fVmaxSigCorssText,fKeyVmaxSigCorss);
  fVmaxSigCorssText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVmaxSigCorss()");
  fLayoutVmaxSigCorssFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVmaxSigCorssFrame->AddFrame(fVmaxSigCorssText, fLayoutVmaxSigCorssFieldText);
  fLayoutVmaxSigCorssFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmSigCorssFrame->AddFrame(fVmaxSigCorssFrame, fLayoutVmaxSigCorssFrame);

  //...................................... Frame for Ymin
  fVminSigCorssFrame = new TGCompositeFrame
    (fVmmSigCorssFrame,60,20, kHorizontalFrame, kSunkenFrame);                       fCnew++;
  //...................................... Button Min + Entry field 
  fVminSigCorssBut = new TGTextButton(fVminSigCorssFrame, xYminButText);             fCnew++;
  fVminSigCorssBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVminSigCorss()");
  fVminSigCorssBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fLayoutVminSigCorssBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVminSigCorssFrame->AddFrame(fVminSigCorssBut,  fLayoutVminSigCorssBut);
  fEntryVminSigCorssNumber = new TGTextBuffer();                                     fCnew++;
  fVminSigCorssText = new TGTextEntry(fVminSigCorssFrame, fEntryVminSigCorssNumber); fCnew++;
  fVminSigCorssText->SetToolTipText("Click and enter ymin");
  fVminSigCorssText->Resize(nb_of_evts_buf_lenght, fVminSigCorssText->GetDefaultHeight());
  DisplayInEntryField(fVminSigCorssText,fKeyVminSigCorss);
  fVminSigCorssText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVminSigCorss()");
  fLayoutVminSigCorssFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVminSigCorssFrame->AddFrame(fVminSigCorssText, fLayoutVminSigCorssFieldText);
  fLayoutVminSigCorssFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmSigCorssFrame->AddFrame(fVminSigCorssFrame, fLayoutVminSigCorssFrame);

  //...................................... Frame for text
  TString xMenuSigCorss = "   Sigma of cor(s,s)    ";
  fMenuSigCorss = new TGPopupMenu(gClient->GetRoot());                               fCnew++;
  fMenuSigCorss->AddEntry("1D, Histo SM channels",fMenuSigCorssGlobalFullC);
  fMenuSigCorss->AddEntry("1D, Histo SM channels SAME",fMenuSigCorssGlobalSameC);
  fMenuSigCorss->AddSeparator();
  fMenuSigCorss->AddEntry("1D, Histo Projection" ,fMenuSigCorssProjFullC);
  fMenuSigCorss->AddEntry("1D, Histo Projection  SAME",fMenuSigCorssProjSameC);
  fMenuSigCorss->AddSeparator();
  fMenuSigCorss->AddEntry("2D, (eta,phi) view SM ",fMenuSigCorssEtaPhiC);
  fMenuSigCorss->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarSigCorss = new TGMenuBar(fVmmSigCorssFrame, 1, 1, kHorizontalFrame);       fCnew++;
  fMenuBarSigCorss->AddPopup(xMenuSigCorss, fMenuSigCorss, fLayoutGeneral);
  fLayoutMenuBarSigCorss = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);         fCnew++;
  fVmmSigCorssFrame->AddFrame(fMenuBarSigCorss, fLayoutMenuBarSigCorss);

  fLayoutVmmSigCorssFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;    // 185
  fSuMoHozSubSigFrame->AddFrame(fVmmSigCorssFrame, fLayoutVmmSigCorssFrame);

  //------------------------------------------------------------------------------------------
  
  fLayoutSuMoHozSubSigFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1, xB1, xB1, xB1);                fCnew++;
  fSuMoHozFrame->AddFrame(fSuMoHozSubSigFrame, fLayoutSuMoHozSubSigFrame);
  
  //######################################################################################################"

  //------------------------------------------- subframe
  fLayoutSuMoHozFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1, xB1, xB1, xB1);                fCnew++;
  fSuMoUpFrame->AddFrame(fSuMoHozFrame, fLayoutSuMoHozFrame);

  AddFrame(fVoidFrame, fLayoutGeneral);

  //########################################### Composite frame corcc in towers
  fVmmEvCovttFrame = new TGCompositeFrame
    (fSuMoUpFrame,60,20, kHorizontalFrame, kSunkenFrame);                            fCnew++;
 
  //...................................... Menu covariances between towers 

  //...................................... Frame for Ymax
  fVmaxEvCovttFrame = new TGCompositeFrame
    (fVmmEvCovttFrame,60,20, kHorizontalFrame, kSunkenFrame);                        fCnew++;
  //...................................... Button Max + Entry field 
  fVmaxEvCovttBut = new TGTextButton(fVmaxEvCovttFrame, xYmaxButText);               fCnew++;
  fVmaxEvCovttBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVmaxEvCovtt()");
  fVmaxEvCovttBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fLayoutVmaxEvCovttBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVmaxEvCovttFrame->AddFrame(fVmaxEvCovttBut,  fLayoutVmaxEvCovttBut);
  fEntryVmaxEvCovttNumber = new TGTextBuffer();                                      fCnew++;
  fVmaxEvCovttText = new TGTextEntry(fVmaxEvCovttFrame, fEntryVmaxEvCovttNumber);    fCnew++;
  fVmaxEvCovttText->SetToolTipText("Click and enter ymax");
  fVmaxEvCovttText->Resize(nb_of_evts_buf_lenght, fVmaxEvCovttText->GetDefaultHeight());
  DisplayInEntryField(fVmaxEvCovttText, fKeyVmaxEvCovtt);
  fVmaxEvCovttText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVmaxEvCovtt()");

  fLayoutVmaxEvCovttFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVmaxEvCovttFrame->AddFrame(fVmaxEvCovttText, fLayoutVmaxEvCovttFieldText);
  fLayoutVmaxEvCovttFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmEvCovttFrame->AddFrame(fVmaxEvCovttFrame, fLayoutVmaxEvCovttFrame);

  //...................................... Frame for Ymin
  fVminEvCovttFrame = new TGCompositeFrame
    (fVmmEvCovttFrame,60,20, kHorizontalFrame, kSunkenFrame);                        fCnew++;
  //...................................... Button Min + Entry field
  fVminEvCovttBut = new TGTextButton(fVminEvCovttFrame, xYminButText);               fCnew++;
  fVminEvCovttBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVminEvCovtt()");
  fVminEvCovttBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fLayoutVminEvCovttBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVminEvCovttFrame->AddFrame(fVminEvCovttBut,  fLayoutVminEvCovttBut);
  fEntryVminEvCovttNumber = new TGTextBuffer();                                      fCnew++;
  fVminEvCovttText = new TGTextEntry(fVminEvCovttFrame, fEntryVminEvCovttNumber);    fCnew++;    // 200
  fVminEvCovttText->SetToolTipText("Click and enter ymin");
  fVminEvCovttText->Resize(nb_of_evts_buf_lenght, fVminEvCovttText->GetDefaultHeight());
  DisplayInEntryField(fVminEvCovttText,fKeyVminEvCovtt);
  fVminEvCovttText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVminEvCovtt()");
  fLayoutVminEvCovttFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVminEvCovttFrame->AddFrame(fVminEvCovttText, fLayoutVminEvCovttFieldText);
  fLayoutVminEvCovttFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmEvCovttFrame->AddFrame(fVminEvCovttFrame, fLayoutVminEvCovttFrame);

  //........................................... Frame for text 
  TString xMenuCovtt = "Cor(c,c) in towers (SM view)";
  fMenuCovtt = new TGPopupMenu(gClient->GetRoot());                                  fCnew++;
  fMenuCovtt->AddEntry("2D, (eta,phi) view SM ",fMenuCovttColzC);
  fMenuCovtt->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarCovtt = new TGMenuBar(fVmmEvCovttFrame, 1, 1, kHorizontalFrame);           fCnew++;
  fMenuBarCovtt->AddPopup(xMenuCovtt, fMenuCovtt, fLayoutGeneral);
  fLayoutMenuBarCovtt = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);            fCnew++;
  fVmmEvCovttFrame->AddFrame(fMenuBarCovtt, fLayoutMenuBarCovtt);
   fLayoutVmmEvCovttFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fSuMoUpFrame->AddFrame(fVmmEvCovttFrame, fLayoutVmmEvCovttFrame);

  //########################################### Composite frame ev of cortt
  fVmmEvCorttFrame = new TGCompositeFrame
    (fSuMoUpFrame,60,20, kHorizontalFrame, kSunkenFrame);                            fCnew++;
 
  //...................................... Menu correlations between towers 
  //...................................... Frame for Ymax
  fVmaxEvCorttFrame = new TGCompositeFrame
    (fVmmEvCorttFrame,60,20, kHorizontalFrame, kSunkenFrame);                        fCnew++;
  //...................................... Button Max + Entry field 
  fVmaxEvCorttBut = new TGTextButton(fVmaxEvCorttFrame, xYmaxButText);               fCnew++;
  fVmaxEvCorttBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVmaxEvCortt()");
  fVmaxEvCorttBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fLayoutVmaxEvCorttBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;    // 210
  fVmaxEvCorttFrame->AddFrame(fVmaxEvCorttBut,  fLayoutVmaxEvCorttBut);
  fEntryVmaxEvCorttNumber = new TGTextBuffer();                                      fCnew++;
  fVmaxEvCorttText = new TGTextEntry(fVmaxEvCorttFrame, fEntryVmaxEvCorttNumber);    fCnew++;
  fVmaxEvCorttText->SetToolTipText("Click and enter ymax");
  fVmaxEvCorttText->Resize(nb_of_evts_buf_lenght, fVmaxEvCorttText->GetDefaultHeight());
  DisplayInEntryField(fVmaxEvCorttText, fKeyVmaxEvCortt);
  fVmaxEvCorttText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVmaxEvCortt()");

  fLayoutVmaxEvCorttFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVmaxEvCorttFrame->AddFrame(fVmaxEvCorttText, fLayoutVmaxEvCorttFieldText);
  fLayoutVmaxEvCorttFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmEvCorttFrame->AddFrame(fVmaxEvCorttFrame, fLayoutVmaxEvCorttFrame);

  //...................................... Frame for Ymin
  fVminEvCorttFrame = new TGCompositeFrame
    (fVmmEvCorttFrame,60,20, kHorizontalFrame, kSunkenFrame);                        fCnew++;
  //...................................... Button Min + Entry field 
  fVminEvCorttBut = new TGTextButton(fVminEvCorttFrame, xYminButText);               fCnew++;
  fVminEvCorttBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonVminEvCortt()");
  fVminEvCorttBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fLayoutVminEvCorttBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVminEvCorttFrame->AddFrame(fVminEvCorttBut,  fLayoutVminEvCorttBut);
  fEntryVminEvCorttNumber = new TGTextBuffer();                                      fCnew++;
  fVminEvCorttText = new TGTextEntry(fVminEvCorttFrame, fEntryVminEvCorttNumber);    fCnew++;
  fVminEvCorttText->SetToolTipText("Click and enter ymin");
  fVminEvCorttText->Resize(nb_of_evts_buf_lenght, fVminEvCorttText->GetDefaultHeight());
  DisplayInEntryField(fVminEvCorttText,fKeyVminEvCortt);
  fVminEvCorttText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonVminEvCortt()");
  fLayoutVminEvCorttFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;    // 220
  fVminEvCorttFrame->AddFrame(fVminEvCorttText, fLayoutVminEvCorttFieldText);
  fLayoutVminEvCorttFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmEvCorttFrame->AddFrame(fVminEvCorttFrame, fLayoutVminEvCorttFrame);

  //...................................... Frame for text 
  TString xMenuCortt = " Correlations between towers";
  fMenuCortt = new TGPopupMenu(gClient->GetRoot());                                  fCnew++;
  fMenuCortt->AddEntry("2D, COLZ ",fMenuCorttColzC);
  fMenuCortt->AddEntry("3D, LEGO2Z" ,fMenuCorttLegoC);
  fMenuCortt->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarCortt = new TGMenuBar(fVmmEvCorttFrame, 1, 1, kHorizontalFrame);           fCnew++;
  fMenuBarCortt->AddPopup(xMenuCortt, fMenuCortt, fLayoutGeneral);
  fLayoutMenuBarCortt = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);            fCnew++;
  fVmmEvCorttFrame->AddFrame(fMenuBarCortt, fLayoutMenuBarCortt);
  fLayoutVmmEvCorttFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fSuMoUpFrame->AddFrame(fVmmEvCorttFrame, fLayoutVmmEvCorttFrame);

 
  //=============================================== "SuMo" frame ===============================================
  fLayoutSuMoUpFrame =
    new TGLayoutHints(kLHintsTop | kLHintsCenterX, xB1, xB1, xB1, xB1);              fCnew++;
  AddFrame(fSuMoUpFrame, fLayoutSuMoUpFrame);

  AddFrame(fVoidFrame, fLayoutGeneral);

  //======================================= Tower X & Tower Y FRAME ========================= 
  fTowSpFrame =
    new TGCompositeFrame(this,60,20,kHorizontalFrame,
			 GetDefaultFrameBackground());                               fCnew++;

  TString xTowXButText  = "  Tower X number-> ";
  TString xTowYButText  = "  Tower Y number-> ";

  Int_t tower_buf_lenght =  65;
 
  //============================= TOWER X =====================================
  fTxSubFrame = new TGCompositeFrame
    (fTowSpFrame,60,20,kVerticalFrame, GetDefaultFrameBackground());     fCnew++;

  fTowXFrame = new TGCompositeFrame
    (fTxSubFrame,60,20,kHorizontalFrame,kSunkenFrame);                    fCnew++;

  fTowXBut = new TGTextButton(fTowXFrame, xTowXButText, fTowXButC);       fCnew++;         // 230
  fTowXBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonTowX()");
  fTowXBut->SetToolTipText("Click here to register the tower X number written on the right");
  fTowXBut->Resize(tower_buf_lenght, fTowXBut->GetDefaultHeight());
  fLayoutTowXBut = new TGLayoutHints(kLHintsLeft, xB1,xB1,xB1,xB1);       fCnew++;
  fTowXFrame->AddFrame(fTowXBut,  fLayoutTowXBut);

  fEntryTowXNumber = new TGTextBuffer();                                  fCnew++;
  fTowXText = new TGTextEntry(fTowXFrame, fEntryTowXNumber);              fCnew++;
  fTowXText->SetToolTipText("Click and enter the Tower number");
  fTowXText->Resize(tower_buf_lenght, fTowXText->GetDefaultHeight());
  DisplayInEntryField(fTowXText,fKeyTowXNumber);
  fTowXText->Connect("ReturnPressed()", "TCnaDialogEB",this, "DoButtonTowX()");
  fLayoutTowXField =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1 );        fCnew++;
  fTowXFrame->AddFrame(fTowXText, fLayoutTowXField);

  fTxSubFrame->AddFrame(fTowXFrame, fLayoutGeneral);

  //========================== TOWER X CRYSTAL NUMBERING VIEW
  TString xChNbButText     = "  Tower X Crystal Numbering  ";
  fButChNb = new TGTextButton(fTxSubFrame, xChNbButText, fButChNbC);      fCnew++;
  fButChNb->Connect("Clicked()","TCnaDialogEB", this, "DoButtonChNb()");
  fLayoutChNbBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);      fCnew++;
  fTxSubFrame->AddFrame(fButChNb, fLayoutChNbBut); 

  //---------------- menus relative to the Tower X subframe 

  //===================== Menus relative to the tower X ======================

  TString xMenuBarCorGlob = " Cor(s,s) in crystals (Tower X view)";
  TString xMenuBarCovGlob = " Cov(s,s) in crystals (Tower X view)";

  //................. Menu correlations between samples for all the channels. Global view
  
  fMenuCorssAll = new TGPopupMenu(gClient->GetRoot());                      fCnew++;
  fMenuCorssAll->AddEntry("2D, COLZ",fMenuCorssAllColzC);
  fMenuCorssAll->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarCorssAll =  new TGMenuBar(fTxSubFrame, 1, 1, kHorizontalFrame);   fCnew++;
  fMenuBarCorssAll->AddPopup(xMenuBarCorGlob, fMenuCorssAll, fLayoutGeneral);
  fTxSubFrame->AddFrame(fMenuBarCorssAll, fLayoutTopLeft);

  //................. Menu covariances between samples for all the channels. Global view

  fMenuCovssAll = new TGPopupMenu(gClient->GetRoot());                      fCnew++;
  fMenuCovssAll->AddEntry("2D, COLZ",fMenuCovssAllColzC);
  fMenuCovssAll->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarCovssAll =  new TGMenuBar(fTxSubFrame, 1, 1, kHorizontalFrame);   fCnew++;         // 240
  fMenuBarCovssAll->AddPopup(xMenuBarCovGlob, fMenuCovssAll, fLayoutGeneral);
  fTxSubFrame->AddFrame(fMenuBarCovssAll, fLayoutTopLeft);

  //------------------ Add Tower X frame to the surframe 
  fLayoutTxSubFrame = 
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);        fCnew++;
  fTowSpFrame->AddFrame(fTxSubFrame, fLayoutTxSubFrame);

  //============================= TOWER Y =====================================

  fTySubFrame = new TGCompositeFrame
    (fTowSpFrame,60,20,kVerticalFrame, GetDefaultFrameBackground());  fCnew++;

  fTowYFrame = new TGCompositeFrame
    (fTySubFrame,60,20,kHorizontalFrame,kSunkenFrame);                fCnew++;

  fTowYBut =
    new TGTextButton(fTowYFrame, xTowYButText, fTowYButC);            fCnew++;
  fTowYBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonTowY()");
  fTowYBut->SetToolTipText("Click here to register the tower Y number written on the right");
  fTowYBut->Resize(tower_buf_lenght, fTowYBut->GetDefaultHeight());
  fLayoutTowYBut = new TGLayoutHints(kLHintsLeft, xB1,xB1,xB1,xB1);   fCnew++;
  fTowYFrame->AddFrame(fTowYBut,  fLayoutTowYBut);

  fEntryTowYNumber = new TGTextBuffer();                              fCnew++;
  fTowYText = new TGTextEntry(fTowYFrame, fEntryTowYNumber);          fCnew++;
  fTowYText->SetToolTipText("Click and enter the Tower number");
  fTowYText->Resize(tower_buf_lenght, fTowYText->GetDefaultHeight());
  DisplayInEntryField(fTowYText,fKeyTowYNumber);
  fTowYText->Connect("ReturnPressed()", "TCnaDialogEB",this, "DoButtonTowY()");
  fLayoutTowYField =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1 );    fCnew++;
  fTowYFrame->AddFrame(fTowYText, fLayoutTowYField);

  fTySubFrame->AddFrame(fTowYFrame, fLayoutGeneral);

  //---------------- menus relative to the Tower Y subframe 

  //                    (no such menus )

  //------------------ Add Tower Y subframe to the frame 
  fLayoutTySubFrame =  new TGLayoutHints
    (kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);                   fCnew++;

  fTowSpFrame->AddFrame(fTySubFrame, fLayoutTySubFrame);

  //---------------------- composite frame (tower X, tower Y)
  fLayoutTowSpFrame =  new TGLayoutHints
    (kLHintsTop | kLHintsCenterX, xB1, xB1, xB1, xB1);                fCnew++;   // 250

  AddFrame(fTowSpFrame, fLayoutTowSpFrame);

  //------------------ menus relatives to the Horizontal frame (Tower_X + Tower_Y)

  TString xMenuBarCorcc = " Cor(Crystal Tower X, Crystal Tower Y). Mean over samples";
  TString xMenuBarCovcc = " Cov(Crystal Tower X, Crystal Tower Y). Mean over samples";

  //...................... Menu correlations between channels

  fMenuCorcc = new TGPopupMenu(gClient->GetRoot());                   fCnew++;
  fMenuCorcc->AddEntry("2D, COLZ",fMenuCorccColzC);
  fMenuCorcc->AddSeparator();
  fMenuCorcc->AddEntry("3D, LEGO2Z",fMenuCorccLegoC);
  fMenuCorcc->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarCorcc = new TGMenuBar(this, 1, 1, kHorizontalFrame);        fCnew++;
  fMenuBarCorcc->AddPopup(xMenuBarCorcc, fMenuCorcc, fLayoutTopRight);
  AddFrame(fMenuBarCorcc, fLayoutGeneral);

  //...................... Menu covariances between channels

  fMenuCovcc = new TGPopupMenu(gClient->GetRoot());                   fCnew++;
  fMenuCovcc->AddEntry("2D, COLZ",fMenuCovccColzC);
  fMenuCovcc->AddSeparator();
  fMenuCovcc->AddEntry("3D, LEGO2Z",fMenuCovccLegoC);
  fMenuCovcc->Connect("Activated(Int_t)", "TCnaDialogEB", this,"HandleMenu(Int_t)");
  fMenuBarCovcc = new TGMenuBar(this, 1, 1, kHorizontalFrame);        fCnew++;
  fMenuBarCovcc->AddPopup(xMenuBarCovcc, fMenuCovcc, fLayoutTopRight);
  AddFrame(fMenuBarCovcc, fLayoutGeneral);

  //=================================== CHANNEL & SAMPLE FRAME ==============

  fChSpFrame =
    new TGCompositeFrame(this,60,20,kHorizontalFrame,
			 GetDefaultFrameBackground());                fCnew++;

  TString xChanButText  = "  Tower X Channel number-> ";
  TString xSampButText  = "  Sample number->   ";

  Int_t chan_buf_lenght =  50;
  Int_t samp_buf_lenght =  50;

  TString xMenuBarCorss    = " Correlations between samples";
  TString xMenuBarCovss    = " Covariances between samples";
  TString xMenuBarEvs      = " Expectation values of the samples";
  TString xMenuBarSigs     = " Sigmas of the samples";

  //=================================== CHANNEL (CRYSTAL)
  fChSubFrame = new TGCompositeFrame
    (fChSpFrame,60,20,kVerticalFrame, GetDefaultFrameBackground());   fCnew++;

  fChanFrame = new TGCompositeFrame
    (fChSubFrame,60,20,kHorizontalFrame,kSunkenFrame);                fCnew++;

  fChanBut =
    new TGTextButton(fChanFrame, xChanButText, fChanButC);            fCnew++;
  fChanBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonChan()");
  fChanBut->SetToolTipText("Click here to register the crystal number written to the right");
  fChanBut->Resize(chan_buf_lenght, fChanBut->GetDefaultHeight());
  fLayoutChanBut = new TGLayoutHints(kLHintsLeft, xB1,xB1,xB1,xB1);   fCnew++;
  fChanFrame->AddFrame(fChanBut,  fLayoutChanBut);

  fEntryChanNumber = new TGTextBuffer();                              fCnew++;       // 260
  fChanText = new TGTextEntry(fChanFrame, fEntryChanNumber);          fCnew++;
  fChanText->SetToolTipText("Click and enter the crystal number");
  fChanText->Resize(chan_buf_lenght, fChanText->GetDefaultHeight());
  DisplayInEntryField(fChanText,fKeyChanNumber);
  fChanText->Connect("ReturnPressed()", "TCnaDialogEB",this, "DoButtonChan()");
  fLayoutChanField =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1 );    fCnew++;
  fChanFrame->AddFrame(fChanText, fLayoutChanField);

  fChSubFrame->AddFrame(fChanFrame, fLayoutGeneral);

  //--------------------- Menus relative to the channel SubFrame -------------
  //...................... Menu correlations between samples

  fMenuCorss = new TGPopupMenu(gClient->GetRoot());                   fCnew++;
  fMenuCorss->AddEntry("2D, COLZ",fMenuCorssColzC);
  fMenuCorss->AddSeparator();
  fMenuCorss->AddEntry("3D, LEGO2Z",fMenuCorssLegoC);
  fMenuCorss->AddEntry("3D, SURF1Z",fMenuCorssSurf1C);
  fMenuCorss->AddEntry("3D, SURF4",fMenuCorssSurf4C);
  fMenuCorss->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarCorss = new TGMenuBar(fChSubFrame, 1, 1, kHorizontalFrame); fCnew++;
  fMenuBarCorss->AddPopup(xMenuBarCorss, fMenuCorss, fLayoutTopLeft);
  fChSubFrame->AddFrame(fMenuBarCorss, fLayoutTopLeft);

  //...................... Menu covariances between samples

  fMenuCovss = new TGPopupMenu(gClient->GetRoot());                   fCnew++;
  fMenuCovss->AddEntry("2D, COLZ",fMenuCovssColzC);
  fMenuCovss->AddSeparator();
  fMenuCovss->AddEntry("3D, LEGO2Z",fMenuCovssLegoC);
  fMenuCovss->AddEntry("3D, SURF1Z",fMenuCovssSurf1C);
  fMenuCovss->AddEntry("3D, SURF4",fMenuCovssSurf4C);
  fMenuCovss->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarCovss = new TGMenuBar(fChSubFrame, 1, 1, kHorizontalFrame); fCnew++;
  fMenuBarCovss->AddPopup(xMenuBarCovss, fMenuCovss, fLayoutTopLeft);
  fChSubFrame->AddFrame(fMenuBarCovss, fLayoutTopLeft);

  //...................... Menu expectation values of the samples

  fMenuEv = new TGPopupMenu(gClient->GetRoot());                      fCnew++;
  fMenuEv->AddEntry("1D, HISTO ",fMenuEvLineFullC);
  fMenuEv->AddEntry("1D, HISTO SAME",fMenuEvLineSameC);
  fMenuEv->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarEv = new TGMenuBar(fChSubFrame, 1, 1, kHorizontalFrame);    fCnew++;
  fMenuBarEv->AddPopup(xMenuBarEvs, fMenuEv, fLayoutTopLeft);
  fChSubFrame->AddFrame(fMenuBarEv, fLayoutTopLeft);

  //...................... Menu sigmas/variances of the samples

  fMenuVar = new TGPopupMenu(gClient->GetRoot());                     fCnew++;
  fMenuVar->AddEntry("1D, HISTO ",fMenuVarLineFullC);
  fMenuVar->AddEntry("1D, HISTO SAME",fMenuVarLineSameC);
  fMenuVar->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarVar = new TGMenuBar(fChSubFrame, 1, 1, kHorizontalFrame);   fCnew++;         // 270
  fMenuBarVar->AddPopup(xMenuBarSigs, fMenuVar, fLayoutTopLeft);
  fChSubFrame->AddFrame(fMenuBarVar, fLayoutTopLeft);

  //------------------ Add Channel subframe to the frame 
  fLayoutChSubFrame =  new TGLayoutHints
    (kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);                   fCnew++;
  fChSpFrame->AddFrame(fChSubFrame, fLayoutChSubFrame);

  //=================================== SAMPLE
  fSpSubFrame = new TGCompositeFrame
    (fChSpFrame,60,20,kVerticalFrame, GetDefaultFrameBackground());   fCnew++;

  fSampFrame = new TGCompositeFrame
    (fSpSubFrame,60,20,kHorizontalFrame, kSunkenFrame);               fCnew++;

  fSampBut = new TGTextButton(fSampFrame, xSampButText, fSampButC);   fCnew++;
  fSampBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonSamp()");
  fSampBut->SetToolTipText("Click here to register the sample number written to the right");
  fSampBut->Resize(samp_buf_lenght, fSampBut->GetDefaultHeight());
  fLayoutSampBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);     fCnew++;
  fSampFrame->AddFrame(fSampBut, fLayoutSampBut);

  fEntrySampNumber = new TGTextBuffer();                              fCnew++;
  fSampText = new TGTextEntry(fSampFrame, fEntrySampNumber);          fCnew++;
  fSampText->SetToolTipText("Click and enter the sample number");
  fSampText->Resize(samp_buf_lenght, fSampText->GetDefaultHeight());
  DisplayInEntryField(fSampText,fKeySampNumber);
  fSampText->Connect("ReturnPressed()", "TCnaDialogEB",this, "DoButtonSamp()");
  fLayoutSampField =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1 );   fCnew++;
  fSampFrame->AddFrame(fSampText, fLayoutSampField);

  fSpSubFrame->AddFrame(fSampFrame,fLayoutGeneral);

  fLayoutSpSubFrame =  new TGLayoutHints
    (kLHintsTop | kLHintsRight, xB1, xB1, xB1, xB1);                  fCnew++;
  fChSpFrame->AddFrame(fSpSubFrame, fLayoutSpSubFrame);

  //---------------------- composite frame (channel/sample+menus)

  fLayoutChSpFrame =  new TGLayoutHints
    (kLHintsTop | kLHintsCenterX, xB1, xB1, xB1, xB1);                fCnew++;      // 280
  AddFrame(fChSpFrame, fLayoutChSpFrame);


 //====================== Menu histogram of the distribution
 //                       for a given (channel, sample)

  fMenuEvts = new TGPopupMenu(gClient->GetRoot());                    fCnew++;
  fMenuEvts->AddEntry("1D, HISTO",fMenuEvtsLineLinyFullC);
  fMenuEvts->AddEntry("1D, HISTO SAME",fMenuEvtsLineLinySameC);
  fMenuEvts->Connect("Activated(Int_t)", "TCnaDialogEB", this,
		     "HandleMenu(Int_t)");

  fMenuBarEvts = new TGMenuBar(this, 1, 1, kHorizontalFrame);         fCnew++;
  fMenuBarEvts->AddPopup("Event distribution for (Tower X Crystal, Sample)",
		     fMenuEvts, fLayoutGeneral);

  fLayoutMenuBarEvts =
    new TGLayoutHints(kLHintsCenterX, xB1,xB1,xB1,xB1);               fCnew++;
  AddFrame(fMenuBarEvts, fLayoutMenuBarEvts);

  AddFrame(fVoidFrame, fLayoutGeneral);

  //======================================== RUN LIST FOR HISTORY EVOLUTION
  TString xRunListButText  = " File name for TIME EVOLUTION plots ->  ";
  TString xMenuBarEvol     = " TIME EVOLUTION menu";
  Int_t run_list_buf_lenght  = 160;

  fRulFrame = new TGCompositeFrame(this,60,20, kHorizontalFrame, kSunkenFrame); fCnew++;
  
  fRulBut= new TGTextButton(fRulFrame, xRunListButText);                        fCnew++;
  fRulBut->Connect("Clicked()","TCnaDialogEB", this, "DoButtonRul()");
  fRulBut->SetToolTipText
    ("Click here to register the name of the file containing the run list (written on the right)");
  fLayoutRulBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);     fCnew++;
  fRulFrame->AddFrame(fRulBut,  fLayoutRulBut);

  fEntryRulNumber = new TGTextBuffer();                               fCnew++;
  fRulText = new TGTextEntry(fRulFrame, fEntryRulNumber);             fCnew++;
  fRulText->SetToolTipText("Click and enter the name of the file containing the run list");
  fRulText->Resize(run_list_buf_lenght, fRulText->GetDefaultHeight());
  fRulText->Connect("ReturnPressed()", "TCnaDialogEB", this, "DoButtonRul()");
  fLayoutRulFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsLeft, xB1,xB1,xB1,xB1);  fCnew++;
  fRulFrame->AddFrame(fRulText, fLayoutRulFieldText);

  fLayoutRulFieldFrame =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);     fCnew++;        // 290
  AddFrame(fRulFrame, fLayoutRulFieldFrame);

  //...................... Menu evolution in time

  fMenuEvol = new TGPopupMenu(gClient->GetRoot());                    fCnew++;
  fMenuEvol->AddSeparator();
  fMenuEvol->AddEntry("1D, Mean Pedestals ",fMenuEvolEvEvPolmFullC);
  fMenuEvol->AddEntry("1D, Mean Pedestals SAME",fMenuEvolEvEvPolmSameC);
  fMenuEvol->AddSeparator();
  fMenuEvol->AddEntry("1D, Mean of Sigmas ",fMenuEvolEvSigPolmFullC);
  fMenuEvol->AddEntry("1D, Mean of Sigmas SAME",fMenuEvolEvSigPolmSameC);
  fMenuEvol->AddSeparator();
  fMenuEvol->AddEntry("1D, Mean of cor(s,s) ",fMenuEvolEvCorssPolmFullC);
  fMenuEvol->AddEntry("1D, Mean of cor(s,s) SAME",fMenuEvolEvCorssPolmSameC);
  fMenuEvol->AddSeparator();
  fMenuEvol->AddEntry("1D, Pedestal a.f.o. evt number ",fMenuEvolSampLineFullC);
  fMenuEvol->AddEntry("1D, Pedestal a.f.o. evt number SAME",fMenuEvolSampLineSameC);

  fMenuEvol->Connect("Activated(Int_t)", "TCnaDialogEB", this, "HandleMenu(Int_t)");
  fMenuBarEvol = new TGMenuBar(this , 1, 1, kHorizontalFrame);        fCnew++;
  fMenuBarEvol->AddPopup(xMenuBarEvol, fMenuEvol, fLayoutTopLeft);
  AddFrame(fMenuBarEvol, fLayoutTopLeft);

  AddFrame(fVoidFrame, fLayoutGeneral);

  //======================== Last Buttons ==================================

  //========================== LinLog frame: buttons: LinX, LinY, LogX, LogY

  fLinLogFrame = new TGCompositeFrame(this,60,20, kHorizontalFrame, kSunkenFrame);    fCnew++;

  //-------------------------- Lin X <-> Log X
  TString xLogxButText     = " LOG X ";
  fButLogx = new TGCheckButton(fLinLogFrame, xLogxButText, fButLogxC);                fCnew++;
  fButLogx->Connect("Clicked()","TCnaDialogEB", this, "DoButtonLogx()");
  fLayoutLogxBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);   fCnew++;
  fLinLogFrame->AddFrame(fButLogx, fLayoutLogxBut);
  //-------------------------- Lin Y <-> Log Y
  TString xLogyButText     = " LOG Y ";
  fButLogy = new TGCheckButton(fLinLogFrame, xLogyButText, fButLogyC);                fCnew++;
  fButLogy->Connect("Clicked()","TCnaDialogEB", this, "DoButtonLogy()");
  fLayoutLogyBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);   fCnew++;
  fLinLogFrame->AddFrame(fButLogy, fLayoutLogyBut);

  AddFrame(fVoidFrame, fLayoutBottRight);
  AddFrame(fLinLogFrame, fLayoutGeneral);

  //========================== EXIT
  TString xExitButText     = " Exit ";
  fButExit = new TGTextButton(this, xExitButText, fButExitC);                         fCnew++;
  fButExit->Connect("Clicked()","TCnaDialogEB", this, "DoButtonExit()");
  //fButExit->SetCommand(".q");
  fLayoutExitBut = new TGLayoutHints(kLHintsTop | kLHintsRight, xB1, xB1, xB1, xB1);  fCnew++;
  AddFrame(fButExit, fLayoutExitBut);

  //========================== Last frame: buttons: ROOT version, Help

  fLastFrame = new TGCompositeFrame(this,60,20, kHorizontalFrame, kSunkenFrame);      fCnew++;   // 300

  //-------------------------- ROOT version
  TString xRootButText     = " ROOT Version ";
  fButRoot = new TGTextButton(fLastFrame, xRootButText, fButRootC);                   fCnew++;
  fButRoot->Connect("Clicked()","TCnaDialogEB", this, "DoButtonRoot()");
  fLayoutRootBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);   fCnew++;
  fLastFrame->AddFrame(fButRoot, fLayoutRootBut);

  //-------------------------- HELP
  TString xHelpButText     = " Help ";
  fButHelp = new TGTextButton(fLastFrame, xHelpButText, fButHelpC);                   fCnew++;
  fButHelp->Connect("Clicked()","TCnaDialogEB", this, "DoButtonHelp()");
  fLayoutHelpBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);   fCnew++;   // 304
  fLastFrame->AddFrame(fButHelp, fLayoutHelpBut);

  AddFrame(fVoidFrame, fLayoutBottRight);
  AddFrame(fLastFrame, fLayoutGeneral);

  //................................. Window

  MapSubwindows();
  Layout();

  SetWindowName("Correlated Noises Analysis (CNA) , ECAL Barrel");
  SetIconName("CNA");
  MapWindow();

  // cout << "TCnaDialogEB> Leaving constructor with arguments:" << endl;
  // cout << "            fCnew = " << fCnew << ", fCdelete = " << fCdelete << endl;

}
//   end of constructor with arguments

//###################################################################
//
//                        M E T H O D S
//
//###################################################################

//===============================================================
//
//                          Buttons
//
//===============================================================
void TCnaDialogEB::DoButtonAna()
{
//Registration of the type of the analysis

  char* bufferchain;
  bufferchain = (char*)fAnaText->GetBuffer()->GetString();

  fKeyAnaType = bufferchain;
  
  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of analysis name ---> "
       << fKeyAnaType << endl;
}

//----------------------------------------------------------------------
void TCnaDialogEB::DoButtonRun()
{
//Register run number or name of the file containing the list of run parameters

  //........................... get info from the entry field
  char* runchain = (char*)fRunText->GetBuffer()->GetString();
  char  tchiffr[10] = {'0', '1', '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9' };

  //............. test of the first character (figure => run number, letter => file name)
  if( runchain[0] == tchiffr [0] || runchain[0] == tchiffr [1] ||
      runchain[0] == tchiffr [2] || runchain[0] == tchiffr [3] ||
      runchain[0] == tchiffr [4] || runchain[0] == tchiffr [5] ||
      runchain[0] == tchiffr [6] || runchain[0] == tchiffr [7] ||
      runchain[0] == tchiffr [8] || runchain[0] == tchiffr [9] )
    {
      fKeyRunNumberTString = (TString)runchain;  
      fKeyRunNumber = atoi(runchain);
      fCnaCommand++;
      cout << "   *CNA [" << fCnaCommand
	   << "]> Registration of run number -------> "
	   << fKeyRunNumber << endl;
    }
  else
    {
      fCnaError++;
      cout << "   !CNA (" << fCnaError << ") *ERROR* ===> "
	   << " Please, enter a number."
	   << fTTBELL << endl;
    }
}

//-------------------------------------------------------------------
void TCnaDialogEB::DoButtonFev()
{
//Registration of the first event number

  char* bufferchain;
  bufferchain = (char*)fFevText->GetBuffer()->GetString();
  fKeyFirstEvt = atoi(bufferchain);

  if ( fKeyFirstEvt < 0)
    {
      fCnaError++;
      cout << "   !CNA (" << fCnaError << ") *WARNING* ===> "
	   << " First event number = " << fKeyFirstEvt
	   << ": negative. " << endl 
	   << "                                 Will be forced to zero."
	   << fTTBELL << endl;
    }

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of first taken event number -> "
       << fKeyFirstEvt << endl;
}
//-------------------------------------------------------------------
void TCnaDialogEB::DoButtonNoe()
{
//Registration of the number of events

  char* bufferchain;
  bufferchain = (char*)fNoeText->GetBuffer()->GetString();
  fKeyNbOfEvts = atoi(bufferchain);

  if ( fKeyNbOfEvts <= 0)
    {
      fCnaError++;
      cout << "   !CNA (" << fCnaError << ") *WARNING* ===> "
	   << " Number of events = " << fKeyNbOfEvts
	   << ": null or negative." << endl
	   << "                 Will be forced to the number of entries "
	   << fTTBELL << endl;
    }

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of number of taken events -> "
       << fKeyNbOfEvts << endl;
}

//-------------------------------------------------------------------
void TCnaDialogEB::DoButtonSuMo()
{
//Registration of the Super-Module number

  char* bufferchain;
  bufferchain = (char*)fSuMoText->GetBuffer()->GetString();
  fKeySuMoNumberTString = (TString)bufferchain;
  fKeySuMoNumber = atoi(bufferchain);
  
  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Super-Module number -> "
       << fKeySuMoNumber << endl;

  TEBParameters* MyEcalParameters = new TEBParameters();    fCnew++;

  if ( (fKeySuMoNumber < 1) || (fKeySuMoNumber > MyEcalParameters->MaxSMInBarrel() )  )
    {
      fCnaError++;
      cout << "   !CNA (" << fCnaError << ") *** ERROR *** ===> "
     	   << " SuperModule number = " << fKeySuMoNumber
     	   << ": out of range ( range = [ 1 ," << MyEcalParameters->MaxSMInBarrel() << " ] )"
     	   << fTTBELL << endl;
    }

  delete MyEcalParameters;                                      fCdelete++;
}
//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVminFoundEvts()
{
//Registration of Ymin for number of found events

  char* bufferchain;
  bufferchain = (char*)fVminFoundEvtsText->GetBuffer()->GetString();

  fKeyVminFoundEvts = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymin for plot of number of found events -> "
       << fKeyVminFoundEvts << endl;
}
//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVmaxFoundEvts()
{
//Registration of Ymax for number of found events

  char* bufferchain;
  bufferchain = (char*)fVmaxFoundEvtsText->GetBuffer()->GetString();

  fKeyVmaxFoundEvts = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymax for plot of number of found events -> "
       << fKeyVmaxFoundEvts << endl;
}
//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVminEvEv()
{
//Registration of Ymin for mean pedestals

  char* bufferchain;
  bufferchain = (char*)fVminEvEvText->GetBuffer()->GetString();

  fKeyVminEvEv = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymin for plot of mean pedestals -> "
       << fKeyVminEvEv << endl;
}
//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVmaxEvEv()
{
//Registration of Ymax for mean pedestals

  char* bufferchain;
  bufferchain = (char*)fVmaxEvEvText->GetBuffer()->GetString();

  fKeyVmaxEvEv = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymax for plot of mean pedestals -> "
       << fKeyVmaxEvEv << endl;
}
//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVminEvSig()
{
//Registration of Ymin for mean of sample sigmas (noise)

  char* bufferchain;
  bufferchain = (char*)fVminEvSigText->GetBuffer()->GetString();

  fKeyVminEvSig = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymin for plot of mean of sample sigmas (noise) -> "
       << fKeyVminEvSig << endl;
}
//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVmaxEvSig()
{
//Registration of Ymax for mean of sample sigmas (noise)

  char* bufferchain;
  bufferchain = (char*)fVmaxEvSigText->GetBuffer()->GetString();

  fKeyVmaxEvSig = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymax for plot of mean of sample sigmas (noise) -> "
       << fKeyVmaxEvSig << endl;
}
//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVminEvCorss()
{
//Registration of Ymin for mean of cor(s,s)

  char* bufferchain;
  bufferchain = (char*)fVminEvCorssText->GetBuffer()->GetString();

  fKeyVminEvCorss = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymin for plot of mean of cor(s,s) -> "
       << fKeyVminEvCorss << endl;
}
//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVmaxEvCorss()
{
//Registration of Ymax for mean of cor(s,s)

  char* bufferchain;
  bufferchain = (char*)fVmaxEvCorssText->GetBuffer()->GetString();

  fKeyVmaxEvCorss = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymax for plot of mean of cor(s,s) -> "
       << fKeyVmaxEvCorss << endl;
}
//-------------------------------------------------------------------


void TCnaDialogEB::DoButtonVminSigEv()
{
//Registration of Ymin for sigmas of sample means

  char* bufferchain;
  bufferchain = (char*)fVminSigEvText->GetBuffer()->GetString();

  fKeyVminSigEv = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymin for plot of sigmas of sample means -> "
       << fKeyVminSigEv << endl;
}
//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVmaxSigEv()
{
//Registration of Ymax for sigmas of sample means 

  char* bufferchain;
  bufferchain = (char*)fVmaxSigEvText->GetBuffer()->GetString();

  fKeyVmaxSigEv = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymax for plot of sigmas of sample means -> "
       << fKeyVmaxSigEv << endl;
}
//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVminSigSig()
{
//Registration of Ymin for sigmas of sample sigmas

  char* bufferchain;
  bufferchain = (char*)fVminSigSigText->GetBuffer()->GetString();

  fKeyVminSigSig = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymin for plot of sigmas of sample sigmas -> "
       << fKeyVminSigSig << endl;
}
//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVmaxSigSig()
{
//Registration of Ymax for sigmas of sample sigmas

  char* bufferchain;
  bufferchain = (char*)fVmaxSigSigText->GetBuffer()->GetString();

  fKeyVmaxSigSig = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymax for plot of sigmas of sample sigmas -> "
       << fKeyVmaxSigSig << endl;
}
//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVminSigCorss()
{
//Registration of Ymin for sigmas of cor(s,s)

  char* bufferchain;
  bufferchain = (char*)fVminSigCorssText->GetBuffer()->GetString();

  fKeyVminSigCorss = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymin for plot of sigmas of cor(s,s) -> "
       << fKeyVminSigCorss << endl;
}
//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVmaxSigCorss()
{
//Registration of Ymax for sigmas of cor(s,s)

  char* bufferchain;
  bufferchain = (char*)fVmaxSigCorssText->GetBuffer()->GetString();

  fKeyVmaxSigCorss = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymax for plot of sigmas of cor(s,s) -> "
       << fKeyVmaxSigCorss << endl;
}
//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVminEvCortt()
{
//Registration of Ymin for cor(t,t)

  char* bufferchain;
  bufferchain = (char*)fVminEvCorttText->GetBuffer()->GetString();

  fKeyVminEvCortt = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymin for plot of cor(t,t) -> "
       << fKeyVminEvCortt << endl;
}
//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVmaxEvCortt()
{
//Registration of Ymax for cor(t,t)

  char* bufferchain;
  bufferchain = (char*)fVmaxEvCorttText->GetBuffer()->GetString();

  fKeyVmaxEvCortt = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymax for plot of cor(t,t) -> "
       << fKeyVmaxEvCortt << endl;
}

//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVminEvCovtt()
{
//Registration of Ymin for cor(c,c) in towers

  char* bufferchain;
  bufferchain = (char*)fVminEvCovttText->GetBuffer()->GetString();

  fKeyVminEvCovtt = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymin for plot of cor(c,c) in towers -> "
       << fKeyVminEvCovtt << endl;
}
//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonVmaxEvCovtt()
{
//Registration of Ymax for cor(c,c) in towers

  char* bufferchain;
  bufferchain = (char*)fVmaxEvCovttText->GetBuffer()->GetString();

  fKeyVmaxEvCovtt = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of Ymax for plot of cor(c,c) in towers -> "
       << fKeyVmaxEvCovtt << endl;
}

//-------------------------------------------------------------------

void TCnaDialogEB::DoButtonSMNb()
{
  //  fCnaCommand++;
  //  cout << "   *CNA [" << fCnaCommand
  //       << "]> Tower numbering global view for SuperModule number "
  //       << fKeySuMoNumber << endl;

  ViewSMTowerNumbering();
}

//-------------------------------------------------------------------
void TCnaDialogEB::DoButtonChan()
{
//Registration of the channel number

  char* bufferchain;
  bufferchain = (char*)fChanText->GetBuffer()->GetString();
  fKeyChanNumber = atoi(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of channel number ---> "
       << fKeyChanNumber << endl;

  TEBParameters* MyEcalParameters = new TEBParameters();    fCnew++;

  if ( (fKeyChanNumber < 0) || (fKeyChanNumber > MyEcalParameters->MaxCrysInTow()-1 )  )
    {
      fCnaError++;
      cout << "   !CNA (" << fCnaError << ") *** ERROR *** ===> "
     	   << " Channel number in tower = " << fKeyChanNumber
     	   << ": out of range ( range = [ 0 ," << MyEcalParameters->MaxCrysInTow()-1 << " ] )"
     	   << fTTBELL << endl;
    }
  delete MyEcalParameters;                                      fCdelete++;
}
//-------------------------------------------------------------------
void TCnaDialogEB::DoButtonSamp()
{
//Registration of the sample number

  char* bufferchain;
  bufferchain = (char*)fSampText->GetBuffer()->GetString();
  fKeySampNumber = atoi(bufferchain);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of sample number ----> "
       << fKeySampNumber << endl;

  TEBParameters* MyEcalParameters = new TEBParameters();    fCnew++;

  if ( (fKeySampNumber < 0) || (fKeySampNumber > MyEcalParameters->MaxSampADC()-1 )  )
    {
      fCnaError++;
      cout << "   !CNA (" << fCnaError << ") *** ERROR *** ===> "
     	   << " Sample number = " << fKeySampNumber
     	   << ": out of range ( range = [ 0 ," << MyEcalParameters->MaxSampADC()-1 << " ] )"
     	   << fTTBELL << endl;
    }
  
  delete MyEcalParameters;                                      fCdelete++;
}
//-------------------------------------------------------------------
void TCnaDialogEB::DoButtonTowX()
{
//Registration of the tower X number

  char* bufferchain;
  bufferchain = (char*)fTowXText->GetBuffer()->GetString();
  fKeyTowXNumber = atoi(bufferchain);
  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of tower X number ---> "
       << fKeyTowXNumber << endl;

  TEBParameters* MyEcalParameters = new TEBParameters();    fCnew++;

  if ( (fKeyTowXNumber < 1) || (fKeyTowXNumber > MyEcalParameters->MaxTowInSM() )  )
    {
      fCnaError++;
      cout << "   !CNA (" << fCnaError << ") *** ERROR *** ===> "
     	   << " Tower number = " << fKeyTowXNumber
     	   << ": out of range ( range = [ 1 ," << MyEcalParameters->MaxTowInSM() << " ] ) "
     	   << fTTBELL << endl;
    }
  
  delete MyEcalParameters;                                      fCdelete++;
}

//-------------------------------------------------------------------
void TCnaDialogEB::DoButtonChNb()
{
  //  fCnaCommand++;
  //  cout << "   *CNA [" << fCnaCommand
  //       << "]> Channel numbering global view for tower number "
  //       << fKeyTowXNumber << endl;

  ViewTowerCrystalNumbering(fKeyTowXNumber);
}

//-------------------------------------------------------------------
void TCnaDialogEB::DoButtonTowY()
{
//Registration of the tower Y number

  char* bufferchain;
  bufferchain = (char*)fTowYText->GetBuffer()->GetString();
  fKeyTowYNumber = atoi(bufferchain);
  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Registration of tower Y number ---> "
       << fKeyTowYNumber << endl;

  TEBParameters* MyEcalParameters = new TEBParameters();    fCnew++;

  if ( (fKeyTowYNumber < 1) || (fKeyTowYNumber > MyEcalParameters->MaxTowInSM() )  )
    {
      fCnaError++;
      cout << "   !CNA (" << fCnaError << ") *** ERROR *** ===> "
     	   << " Tower number = " << fKeyTowYNumber
     	   << ": out of range ( range = [ 1 ," << MyEcalParameters->MaxTowInSM() << " ] )"
     	   << fTTBELL << endl;
    }
  
  delete MyEcalParameters;                                      fCdelete++;
}

//----------------------------------------------------------------------
void TCnaDialogEB::DoButtonRul()
{
//Register the name of the file containing the list of run parameters

  //........................... get info from the entry field
  char* listchain = (char*)fRulText->GetBuffer()->GetString();
  char tchiffr[10] = {'0', '1', '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9' };

  //............. test of the first character (figure => run number, letter => file name)
  if( listchain[0] == tchiffr [0] || listchain[0] == tchiffr [1] ||
      listchain[0] == tchiffr [2] || listchain[0] == tchiffr [3] ||
      listchain[0] == tchiffr [4] || listchain[0] == tchiffr [5] ||
      listchain[0] == tchiffr [6] || listchain[0] == tchiffr [7] ||
      listchain[0] == tchiffr [8] || listchain[0] == tchiffr [9] )
    {
      fCnaError++;
      cout << "   !CNA (" << fCnaError << ") *ERROR* ===> "
	   << " Please, enter a file name."
	   << fTTBELL << endl;
    }
  else
    {
      fKeyFileNameRunList = listchain;

      fCnaCommand++;
      cout << "   *CNA [" << fCnaCommand
	   << "]> Registration of file name for list of run parameters -------> "
	   << fKeyFileNameRunList.Data() << endl;
    }
}

//===================================================================
//
//                       HandleMenu
//
//===================================================================
void TCnaDialogEB::HandleMenu(Int_t id)
{
  //HandleMenu
  
  //.................... Found events in the Super-Module
  if( id == fMenuFoundEvtsGlobalFullC ){ViewHistoSuperModuleFoundEventsOfCrystals(fOptPlotFull);}
  if( id == fMenuFoundEvtsGlobalSameC ){ViewHistoSuperModuleFoundEventsOfCrystals(fOptPlotSame);}
  if( id == fMenuFoundEvtsProjFullC   ){ViewHistoSuperModuleFoundEventsDistribution(fOptPlotFull);}
  if( id == fMenuFoundEvtsProjSameC   ){ViewHistoSuperModuleFoundEventsDistribution(fOptPlotSame);}

  //.................... Ev of Ev in the Super-Module
  if( id == fMenuEvEvGlobalFullC ){ViewHistoSuperModuleMeanPedestalsOfCrystals(fOptPlotFull);}
  if( id == fMenuEvEvGlobalSameC ){ViewHistoSuperModuleMeanPedestalsOfCrystals(fOptPlotSame);}
  if( id == fMenuEvEvProjFullC   ){ViewHistoSuperModuleMeanPedestalsDistribution(fOptPlotFull);}
  if( id == fMenuEvEvProjSameC   ){ViewHistoSuperModuleMeanPedestalsDistribution(fOptPlotSame);}

  //.................... Ev of Sig in the Super-Module
  if( id == fMenuEvSigGlobalFullC ){ViewHistoSuperModuleMeanOfSampleSigmasOfCrystals(fOptPlotFull);}
  if( id == fMenuEvSigGlobalSameC ){ViewHistoSuperModuleMeanOfSampleSigmasOfCrystals(fOptPlotSame);}
  if( id == fMenuEvSigProjFullC   ){ViewHistoSuperModuleMeanOfSampleSigmasDistribution(fOptPlotFull);}
  if( id == fMenuEvSigProjSameC   ){ViewHistoSuperModuleMeanOfSampleSigmasDistribution(fOptPlotSame);}

  //.................... Ev of Corss in the Super-Module
  if( id == fMenuEvCorssGlobalFullC ){ViewHistoSuperModuleMeanOfCorssOfCrystals(fOptPlotFull);}
  if( id == fMenuEvCorssGlobalSameC ){ViewHistoSuperModuleMeanOfCorssOfCrystals(fOptPlotSame);}
  if( id == fMenuEvCorssProjFullC   ){ViewHistoSuperModuleMeanOfCorssDistribution(fOptPlotFull);}
  if( id == fMenuEvCorssProjSameC   ){ViewHistoSuperModuleMeanOfCorssDistribution(fOptPlotSame);}

  //.................... Sig of Ev in the Super-Module
  if( id == fMenuSigEvGlobalFullC ){ViewHistoSuperModuleSigmaPedestalsOfCrystals(fOptPlotFull);}
  if( id == fMenuSigEvGlobalSameC ){ViewHistoSuperModuleSigmaPedestalsOfCrystals(fOptPlotSame);}
  if( id == fMenuSigEvProjFullC   ){ViewHistoSuperModuleSigmaPedestalsDistribution(fOptPlotFull);}
  if( id == fMenuSigEvProjSameC   ){ViewHistoSuperModuleSigmaPedestalsDistribution(fOptPlotSame);}

  //.................... Sig of Sig in the Super-Module
  if( id == fMenuSigSigGlobalFullC ){ViewHistoSuperModuleSigmaOfSampleSigmasOfCrystals(fOptPlotFull);}
  if( id == fMenuSigSigGlobalSameC ){ViewHistoSuperModuleSigmaOfSampleSigmasOfCrystals(fOptPlotSame);}
  if( id == fMenuSigSigProjFullC   ){ViewHistoSuperModuleSigmaOfSampleSigmasDistribution(fOptPlotFull);}
  if( id == fMenuSigSigProjSameC   ){ViewHistoSuperModuleSigmaOfSampleSigmasDistribution(fOptPlotSame);}

  //.................... Sig of Corss in the Super-Module
  if( id == fMenuSigCorssGlobalFullC ){ViewHistoSuperModuleSigmaOfCorssOfCrystals(fOptPlotFull);}
  if( id == fMenuSigCorssGlobalSameC ){ViewHistoSuperModuleSigmaOfCorssOfCrystals(fOptPlotSame);}
  if( id == fMenuSigCorssProjFullC   ){ViewHistoSuperModuleSigmaOfCorssDistribution(fOptPlotFull);}
  if( id == fMenuSigCorssProjSameC   ){ViewHistoSuperModuleSigmaOfCorssDistribution(fOptPlotSame);}

  //....................... (eta,phi) super-module viewing
  if( id ==  fMenuFoundEvtsEtaPhiC){ViewSuperModuleFoundEvents();}
  if( id ==  fMenuEvEvEtaPhiC     ){ViewSuperModuleMeanPedestals();}
  if( id ==  fMenuEvSigEtaPhiC    ){ViewSuperModuleMeanOfSampleSigmas();}
  if( id ==  fMenuEvCorssEtaPhiC  ){ViewSuperModuleMeanOfCorss();}
  if( id ==  fMenuSigEvEtaPhiC    ){ViewSuperModuleSigmaPedestals();}
  if( id ==  fMenuSigSigEtaPhiC   ){ViewSuperModuleSigmaOfSampleSigmas();}
  if( id ==  fMenuSigCorssEtaPhiC ){ViewSuperModuleSigmaOfCorss();}

  //............................... Correlations and covariances between towers
  if( id == fMenuCorttColzC ){ViewMatrixCorrelationTowers("COLZ");}
  if( id == fMenuCorttLegoC ){ViewMatrixCorrelationTowers("LEGO2Z");}

  if( id == fMenuCovttColzC ){ViewSuperModuleCorccMeanOverSamples();}  // noms a reprendre Covtt-> Corcc ?
  //if( id == fMenuCovttColzC ){ViewMatrixCovarianceTowers("COLZ");}
  //if( id == fMenuCovttLegoC ){ViewMatrixCovarianceTowers("LEGO2Z");}

  //............................... Correlations and covariances between channels
  if( id == fMenuCorccColzC ){ViewMatrixCorrelationCrystals(fKeyTowXNumber, fKeyTowYNumber, "COLZ");}
  if( id == fMenuCorccLegoC ){ViewMatrixCorrelationCrystals(fKeyTowXNumber, fKeyTowYNumber, "LEGO2Z");}

  if( id == fMenuCovccColzC ){ViewMatrixCovarianceCrystals(fKeyTowXNumber, fKeyTowYNumber, "COLZ");}
  if( id == fMenuCovccLegoC ){ViewMatrixCovarianceCrystals(fKeyTowXNumber, fKeyTowYNumber, "LEGO2Z");}

  //.................................... Correlations and covariances between samples

  if( id == fMenuCorssColzC  ){ViewMatrixCorrelationSamples(fKeyTowXNumber, fKeyChanNumber, "COLZ");}
  if( id == fMenuCorssLegoC  ){ViewMatrixCorrelationSamples(fKeyTowXNumber, fKeyChanNumber, "LEGO2Z");}
  if( id == fMenuCorssSurf1C ){ViewMatrixCorrelationSamples(fKeyTowXNumber, fKeyChanNumber, "SURF1Z");}
  if( id == fMenuCorssSurf4C ){ViewMatrixCorrelationSamples(fKeyTowXNumber, fKeyChanNumber, "SURF4");}

  if( id == fMenuCovssColzC  ){ViewMatrixCovarianceSamples(fKeyTowXNumber, fKeyChanNumber, "COLZ");}
  if( id == fMenuCovssLegoC  ){ViewMatrixCovarianceSamples(fKeyTowXNumber, fKeyChanNumber, "LEGO2Z");}
  if( id == fMenuCovssSurf1C ){ViewMatrixCovarianceSamples(fKeyTowXNumber, fKeyChanNumber, "SURF1Z");}
  if( id == fMenuCovssSurf4C ){ViewMatrixCovarianceSamples(fKeyTowXNumber, fKeyChanNumber, "SURF4");}

  //.................... Correlations and covariances between samples for all channels of a tower
  if( id == fMenuCorssAllColzC ){ViewTowerCorrelationSamples(fKeyTowXNumber);}
  if( id == fMenuCovssAllColzC ){ViewTowerCovarianceSamples(fKeyTowXNumber);}
     
  //..................................... Expectation values of the samples (mean pedestals)
  if( id == fMenuEvLineFullC )
    {ViewHistoCrystalExpectationValuesOfSamples(fKeyTowXNumber, fKeyChanNumber, fOptPlotFull);}
  if( id == fMenuEvLineSameC )
    {ViewHistoCrystalExpectationValuesOfSamples(fKeyTowXNumber, fKeyChanNumber, fOptPlotSame);}
  
  //..................................... Sigmas of the samples
  if( id == fMenuVarLineFullC )
    {ViewHistoCrystalSigmasOfSamples(fKeyTowXNumber, fKeyChanNumber, fOptPlotFull);}
  if( id == fMenuVarLineSameC )
    {ViewHistoCrystalSigmasOfSamples(fKeyTowXNumber, fKeyChanNumber, fOptPlotSame);}

  //..................................... Evolution in time (ViewHistime, except EvolSamp -> Viewhisto)

  if( id == fMenuEvolEvEvPolmFullC )
    {TString run_par_file_name = fKeyFileNameRunList.Data();
    ViewHistimeCrystalMeanPedestals(run_par_file_name, fKeyTowXNumber, fKeyChanNumber, fOptPlotFull);}
  if( id == fMenuEvolEvEvPolmSameC )
    {TString run_par_file_name = fKeyFileNameRunList.Data();
    ViewHistimeCrystalMeanPedestals(run_par_file_name, fKeyTowXNumber, fKeyChanNumber, fOptPlotSame);}

  if( id == fMenuEvolEvSigPolmFullC )
    {TString run_par_file_name = fKeyFileNameRunList.Data();
    ViewHistimeCrystalMeanSigmas(run_par_file_name, fKeyTowXNumber, fKeyChanNumber, fOptPlotFull);}
  if( id == fMenuEvolEvSigPolmSameC )
    {TString run_par_file_name = fKeyFileNameRunList.Data();
    ViewHistimeCrystalMeanSigmas(run_par_file_name, fKeyTowXNumber, fKeyChanNumber, fOptPlotSame);}
  
  if( id == fMenuEvolEvCorssPolmFullC )
    {TString run_par_file_name = fKeyFileNameRunList.Data();
    ViewHistimeCrystalMeanCorss(run_par_file_name, fKeyTowXNumber, fKeyChanNumber, fOptPlotFull);}
  if( id == fMenuEvolEvCorssPolmSameC )
    {TString run_par_file_name = fKeyFileNameRunList.Data();
    ViewHistimeCrystalMeanCorss(run_par_file_name, fKeyTowXNumber, fKeyChanNumber, fOptPlotSame);}
  
  if( id == fMenuEvolSampLineFullC )
    {ViewHistoCrystalPedestalEventNumber(fKeyTowXNumber, fKeyChanNumber, fOptPlotFull);}
  if( id == fMenuEvolSampLineSameC )
    {ViewHistoCrystalPedestalEventNumber(fKeyTowXNumber, fKeyChanNumber, fOptPlotSame);}

  //..................................... Event distribution
  if( id == fMenuEvtsLineLinyFullC )
    {ViewHistoSampleEventDistribution(fKeyTowXNumber, fKeyChanNumber, fKeySampNumber, fOptPlotFull);}
  if( id == fMenuEvtsLineLinySameC )
    {ViewHistoSampleEventDistribution(fKeyTowXNumber, fKeyChanNumber, fKeySampNumber, fOptPlotSame);}
}

//-------------------------------------------------------------------
//
//                        Last buttons methods
//
//-------------------------------------------------------------------
//======================= LIN-LOG FRAME

void TCnaDialogEB::DoButtonLogx()
{
  if( fMemoScaleX == "LOG") {fKeyScaleX = "LIN";}
  if( fMemoScaleX == "LIN") {fKeyScaleX = "LOG";}
  fMemoScaleX = fKeyScaleX;

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Set X axis on " << fKeyScaleX << " scale " << endl;
}
void TCnaDialogEB::DoButtonLogy()
{
  if( fMemoScaleY == "LOG" ) {fKeyScaleY = "LIN";}
  if( fMemoScaleY == "LIN" ) {fKeyScaleY = "LOG";}
  fMemoScaleY = fKeyScaleY;

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Set Y axis on " << fKeyScaleY << " scale " << endl;
}

//======================= LAST FRAME

//-------------------------------------------------------------------
void TCnaDialogEB::DoButtonRoot()
{
  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> This is ROOT version " << gROOT->GetVersion()
       << endl;

#define JBAT
#ifndef JBAT

  //====================== prepa commande de soumission de job en batch
  //...... fabrication du script
  
  TString submit_script_name = "b";
  Text_t *t_run_number = (Text_t *)fKeyRunNumberTString.Data();
  submit_script_name.Append(t_run_number);
  submit_script_name.Append('_');
  Text_t *t_SM_number = (Text_t *)fKeySuMoNumberTString.Data();
  submit_script_name.Append(t_SM_number);

  cout << "TCnaDialogEB> CONTROLE 1. Open file " << submit_script_name.Data() << endl;

  TString s_path_name = gSystem->Getenv("HOME");
  TString s_cmssw_version = "/cmssw/ECALTBH4_0_4_0_pre1"; 

  fFcout_f.open(submit_script_name); 
  fFcout_f << "cd " << s_path_name.Data() << s_cmssw_version.Data() << endl
	   << "eval `scramv1 runtime -sh`" << endl
	   << "cd " << fCfgResultsRootFilePath.Data() << endl
	   << "cmsRun " << s_path_name.Data() << s_cmssw_version.Data() << "/src/CalibCalorimetry/EcalCorrelatedNoiseAnalysisModules/data/EcalPedestalRun"<< fKeyRunNumberTString << ".cfg" << endl;

  //...... envoi de la commande
  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Submit job from script cna001 "
       << endl;

  //system("qsub -q 1nh bsubmit");
  //Int_t i_exec = gSystem->Exec("qsub -q 1nh cna001");
  //cout << "TCnaDialogEB> CONTROLE 1. i_exec = " << i_exec << endl;

  fFcout_f.close();
  cout << "TCnaDialogEB> CONTROLE 2. Close file " <<  submit_script_name.Data() << endl;

#endif // JBAT

}

//-------------------------------------------------------------------
void TCnaDialogEB::DoButtonHelp()
{
  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> YOU NEED HELP. Please, have a look at the web page -> "
       << "http://www.cern.ch/cms-fabbro/cna"
       << endl;
}
//-------------------------------------------------------------------
void TCnaDialogEB::DoButtonExit()
{
  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Exit CNA session."
       << endl;
  //............................ Quit the ROOT session
  fButExit->SetCommand(".q");
}

//==========================================================================
//
//                  View  Matrix
//
//==========================================================================
void TCnaDialogEB::ViewMatrixCorrelationTowers(const TString option_plot)
{
// Plot of correlation matrix between towers (mean over crystals and samples)

  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Plot of correlation matrix between towers (mean over crystals and samples). Option: "
       << option_plot << endl;

  MyView->PutYmin("SMEvCorttMatrix",fKeyVminEvCortt);
  MyView->PutYmax("SMEvCorttMatrix",fKeyVmaxEvCortt);     
  MyView->CorrelationsBetweenTowers(option_plot);
  
  delete MyView;                       fCdelete++;
}

void TCnaDialogEB::ViewMatrixCovarianceTowers(const TString option_plot)
{
// Plot of covariance matrix between towers (mean over crystals and samples)

  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);
  
  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Plot of covariance matrix between towers (mean over crystals and samples). Option: "
       << option_plot << endl;

  MyView->PutYmin("SMEvCovttMatrix",fKeyVminEvCovtt);
  MyView->PutYmax("SMEvCovttMatrix",fKeyVmaxEvCovtt);   
  MyView->CovariancesBetweenTowers(option_plot);
  
  delete MyView;                       fCdelete++;
}

void TCnaDialogEB::ViewMatrixCorrelationCrystals(const Int_t&  SMtower_X, const Int_t& SMtower_Y,
					       const TString option_plot)
{
// Plot of correlation matrix (crystal of tower X, crystal of tower Y) (mean over samples)

  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);
  
  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Plot of correlation matrix between crystals (mean over samples). Tower X: "
       << SMtower_X  << ", Tower Y: " << SMtower_Y << ", option: " << option_plot << endl;

  MyView->PutYmin("SMEvCorttMatrix",fKeyVminEvCortt);
  MyView->PutYmax("SMEvCorttMatrix",fKeyVmaxEvCortt);   
  MyView->CorrelationsBetweenCrystals( SMtower_X, SMtower_Y, option_plot);
  
  delete MyView;                       fCdelete++;
}

void TCnaDialogEB::ViewMatrixCovarianceCrystals(const Int_t&  SMtower_X, const Int_t& SMtower_Y,
					      const TString option_plot)
{
// Plot of covariance matrix (crystal of tower X, crystal of tower Y) (mean over samples)

  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);
  
  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Plot of covariance matrix between crystals (mean over samples). Tower X: "
       << SMtower_X  << ", Tower Y: " << SMtower_Y << ", option: " << option_plot << endl;

  MyView->PutYmin("SMEvCovttMatrix",fKeyVminEvCovtt);
  MyView->PutYmax("SMEvCovttMatrix",fKeyVmaxEvCovtt);    
  MyView->CovariancesBetweenCrystals( SMtower_X, SMtower_Y, option_plot);
  
  delete MyView;                       fCdelete++;
}

void TCnaDialogEB::ViewMatrixCorrelationSamples(const Int_t&  SMtower_X, const Int_t& crystal,
					      const TString option_plot)
{
// Plot of correlation matrix between samples for a given crystal

  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);
  
  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Plot of correlation matrix between samples. Tower "
       << SMtower_X  << ", crystal " << crystal << ", option: " << option_plot << endl;
 
  MyView->PutYmin("SMEvCorssGlobal",fKeyVminEvCorss);
  MyView->PutYmax("SMEvCorssGlobal",fKeyVmaxEvCorss); 
  MyView->CorrelationsBetweenSamples(SMtower_X, crystal, option_plot);
  
  delete MyView;                       fCdelete++;
}

void TCnaDialogEB::ViewMatrixCovarianceSamples(const Int_t&  SMtower_X, const Int_t& crystal,
					     const TString option_plot)
{
// Plot of covariance matrix between samples for a given crystal

  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);
  
  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Plot of covariance matrix between samples. Tower "
       << SMtower_X  << ", crystal " << crystal << ", option: " << option_plot << endl;

  MyView->PutYmin("SMEvSigGlobal",fKeyVminEvSig);  // same as mean of sample sigmas
  MyView->PutYmax("SMEvSigGlobal",fKeyVmaxEvSig);      
  MyView->CovariancesBetweenSamples( SMtower_X, crystal, option_plot);
  
  delete MyView;                       fCdelete++;
}

//==========================================================================
//
//                         ViewTower      
//
//     SMtower ==> (sample,sample) matrices for all the crystal of SMtower              
//
//==========================================================================
void TCnaDialogEB::ViewTowerCorrelationSamples(const Int_t& SMtower)
{
  // Plot of (sample,sample) correlation matrices for all the crystal of a given tower  
  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);
  
  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Plot of correlation matrices between samples for each crystal of tower " << SMtower << endl;
  
  MyView->PutYmin("SMEvCorssGlobal",fKeyVminEvCorss);
  MyView->PutYmax("SMEvCorssGlobal",fKeyVmaxEvCorss); 
  MyView->CorrelationsBetweenSamples(SMtower);
  
  delete MyView;                       fCdelete++;
}

void TCnaDialogEB::ViewTowerCovarianceSamples(const Int_t& SMtower)
{
  // Plot of (sample,sample) covariance matrices for all the crystal of a given tower  
  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);
  
  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Plot of covariance matrices between samples for each crystal of tower "
       << SMtower << endl;

  MyView->PutYmin("SMEvSigGlobal",fKeyVminEvSig);   // same as mean of sample sigmas
  MyView->PutYmax("SMEvSigGlobal",fKeyVmaxEvSig);   
  MyView->CovariancesBetweenSamples(SMtower);

  delete MyView;                       fCdelete++;
}
//==========================================================================
//
//                         ViewSuperModule (eta,phi)
//
//==========================================================================

void TCnaDialogEB::ViewSuperModuleFoundEvents()
{
  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> 2D plot of Found Events. Supermodule: " << fKeySuMoNumber << endl;
  
  MyView->PutYmin("SMFoundEvtsGlobal",fKeyVminFoundEvts);
  MyView->PutYmax("SMFoundEvtsGlobal",fKeyVmaxFoundEvts);   
  MyView->EtaPhiSuperModuleFoundEvents();
  
  delete MyView;                       fCdelete++;
}

void TCnaDialogEB::ViewSuperModuleMeanPedestals()
{
  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> 2D plot of Mean Pedestals. Supermodule: " << fKeySuMoNumber << endl;

  MyView->PutYmin("SMEvEvGlobal",fKeyVminEvEv);
  MyView->PutYmax("SMEvEvGlobal",fKeyVmaxEvEv);     
  MyView->EtaPhiSuperModuleMeanPedestals();
  
  delete MyView;                       fCdelete++;
}

void TCnaDialogEB::ViewSuperModuleMeanOfSampleSigmas()
{
  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> 2D plot of Mean of Sample Sigmas. Supermodule: " << fKeySuMoNumber << endl;

  MyView->PutYmin("SMEvSigGlobal",fKeyVminEvSig);
  MyView->PutYmax("SMEvSigGlobal",fKeyVmaxEvSig);     
  MyView->EtaPhiSuperModuleMeanOfSampleSigmas();
  
  delete MyView;                      fCdelete++;
}

void TCnaDialogEB::ViewSuperModuleMeanOfCorss()
{
  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> 2D plot of Mean of Corss. Supermodule: " << fKeySuMoNumber << endl;

  MyView->PutYmin("SMEvCorssGlobal",fKeyVminEvCorss);
  MyView->PutYmax("SMEvCorssGlobal",fKeyVmaxEvCorss);   
  MyView->EtaPhiSuperModuleMeanOfCorss();
  
  delete MyView;                      fCdelete++;
}

void TCnaDialogEB::ViewSuperModuleSigmaPedestals()
{
  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> 2D plot of Sigma Pedestals. Supermodule: " << fKeySuMoNumber << endl;

  MyView->PutYmin("SMSigEvGlobal",fKeyVminSigEv);
  MyView->PutYmax("SMSigEvGlobal",fKeyVmaxSigEv);     
  MyView->EtaPhiSuperModuleSigmaPedestals();
  
  delete MyView;                      fCdelete++;
}

void TCnaDialogEB::ViewSuperModuleSigmaOfSampleSigmas()
{
  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> 2D plot of Sigma of Sample Sigmas. Supermodule: " << fKeySuMoNumber << endl;

  MyView->PutYmin("SMSigSigGlobal",fKeyVminSigSig);
  MyView->PutYmax("SMSigSigGlobal",fKeyVmaxSigSig);       
  MyView->EtaPhiSuperModuleSigmaOfSampleSigmas();
  
  delete MyView;                      fCdelete++;
}

void TCnaDialogEB::ViewSuperModuleSigmaOfCorss()
{
  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> 2D plot of Sigma of Corss. Supermodule: " << fKeySuMoNumber << endl;
  
  MyView->PutYmin("SMSigCorssGlobal",fKeyVminSigCorss);
  MyView->PutYmax("SMSigCorssGlobal",fKeyVmaxSigCorss);
 
  MyView->EtaPhiSuperModuleSigmaOfCorss();  

  delete MyView;                      fCdelete++;
}

void TCnaDialogEB::ViewSuperModuleCorccMeanOverSamples()
{
  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> 2D plot of cor(c,c), mean over samples. Supermodule: " << fKeySuMoNumber << endl;
  
  MyView->PutYmin("SMCorccInTowers",fKeyVminEvCovtt);
  MyView->PutYmax("SMCorccInTowers",fKeyVmaxEvCovtt);

  MyView->EtaPhiSuperModuleCorccMeanOverSamples();

  delete MyView;                      fCdelete++;
}


//=======================================================================================
//
//                        ViewTowerCrystalNumbering
//
//=======================================================================================  

void TCnaDialogEB::ViewTowerCrystalNumbering(const Int_t& SMtower)
{
  // Plot the crystal numbering of one tower

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Plot of 2D matrix. ViewTowerCrystalNumbering. SuperModule "
       << fKeySuMoNumber << ", tower " << SMtower << endl;

  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);

  MyView->TowerCrystalNumbering(fKeySuMoNumber, SMtower);
  delete MyView;                      fCdelete++;
}
//---------------->  end of ViewTowerCrystalNumbering()

//===========================================================================
//
//                        ViewSMTowerNumbering
//
//===========================================================================  

void TCnaDialogEB::ViewSMTowerNumbering()
{
  // Plot the tower numbering of one supermodule.
  // No argument here since the SM number is a part of the ROOT file name
  // and is in the entry field of the SuperModule button (fKeySuMoNumber)

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Plot of 2D matrix. ViewSMTowerNumbering. SuperModule "
       << fKeySuMoNumber << endl;

  TCnaViewEB* MyView = new TCnaViewEB();    fCnew++;
  MyView->GetPathForResultsRootFiles();
  MyView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);

  MyView->SuperModuleTowerNumbering(fKeySuMoNumber);
  delete MyView;                      fCdelete++;
}
//---------------->  end of ViewSMTowerNumbering()

//===============================================================================
//
//                         ViewHisto...
//
//===============================================================================
//......................... Found evts

void TCnaDialogEB::ViewHistoSuperModuleFoundEventsOfCrystals(const TString first_same_plot)
{
// Plot the 1D histogram of the found events as a function of crystals (grouped by towers)

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Found Events for crystals"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMFoundEvtsGlobal",fKeyVminFoundEvts);
  fView->PutYmax("SMFoundEvtsGlobal",fKeyVmaxFoundEvts);
  fView->SetHistoScaleY(fKeyScaleY);
  fView->HistoSuperModuleFoundEventsOfCrystals(first_same_plot);
}

void TCnaDialogEB::ViewHistoSuperModuleFoundEventsDistribution(const TString first_same_plot)
{
// Plot the 1D histogram of the found events distribution for a supermodule

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Found Events distribution"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMFoundEvtsGlobal",fKeyVminFoundEvts);
  fView->PutYmax("SMFoundEvtsGlobal",fKeyVmaxFoundEvts);  
  fView->SetHistoScaleY(fKeyScaleY);
  fView->HistoSuperModuleFoundEventsDistribution(first_same_plot);
}

//........................... Ev

void TCnaDialogEB::ViewHistoSuperModuleMeanPedestalsOfCrystals(const TString first_same_plot)
{
// Plot the 1D histogram of the mean pedestals as a function of crystals (grouped by towers)

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Mean pedestals for crystals"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMEvEvGlobal",fKeyVminEvEv);
  fView->PutYmax("SMEvEvGlobal",fKeyVmaxEvEv);
  fView->SetHistoScaleY(fKeyScaleY);  
  fView->HistoSuperModuleMeanPedestalsOfCrystals(first_same_plot);
}

void TCnaDialogEB::ViewHistoSuperModuleMeanPedestalsDistribution(const TString first_same_plot)
{
// Plot the 1D histogram of the mean pedestals distribution for a supermodule

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Mean pedestals distribution"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMEvEvGlobal",fKeyVminEvEv);
  fView->PutYmax("SMEvEvGlobal",fKeyVmaxEvEv);
  fView->SetHistoScaleY(fKeyScaleY);  
  fView->HistoSuperModuleMeanPedestalsDistribution(first_same_plot);
}

void TCnaDialogEB::ViewHistoSuperModuleMeanOfSampleSigmasOfCrystals(const TString first_same_plot)
{
// Plot the 1D histogram of the mean of sample sigmas as a function of crystals (grouped by towers)

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Mean of sample sigmas for crystals"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMEvSigGlobal",fKeyVminEvSig);
  fView->PutYmax("SMEvSigGlobal",fKeyVmaxEvSig);
  fView->SetHistoScaleY(fKeyScaleY);  
  fView->HistoSuperModuleMeanOfSampleSigmasOfCrystals(first_same_plot);
}

void TCnaDialogEB::ViewHistoSuperModuleMeanOfSampleSigmasDistribution(const TString first_same_plot)
{
// Plot the 1D histogram of the mean of sample sigmas distribution for a supermodule

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber);  

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Mean of sample sigmas distribution"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMEvSigGlobal",fKeyVminEvSig);
  fView->PutYmax("SMEvSigGlobal",fKeyVmaxEvSig);  
  fView->SetHistoScaleY(fKeyScaleY);  
  fView->HistoSuperModuleMeanOfSampleSigmasDistribution(first_same_plot);
}

void TCnaDialogEB::ViewHistoSuperModuleMeanOfCorssOfCrystals(const TString first_same_plot)
{
// Plot the 1D histogram of the mean of cor(s,s) as a function of crystals (grouped by towers)

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber); 

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Mean of cor(s,s) of crystals"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMEvCorssGlobal",fKeyVminEvCorss);
  fView->PutYmax("SMEvCorssGlobal",fKeyVmaxEvCorss);
  fView->SetHistoScaleY(fKeyScaleY);
  fView->HistoSuperModuleMeanOfCorssOfCrystals(first_same_plot);
}

void TCnaDialogEB::ViewHistoSuperModuleMeanOfCorssDistribution(const TString first_same_plot)
{
// Plot the 1D histogram of the mean of cor(s,s) sigmas distribution for a supermodule

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber); 

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Mean of cor(s,s) distribution"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMEvCorssGlobal",fKeyVminEvCorss);
  fView->PutYmax("SMEvCorssGlobal",fKeyVmaxEvCorss);
  fView->SetHistoScaleY(fKeyScaleY);  
  fView->HistoSuperModuleMeanOfCorssDistribution(first_same_plot);
}

//............................ Sig

void TCnaDialogEB::ViewHistoSuperModuleSigmaPedestalsOfCrystals(const TString first_same_plot)
{
// Plot the 1D histogram of the mean pedestals as a function of crystals (grouped by towers)

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber); 

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Sigma pedestals of crystals"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMSigEvGlobal",fKeyVminSigEv);
  fView->PutYmax("SMSigEvGlobal",fKeyVmaxSigEv);
  fView->SetHistoScaleY(fKeyScaleY);  
  fView->HistoSuperModuleSigmaPedestalsOfCrystals(first_same_plot);
}

void TCnaDialogEB::ViewHistoSuperModuleSigmaPedestalsDistribution(const TString first_same_plot)
{
// Plot the 1D histogram of the mean pedestals distribution for a supermodule

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber); 

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Sigma pedestals distribution"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMSigEvGlobal",fKeyVminSigEv);
  fView->PutYmax("SMSigEvGlobal",fKeyVmaxSigEv);
  fView->SetHistoScaleY(fKeyScaleY);  
  fView->HistoSuperModuleSigmaPedestalsDistribution(first_same_plot);
}

void TCnaDialogEB::ViewHistoSuperModuleSigmaOfSampleSigmasOfCrystals(const TString first_same_plot)
{
// Plot the 1D histogram of the mean of sample sigmas as a function of crystals (grouped by towers)

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber); 

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Sigma of sample sigmas for crystals"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMSigSigGlobal",fKeyVminSigSig);
  fView->PutYmax("SMSigSigGlobal",fKeyVmaxSigSig);
  fView->SetHistoScaleY(fKeyScaleY);  
  fView->HistoSuperModuleSigmaOfSampleSigmasOfCrystals(first_same_plot);
}

void TCnaDialogEB::ViewHistoSuperModuleSigmaOfSampleSigmasDistribution(const TString first_same_plot)
{
// Plot the 1D histogram of the mean of sample sigmas distribution for a supermodule

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber); 

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Sigma of sample sigmas distribution"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMSigSigGlobal",fKeyVminSigSig);
  fView->PutYmax("SMSigSigGlobal",fKeyVmaxSigSig);  
  fView->SetHistoScaleY(fKeyScaleY);
  fView->HistoSuperModuleSigmaOfSampleSigmasDistribution(first_same_plot);
}

void TCnaDialogEB::ViewHistoSuperModuleSigmaOfCorssOfCrystals(const TString first_same_plot)
{
// Plot the 1D histogram of the mean of cor(s,s) as a function of crystals (grouped by towers)

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber); 

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Sigma of cor(s,s) for crystals"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMSigCorssGlobal",fKeyVminSigCorss);
  fView->PutYmax("SMSigCorssGlobal",fKeyVmaxSigCorss);
  fView->SetHistoScaleY(fKeyScaleY);  
  fView->HistoSuperModuleSigmaOfCorssOfCrystals(first_same_plot);
}

void TCnaDialogEB::ViewHistoSuperModuleSigmaOfCorssDistribution(const TString first_same_plot)
{
// Plot the 1D histogram of the mean of cor(s,s) sigmas distribution for a supermodule

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber); 

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Sigma of cor(s,s) distribution"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMSigCorssGlobal",fKeyVminSigCorss);
  fView->PutYmax("SMSigCorssGlobal",fKeyVmaxSigCorss);
  fView->SetHistoScaleY(fKeyScaleY);  
  fView->HistoSuperModuleSigmaOfCorssDistribution(first_same_plot);
}


void TCnaDialogEB::ViewHistoCrystalExpectationValuesOfSamples(const Int_t& SMtower_X, const Int_t& crystal,
							      const TString first_same_plot)
{
// Plot the 1D histogram of the mean of the sample ADC for a crystal

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber); 

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Expectation values of the samples"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", Tower: " << SMtower_X << ", crystal" << crystal
       << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMEvEvGlobal",fKeyVminEvEv);
  fView->PutYmax("SMEvEvGlobal",fKeyVmaxEvEv); 
  fView->SetHistoScaleY(fKeyScaleY);  
  fView->HistoCrystalExpectationValuesOfSamples(SMtower_X, crystal, first_same_plot);
}

void TCnaDialogEB::ViewHistoCrystalSigmasOfSamples(const Int_t& SMtower_X, const Int_t& crystal,
						   const TString first_same_plot)
{
// Plot the 1D histogram of the sigmas of the sample ADC for a crystal

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber); 

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Sigmas of the samples"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", Tower: " << SMtower_X << ", crystal" << crystal
       << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMEvSigGlobal",fKeyVminEvSig);
  fView->PutYmax("SMEvSigGlobal",fKeyVmaxEvSig); 
  fView->SetHistoScaleY(fKeyScaleY);  
  fView->HistoCrystalSigmasOfSamples(SMtower_X, crystal, first_same_plot);
}

void TCnaDialogEB::ViewHistoSampleEventDistribution(const Int_t& SMtower_X, const Int_t& crystal,
						    const Int_t& sample,    const TString first_same_plot)
{
// Plot the 1D histogram of the ADC event distribution for a sample

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber); 

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> ADC event distribution"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", Tower: " << SMtower_X << ", crystal: " << crystal
       << " Sample " << sample << ", option: " << first_same_plot << endl;
 
  fView->SetHistoScaleY(fKeyScaleY);  
  fView->HistoSampleEventDistribution(SMtower_X, crystal, sample, first_same_plot);
}

//............................ Evolution in time (as a function of event number). EvolSamp

void TCnaDialogEB::ViewHistoCrystalPedestalEventNumber(const Int_t& SMtower_X, const Int_t& crystal,
						       const TString first_same_plot)
{
// Plot the 1D histogram of the pedestals as a function of the event number for a crystal

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}
  fView->GetPathForResultsRootFiles();
  fView->SetFile(fKeyAnaType, fKeyRunNumber, fKeyFirstEvt, fKeyNbOfEvts, fKeySuMoNumber); 

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Pedestals as a function of the event number"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st evt: " << fKeyFirstEvt << ", Nb of evts: " << fKeyNbOfEvts
       << ", SM: " << fKeySuMoNumber << ", Tower: " << SMtower_X << ", crystal" << crystal
       << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMEvEvGlobal",fKeyVminEvEv);
  fView->PutYmax("SMEvEvGlobal",fKeyVmaxEvEv); 
  fView->SetHistoScaleY(fKeyScaleY);  
  fView->HistoCrystalPedestalEventNumber(SMtower_X, crystal, first_same_plot);
}

//.................. Evolution in time (as a function of run date)

void TCnaDialogEB::ViewHistimeCrystalMeanPedestals(const TString  run_par_file_name,
						   const Int_t&   SMtower_X,    const Int_t&   towEcha,
						   const TString  first_same_plot)
{
// Plot the 2D (1D-like) histogram of Mean Pedestals evolution for a given towEcha

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Mean pedestals evolution"
       << ". Run parameters file name: " << run_par_file_name
       << ", tower: " << SMtower_X << ", towEcha: " << towEcha
       << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMEvEvGlobal",fKeyVminEvEv);
  fView->PutYmax("SMEvEvGlobal",fKeyVmaxEvEv); 
  fView->SetHistoScaleY(fKeyScaleY);
  fView->HistimeCrystalMeanPedestals(run_par_file_name, SMtower_X, towEcha, first_same_plot);
}

void TCnaDialogEB::ViewHistimeCrystalMeanSigmas(const TString  run_par_file_name,
						const Int_t&   SMtower_X,    const Int_t&  towEcha,
						const TString  first_same_plot)
{
// Plot the 2D (1D-like) histogram of Mean Sigmas evolution for a given towEcha

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Mean sigmas evolution"
       << ". Run parameters file name: " << run_par_file_name
       << ", tower: " << SMtower_X << ", towEcha: " << towEcha
       << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMEvSigGlobal",fKeyVminEvSig);
  fView->PutYmax("SMEvSigGlobal",fKeyVmaxEvSig); 
  fView->SetHistoScaleY(fKeyScaleY);
  fView->HistimeCrystalMeanSigmas(run_par_file_name, SMtower_X, towEcha, first_same_plot);
}

void TCnaDialogEB::ViewHistimeCrystalMeanCorss(const TString  run_par_file_name,
					       const Int_t&   SMtower_X,    const Int_t&   towEcha,
					       const TString  first_same_plot)
{
// Plot the 2D (1D-like) histogram of Mean Corss evolution for a given towEcha

  if ( fView == 0 ){fView = new TCnaViewEB();       fCnew++;}

  fCnaCommand++;
  cout << "   *CNA [" << fCnaCommand
       << "]> Mean corss evolution"
       << ". Run parameters file name: " << run_par_file_name
       << ", tower: " << SMtower_X << ", towEcha: " << towEcha
       << ", option: " << first_same_plot << endl;

  fView->PutYmin("SMEvCorssGlobal",fKeyVminEvCorss);
  fView->PutYmax("SMEvCorssGlobal",fKeyVmaxEvCorss); 
  fView->SetHistoScaleY(fKeyScaleY);
  fView->HistimeCrystalMeanCorss(run_par_file_name, SMtower_X, towEcha, first_same_plot);
}

//=======================================================================================================
//
//            
//
//=======================================================================================================

void TCnaDialogEB::InitKeys()
{
  //.....Input widgets for: run, channel, sample,
  //                        number of events, first event number
  
  fKeyAnaType = "CnP";
  Int_t MaxCar = fgMaxCar;
  fKeyFileNameRunList.Resize(MaxCar);
  fKeyFileNameRunList = "(no file name run list info)";
  fKeyRunNumber  = 0;

  fKeyNbOfEvts   = 150;

  fKeyFirstEvt   = 0;
  fKeySuMoNumber = 1;

  fKeyChanNumber = 0;
  fKeySampNumber = 0;
  fKeyTowXNumber = 1;
  fKeyTowYNumber = 1;

  MaxCar = fgMaxCar;
  fKeyScaleX.Resize(MaxCar); 
  fKeyScaleX = "LIN";
  MaxCar = fgMaxCar;
  fKeyScaleY.Resize(MaxCar); 
  fKeyScaleY = "LIN";

  //.... ymin and ymax values => values which are displayed on the dialog box

  fKeyVminFoundEvts = (Double_t)0.; 
  fKeyVmaxFoundEvts = (Double_t)250.;
 
  fKeyVminEvEv = (Double_t)150.; 
  fKeyVmaxEvEv = (Double_t)250.; 

  fKeyVminEvSig = (Double_t)0.; 
  fKeyVmaxEvSig = (Double_t)1.5; 

  fKeyVminEvCorss = (Double_t)(-1.); 
  fKeyVmaxEvCorss = (Double_t)1.;

  fKeyVminSigEv = (Double_t)0.; 
  fKeyVmaxSigEv = (Double_t)0.5; 

  fKeyVminSigSig = (Double_t)0.; 
  fKeyVmaxSigSig = (Double_t)0.1; 

  fKeyVminSigCorss = (Double_t)0.; 
  fKeyVmaxSigCorss = (Double_t)0.2; 

  fKeyVminEvCortt = (Double_t)(-1.); 
  fKeyVmaxEvCortt = (Double_t)1.;

  fKeyVminEvCovtt = (Double_t)(-1.); 
  fKeyVmaxEvCovtt = (Double_t)1.;
}

void  TCnaDialogEB::DisplayInEntryField(TGTextEntry* StringOfField, Int_t& value)
{
  char* f_in = new char[20];          fCnew++;
  sprintf( f_in, "%d", value );
  StringOfField->Insert(f_in);
  delete [] f_in;                     fCdelete++;
}

void  TCnaDialogEB::DisplayInEntryField(TGTextEntry* StringOfField, Double_t& value)
{
  char* f_in = new char[20];          fCnew++;
  sprintf( f_in, "%g", value );
  StringOfField->Insert(f_in);
  delete [] f_in;                     fCdelete++;
}
void  TCnaDialogEB::DisplayInEntryField(TGTextEntry* StringOfField, const TString value)
{
  StringOfField->Insert(value);
}

//====================================================================================================

void TCnaDialogEB::GetPathForResultsRootFiles()
{
  GetPathForResultsRootFiles("");
}

void TCnaDialogEB::GetPathForListOfRunFiles()
{
  GetPathForListOfRunFiles("");
}

void TCnaDialogEB::GetPathForResultsRootFiles(const TString argFileName)
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
      cout << "   !CNA(TCnaDialogEB) (" << fCnaError << ") *** ERROR *** " << endl << endl
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

void TCnaDialogEB::GetPathForListOfRunFiles(const TString argFileName)
{
  // Init fCfgListOfRunsFilePath and get it from the file named argFileName
  // argFileName = complete name of the file (/afs/cern.ch/...)
  // if string is empty, file name = "cna_results_root.cfg" and file located in $HOME user's directory (default)

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
      fCfgListOfRunsFilePath  = xListOfRunsP.c_str();

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

