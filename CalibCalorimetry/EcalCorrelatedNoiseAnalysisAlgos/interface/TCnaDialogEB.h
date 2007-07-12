#ifndef ZTR_TCnaDialogEB
#define ZTR_TCnaDialogEB

#include "TObject.h"
#include "TSystem.h"

#include "TGButton.h"
#include "TGCanvas.h"
#include "TGWindow.h"
#include "TGMenu.h"
#include "TGFrame.h"
#include "TGTextEntry.h"
#include "TGLayout.h"

#include "TROOT.h"

#include "TApplication.h"
#include "TGClient.h"
#include "TRint.h"

#include "Riostream.h"
#include "TString.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaParameters.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaViewEB.h"

//------------------------ TCnaDialogEB.h -----------------
//
//   For questions or comments, please send e-mail to:
//
//   Bernard Fabbro             
//   fabbro@hep.saclay.cea.fr 
//--------------------------------------------------------

class TCnaDialogEB : public TGMainFrame {

 private:

  //..... Attributes

  Int_t              fgMaxCar;   // Max nb of caracters for char*

  Int_t              fCnew,        fCdelete;
  Int_t              fCnewRoot,    fCdeleteRoot;

  TString            fTTBELL;

  Int_t              fCnaCommand,  fCnaError;

  //==============================================. GUI box: menus, buttons,..

  TGWindow*          fCnaP;
  UInt_t             fCnaW,    fCnaH;

  //------------------------------------------------------------------------------------

  TCnaParameters*    fParameters;
  TCnaViewEB*        fView;

  //------------------- General frame, void frame, standard layout

  TGLayoutHints      *fLayoutGeneral,     *fLayoutBottLeft,  *fLayoutBottRight;
  TGLayoutHints      *fLayoutTopLeft,     *fLayoutTopRight;  
  TGLayoutHints      *fLayoutCenterYLeft, *fLayoutCenterYRight;      
 
  TGCompositeFrame   *fVoidFrame;     // 010
 
  //++++++++++++++++++++++++++++++++++++++++++ Horizontal frame Ana + Run 
  TGCompositeFrame   *fAnaRunFrame;

  //--------------------------------- Sub-Frame Analysis Name (Button+Entry Field)
  TGCompositeFrame   *fAnaFrame;
  TGTextButton       *fAnaBut;
  Int_t               fAnaButC;
  TGLayoutHints      *fLayoutAnaBut;
  TGTextEntry        *fAnaText;
  TGTextBuffer       *fEntryAnaNumber;
  TGLayoutHints      *fLayoutAnaField;

  //-------------------------------- Sub-Frame Run number (Button+Entry Field)  
  TGCompositeFrame   *fRunFrame;  
  TGTextButton       *fRunBut;
  Int_t               fRunButC;
  TGLayoutHints      *fLayoutRunBut;
  TGTextEntry        *fRunText;
  TGTextBuffer       *fEntryRunNumber;
  TGLayoutHints      *fLayoutRunField;

  TGLayoutHints      *fLayoutAnaRunFrame;

  //+++++++++++++++++++++++++++++++++++++++++++ Frames: first taken evt + nb of events + super-module
  TGCompositeFrame   *fFevFrame;
  TGTextButton       *fFevBut;
  TGLayoutHints      *fLayoutFevBut;
  TGTextEntry        *fFevText;
  TGTextBuffer       *fEntryFevNumber;
  TGLayoutHints      *fLayoutFevFieldText;    // 030
  TGLayoutHints      *fLayoutFevFieldFrame;

  TGCompositeFrame   *fNoeFrame;
  TGTextButton       *fNoeBut;
  TGLayoutHints      *fLayoutNoeBut;
  TGTextEntry        *fNoeText;
  TGTextBuffer       *fEntryNoeNumber;
  TGLayoutHints      *fLayoutNoeFieldText;
  TGLayoutHints      *fLayoutNoeFieldFrame;
 
  TGCompositeFrame   *fSuMoFrame;
  TGTextButton       *fSuMoBut;              // 040
  TGLayoutHints      *fLayoutSuMoBut;
  TGTextEntry        *fSuMoText;
  TGTextBuffer       *fEntrySuMoNumber;
  TGLayoutHints      *fLayoutSuMoFieldText;
  TGLayoutHints      *fLayoutSuMoFieldFrame;

  //............................ SuperModule Tower Numbering view (Button)
  TGTextButton       *fButSMNb;
  Int_t               fButSMNbC;
  TGLayoutHints      *fLayoutSMNbBut;

  //+++++++++++++++++++++++++++++++++++++++++++ Frame for quantities relative to the Super-Module
  TGCompositeFrame   *fSuMoUpFrame;

  //................................ Menus+Ymin+Ymax for the Super-Module ............................

  //...................................... Found evts in the data

  TGCompositeFrame   *fVmmFoundEvtsFrame;

  TGCompositeFrame   *fVmaxFoundEvtsFrame;      // 050
  TGTextButton       *fVmaxFoundEvtsBut;
  TGLayoutHints      *fLayoutVmaxFoundEvtsBut;
  TGTextBuffer       *fEntryVmaxFoundEvtsNumber;
  TGTextEntry        *fVmaxFoundEvtsText;
  TGLayoutHints      *fLayoutVmaxFoundEvtsFieldText;
  TGLayoutHints      *fLayoutVmaxFoundEvtsFrame;
 
  TGCompositeFrame   *fVminFoundEvtsFrame;
  TGTextButton       *fVminFoundEvtsBut;
  TGLayoutHints      *fLayoutVminFoundEvtsBut;
  TGTextBuffer       *fEntryVminFoundEvtsNumber;        // 060
  TGTextEntry        *fVminFoundEvtsText;
  TGLayoutHints      *fLayoutVminFoundEvtsFieldText;
  TGLayoutHints      *fLayoutVminFoundEvtsFrame;

  TGPopupMenu        *fMenuFoundEvts;
  TGMenuBar          *fMenuBarFoundEvts;
  TGLayoutHints      *fLayoutMenuBarFoundEvts;
  Int_t               fMenuFoundEvtsGlobalFullC, fMenuFoundEvtsProjFullC;
  Int_t               fMenuFoundEvtsGlobalSameC, fMenuFoundEvtsProjSameC;
  Int_t               fMenuFoundEvtsEtaPhiC;

  TGLayoutHints      *fLayoutVmmFoundEvtsFrame;

  //................................... Horizontal frame ev + sig
  TGCompositeFrame   *fSuMoHozFrame;

  //............................................... Frame Ev + Menus Ev
  TGCompositeFrame   *fSuMoHozSubEvFrame; 

  //-------------------------------------------------------------
  TGCompositeFrame   *fVmmEvEvFrame;              // 070

  TGCompositeFrame   *fVmaxEvEvFrame;
  TGTextButton       *fVmaxEvEvBut;
  TGLayoutHints      *fLayoutVmaxEvEvBut;
  TGTextEntry        *fVmaxEvEvText;
  TGTextBuffer       *fEntryVmaxEvEvNumber;
  TGLayoutHints      *fLayoutVmaxEvEvFieldText;
  TGLayoutHints      *fLayoutVmaxEvEvFrame;
 
  TGCompositeFrame   *fVminEvEvFrame;
  TGTextButton       *fVminEvEvBut;
  TGLayoutHints      *fLayoutVminEvEvBut;          // 080
  TGTextEntry        *fVminEvEvText;
  TGTextBuffer       *fEntryVminEvEvNumber;
  TGLayoutHints      *fLayoutVminEvEvFieldText;
  TGLayoutHints      *fLayoutVminEvEvFrame;

  TGPopupMenu        *fMenuEvEv;
  TGMenuBar          *fMenuBarEvEv;
  TGLayoutHints      *fLayoutMenuBarEvEv;
  Int_t               fMenuEvEvGlobalFullC,  fMenuEvEvProjFullC;
  Int_t               fMenuEvEvGlobalSameC,  fMenuEvEvProjSameC;
  Int_t               fMenuEvEvEtaPhiC;

  TGLayoutHints      *fLayoutVmmEvEvFrame;

  //-------------------------------------------------------------
  TGCompositeFrame   *fVmmEvSigFrame;

  TGCompositeFrame   *fVmaxEvSigFrame;           // 090
  TGTextButton       *fVmaxEvSigBut;
  TGLayoutHints      *fLayoutVmaxEvSigBut;
  TGTextBuffer       *fEntryVmaxEvSigNumber;
  TGTextEntry        *fVmaxEvSigText;
  TGLayoutHints      *fLayoutVmaxEvSigFieldText;
  TGLayoutHints      *fLayoutVmaxEvSigFrame;

  TGCompositeFrame   *fVminEvSigFrame;
  TGTextButton       *fVminEvSigBut;
  TGLayoutHints      *fLayoutVminEvSigBut;
  TGTextBuffer       *fEntryVminEvSigNumber;            // 100
  TGTextEntry        *fVminEvSigText;
  TGLayoutHints      *fLayoutVminEvSigFieldText;
  TGLayoutHints      *fLayoutVminEvSigFrame;

  TGPopupMenu        *fMenuEvSig;
  TGMenuBar          *fMenuBarEvSig;
  TGLayoutHints      *fLayoutMenuBarEvSig; 
  Int_t               fMenuEvSigGlobalFullC, fMenuEvSigProjFullC; 
  Int_t               fMenuEvSigGlobalSameC, fMenuEvSigProjSameC;
  Int_t               fMenuEvSigEtaPhiC;

  TGLayoutHints      *fLayoutVmmEvSigFrame;

  //-------------------------------------------------------------
  TGCompositeFrame   *fVmmEvCorssFrame;

  TGCompositeFrame   *fVmaxEvCorssFrame;
  TGTextButton       *fVmaxEvCorssBut;            // 110
  TGLayoutHints      *fLayoutVmaxEvCorssBut;
  TGTextEntry        *fVmaxEvCorssText;
  TGTextBuffer       *fEntryVmaxEvCorssNumber;
  TGLayoutHints      *fLayoutVmaxEvCorssFieldText;
  TGLayoutHints      *fLayoutVmaxEvCorssFrame;

  TGCompositeFrame   *fVminEvCorssFrame;
  TGTextButton       *fVminEvCorssBut;
  TGLayoutHints      *fLayoutVminEvCorssBut;
  TGTextBuffer       *fEntryVminEvCorssNumber;
  TGTextEntry        *fVminEvCorssText;            // 120
  TGLayoutHints      *fLayoutVminEvCorssFieldText;
  TGLayoutHints      *fLayoutVminEvCorssFrame;

  TGPopupMenu        *fMenuEvCorss;
  TGMenuBar          *fMenuBarEvCorss;
  TGLayoutHints      *fLayoutMenuBarEvCorss;
  Int_t               fMenuEvCorssGlobalFullC, fMenuEvCorssProjFullC; 
  Int_t               fMenuEvCorssGlobalSameC, fMenuEvCorssProjSameC; 
  Int_t               fMenuEvCorssEtaPhiC;

  TGLayoutHints      *fLayoutVmmEvCorssFrame;

  TGLayoutHints      *fLayoutSuMoHozSubEvFrame;

  //............................................... Frame Sig + Menus Sig
  TGCompositeFrame   *fSuMoHozSubSigFrame; 

  //-------------------------------------------------------------
  TGCompositeFrame   *fVmmSigEvFrame;

  TGCompositeFrame   *fVmaxSigEvFrame;    // 130
  TGTextButton       *fVmaxSigEvBut;
  TGLayoutHints      *fLayoutVmaxSigEvBut;
  TGTextEntry        *fVmaxSigEvText;
  TGTextBuffer       *fEntryVmaxSigEvNumber;
  TGLayoutHints      *fLayoutVmaxSigEvFieldText;
  TGLayoutHints      *fLayoutVmaxSigEvFrame;

  TGCompositeFrame   *fVminSigEvFrame;
  TGTextButton       *fVminSigEvBut;
  TGLayoutHints      *fLayoutVminSigEvBut;
  TGTextBuffer       *fEntryVminSigEvNumber;         // 140
  TGTextEntry        *fVminSigEvText;
  TGLayoutHints      *fLayoutVminSigEvFieldText;
  TGLayoutHints      *fLayoutVminSigEvFrame;

  TGPopupMenu        *fMenuSigEv;
  TGMenuBar          *fMenuBarSigEv;
  TGLayoutHints      *fLayoutMenuBarSigEv;
  Int_t               fMenuSigEvGlobalFullC, fMenuSigEvProjFullC; 
  Int_t               fMenuSigEvGlobalSameC, fMenuSigEvProjSameC; 
  Int_t               fMenuSigEvEtaPhiC;

  TGLayoutHints      *fLayoutVmmSigEvFrame;

  //-------------------------------------------------------------
  TGCompositeFrame   *fVmmSigSigFrame;

  TGCompositeFrame   *fVmaxSigSigFrame;
  TGTextButton       *fVmaxSigSigBut;           // 150
  TGLayoutHints      *fLayoutVmaxSigSigBut;
  TGTextEntry        *fVmaxSigSigText;
  TGTextBuffer       *fEntryVmaxSigSigNumber;
  TGLayoutHints      *fLayoutVmaxSigSigFieldText;
  TGLayoutHints      *fLayoutVmaxSigSigFrame;

  TGCompositeFrame   *fVminSigSigFrame;
  TGTextButton       *fVminSigSigBut;
  TGLayoutHints      *fLayoutVminSigSigBut;
  TGTextBuffer       *fEntryVminSigSigNumber;
  TGTextEntry        *fVminSigSigText;          // 160
  TGLayoutHints      *fLayoutVminSigSigFieldText;
  TGLayoutHints      *fLayoutVminSigSigFrame;

  TGPopupMenu        *fMenuSigSig;
  TGMenuBar          *fMenuBarSigSig;
  TGLayoutHints      *fLayoutMenuBarSigSig;
  Int_t               fMenuSigSigGlobalFullC, fMenuSigSigProjFullC; 
  Int_t               fMenuSigSigGlobalSameC, fMenuSigSigProjSameC;
  Int_t               fMenuSigSigEtaPhiC;

  TGLayoutHints      *fLayoutVmmSigSigFrame;

  //-------------------------------------------------------------
  TGCompositeFrame   *fVmmSigCorssFrame;

  TGCompositeFrame   *fVmaxSigCorssFrame;
  TGTextButton       *fVmaxSigCorssBut;
  TGLayoutHints      *fLayoutVmaxSigCorssBut;   // 170
  TGTextEntry        *fVmaxSigCorssText;
  TGTextBuffer       *fEntryVmaxSigCorssNumber;
  TGLayoutHints      *fLayoutVmaxSigCorssFieldText;
  TGLayoutHints      *fLayoutVmaxSigCorssFrame;

  TGCompositeFrame   *fVminSigCorssFrame;
  TGTextButton       *fVminSigCorssBut;
  TGLayoutHints      *fLayoutVminSigCorssBut;
  TGTextEntry        *fVminSigCorssText;
  TGTextBuffer       *fEntryVminSigCorssNumber;
  TGLayoutHints      *fLayoutVminSigCorssFieldText;
  TGLayoutHints      *fLayoutVminSigCorssFrame;

  TGPopupMenu        *fMenuSigCorss;
  TGMenuBar          *fMenuBarSigCorss;
  TGLayoutHints      *fLayoutMenuBarSigCorss;
  Int_t               fMenuSigCorssGlobalFullC, fMenuSigCorssProjFullC;  
  Int_t               fMenuSigCorssGlobalSameC, fMenuSigCorssProjSameC;
  Int_t               fMenuSigCorssEtaPhiC;

  TGLayoutHints      *fLayoutVmmSigCorssFrame;     // 185

  //----------------------------------------------------------------------------------

  TGLayoutHints      *fLayoutSuMoHozSubSigFrame;

  TGLayoutHints      *fLayoutSuMoHozFrame;

  //...................................... Covariances between towers
  TGCompositeFrame   *fVmmEvCovttFrame;

  TGCompositeFrame   *fVmaxEvCovttFrame;
  TGTextButton       *fVmaxEvCovttBut;
  TGLayoutHints      *fLayoutVmaxEvCovttBut;
  TGTextEntry        *fVmaxEvCovttText;
  TGTextBuffer       *fEntryVmaxEvCovttNumber;
  TGLayoutHints      *fLayoutVmaxEvCovttFieldText;
  TGLayoutHints      *fLayoutVmaxEvCovttFrame;

  TGCompositeFrame   *fVminEvCovttFrame;
  TGTextButton       *fVminEvCovttBut;
  TGLayoutHints      *fLayoutVminEvCovttBut;
  TGTextBuffer       *fEntryVminEvCovttNumber;
  TGTextEntry        *fVminEvCovttText;         // 200
  TGLayoutHints      *fLayoutVminEvCovttFieldText;
  TGLayoutHints      *fLayoutVminEvCovttFrame;

  TGPopupMenu        *fMenuCovtt;
  TGMenuBar          *fMenuBarCovtt;
  TGLayoutHints      *fLayoutMenuBarCovtt;
  Int_t               fMenuCovttColzC, fMenuCovttLegoC;

  TGLayoutHints      *fLayoutVmmEvCovttFrame;

  //...................................... Correlations between towers
  TGCompositeFrame   *fVmmEvCorttFrame;

  TGCompositeFrame   *fVmaxEvCorttFrame;
  TGTextButton       *fVmaxEvCorttBut;
  TGLayoutHints      *fLayoutVmaxEvCorttBut;       // 210
  TGTextEntry        *fVmaxEvCorttText;
  TGTextBuffer       *fEntryVmaxEvCorttNumber;
  TGLayoutHints      *fLayoutVmaxEvCorttFieldText;
  TGLayoutHints      *fLayoutVmaxEvCorttFrame;

  TGCompositeFrame   *fVminEvCorttFrame;
  TGTextButton       *fVminEvCorttBut;
  TGLayoutHints      *fLayoutVminEvCorttBut;
  TGTextEntry        *fVminEvCorttText;
  TGTextBuffer       *fEntryVminEvCorttNumber;
  TGLayoutHints      *fLayoutVminEvCorttFieldText;   // 220
  TGLayoutHints      *fLayoutVminEvCorttFrame;

  TGPopupMenu        *fMenuCortt;
  TGMenuBar          *fMenuBarCortt;
  TGLayoutHints      *fLayoutMenuBarCortt;
  Int_t               fMenuCorttColzC, fMenuCorttLegoC;

  TGLayoutHints      *fLayoutVmmEvCorttFrame;

  TGLayoutHints      *fLayoutSuMoUpFrame;

  //+++++++++++++++++++++++++++++++++++++++++ Horizontal frame Tower_X + Tower_Y
  TGCompositeFrame   *fTowSpFrame; 

  //----------------------------------- SubFrame Tower_X (Button + EntryField)

  TGCompositeFrame   *fTxSubFrame;

  TGCompositeFrame   *fTowXFrame;          
  TGTextButton       *fTowXBut;        // 230
  Int_t               fTowXButC;
  TGLayoutHints      *fLayoutTowXBut; 
  TGTextBuffer       *fEntryTowXNumber; 
  TGTextEntry        *fTowXText;
  TGLayoutHints      *fLayoutTowXField;
  
  //............................ Tower Crystal Numbering view (Button)
  TGTextButton       *fButChNb;
  Int_t               fButChNbC;
  TGLayoutHints      *fLayoutChNbBut;

  //............................ Menus Tower_X
  TGPopupMenu        *fMenuCorssAll;
  TGMenuBar          *fMenuBarCorssAll;
  Int_t               fMenuCorssAllColzC;

  TGPopupMenu        *fMenuCovssAll;
  TGMenuBar          *fMenuBarCovssAll;         // 240
  Int_t               fMenuCovssAllColzC;

  TGLayoutHints      *fLayoutTxSubFrame;

  //----------------------------------- SubFrame Tower_Y (Button + EntryField)
  TGCompositeFrame   *fTySubFrame;

  TGCompositeFrame   *fTowYFrame;
  TGTextButton       *fTowYBut;
  Int_t               fTowYButC;
  TGLayoutHints      *fLayoutTowYBut;
  TGTextBuffer       *fEntryTowYNumber;
  TGTextEntry        *fTowYText;
  TGLayoutHints      *fLayoutTowYField;

  TGLayoutHints      *fLayoutTySubFrame;

  TGLayoutHints      *fLayoutTowSpFrame;         // 250

  //.................................. Menus for Horizontal frame (Tower_X + Tower_Y)
  TGPopupMenu        *fMenuCorcc; 
  TGMenuBar          *fMenuBarCorcc;   
  Int_t               fMenuCorccColzC, fMenuCorccLegoC; 

  TGPopupMenu        *fMenuCovcc;
  TGMenuBar          *fMenuBarCovcc;   
  Int_t               fMenuCovccColzC, fMenuCovccLegoC;    

  //++++++++++++++++++++++++ Horizontal frame channel number (tower_X crystal number) + sample number
  TGCompositeFrame   *fChSpFrame;

  //------------------------------------- SubFrame Channel (Button + EntryField)
  TGCompositeFrame   *fChSubFrame;

  TGCompositeFrame   *fChanFrame;
  TGTextButton       *fChanBut;
  Int_t               fChanButC;
  TGLayoutHints      *fLayoutChanBut;
  TGTextBuffer       *fEntryChanNumber;    // 260
  TGTextEntry        *fChanText;
  TGLayoutHints      *fLayoutChanField;

  //................................ Menus Tower_X crystal number
  TGPopupMenu        *fMenuCorss;
  TGMenuBar          *fMenuBarCorss;
  Int_t               fMenuCorssColzC, fMenuCorssLegoC, fMenuCorssSurf1C, fMenuCorssSurf4C;

  TGPopupMenu        *fMenuCovss;
  TGMenuBar          *fMenuBarCovss;
  Int_t               fMenuCovssColzC, fMenuCovssLegoC, fMenuCovssSurf1C, fMenuCovssSurf4C;

  TGPopupMenu        *fMenuEv;
  TGMenuBar          *fMenuBarEv;
  Int_t               fMenuEvLineFullC,  fMenuEvLineSameC;

  TGPopupMenu        *fMenuVar;
  TGMenuBar          *fMenuBarVar;          // 270  
  Int_t               fMenuVarLineFullC, fMenuVarLineSameC;

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

  TGLayoutHints      *fLayoutChSpFrame;       // 280

  //++++++++++++++++++++++++++++++++++++ Menu Event Distribution
  TGPopupMenu        *fMenuEvts;
  TGMenuBar          *fMenuBarEvts;
  TGLayoutHints      *fLayoutMenuBarEvts;
  Int_t               fMenuEvtsLineLinyFullC;
  Int_t               fMenuEvtsLineLinySameC;

  //++++++++++++++++++++++++++++++++++++ Frame: Run List (Rul) (Button + EntryField)

  TGCompositeFrame   *fRulFrame;
  TGTextButton       *fRulBut;
  TGLayoutHints      *fLayoutRulBut;
  TGTextEntry        *fRulText;
  TGTextBuffer       *fEntryRulNumber;
  TGLayoutHints      *fLayoutRulFieldText;
  TGLayoutHints      *fLayoutRulFieldFrame;      // 290

  //................................ Menus for time evolution
  TGPopupMenu        *fMenuEvol;
  TGMenuBar          *fMenuBarEvol;
  Int_t               fMenuEvolEvEvPolmFullC,     fMenuEvolEvEvPolmSameC;
  Int_t               fMenuEvolEvSigPolmFullC,    fMenuEvolEvSigPolmSameC;
  Int_t               fMenuEvolEvCorssPolmFullC,  fMenuEvolEvCorssPolmSameC;
  Int_t               fMenuEvolSampLineFullC,     fMenuEvolSampLineSameC;

  //++++++++++++++++++++++++++++++++++++ LinLog Frame
  TGCompositeFrame   *fLinLogFrame;

  //---------------------------------- Lin/Log X
  TGCheckButton      *fButLogx;
  Int_t               fButLogxC; 
  TGLayoutHints      *fLayoutLogxBut;
  //---------------------------------- Lin/Log Y
  TGCheckButton      *fButLogy;
  Int_t               fButLogyC; 
  TGLayoutHints      *fLayoutLogyBut;

  //++++++++++++++++++++++++++++++++++++ EXIT BUTTON
  TGTextButton       *fButExit;
  Int_t               fButExitC;      
  TGLayoutHints      *fLayoutExitBut;

  //++++++++++++++++++++++++++++++++++++ Last Frame
  TGCompositeFrame   *fLastFrame;     // 300 

  //--------------------------------- Root version (Button)
  TGTextButton       *fButRoot;
  Int_t               fButRootC;
  TGLayoutHints      *fLayoutRootBut;

  //--------------------------------- Help (Button)
  TGTextButton       *fButHelp;
  Int_t               fButHelpC;
  TGLayoutHints      *fLayoutHelpBut;    // 304

  //==================================================== Miscellaneous parameters
  ifstream fFcin_rr;
  ifstream fFcin_lor;

  ofstream fFcout_f;

  //  TString  fFileNameForCnaPaths;     // name of the file containing the paths (cna_paths.ascii)

  TString  fFileForResultsRootFilePath; // name of the file containing the results .root file path
  TString  fFileForListOfRunFilePath;   // name of the file containing the list-of-run file path

  TString  fCfgResultsRootFilePath; // absolute path for the results .root files (/afs/etc...)
  TString  fCfgListOfRunsFilePath;  // absolute path for the list-of-runs .ascii  files (/afs/etc...)

  //  TString  fCfgResultsAsciiFilePath;    // absolute path for the ASCII files (/afs/etc...)

  TString  fKeyAnaType;         // Type of analysis
  TString  fKeyFileNameRunList; // Name of the file containing the run parameters list
  Int_t    fKeyRunNumber;       // Run number
  Int_t    fKeyFirstEvt;        // First Event number (to be analyzed)
  Int_t    fKeyNbOfEvts;        // Number of required events (events to be analyzed)
  TString  fKeyScaleX;
  TString  fKeyScaleY;

  //................... VISUALIZATION PARAMETERS  
  Int_t    fKeySuMoNumber;  // Super-Module number
  Int_t    fKeyTowXNumber;  // Tower X number
  Int_t    fKeyTowYNumber;  // Tower Y number
  Int_t    fKeyChanNumber;  // Channel number
  Int_t    fKeySampNumber;  // Sample number

  //................ TString run number + SM number for submit job batch
  TString  fKeyRunNumberTString;
  TString  fKeySuMoNumberTString;

  //................... ymin and ymax values

  Double_t fKeyVminFoundEvts; 
  Double_t fKeyVmaxFoundEvts;
 
  Double_t fKeyVminEvEv; 
  Double_t fKeyVmaxEvEv; 

  Double_t fKeyVminEvSig; 
  Double_t fKeyVmaxEvSig; 

  Double_t fKeyVminEvCorss; 
  Double_t fKeyVmaxEvCorss;

  Double_t fKeyVminSigEv; 
  Double_t fKeyVmaxSigEv; 

  Double_t fKeyVminSigSig; 
  Double_t fKeyVmaxSigSig; 

  Double_t fKeyVminSigCorss; 
  Double_t fKeyVmaxSigCorss; 

  Double_t fKeyVminEvCortt; 
  Double_t fKeyVmaxEvCortt;

  Double_t fKeyVminEvCovtt; 
  Double_t fKeyVmaxEvCovtt;

  //................... plot parameters (for onlyone,same options)

  TString  fMemoScaleX;
  TString  fMemoScaleY;

  TString  fOptPlotFull;
  TString  fOptPlotSame;

  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

 public:
           TCnaDialogEB(const TGWindow *, UInt_t, UInt_t);
  virtual  ~TCnaDialogEB();

  void     InitKeys();

  void     GetPathForResultsRootFiles();
  void     GetPathForResultsRootFiles(const TString);
  void     GetPathForListOfRunFiles();
  void     GetPathForListOfRunFiles(const TString);

  void     DisplayInEntryField(TGTextEntry*, Int_t&);
  void     DisplayInEntryField(TGTextEntry*, Double_t&);
  void     DisplayInEntryField(TGTextEntry*, const TString);

  void     DoButtonAna();
  void     DoButtonRun();

  void     DoButtonFev();
  void     DoButtonNoe();
  void     DoButtonSuMo();
  void     DoButtonSMNb();

  //########################################
  void     DoButtonVminFoundEvts();
  void     DoButtonVmaxFoundEvts();

  void     DoButtonVminEvEv();
  void     DoButtonVmaxEvEv();

  void     DoButtonVminEvSig();
  void     DoButtonVmaxEvSig();

  void     DoButtonVminEvCorss();
  void     DoButtonVmaxEvCorss();

  void     DoButtonVminSigEv();
  void     DoButtonVmaxSigEv();

  void     DoButtonVminSigSig();
  void     DoButtonVmaxSigSig();

  void     DoButtonVminSigCorss();
  void     DoButtonVmaxSigCorss();

  void     DoButtonVminEvCortt();
  void     DoButtonVmaxEvCortt();

  void     DoButtonVminEvCovtt();
  void     DoButtonVmaxEvCovtt();

  //########################################

  void     DoButtonTowX();
  void     DoButtonTowY();

  void     DoButtonChNb();
  void     DoButtonChan();
  void     DoButtonSamp();

  void     DoButtonRul();

  void     DoButtonLogx();
  void     DoButtonLogy();

  void     DoButtonHelp();
  void     DoButtonRoot();
  void     DoButtonExit();

  void     HandleMenu(Int_t);
  //  void     CloseWindow();


  //------------------- VISUALIZATION METHODS

  void      ViewMatrixCorrelationTowers(const TString);
  void      ViewMatrixCovarianceTowers(const TString);
  void      ViewMatrixCorrelationCrystals(const Int_t&, const Int_t&, const TString);
  void      ViewMatrixCovarianceCrystals(const Int_t&, const Int_t&, const TString);
  void      ViewMatrixCorrelationSamples(const Int_t&, const Int_t&, const TString);
  void      ViewMatrixCovarianceSamples(const Int_t&, const Int_t&, const TString);

  void      ViewSuperModuleFoundEvents();
  void      ViewSuperModuleMeanPedestals();
  void      ViewSuperModuleMeanOfSampleSigmas();
  void      ViewSuperModuleMeanOfCorss();
  void      ViewSuperModuleSigmaPedestals();
  void      ViewSuperModuleSigmaOfSampleSigmas();
  void      ViewSuperModuleSigmaOfCorss();
  void      ViewSuperModuleCorccMeanOverSamples();

  void      ViewTowerCorrelationSamples(const Int_t&);
  void      ViewTowerCovarianceSamples(const Int_t&);
  void      ViewTowerCrystalNumbering(const Int_t&);
  void      ViewSMTowerNumbering();

  void      ViewHistoSuperModuleFoundEventsOfCrystals(const TString);
  void      ViewHistoSuperModuleFoundEventsDistribution(const TString);
  void      ViewHistoSuperModuleMeanPedestalsOfCrystals(const TString);
  void      ViewHistoSuperModuleMeanPedestalsDistribution(const TString);
  void      ViewHistoSuperModuleMeanOfSampleSigmasOfCrystals(const TString);
  void      ViewHistoSuperModuleMeanOfSampleSigmasDistribution(const TString);
  void      ViewHistoSuperModuleMeanOfCorssOfCrystals(const TString);
  void      ViewHistoSuperModuleMeanOfCorssDistribution(const TString);
  void      ViewHistoSuperModuleSigmaPedestalsOfCrystals(const TString);
  void      ViewHistoSuperModuleSigmaPedestalsDistribution(const TString);
  void      ViewHistoSuperModuleSigmaOfSampleSigmasOfCrystals(const TString);
  void      ViewHistoSuperModuleSigmaOfSampleSigmasDistribution(const TString);
  void      ViewHistoSuperModuleSigmaOfCorssOfCrystals(const TString);
  void      ViewHistoSuperModuleSigmaOfCorssDistribution(const TString);

  void      ViewHistoCrystalExpectationValuesOfSamples(const Int_t&, const Int_t&, const TString);
  void      ViewHistoCrystalSigmasOfSamples(const Int_t&, const Int_t&, const TString);
  void      ViewHistoCrystalPedestalEventNumber(const Int_t&, const Int_t&, const TString);

  void      ViewHistoSampleEventDistribution(const Int_t&, const Int_t&, const Int_t&, const TString);

  void      ViewHistimeCrystalMeanPedestals(const TString, const Int_t&, const Int_t&, const TString);
  void      ViewHistimeCrystalMeanSigmas(const TString, const Int_t&, const Int_t&, const TString);
  void      ViewHistimeCrystalMeanCorss(const TString, const Int_t&, const Int_t&, const TString);


ClassDef(TCnaDialogEB,1)// Dialog box + methods for CNA (Correlated Noises Analysis)

};

#endif   //    ZTR_TCnaDialogEB
