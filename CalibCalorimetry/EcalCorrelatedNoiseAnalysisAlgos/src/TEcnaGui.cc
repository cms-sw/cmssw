//----------Author's Name: B.Fabbro DSM/IRFU/SPP CEA-Saclay
//----------Copyright: Those valid for CEA sofware
//----------Modified: 13/04/2011

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaGui.h"
#include <cstdlib>

//--------------------------------------
//  TEcnaGui.cc
//  Class creation: 03 Dec 2002
//  Documentation: see TEcnaGui.h
//--------------------------------------

ClassImp(TEcnaGui)
//______________________________________________________________________________
//

  TEcnaGui::~TEcnaGui()
{
  //destructor

#define DEST
#ifdef DEST
  // cout << "TEcnaGui> Entering destructor" << endl;
  // cout << "            fCnew = " << fCnew << ", fCdelete = " << fCdelete << endl;
 
  //.... general variables 
  //if ( fHistos             != 0 ) {delete fHistos;             fCdelete++;}

  //if ( fCnaParHistos       != 0 ) {delete fCnaParHistos;       fCdelete++;}
  //if ( fCnaParPaths        != 0 ) {delete fCnaParPaths;        fCdelete++;}
  //if ( fEcalNumbering      != 0 ) {delete fEcalNumbering;      fCdelete++;}
  //if ( fEcal               != 0 ) {delete fEcal;               fCdelete++;}

  //.... general frames

  if ( fLayoutGeneral      != 0 ) {delete fLayoutGeneral;      fCdelete++;}
  if ( fLayoutBottLeft     != 0 ) {delete fLayoutBottLeft;     fCdelete++;}
  if ( fLayoutBottRight    != 0 ) {delete fLayoutBottRight;    fCdelete++;}
  if ( fLayoutTopLeft      != 0 ) {delete fLayoutTopLeft;      fCdelete++;}
  if ( fLayoutTopRight     != 0 ) {delete fLayoutTopRight;     fCdelete++;}
  if ( fLayoutCenterYLeft  != 0 ) {delete fLayoutCenterYLeft;  fCdelete++;}
  if ( fLayoutCenterYRight != 0 ) {delete fLayoutCenterYRight; fCdelete++;}
  if ( fLayoutCenterXTop   != 0 ) {delete fLayoutCenterXTop;   fCdelete++;}

  if ( fVoidFrame          != 0 ) {delete fVoidFrame;          fCdelete++;}

  //..... specific frames + buttons + menus

  //++++++++++++++++++++++++++++++ Horizontal frame Analysis + First requested evt number + Run number
  if ( fAnaNorsRunFrame       != 0 ) {delete fAnaNorsRunFrame;       fCdelete++;}
  if ( fLayoutAnaNorsRunFrame != 0 ) {delete fLayoutAnaNorsRunFrame; fCdelete++;}

  //--------------------------------- Sub-Frame Analysis Name (Button+Entry Field)
  if ( fAnaFrame       != 0 ) {delete fAnaFrame;       fCdelete++;}
  if ( fAnaBut         != 0 ) {delete fAnaBut;         fCdelete++;}
  if ( fLayoutAnaBut   != 0 ) {delete fLayoutAnaBut;   fCdelete++;}
  if ( fEntryAnaNumber != 0 ) {delete fEntryAnaNumber; fCdelete++;}
  if ( fAnaText        != 0 ) {fAnaText->Delete();     fCdelete++;}
  if ( fLayoutAnaField != 0 ) {delete fLayoutAnaField; fCdelete++;}

  //------------------- subframe first requested evt number
  if ( fFevFrame            != 0 ) {delete fFevFrame;            fCdelete++;}
  if ( fFevBut              != 0 ) {delete fFevBut;              fCdelete++;}
  if ( fLayoutFevBut        != 0 ) {delete fLayoutFevBut;        fCdelete++;}
  if ( fEntryFevNumber      != 0 ) {delete fEntryFevNumber;      fCdelete++;}
  if ( fFevText             != 0 ) {fFevText->Delete();          fCdelete++;}  
  if ( fLayoutFevFieldText  != 0 ) {delete fLayoutFevFieldText;  fCdelete++;}
  if ( fLayoutFevFieldFrame != 0 ) {delete fLayoutFevFieldFrame; fCdelete++;}

  //-------------------------------- Sub-Frame Run number (Button+Entry Field)  
  if ( fRunFrame       != 0 ) {delete fRunFrame;           fCdelete++;}
  if ( fRunBut         != 0 ) {delete fRunBut;             fCdelete++;}
  if ( fLayoutRunBut   != 0 ) {delete fLayoutRunBut;       fCdelete++;}
  if ( fEntryRunNumber != 0 ) {delete fEntryRunNumber;     fCdelete++;}
  if ( fRunText        != 0 ) {fRunText->Delete();         fCdelete++;}
  if ( fLayoutRunField != 0 ) {delete fLayoutRunField;     fCdelete++;}

  //+++++++++++++++++++++++++++++ Horizontal frame Nb Of Samples + last requested events + Clean + Submit
  if ( fFevLevStexFrame       != 0 ) {delete fFevLevStexFrame;       fCdelete++;}
  if ( fLayoutFevLevStexFrame != 0 ) {delete fLayoutFevLevStexFrame; fCdelete++;}

  //------------------- Sub-Frame Nb of Required Samples (Button+Entry Field)
  if ( fNorsFrame          != 0 ) {delete fNorsFrame;          fCdelete++;}
  if ( fNorsBut            != 0 ) {delete fNorsBut;            fCdelete++;}
  if ( fLayoutNorsBut      != 0 ) {delete fLayoutNorsBut;      fCdelete++;}
  if ( fEntryNorsNumber    != 0 ) {delete fEntryNorsNumber;    fCdelete++;}
  if ( fNorsText           != 0 ) {fNorsText->Delete();        fCdelete++;}
  if ( fLayoutNorsField    != 0 ) {delete fLayoutNorsField;    fCdelete++;}

  //------------------- subframe last requested evt number
  if ( fLevFrame            != 0 ) {delete fLevFrame;            fCdelete++;}
  if ( fLevBut              != 0 ) {delete fLevBut;              fCdelete++;}
  if ( fLayoutLevBut        != 0 ) {delete fLayoutLevBut;        fCdelete++;}
  if ( fEntryLevNumber      != 0 ) {delete fEntryLevNumber;      fCdelete++;}
  if ( fLevText             != 0 ) {fLevText->Delete();          fCdelete++;}
  if ( fLayoutLevFieldText  != 0 ) {delete fLayoutLevFieldText;  fCdelete++;}
  if ( fLayoutLevFieldFrame != 0 ) {delete fLayoutLevFieldFrame; fCdelete++;}

  //................................ Menu for Clean
  if ( fMenuClean          != 0 ) {delete fMenuClean;          fCdelete++;}
  if ( fMenuBarClean       != 0 ) {fMenuBarClean->Delete();    fCdelete++;}
  //................................ Menu for Submit jobs on batch system
  if ( fMenuSubmit         != 0 ) {delete fMenuSubmit;         fCdelete++;}
  if ( fMenuBarSubmit      != 0 ) {fMenuBarSubmit->Delete();   fCdelete++;}

  //+++++++++++++++++++++++++++++++++++++++++++++++++  Horizontal Frame:Stex number + NbOfReqEvts
  if ( fCompStRqFrame       != 0 ) {delete fCompStRqFrame;       fCdelete++;}
  if ( fLayoutCompStRqFrame != 0 ) {delete fLayoutCompStRqFrame; fCdelete++;}

  //------------------- subframe stex number
  if ( fStexFrame            != 0 ) {delete fStexFrame;            fCdelete++;}
  if ( fStexBut              != 0 ) {delete fStexBut;              fCdelete++;}
  if ( fLayoutStexBut        != 0 ) {delete fLayoutStexBut;        fCdelete++;}
  if ( fEntryStexNumber      != 0 ) {delete fEntryStexNumber;      fCdelete++;}
  if ( fStexText             != 0 ) {fStexText->Delete();          fCdelete++;}  
  if ( fLayoutStexFieldText  != 0 ) {delete fLayoutStexFieldText;  fCdelete++;}
  if ( fLayoutStexFieldFrame != 0 ) {delete fLayoutStexFieldFrame; fCdelete++;}

  //------------------- subframe number of requested evts
  if ( fRevFrame            != 0 ) {delete fRevFrame;            fCdelete++;}
  if ( fRevBut              != 0 ) {delete fRevBut;              fCdelete++;}
  if ( fLayoutRevBut        != 0 ) {delete fLayoutRevBut;        fCdelete++;}
  if ( fEntryRevNumber      != 0 ) {delete fEntryRevNumber;      fCdelete++;}
  if ( fRevText             != 0 ) {fRevText->Delete();          fCdelete++;}
  if ( fLayoutRevFieldText  != 0 ) {delete fLayoutRevFieldText;  fCdelete++;}
  if ( fLayoutRevFieldFrame != 0 ) {delete fLayoutRevFieldFrame; fCdelete++;}

  //+++++++++++++++++++++++  Horizontal Frame StexStin numbering + Nb Samp for calc + Calculations
  if ( fCompStnbFrame       != 0 ) {delete fCompStnbFrame;       fCdelete++;}
  if ( fLayoutCompStnbFrame != 0 ) {delete fLayoutCompStnbFrame; fCdelete++;}

  //............................ Stex Stin Numbering view (Button)
  if ( fButStexNb           != 0 ) {delete fButStexNb;           fCdelete++;}
  if ( fLayoutStexNbBut     != 0 ) {delete fLayoutStexNbBut;     fCdelete++;}
  //------------------- subframe NbSampForCalc
  if ( fNbSampForCalcFrame            != 0 ) {delete fNbSampForCalcFrame;            fCdelete++;}
  if ( fNbSampForCalcBut              != 0 ) {delete fNbSampForCalcBut;              fCdelete++;}
  if ( fLayoutNbSampForCalcBut        != 0 ) {delete fLayoutNbSampForCalcBut;        fCdelete++;}
  if ( fEntryNbSampForCalcNumber      != 0 ) {delete fEntryNbSampForCalcNumber;      fCdelete++;}
  if ( fNbSampForCalcText             != 0 ) {fNbSampForCalcText->Delete();          fCdelete++;}  
  if ( fLayoutNbSampForCalcFieldText  != 0 ) {delete fLayoutNbSampForCalcFieldText;  fCdelete++;}
  if ( fLayoutNbSampForCalcFieldFrame != 0 ) {delete fLayoutNbSampForCalcFieldFrame; fCdelete++;}
  //................................ Menus for CALCULATIONS
  if ( fMenuComput          != 0 ) {delete fMenuComput;          fCdelete++;}
  if ( fMenuBarComput       != 0 ) {fMenuBarComput->Delete();    fCdelete++;}

  //+++++++++++++++++++++++++++++++++++++++++++ Frame for quantities relative to the Stex 
  if ( fStexUpFrame          != 0 ) {delete fStexUpFrame;          fCdelete++;}

  //................................ Menus+Ymin+Ymax for the Stex ............................

  //...................................... Nb of evts in the data

  if ( fVmmD_NOE_ChNbFrame            != 0 ) {delete fVmmD_NOE_ChNbFrame;            fCdelete++;}

  if ( fVmaxD_NOE_ChNbFrame           != 0 ) {delete fVmaxD_NOE_ChNbFrame;           fCdelete++;}
  if ( fVmaxD_NOE_ChNbBut             != 0 ) {delete fVmaxD_NOE_ChNbBut;             fCdelete++;}
  if ( fLayoutVmaxD_NOE_ChNbBut       != 0 ) {delete fLayoutVmaxD_NOE_ChNbBut;       fCdelete++;}
  if ( fEntryVmaxD_NOE_ChNbNumber     != 0 ) {delete fEntryVmaxD_NOE_ChNbNumber;     fCdelete++;}
  if ( fVmaxD_NOE_ChNbText            != 0 ) {fVmaxD_NOE_ChNbText->Delete();         fCdelete++;}
  if ( fLayoutVmaxD_NOE_ChNbFieldText != 0 ) {delete fLayoutVmaxD_NOE_ChNbFieldText; fCdelete++;}
  if ( fLayoutVmaxD_NOE_ChNbFrame     != 0 ) {delete fLayoutVmaxD_NOE_ChNbFrame;     fCdelete++;}

  if ( fVminD_NOE_ChNbFrame           != 0 ) {delete fVminD_NOE_ChNbFrame;           fCdelete++;}
  if ( fVminD_NOE_ChNbBut             != 0 ) {delete fVminD_NOE_ChNbBut;             fCdelete++;}
  if ( fLayoutVminD_NOE_ChNbBut       != 0 ) {delete fLayoutVminD_NOE_ChNbBut;       fCdelete++;}
  if ( fEntryVminD_NOE_ChNbNumber     != 0 ) {delete fEntryVminD_NOE_ChNbNumber;     fCdelete++;}
  if ( fVminD_NOE_ChNbText            != 0 ) {fVminD_NOE_ChNbText->Delete();         fCdelete++;}
  if ( fLayoutVminD_NOE_ChNbFieldText != 0 ) {delete fLayoutVminD_NOE_ChNbFieldText; fCdelete++;}
  if ( fLayoutVminD_NOE_ChNbFrame     != 0 ) {delete fLayoutVminD_NOE_ChNbFrame;     fCdelete++;}

  if ( fMenuD_NOE_ChNb                != 0 ) {delete fMenuD_NOE_ChNb;                fCdelete++;}
  if ( fMenuBarD_NOE_ChNb             != 0 ) {fMenuBarD_NOE_ChNb->Delete();          fCdelete++;}
  if ( fVminD_NOE_ChNbText            != 0 ) {fVminD_NOE_ChNbText->Delete();         fCdelete++;}

  if ( fLayoutVmmD_NOE_ChNbFrame      != 0 ) {delete fLayoutVmmD_NOE_ChNbFrame;      fCdelete++;}

  //---------------------------------------------------
  if ( fVmmD_Ped_ChNbFrame            != 0 ) {delete fVmmD_Ped_ChNbFrame;            fCdelete++;}

  if ( fVmaxD_Ped_ChNbFrame           != 0 ) {delete fVmaxD_Ped_ChNbFrame;           fCdelete++;}
  if ( fVmaxD_Ped_ChNbBut             != 0 ) {delete fVmaxD_Ped_ChNbBut;             fCdelete++;}
  if ( fLayoutVmaxD_Ped_ChNbBut       != 0 ) {delete fLayoutVmaxD_Ped_ChNbBut;       fCdelete++;}
  if ( fVmaxD_Ped_ChNbText            != 0 ) {fVmaxD_Ped_ChNbText->Delete();         fCdelete++;}
  if ( fEntryVmaxD_Ped_ChNbNumber     != 0 ) {delete fEntryVmaxD_Ped_ChNbNumber;     fCdelete++;}
  if ( fLayoutVmaxD_Ped_ChNbFieldText != 0 ) {delete fLayoutVmaxD_Ped_ChNbFieldText; fCdelete++;}
  if ( fLayoutVmaxD_Ped_ChNbFrame     != 0 ) {delete fLayoutVmaxD_Ped_ChNbFrame;     fCdelete++;}

  if ( fVminD_Ped_ChNbFrame           != 0 ) {delete fVminD_Ped_ChNbFrame;           fCdelete++;}
  if ( fVminD_Ped_ChNbBut             != 0 ) {delete fVminD_Ped_ChNbBut;             fCdelete++;}
  if ( fLayoutVminD_Ped_ChNbBut       != 0 ) {delete fLayoutVminD_Ped_ChNbBut;       fCdelete++;}
  if ( fVminD_Ped_ChNbText            != 0 ) {fVminD_Ped_ChNbText->Delete();         fCdelete++;}
  if ( fEntryVminD_Ped_ChNbNumber     != 0 ) {delete fEntryVminD_Ped_ChNbNumber;     fCdelete++;}
  if ( fLayoutVminD_Ped_ChNbFieldText != 0 ) {delete fLayoutVminD_Ped_ChNbFieldText; fCdelete++;}
  if ( fLayoutVminD_Ped_ChNbFrame     != 0 ) {delete fLayoutVminD_Ped_ChNbFrame;     fCdelete++;}

  if ( fMenuD_Ped_ChNb                != 0 ) {delete fMenuD_Ped_ChNb;                fCdelete++;}
  if ( fMenuBarD_Ped_ChNb             != 0 ) {fMenuBarD_Ped_ChNb->Delete();          fCdelete++;}
  if ( fLayoutMenuBarD_Ped_ChNb       != 0 ) {delete fLayoutMenuBarD_Ped_ChNb;       fCdelete++;}

  if ( fLayoutVmmD_Ped_ChNbFrame      != 0 ) {delete fLayoutVmmD_Ped_ChNbFrame;      fCdelete++;}

  //----------------------------------------------------
  if ( fVmmD_TNo_ChNbFrame            != 0 ) {delete fVmmD_TNo_ChNbFrame;            fCdelete++;}

  if ( fVmaxD_TNo_ChNbFrame           != 0 ) {delete fVmaxD_TNo_ChNbFrame;           fCdelete++;}
  if ( fVmaxD_TNo_ChNbBut             != 0 ) {delete fVmaxD_TNo_ChNbBut;             fCdelete++;}
  if ( fLayoutVmaxD_TNo_ChNbBut       != 0 ) {delete fLayoutVmaxD_TNo_ChNbBut;       fCdelete++;}
  if ( fVmaxD_TNo_ChNbText            != 0 ) {fVmaxD_TNo_ChNbText->Delete();         fCdelete++;}
  if ( fEntryVmaxD_TNo_ChNbNumber     != 0 ) {delete fEntryVmaxD_TNo_ChNbNumber;     fCdelete++;}
  if ( fLayoutVmaxD_TNo_ChNbFieldText != 0 ) {delete fLayoutVmaxD_TNo_ChNbFieldText; fCdelete++;}
  if ( fLayoutVmaxD_TNo_ChNbFrame     != 0 ) {delete fLayoutVmaxD_TNo_ChNbFrame;     fCdelete++;}
 
  if ( fVminD_TNo_ChNbFrame           != 0 ) {delete fVminD_TNo_ChNbFrame;           fCdelete++;}
  if ( fVminD_TNo_ChNbBut             != 0 ) {delete fVminD_TNo_ChNbBut;             fCdelete++;}
  if ( fLayoutVminD_TNo_ChNbBut       != 0 ) {delete fLayoutVminD_TNo_ChNbBut;       fCdelete++;}
  if ( fVminD_TNo_ChNbText            != 0 ) {fVminD_TNo_ChNbText->Delete();         fCdelete++;}
  if ( fEntryVminD_TNo_ChNbNumber     != 0 ) {delete fEntryVminD_TNo_ChNbNumber;     fCdelete++;}
  if ( fLayoutVminD_TNo_ChNbFieldText != 0 ) {delete fLayoutVminD_TNo_ChNbFieldText; fCdelete++;}
  if ( fLayoutVminD_TNo_ChNbFrame     != 0 ) {delete fLayoutVminD_TNo_ChNbFrame;     fCdelete++;}
 
  if ( fMenuD_TNo_ChNb                != 0 ) {delete fMenuD_TNo_ChNb;                fCdelete++;}
  if ( fMenuBarD_TNo_ChNb             != 0 ) {fMenuBarD_TNo_ChNb->Delete();          fCdelete++;}
  if ( fLayoutMenuBarD_TNo_ChNb       != 0 ) {delete fLayoutMenuBarD_TNo_ChNb;       fCdelete++;}

  if ( fLayoutVmmD_TNo_ChNbFrame      != 0 ) {delete fLayoutVmmD_TNo_ChNbFrame;      fCdelete++;}

  //-----------------------------------------------------------
  if ( fVmmD_MCs_ChNbFrame            != 0 ) {delete fVmmD_MCs_ChNbFrame;            fCdelete++;}

  if ( fVmaxD_MCs_ChNbFrame           != 0 ) {delete fVmaxD_MCs_ChNbFrame;           fCdelete++;}
  if ( fVmaxD_MCs_ChNbBut             != 0 ) {delete fVmaxD_MCs_ChNbBut;             fCdelete++;}
  if ( fLayoutVmaxD_MCs_ChNbBut       != 0 ) {delete fLayoutVmaxD_MCs_ChNbBut;       fCdelete++;}
  if ( fVmaxD_MCs_ChNbText            != 0 ) {fVmaxD_MCs_ChNbText->Delete();         fCdelete++;}
  if ( fEntryVmaxD_MCs_ChNbNumber     != 0 ) {delete fEntryVmaxD_MCs_ChNbNumber;     fCdelete++;}
  if ( fLayoutVmaxD_MCs_ChNbFieldText != 0 ) {delete fLayoutVmaxD_MCs_ChNbFieldText; fCdelete++;}
  if ( fLayoutVmaxD_MCs_ChNbFrame     != 0 ) {delete fLayoutVmaxD_MCs_ChNbFrame;     fCdelete++;}

  if ( fVminD_MCs_ChNbFrame           != 0 ) {delete fVminD_MCs_ChNbFrame;           fCdelete++;}
  if ( fVminD_MCs_ChNbBut             != 0 ) {delete fVminD_MCs_ChNbBut;             fCdelete++;}
  if ( fLayoutVminD_MCs_ChNbBut       != 0 ) {delete fLayoutVminD_MCs_ChNbBut;       fCdelete++;}
  if ( fVminD_MCs_ChNbText            != 0 ) {fVminD_MCs_ChNbText->Delete();         fCdelete++;}
  if ( fEntryVminD_MCs_ChNbNumber     != 0 ) {delete fEntryVminD_MCs_ChNbNumber;     fCdelete++;}
  if ( fLayoutVminD_MCs_ChNbFieldText != 0 ) {delete fLayoutVminD_MCs_ChNbFieldText; fCdelete++;}
  if ( fLayoutVminD_MCs_ChNbFrame     != 0 ) {delete fLayoutVminD_MCs_ChNbFrame;     fCdelete++;}

  if ( fMenuD_MCs_ChNb                != 0 ) {delete fMenuD_MCs_ChNb;                fCdelete++;}
  if ( fMenuBarD_MCs_ChNb             != 0 ) {fMenuBarD_MCs_ChNb->Delete();          fCdelete++;}
  if ( fLayoutMenuBarD_MCs_ChNb       != 0 ) {delete fLayoutMenuBarD_MCs_ChNb;       fCdelete++;}

  if ( fLayoutVmmD_MCs_ChNbFrame      != 0 ) {delete fLayoutVmmD_MCs_ChNbFrame;      fCdelete++;}
  
  //............................................... Frame Sig + Menus Sig
  if ( fStexHozFrame         != 0 ) {delete fStexHozFrame;         fCdelete++;}

  //------------------------------------------------------------- 
  if ( fVmmD_LFN_ChNbFrame            != 0 ) {delete fVmmD_LFN_ChNbFrame;            fCdelete++;}

  if ( fVmaxD_LFN_ChNbFrame           != 0 ) {delete fVmaxD_LFN_ChNbFrame;           fCdelete++;}
  if ( fVmaxD_LFN_ChNbBut             != 0 ) {delete fVmaxD_LFN_ChNbBut;             fCdelete++;}
  if ( fLayoutVmaxD_LFN_ChNbBut       != 0 ) {delete fLayoutVmaxD_LFN_ChNbBut;       fCdelete++;}
  if ( fVmaxD_LFN_ChNbText            != 0 ) {fVmaxD_LFN_ChNbText->Delete();         fCdelete++;}
  if ( fEntryVmaxD_LFN_ChNbNumber     != 0 ) {delete fEntryVmaxD_LFN_ChNbNumber;     fCdelete++;}
  if ( fLayoutVmaxD_LFN_ChNbFieldText != 0 ) {delete fLayoutVmaxD_LFN_ChNbFieldText; fCdelete++;}
  if ( fLayoutVmaxD_LFN_ChNbFrame     != 0 ) {delete fLayoutVmaxD_LFN_ChNbFrame;     fCdelete++;}

  if ( fVminD_LFN_ChNbFrame           != 0 ) {delete fVminD_LFN_ChNbFrame;           fCdelete++;}
  if ( fVminD_LFN_ChNbBut             != 0 ) {delete fVminD_LFN_ChNbBut;             fCdelete++;}
  if ( fLayoutVminD_LFN_ChNbBut       != 0 ) {delete fLayoutVminD_LFN_ChNbBut;       fCdelete++;}
  if ( fVminD_LFN_ChNbText            != 0 ) {fVminD_LFN_ChNbText->Delete();         fCdelete++;}
  if ( fEntryVminD_LFN_ChNbNumber     != 0 ) {delete fEntryVminD_LFN_ChNbNumber;     fCdelete++;}
  if ( fLayoutVminD_LFN_ChNbFieldText != 0 ) {delete fLayoutVminD_LFN_ChNbFieldText; fCdelete++;}
  if ( fLayoutVminD_LFN_ChNbFrame     != 0 ) {delete fLayoutVminD_LFN_ChNbFrame;     fCdelete++;}

  if ( fMenuD_LFN_ChNb                != 0 ) {delete fMenuD_LFN_ChNb;                fCdelete++;}
  if ( fMenuBarD_LFN_ChNb             != 0 ) {fMenuBarD_LFN_ChNb->Delete();          fCdelete++;}
  if ( fLayoutMenuBarD_LFN_ChNb       != 0 ) {delete fLayoutMenuBarD_LFN_ChNb;       fCdelete++;}

  if ( fLayoutVmmD_LFN_ChNbFrame      != 0 ) {delete fLayoutVmmD_LFN_ChNbFrame;      fCdelete++;}

  //-------------------------------------------------------------
  if ( fVmmD_HFN_ChNbFrame            != 0 ) {delete fVmmD_HFN_ChNbFrame;            fCdelete++;}

  if ( fVmaxD_HFN_ChNbFrame           != 0 ) {delete fVmaxD_HFN_ChNbFrame;           fCdelete++;}
  if ( fVmaxD_HFN_ChNbBut             != 0 ) {delete fVmaxD_HFN_ChNbBut;             fCdelete++;}
  if ( fLayoutVmaxD_HFN_ChNbBut       != 0 ) {delete fLayoutVmaxD_HFN_ChNbBut;       fCdelete++;}
  if ( fVmaxD_HFN_ChNbText            != 0 ) {fVmaxD_HFN_ChNbText->Delete();         fCdelete++;}
  if ( fEntryVmaxD_HFN_ChNbNumber     != 0 ) {delete fEntryVmaxD_HFN_ChNbNumber;     fCdelete++;}
  if ( fLayoutVmaxD_HFN_ChNbFieldText != 0 ) {delete fLayoutVmaxD_HFN_ChNbFieldText; fCdelete++;}
  if ( fLayoutVmaxD_HFN_ChNbFrame     != 0 ) {delete fLayoutVmaxD_HFN_ChNbFrame;     fCdelete++;}

  if ( fVminD_HFN_ChNbFrame           != 0 ) {delete fVminD_HFN_ChNbFrame;           fCdelete++;}
  if ( fVminD_HFN_ChNbBut             != 0 ) {delete fVminD_HFN_ChNbBut;             fCdelete++;}
  if ( fLayoutVminD_HFN_ChNbBut       != 0 ) {delete fLayoutVminD_HFN_ChNbBut;       fCdelete++;}
  if ( fVminD_HFN_ChNbText            != 0 ) {fVminD_HFN_ChNbText->Delete();         fCdelete++;}
  if ( fEntryVminD_HFN_ChNbNumber     != 0 ) {delete fEntryVminD_HFN_ChNbNumber;     fCdelete++;}
  if ( fLayoutVminD_HFN_ChNbFieldText != 0 ) {delete fLayoutVminD_HFN_ChNbFieldText; fCdelete++;}
  if ( fLayoutVminD_HFN_ChNbFrame     != 0 ) {delete fLayoutVminD_HFN_ChNbFrame;     fCdelete++;}

  if ( fMenuD_HFN_ChNb                != 0 ) {delete fMenuD_HFN_ChNb;                fCdelete++;}
  if ( fMenuBarD_HFN_ChNb             != 0 ) {fMenuBarD_HFN_ChNb->Delete();          fCdelete++;}
  if ( fLayoutMenuBarD_HFN_ChNb       != 0 ) {delete fLayoutMenuBarD_HFN_ChNb;       fCdelete++;}

  if ( fLayoutVmmD_HFN_ChNbFrame      != 0 ) {delete fLayoutVmmD_HFN_ChNbFrame;      fCdelete++;}

  //-------------------------------------------------------------
  if ( fVmmD_SCs_ChNbFrame            != 0 ) {delete fVmmD_SCs_ChNbFrame;            fCdelete++;}

  if ( fVmaxD_SCs_ChNbFrame           != 0 ) {delete fVmaxD_SCs_ChNbFrame;           fCdelete++;}
  if ( fVmaxD_SCs_ChNbBut             != 0 ) {delete fVmaxD_SCs_ChNbBut;             fCdelete++;}
  if ( fLayoutVmaxD_SCs_ChNbBut       != 0 ) {delete fLayoutVmaxD_SCs_ChNbBut;       fCdelete++;}
  if ( fVmaxD_SCs_ChNbText            != 0 ) {fVmaxD_SCs_ChNbText->Delete();         fCdelete++;}
  if ( fEntryVmaxD_SCs_ChNbNumber     != 0 ) {delete fEntryVmaxD_SCs_ChNbNumber;     fCdelete++;}
  if ( fLayoutVmaxD_SCs_ChNbFieldText != 0 ) {delete fLayoutVmaxD_SCs_ChNbFieldText; fCdelete++;}
  if ( fLayoutVmaxD_SCs_ChNbFrame     != 0 ) {delete fLayoutVmaxD_SCs_ChNbFrame;     fCdelete++;}

  if ( fVminD_SCs_ChNbFrame           != 0 ) {delete fVminD_SCs_ChNbFrame;           fCdelete++;}
  if ( fVminD_SCs_ChNbBut             != 0 ) {delete fVminD_SCs_ChNbBut;             fCdelete++;}
  if ( fLayoutVminD_SCs_ChNbBut       != 0 ) {delete fLayoutVminD_SCs_ChNbBut;       fCdelete++;}
  if ( fVminD_SCs_ChNbText            != 0 ) {fVminD_SCs_ChNbText->Delete();         fCdelete++;}
  if ( fEntryVminD_SCs_ChNbNumber     != 0 ) {delete fEntryVminD_SCs_ChNbNumber;     fCdelete++;}
  if ( fLayoutVminD_SCs_ChNbFieldText != 0 ) {delete fLayoutVminD_SCs_ChNbFieldText; fCdelete++;}
  if ( fLayoutVminD_SCs_ChNbFrame     != 0 ) {delete fLayoutVminD_SCs_ChNbFrame;     fCdelete++;}

  if ( fMenuD_SCs_ChNb                != 0 ) {delete fMenuD_SCs_ChNb;                fCdelete++;}
  if ( fMenuBarD_SCs_ChNb             != 0 ) {fMenuBarD_SCs_ChNb->Delete();          fCdelete++;}
  if ( fLayoutMenuBarD_SCs_ChNb       != 0 ) {delete fLayoutMenuBarD_SCs_ChNb;       fCdelete++;}

  if ( fLayoutVmmD_SCs_ChNbFrame      != 0 ) {delete fLayoutVmmD_SCs_ChNbFrame;      fCdelete++;}
  //-------------------------------------------------------------
  if ( fLayoutStexHozFrame          != 0 ) {delete fLayoutStexHozFrame;          fCdelete++;}

  //----------------------------------------------------------------------------------------------

  //...................................... Covariances between Stins
  if ( fVmmLHFccFrame            != 0 ) {delete fVmmLHFccFrame;            fCdelete++;}

  if ( fVmaxLHFccFrame           != 0 ) {delete fVmaxLHFccFrame;           fCdelete++;}
  if ( fVmaxLHFccBut             != 0 ) {delete fVmaxLHFccBut;             fCdelete++;}
  if ( fLayoutVmaxLHFccBut       != 0 ) {delete fLayoutVmaxLHFccBut;       fCdelete++;}
  if ( fVmaxLHFccText            != 0 ) {fVmaxLHFccText->Delete();         fCdelete++;}
  if ( fEntryVmaxLHFccNumber     != 0 ) {delete fEntryVmaxLHFccNumber;     fCdelete++;}
  if ( fLayoutVmaxLHFccFieldText != 0 ) {delete fLayoutVmaxLHFccFieldText; fCdelete++;}
  if ( fLayoutVmaxLHFccFrame     != 0 ) {delete fLayoutVmaxLHFccFrame;     fCdelete++;}

  if ( fVminLHFccFrame           != 0 ) {delete fVminLHFccFrame;           fCdelete++;}
  if ( fVminLHFccBut             != 0 ) {delete fVminLHFccBut;             fCdelete++;}
  if ( fLayoutVminLHFccBut       != 0 ) {delete fLayoutVminLHFccBut;       fCdelete++;}
  if ( fVminLHFccText            != 0 ) {fVminLHFccText->Delete();         fCdelete++;}
  if ( fEntryVminLHFccNumber     != 0 ) {delete fEntryVminLHFccNumber;     fCdelete++;}
  if ( fLayoutVminLHFccFieldText != 0 ) {delete fLayoutVminLHFccFieldText; fCdelete++;}
  if ( fLayoutVminLHFccFrame     != 0 ) {delete fLayoutVminLHFccFrame;     fCdelete++;}

  if ( fMenuLHFcc             != 0 ) {delete fMenuLHFcc;             fCdelete++;}
  if ( fMenuBarLHFcc          != 0 ) {fMenuBarLHFcc->Delete();       fCdelete++;}
  if ( fLayoutMenuBarLHFcc    != 0 ) {delete fLayoutMenuBarLHFcc;    fCdelete++;}

  if ( fLayoutVmmLHFccFrame      != 0 ) {delete fLayoutVmmLHFccFrame;      fCdelete++;}

  //...................................... Low Freq Cor(c,c') for each pair of Stins  
  if ( fVmmLFccMosFrame            != 0 ) {delete fVmmLFccMosFrame;            fCdelete++;}

  if ( fVmaxLFccMosFrame           != 0 ) {delete fVmaxLFccMosFrame;           fCdelete++;}
  if ( fVmaxLFccMosBut             != 0 ) {delete fVmaxLFccMosBut;             fCdelete++;}
  if ( fLayoutVmaxLFccMosBut       != 0 ) {delete fLayoutVmaxLFccMosBut;       fCdelete++;}
  if ( fVmaxLFccMosText            != 0 ) {fVmaxLFccMosText->Delete();         fCdelete++;}
  if ( fEntryVmaxLFccMosNumber     != 0 ) {delete fEntryVmaxLFccMosNumber;     fCdelete++;}
  if ( fLayoutVmaxLFccMosFieldText != 0 ) {delete fLayoutVmaxLFccMosFieldText; fCdelete++;}
  if ( fLayoutVmaxLFccMosFrame     != 0 ) {delete fLayoutVmaxLFccMosFrame;     fCdelete++;}

  if ( fVminLFccMosFrame           != 0 ) {delete fVminLFccMosFrame;           fCdelete++;}
  if ( fVminLFccMosBut             != 0 ) {delete fVminLFccMosBut;             fCdelete++;}
  if ( fLayoutVminLFccMosBut       != 0 ) {delete fLayoutVminLFccMosBut;       fCdelete++;}
  if ( fVminLFccMosText            != 0 ) {fVminLFccMosText->Delete();         fCdelete++;}
  if ( fEntryVminLFccMosNumber     != 0 ) {delete fEntryVminLFccMosNumber;     fCdelete++;}
  if ( fLayoutVminLFccMosFieldText != 0 ) {delete fLayoutVminLFccMosFieldText; fCdelete++;}
  if ( fLayoutVminLFccMosFrame     != 0 ) {delete fLayoutVminLFccMosFrame;     fCdelete++;}

  if ( fMenuLFccMos                  != 0 ) {delete fMenuLFccMos;                  fCdelete++;}
  if ( fMenuBarLFccMos               != 0 ) {fMenuBarLFccMos->Delete();            fCdelete++;}
  if ( fLayoutMenuBarLFccMos         != 0 ) {delete fLayoutMenuBarLFccMos;         fCdelete++;}

  if ( fLayoutVmmLFccMosFrame      != 0 ) {delete fLayoutVmmLFccMosFrame;      fCdelete++;}

  //...................................... High Freq Cor(c,c') for each pair of Stins  
  if ( fVmmHFccMosFrame            != 0 ) {delete fVmmHFccMosFrame;            fCdelete++;}

  if ( fVmaxHFccMosFrame           != 0 ) {delete fVmaxHFccMosFrame;           fCdelete++;}
  if ( fVmaxHFccMosBut             != 0 ) {delete fVmaxHFccMosBut;             fCdelete++;}
  if ( fLayoutVmaxHFccMosBut       != 0 ) {delete fLayoutVmaxHFccMosBut;       fCdelete++;}
  if ( fVmaxHFccMosText            != 0 ) {fVmaxHFccMosText->Delete();         fCdelete++;}
  if ( fEntryVmaxHFccMosNumber     != 0 ) {delete fEntryVmaxHFccMosNumber;     fCdelete++;}
  if ( fLayoutVmaxHFccMosFieldText != 0 ) {delete fLayoutVmaxHFccMosFieldText; fCdelete++;}
  if ( fLayoutVmaxHFccMosFrame     != 0 ) {delete fLayoutVmaxHFccMosFrame;     fCdelete++;}

  if ( fVminHFccMosFrame           != 0 ) {delete fVminHFccMosFrame;           fCdelete++;}
  if ( fVminHFccMosBut             != 0 ) {delete fVminHFccMosBut;             fCdelete++;}
  if ( fLayoutVminHFccMosBut       != 0 ) {delete fLayoutVminHFccMosBut;       fCdelete++;}
  if ( fVminHFccMosText            != 0 ) {fVminHFccMosText->Delete();         fCdelete++;}
  if ( fEntryVminHFccMosNumber     != 0 ) {delete fEntryVminHFccMosNumber;     fCdelete++;}
  if ( fLayoutVminHFccMosFieldText != 0 ) {delete fLayoutVminHFccMosFieldText; fCdelete++;}
  if ( fLayoutVminHFccMosFrame     != 0 ) {delete fLayoutVminHFccMosFrame;     fCdelete++;}

  if ( fMenuHFccMos                != 0 ) {delete fMenuHFccMos;                fCdelete++;}
  if ( fMenuBarHFccMos             != 0 ) {fMenuBarHFccMos->Delete();          fCdelete++;}
  if ( fLayoutMenuBarHFccMos       != 0 ) {delete fLayoutMenuBarHFccMos;       fCdelete++;}

  if ( fLayoutVmmHFccMosFrame      != 0 ) {delete fLayoutVmmHFccMosFrame;      fCdelete++;}

  if ( fLayoutStexUpFrame          != 0 ) {delete fLayoutStexUpFrame;          fCdelete++;}

  //+++++++++++++++++++++++++++++++++++++++++ Horizontal frame Stin_A + Stin_B
  if ( fStinSpFrame       != 0 ) {delete fStinSpFrame;         fCdelete++;}
  
  //----------------------------------- SubFrame Stin_A (Button + EntryField)
  if ( fTxSubFrame        != 0 ) {delete fTxSubFrame;          fCdelete++;}

  if ( fStinAFrame        != 0 ) {delete fStinAFrame;          fCdelete++;}
  if ( fStinABut          != 0 ) {delete fStinABut;            fCdelete++;}
  if ( fLayoutStinABut    != 0 ) {delete fLayoutStinABut;      fCdelete++;} 
  if ( fEntryStinANumber  != 0 ) {delete fEntryStinANumber;    fCdelete++;}
  if ( fStinAText         != 0 ) {fStinAText->Delete();        fCdelete++;} 
  if ( fLayoutStinAField  != 0 ) {delete fLayoutStinAField;    fCdelete++;} 

  //............................ Stin Crystal Numbering view (Button)
  if ( fButChNb          != 0 ) {delete fButChNb;            fCdelete++;}
  if ( fLayoutChNbBut    != 0 ) {delete fLayoutChNbBut;      fCdelete++;} 

  //............................ Menus Stin_A
  if ( fMenuCorssAll     != 0 ) {delete fMenuCorssAll;       fCdelete++;}
  if ( fMenuBarCorssAll  != 0 ) {fMenuBarCorssAll->Delete(); fCdelete++;}

  //if ( fMenuCovssAll     != 0 ) {delete fMenuCovssAll;       fCdelete++;}
  //if ( fMenuBarCovssAll  != 0 ) {fMenuBarCovssAll->Delete(); fCdelete++;}

  if ( fLayoutTxSubFrame != 0 ) {delete fLayoutTxSubFrame;   fCdelete++;}

  //----------------------------------- SubFrame Stin_B (Button + EntryField)

  if ( fTySubFrame        != 0 ) {delete fTySubFrame;        fCdelete++;}

  if ( fStinBFrame        != 0 ) {delete fStinBFrame;        fCdelete++;}
  if ( fStinBBut          != 0 ) {delete fStinBBut;          fCdelete++;}
  if ( fLayoutStinBBut    != 0 ) {delete fLayoutStinBBut;    fCdelete++;}
  if ( fEntryStinBNumber  != 0 ) {delete fEntryStinBNumber;  fCdelete++;}
  if ( fStinBText         != 0 ) {fStinBText->Delete();      fCdelete++;}
  if ( fLayoutStinBField  != 0 ) {delete fLayoutStinBField;  fCdelete++;}

  if ( fLayoutTySubFrame  != 0 ) {delete fLayoutTySubFrame;  fCdelete++;}

  if ( fLayoutStinSpFrame != 0 ) {delete fLayoutStinSpFrame; fCdelete++;}

  //.................................. Menus for Horizontal frame (Stin_A + Stin_B)

  if ( fMenuLFCorcc     != 0 ) {delete fMenuLFCorcc;        fCdelete++;}
  if ( fMenuBarLFCorcc  != 0 ) {fMenuBarLFCorcc->Delete();  fCdelete++;}

  if ( fMenuHFCorcc     != 0 ) {delete fMenuHFCorcc;        fCdelete++;}
  if ( fMenuBarHFCorcc  != 0 ) {fMenuBarHFCorcc->Delete();  fCdelete++;}

  //++++++++++++++++++++++++ Horizontal frame channel number (Stin_A crystal number) + sample number
  if ( fChSpFrame        != 0 ) {delete fChSpFrame;         fCdelete++;}

  //------------------------------------- SubFrame Channel (Button + EntryField)

  if ( fChSubFrame       != 0 ) {delete fChSubFrame;        fCdelete++;}

  if ( fChanFrame        != 0 ) {delete fChanFrame;         fCdelete++;}
  if ( fChanBut          != 0 ) {delete fChanBut;           fCdelete++;}
  if ( fLayoutChanBut    != 0 ) {delete fLayoutChanBut;     fCdelete++;}
  if ( fEntryChanNumber  != 0 ) {delete fEntryChanNumber;   fCdelete++;}
  if ( fChanText         != 0 ) {fChanText->Delete();       fCdelete++;}
  if ( fLayoutChanField  != 0 ) {delete fLayoutChanField;   fCdelete++;}

  //................................ Menus Stin_A crystal number
  if ( fMenuCorss        != 0 ) {delete fMenuCorss;         fCdelete++;}
  if ( fMenuBarCorss     != 0 ) {fMenuBarCorss->Delete();   fCdelete++;}

  if ( fMenuCovss        != 0 ) {delete fMenuCovss;         fCdelete++;}
  if ( fMenuBarCovss     != 0 ) {fMenuBarCovss->Delete();   fCdelete++;}

  if ( fMenuD_MSp_SpNb    != 0 ) {delete fMenuD_MSp_SpNb;       fCdelete++;}
  if ( fMenuBarD_MSp_SpNb != 0 ) {fMenuBarD_MSp_SpNb->Delete(); fCdelete++;}
  if ( fMenuD_MSp_SpDs    != 0 ) {delete fMenuD_MSp_SpDs;       fCdelete++;}
  if ( fMenuBarD_MSp_SpDs != 0 ) {fMenuBarD_MSp_SpDs->Delete(); fCdelete++;}

  if ( fMenuD_SSp_SpNb    != 0 ) {delete fMenuD_SSp_SpNb;       fCdelete++;}
  if ( fMenuBarD_SSp_SpNb != 0 ) {fMenuBarD_SSp_SpNb->Delete(); fCdelete++;}
  if ( fMenuD_SSp_SpDs    != 0 ) {delete fMenuD_SSp_SpDs;       fCdelete++;}
  if ( fMenuBarD_SSp_SpDs != 0 ) {fMenuBarD_SSp_SpDs->Delete(); fCdelete++;}

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

  if ( fLayoutChSpFrame  != 0 ) {delete fLayoutChSpFrame;   fCdelete++;}

  //++++++++++++++++++++++++++++++++++++ Menu Event Distribution
  if ( fMenuAdcProj            != 0 ) {delete fMenuAdcProj;            fCdelete++;}
  if ( fMenuBarAdcProj         != 0 ) {fMenuBarAdcProj->Delete();      fCdelete++;}
  if ( fLayoutMenuBarAdcProj   != 0 ) {delete fLayoutMenuBarAdcProj;   fCdelete++;}

  //++++++++++++++++++++++++++++++++++++ Frame: Run List (Rul) (Button + EntryField)
  if ( fRulFrame            != 0 ) {delete fRulFrame;            fCdelete++;}
  if ( fRulBut              != 0 ) {delete fRulBut;              fCdelete++;}
  if ( fLayoutRulBut        != 0 ) {delete fLayoutRulBut;        fCdelete++;}
  if ( fEntryRulNumber      != 0 ) {delete fEntryRulNumber;      fCdelete++;}
  if ( fRulText             != 0 ) {fRulText->Delete();          fCdelete++;}
  if ( fLayoutRulFieldText  != 0 ) {delete fLayoutRulFieldText;  fCdelete++;}
  if ( fLayoutRulFieldFrame != 0 ) {delete fLayoutRulFieldFrame; fCdelete++;}

  //................................ Menus for time evolution
  if ( fMenuHistory         != 0 ) {delete fMenuHistory;         fCdelete++;}
  if ( fMenuBarHistory      != 0 ) {fMenuBarHistory->Delete();   fCdelete++;}

  //++++++++++++++++++++++++++++++++++++ LinLog Frame
  if ( fLinLogFrame   != 0 ) {delete fLinLogFrame;   fCdelete++;}

  //---------------------------------- Lin/Log X
  if ( fButLogx         != 0 ) {delete fButLogx;         fCdelete++;}
  if ( fLayoutLogxBut   != 0 ) {delete fLayoutLogxBut;   fCdelete++;}
  //---------------------------------- Lin/Log Y
  if ( fButLogy         != 0 ) {delete fButLogy;         fCdelete++;}
  if ( fLayoutLogyBut   != 0 ) {delete fLayoutLogyBut;   fCdelete++;} 
  //---------------------------------- Proj Y
  if ( fButProjy        != 0 ) {delete fButProjy;        fCdelete++;}
  if ( fLayoutProjyBut  != 0 ) {delete fLayoutProjyBut;  fCdelete++;} 

  //++++++++++++++++++++++++++++++++++++ Frame: General Title (Gent) (Button + EntryField)
  if ( fGentFrame            != 0 ) {delete fGentFrame;            fCdelete++;}
  if ( fGentBut              != 0 ) {delete fGentBut;              fCdelete++;}
  if ( fLayoutGentBut        != 0 ) {delete fLayoutGentBut;        fCdelete++;}
  if ( fEntryGentNumber      != 0 ) {delete fEntryGentNumber;      fCdelete++;}
  if ( fGentText             != 0 ) {fGentText->Delete();          fCdelete++;}
  if ( fLayoutGentFieldText  != 0 ) {delete fLayoutGentFieldText;  fCdelete++;}
  if ( fLayoutGentFieldFrame != 0 ) {delete fLayoutGentFieldFrame; fCdelete++;}

  //++++++++++++++++++++++++++++++++++++ Color + EXIT BUTTON
  if ( fColorExitFrame       != 0 ) {delete fColorExitFrame;       fCdelete++;}
  if ( fLayoutColorExitFrame != 0 ) {delete fLayoutColorExitFrame; fCdelete++;}

  //---------------------------------- Color Palette
  if ( fButColPal       != 0 ) {delete fButColPal;       fCdelete++;}
  if ( fLayoutColPalBut != 0 ) {delete fLayoutColPalBut; fCdelete++;}
  //---------------------------------- Exit
  if ( fButExit       != 0 ) {delete fButExit;       fCdelete++;}
  if ( fLayoutExitBut != 0 ) {delete fLayoutExitBut; fCdelete++;}
 
  //++++++++++++++++++++++++++++++++++++ Last Frame
  if ( fLastFrame     != 0 ) {delete fLastFrame;     fCdelete++;}

  //--------------------------------- Clone Last Canvas (Button)
  if ( fButClone       != 0 ) {delete fButClone;       fCdelete++;}
  if ( fLayoutCloneBut != 0 ) {delete fLayoutCloneBut; fCdelete++;}

  //--------------------------------- Root version (Button)
  if ( fButRoot       != 0 ) {delete fButRoot;       fCdelete++;}
  if ( fLayoutRootBut != 0 ) {delete fLayoutRootBut; fCdelete++;}

  //--------------------------------- Help (Button)
  if ( fButHelp       != 0 ) {delete fButHelp;       fCdelete++;}
  if ( fLayoutHelpBut != 0 ) {delete fLayoutHelpBut; fCdelete++;}

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if ( fCnew != fCdelete )
    {
      cout << "*TEcnaGui> WRONG MANAGEMENT OF ALLOCATIONS: fCnew = "
	   << fCnew << ", fCdelete = " << fCdelete << endl;
    }
  else
    {
      //cout << "*TEcnaGui> BRAVO! GOOD MANAGEMENT OF ALLOCATIONS: fCnew = "
      //   << fCnew << ", fCdelete = " << fCdelete << endl;
    }

#endif // DEST

#define MGRA
#ifndef MGRA
  if ( fCnewRoot != fCdeleteRoot )
    {
      cout << "*TEcnaGui> WRONG MANAGEMENT OF ROOT ALLOCATIONS: fCnewRoot = "
	   << fCnewRoot << ", fCdeleteRoot = " << fCdeleteRoot << endl;
    }
  else
    {
      cout << "*TEcnaGui> BRAVO! GOOD MANAGEMENT OF ROOT ALLOCATIONS:"
	   << " fCnewRoot = " << fCnewRoot <<", fCdeleteRoot = "
	   << fCdeleteRoot << endl;
    }
#endif // MGRA

  // cout << "TEcnaGui> Leaving destructor" << endl;
  // cout << "            fCnew = " << fCnew << ", fCdelete = " << fCdelete << endl;

  // cout << "[Info Management] CLASS: TEcnaGui.           DESTROY OBJECT: this = " << this << endl;

}
//   end of destructor

//===================================================================
//
//                   Constructors
//
//===================================================================

TEcnaGui::TEcnaGui()
{
  Init();
}

TEcnaGui::TEcnaGui(TEcnaObject* pObjectManager, const TString& SubDet, const TGWindow *p, UInt_t w, UInt_t h):
TGMainFrame(p, w, h) 
{
  // cout << "[Info Management] CLASS: TEcnaGui.           CREATE OBJECT: this = " << this << endl;

  // cout << "TEcnaGui> Entering constructor with arguments" << endl;
  // cout << "            fCnew = " << fCnew << ", fCdelete = " << fCdelete << endl;

  fObjectManager = (TEcnaObject*)pObjectManager;
  Long_t i_this = (Long_t)this;
  pObjectManager->RegisterPointer("TEcnaGui", i_this);

  Int_t MaxCar = fgMaxCar;
  fSubDet.Resize(MaxCar); 
  fSubDet = SubDet.Data();

  //............................ fCnaParCout
  fCnaParCout = 0;
  Long_t iCnaParCout = pObjectManager->GetPointerValue("TEcnaParCout");
  if( iCnaParCout == 0 )
    {fCnaParCout = new TEcnaParCout(pObjectManager); /*fCnew++*/}
  else
    {fCnaParCout = (TEcnaParCout*)iCnaParCout;}

  //fCnaParPaths = 0; fCnaParPaths = new TEcnaParPaths();                        //fCnew++;
  //fCnaParPaths->GetPaths();
  //if( fCnaParPaths->GetPaths() == kTRUE )
  // {
  //fCnaParPaths->GetCMSSWParameters();

  //............................ fCnaParPaths
  fCnaParPaths = 0;
  Long_t iCnaParPaths = pObjectManager->GetPointerValue("TEcnaParPaths");
  if( iCnaParPaths == 0 )
    {fCnaParPaths = new TEcnaParPaths(pObjectManager); /*fCnew++*/}
  else
    {fCnaParPaths = (TEcnaParPaths*)iCnaParPaths;}

  fCnaParPaths->GetPaths();
  fCnaParPaths->GetCMSSWParameters();

  //............................ fEcal  => to be changed in fParEcal
  fEcal = 0;
  Long_t iParEcal = pObjectManager->GetPointerValue("TEcnaParEcal");
  if( iParEcal == 0 )
    {fEcal = new TEcnaParEcal(pObjectManager, SubDet.Data()); /*fCnew++*/}
  else
    {fEcal = (TEcnaParEcal*)iParEcal;}

  //............................ fEcalNumbering
  fEcalNumbering = 0;
  Long_t iEcalNumbering = pObjectManager->GetPointerValue("TEcnaNumbering");
  if( iEcalNumbering == 0 )
    {fEcalNumbering = new TEcnaNumbering(pObjectManager, SubDet.Data()); /*fCnew++*/}
  else
    {fEcalNumbering = (TEcnaNumbering*)iEcalNumbering;}

  //............................ fCnaParHistos
  fCnaParHistos = 0;
  Long_t iCnaParHistos = pObjectManager->GetPointerValue("TEcnaParHistos");
  if( iCnaParHistos == 0 )
    {fCnaParHistos = new TEcnaParHistos(pObjectManager, SubDet.Data()); /*fCnew++*/}
  else
    {fCnaParHistos = (TEcnaParHistos*)iCnaParHistos;}

  //............................ fCnaWrite
  fCnaWrite = 0;
  Long_t iCnaWrite = pObjectManager->GetPointerValue("TEcnaWrite");
  if( iCnaWrite == 0 )
    {fCnaWrite = new TEcnaWrite(pObjectManager, SubDet.Data()); /*fCnew++*/}
  else
    {fCnaWrite = (TEcnaWrite*)iCnaWrite;}

  //............................ fHistos
  //fHistos     = 0;
  //fHistos = new TEcnaHistos(fSubDet.Data(), fCnaParPaths, fCnaParCout,
  //		              fEcal, fCnaParHistos, fEcalNumbering, fCnaWrite);       fCnew++;

  fHistos = 0;
  Long_t iHistos = pObjectManager->GetPointerValue("TEcnaHistos");
  if( iHistos == 0 )
    {fHistos = new TEcnaHistos(pObjectManager, SubDet.Data()); /*fCnew++*/}
  else
    {fHistos = (TEcnaHistos*)iHistos;}

  //fMyRootFile = 0;

  Init();
}

void TEcnaGui::Init()
{
  //========================= GENERAL INITIALISATION 
  fCnew        = 0;
  fCdelete     = 0;
  fCnewRoot    = 0;
  fCdeleteRoot = 0;


  Int_t fgMaxCar = (Int_t)512;
  fTTBELL = '\007';

  //........................ init View and Cna parameters

  //............................................................................

  if( fSubDet == "EB" ){fStexName = "SM";  fStinName = "tower";}
  if( fSubDet == "EE" ){fStexName = "Dee"; fStinName = "SC";}

  //................. Init Keys
  InitKeys();

  //................ Init CNA Command and error numbering
  fCnaCommand = 0;
  fCnaError   = 0;
  //................ Init Confirm flags
  fConfirmSubmit    = 0;
  fConfirmRunNumber = 0;
  fConfirmCalcScc   = 0;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Init GUI DIALOG BOX pointers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  fLayoutGeneral      = 0;
  fLayoutBottLeft     = 0;
  fLayoutBottRight    = 0;
  fLayoutTopLeft      = 0;
  fLayoutTopRight     = 0;  
  fLayoutCenterYLeft  = 0;
  fLayoutCenterYRight = 0; 

  fVoidFrame = 0;

  //+++++++++++++++++++++++++++++++++ Horizontal frame Analysis + 1st requested evt number + Run number
  fAnaNorsRunFrame       = 0;
  fLayoutAnaNorsRunFrame = 0;
  //--------------------------------- Sub-Frame Analysis Name (Button+Entry Field)
  fAnaFrame       = 0;
  fAnaBut         = 0;
  fLayoutAnaBut   = 0;
  fAnaText        = 0;
  fEntryAnaNumber = 0;
  fLayoutAnaField = 0;
  //--------------------------------- SubFrame: first requested evt number
  fFevFrame            = 0;
  fFevBut              = 0;
  fLayoutFevBut        = 0;
  fFevText             = 0;
  fEntryFevNumber      = 0;
  fLayoutFevFieldText  = 0;
  fLayoutFevFieldFrame = 0;
  //-------------------------------- Sub-Frame Run number (Button+Entry Field)  
  fRunFrame       = 0;  
  fRunBut         = 0;
  fLayoutRunBut   = 0;
  fRunText        = 0;
  fEntryRunNumber = 0;
  fLayoutRunField = 0;

  //+++++++++++++++++++++++++ Horizontal frame Nb Of Samples + Last requested evt number + Clean + submit
  fFevLevStexFrame       = 0;
  fLayoutFevLevStexFrame = 0;
  //-------------------------------- Sub-Frame: Nb Of Requesred Samples (Button+Entry Field)  
  fNorsFrame       = 0;
  fNorsBut         = 0;
  fLayoutNorsBut   = 0;
  fNorsText        = 0;
  fEntryNorsNumber = 0;
  fLayoutNorsField = 0;
  //---------------------------- SubFrame: last requested event number
  fLevFrame            = 0;
  fLevBut              = 0;
  fLayoutLevBut        = 0;
  fLevText             = 0;
  fEntryLevNumber      = 0;
  fLayoutLevFieldText  = 0;
  fLayoutLevFieldFrame = 0;
  //................................ Menu for Clean
  fMenuClean    = 0;
  fMenuBarClean = 0;
  //................................ Menu for SUBMIT
  fMenuSubmit    = 0;
  fMenuBarSubmit = 0;

  //+++++++++++++++++++++++++++++++++++ Horizontal frame StexStin number +  Nb of Req evts
  fCompStRqFrame       = 0;
  fLayoutCompStRqFrame = 0;

  //---------------------------- SubFrame: Stex number
  fStexFrame            = 0;
  fStexBut              = 0;
  fLayoutStexBut        = 0;
  fStexText             = 0;
  fEntryStexNumber      = 0;
  fLayoutStexFieldText  = 0;
  fLayoutStexFieldFrame = 0;

  //---------------------------- SubFrame: number of requested events
  fRevFrame            = 0;
  fRevBut              = 0;
  fLayoutRevBut        = 0;
  fRevText             = 0;
  fEntryRevNumber      = 0;
  fLayoutRevFieldText  = 0;
  fLayoutRevFieldFrame = 0;

  //+++++++++++++++++++++++++++++++++++ Horizontal frame StexStin numbering +  Nb of samp for Calc + Calculations
  fCompStnbFrame       = 0;
  fLayoutCompStnbFrame = 0;

  //................................ Stex Stin Numbering view (Button)
  fButStexNb       = 0;
  fLayoutStexNbBut = 0;

  //---------------------------- SubFrame: NbSampForCalc
  fNbSampForCalcFrame            = 0;
  fNbSampForCalcBut              = 0;
  fLayoutNbSampForCalcBut        = 0;
  fNbSampForCalcText             = 0;
  fEntryNbSampForCalcNumber      = 0;
  fLayoutNbSampForCalcFieldText  = 0;
  fLayoutNbSampForCalcFieldFrame = 0;

  //................................ Menu for Calculations
  fMenuComput    = 0;
  fMenuBarComput = 0;

  //=====================================================================================

  //+++++++++++++++++++++++++++++++++++++++++++ Frame for quantities relative to the Stex
  fStexUpFrame       = 0; 

  //................................ Menus+Ymin+Ymax for the Stex ............................

  //...................................... Nb of evts in the data
  fVmmD_NOE_ChNbFrame       = 0;

  fVmaxD_NOE_ChNbFrame           = 0;
  fVmaxD_NOE_ChNbBut             = 0;
  fLayoutVmaxD_NOE_ChNbBut       = 0;
  fVmaxD_NOE_ChNbText            = 0;
  fEntryVmaxD_NOE_ChNbNumber     = 0;
  fLayoutVmaxD_NOE_ChNbFieldText = 0;
  fLayoutVmaxD_NOE_ChNbFrame     = 0;

  fVminD_NOE_ChNbFrame           = 0;
  fVminD_NOE_ChNbBut             = 0;
  fLayoutVminD_NOE_ChNbBut       = 0;
  fVminD_NOE_ChNbText            = 0;
  fEntryVminD_NOE_ChNbNumber     = 0;
  fLayoutVminD_NOE_ChNbFieldText = 0;
  fLayoutVminD_NOE_ChNbFrame     = 0;

  fMenuD_NOE_ChNb           = 0;
  fMenuBarD_NOE_ChNb        = 0;
  fLayoutMenuBarD_NOE_ChNb  = 0;

  fLayoutVmmD_NOE_ChNbFrame = 0;

  //-------------------------------------------------------------
  fVmmD_Ped_ChNbFrame       = 0;

  fVmaxD_Ped_ChNbFrame           = 0;
  fVmaxD_Ped_ChNbBut             = 0;
  fLayoutVmaxD_Ped_ChNbBut       = 0;
  fVmaxD_Ped_ChNbText            = 0;
  fEntryVmaxD_Ped_ChNbNumber     = 0;
  fLayoutVmaxD_Ped_ChNbFieldText = 0;
  fLayoutVmaxD_Ped_ChNbFrame     = 0;

  fVminD_Ped_ChNbFrame           = 0;
  fVminD_Ped_ChNbBut             = 0;
  fLayoutVminD_Ped_ChNbBut       = 0;
  fVminD_Ped_ChNbText            = 0;
  fEntryVminD_Ped_ChNbNumber     = 0;
  fLayoutVminD_Ped_ChNbFieldText = 0;
  fLayoutVminD_Ped_ChNbFrame     = 0;

  fMenuD_Ped_ChNb          = 0;
  fMenuBarD_Ped_ChNb       = 0;
  fLayoutMenuBarD_Ped_ChNb = 0;

  fLayoutVmmD_Ped_ChNbFrame = 0;

  //-------------------------------------------------------------
  fVmmD_TNo_ChNbFrame       = 0;

  fVmaxD_TNo_ChNbFrame           = 0;
  fVmaxD_TNo_ChNbBut             = 0;
  fLayoutVmaxD_TNo_ChNbBut       = 0;
  fVmaxD_TNo_ChNbText            = 0;
  fEntryVmaxD_TNo_ChNbNumber     = 0;
  fLayoutVmaxD_TNo_ChNbFieldText = 0;

  fVminD_TNo_ChNbFrame           = 0;
  fVminD_TNo_ChNbBut             = 0;
  fLayoutVminD_TNo_ChNbBut       = 0;
  fVminD_TNo_ChNbText            = 0;
  fEntryVminD_TNo_ChNbNumber     = 0;
  fLayoutVminD_TNo_ChNbFieldText = 0;
  fLayoutVminD_TNo_ChNbFrame     = 0;

  fMenuD_TNo_ChNb          = 0;
  fMenuBarD_TNo_ChNb       = 0;
  fLayoutMenuBarD_TNo_ChNb = 0;
  fLayoutVmaxD_TNo_ChNbFrame = 0;

  fLayoutVmmD_TNo_ChNbFrame = 0;

  //-------------------------------------------------------------
  fVmmD_MCs_ChNbFrame       = 0;

  fVmaxD_MCs_ChNbFrame           = 0;
  fVmaxD_MCs_ChNbBut             = 0;
  fLayoutVmaxD_MCs_ChNbBut       = 0;
  fVmaxD_MCs_ChNbText            = 0;
  fEntryVmaxD_MCs_ChNbNumber     = 0;
  fLayoutVmaxD_MCs_ChNbFieldText = 0;
  fLayoutVmaxD_MCs_ChNbFrame     = 0;

  fVminD_MCs_ChNbFrame           = 0;
  fVminD_MCs_ChNbBut             = 0;
  fLayoutVminD_MCs_ChNbBut       = 0;
  fVminD_MCs_ChNbText            = 0;
  fEntryVminD_MCs_ChNbNumber     = 0;
  fLayoutVminD_MCs_ChNbFieldText = 0;
  fLayoutVminD_MCs_ChNbFrame     = 0;

  fMenuD_MCs_ChNb          = 0;
  fMenuBarD_MCs_ChNb       = 0;
  fLayoutMenuBarD_MCs_ChNb = 0;
  fLayoutVmmD_MCs_ChNbFrame = 0;

  //............................................... Frame Sig + Menus Sig
  fStexHozFrame       = 0; 

  //-------------------------------------------------------------
  fVmmD_LFN_ChNbFrame       = 0;

  fVmaxD_LFN_ChNbFrame           = 0;
  fVmaxD_LFN_ChNbBut             = 0;
  fLayoutVmaxD_LFN_ChNbBut       = 0;
  fVmaxD_LFN_ChNbText            = 0;
  fEntryVmaxD_LFN_ChNbNumber     = 0;
  fLayoutVmaxD_LFN_ChNbFieldText = 0;
  fLayoutVmaxD_LFN_ChNbFrame     = 0;

  fVminD_LFN_ChNbFrame           = 0;
  fVminD_LFN_ChNbBut             = 0;
  fLayoutVminD_LFN_ChNbBut       = 0;
  fVminD_LFN_ChNbText            = 0;
  fEntryVminD_LFN_ChNbNumber     = 0;
  fLayoutVminD_LFN_ChNbFieldText = 0;
  fLayoutVminD_LFN_ChNbFrame     = 0;

  fMenuD_LFN_ChNb          = 0;
  fMenuBarD_LFN_ChNb       = 0;
  fLayoutMenuBarD_LFN_ChNb = 0;

  fLayoutVmmD_LFN_ChNbFrame = 0;

  //-------------------------------------------------------------
  fVmmD_HFN_ChNbFrame       = 0;

  fVmaxD_HFN_ChNbFrame           = 0;
  fVmaxD_HFN_ChNbBut             = 0;
  fLayoutVmaxD_HFN_ChNbBut       = 0;
  fVmaxD_HFN_ChNbText            = 0;
  fEntryVmaxD_HFN_ChNbNumber     = 0;
  fLayoutVmaxD_HFN_ChNbFieldText = 0;
  fLayoutVmaxD_HFN_ChNbFrame     = 0;

  fVminD_HFN_ChNbFrame           = 0;
  fVminD_HFN_ChNbBut             = 0;
  fLayoutVminD_HFN_ChNbBut       = 0;
  fVminD_HFN_ChNbText            = 0;
  fEntryVminD_HFN_ChNbNumber     = 0;
  fLayoutVminD_HFN_ChNbFieldText = 0;
  fLayoutVminD_HFN_ChNbFrame     = 0;

  fMenuD_HFN_ChNb          = 0; 
  fMenuBarD_HFN_ChNb       = 0;
  fLayoutMenuBarD_HFN_ChNb = 0;

  fLayoutVmmD_HFN_ChNbFrame = 0;

  //-------------------------------------------------------------
  fVmmD_SCs_ChNbFrame       = 0;

  fVmaxD_SCs_ChNbFrame           = 0;
  fVmaxD_SCs_ChNbBut             = 0;
  fLayoutVmaxD_SCs_ChNbBut       = 0;
  fVmaxD_SCs_ChNbText            = 0;
  fEntryVmaxD_SCs_ChNbNumber     = 0;
  fLayoutVmaxD_SCs_ChNbFieldText = 0;
  fLayoutVmaxD_SCs_ChNbFrame     = 0;

  fVminD_SCs_ChNbFrame           = 0;
  fVminD_SCs_ChNbBut             = 0;
  fLayoutVminD_SCs_ChNbBut       = 0;
  fVminD_SCs_ChNbText            = 0;
  fEntryVminD_SCs_ChNbNumber     = 0;
  fLayoutVminD_SCs_ChNbFieldText = 0;
  fLayoutVminD_SCs_ChNbFrame     = 0;

  fMenuD_SCs_ChNb          = 0;
  fMenuBarD_SCs_ChNb       = 0;
  fLayoutMenuBarD_SCs_ChNb = 0;

  fLayoutVmmD_SCs_ChNbFrame = 0;

  //----------------------------------------------------------------------------------

  //...................................... Low Freq Cor(c,c') for each pair of  Stins
  fVmmLFccMosFrame       = 0;

  fVmaxLFccMosFrame           = 0;
  fVmaxLFccMosBut             = 0;
  fLayoutVmaxLFccMosBut       = 0;
  fVmaxLFccMosText            = 0;
  fEntryVmaxLFccMosNumber     = 0;
  fLayoutVmaxLFccMosFieldText = 0;
  fLayoutVmaxLFccMosFrame     = 0;

  fVminLFccMosFrame           = 0;
  fVminLFccMosBut             = 0;
  fLayoutVminLFccMosBut       = 0;
  fVminLFccMosText            = 0;
  fEntryVminLFccMosNumber     = 0;
  fLayoutVminLFccMosFieldText = 0;
  fLayoutVminLFccMosFrame     = 0;

  fMenuLFccMos          = 0;
  fMenuBarLFccMos       = 0;
  fLayoutMenuBarLFccMos = 0;

  fLayoutVmmLFccMosFrame = 0;

  //...................................... High Freq Cor(c,c') for each pair of  Stins
  fVmmHFccMosFrame       = 0;

  fVmaxHFccMosFrame           = 0;
  fVmaxHFccMosBut             = 0;
  fLayoutVmaxHFccMosBut       = 0;
  fVmaxHFccMosText            = 0;
  fEntryVmaxHFccMosNumber     = 0;
  fLayoutVmaxHFccMosFieldText = 0;
  fLayoutVmaxHFccMosFrame     = 0;

  fVminHFccMosFrame           = 0;
  fVminHFccMosBut             = 0;
  fLayoutVminHFccMosBut       = 0;
  fVminHFccMosText            = 0;
  fEntryVminHFccMosNumber     = 0;
  fLayoutVminHFccMosFieldText = 0;
  fLayoutVminHFccMosFrame     = 0;

  fMenuHFccMos          = 0;
  fMenuBarHFccMos       = 0;
  fLayoutMenuBarHFccMos = 0;

  fLayoutVmmHFccMosFrame = 0;

  //...................................... LF and HF Cor(c,c')
  fVmmLHFccFrame            = 0;

  fVmaxLHFccFrame           = 0;
  fVmaxLHFccBut             = 0;
  fLayoutVmaxLHFccBut       = 0;
  fVmaxLHFccText            = 0;
  fEntryVmaxLHFccNumber     = 0;
  fLayoutVmaxLHFccFieldText = 0;
  fLayoutVmaxLHFccFrame     = 0;

  fVminLHFccFrame           = 0;
  fVminLHFccBut             = 0;
  fLayoutVminLHFccBut       = 0;
  fVminLHFccText            = 0;
  fEntryVminLHFccNumber     = 0;
  fLayoutVminLHFccFieldText = 0;
  fLayoutVminLHFccFrame     = 0;

  fMenuLHFcc          = 0;
  fMenuBarLHFcc       = 0;
  fLayoutMenuBarLHFcc = 0;

  fLayoutVmmLHFccFrame = 0;

  fLayoutStexHozFrame = 0;

  fLayoutStexUpFrame = 0;

  //+++++++++++++++++++++++++++++++++++++++++ Horizontal frame Stin_A + Stin_B
  fStinSpFrame      = 0;
  fLayoutStinSpFrame = 0;
  
  //----------------------------------- SubFrame Stin_A (Button + EntryField)

  fTxSubFrame       = 0; 
  fLayoutTxSubFrame = 0;

  fStinAFrame     = 0;
  fStinABut       = 0;
  fLayoutStinABut = 0; 

  fStinAText        = 0;
  fEntryStinANumber = 0; 
  fLayoutStinAField = 0;
  
  //............................ Stin Crystal Numbering view (Button)
  fButChNb       = 0;
  fLayoutChNbBut = 0;

  //............................ Menus Stin_A
  fMenuCorssAll    = 0;
  fMenuBarCorssAll = 0;

  //fMenuCovssAll    = 0;
  //fMenuBarCovssAll = 0;

  //----------------------------------- SubFrame Stin_B (Button + EntryField)
  fTySubFrame       = 0;
  fLayoutTySubFrame = 0;

  fStinBFrame     = 0;
  fStinBBut       = 0;
  fLayoutStinBBut = 0;

  fStinBText        = 0;  
  fEntryStinBNumber = 0;
  fLayoutStinBField = 0;

  //.................................. Menus for Horizontal frame (Stin_A + Stin_B)
  fMenuBarLFCorcc = 0;
  fMenuLFCorcc    = 0; 

  fMenuBarHFCorcc = 0;  
  fMenuHFCorcc    = 0;

  //++++++++++++++++++++++++ Horizontal frame channel number (Stin_A crystal number) + sample number
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

  //................................ Menus Stin_A crystal number
  fMenuCorss    = 0;
  fMenuBarCorss = 0;

  fMenuCovss    = 0; 
  fMenuBarCovss = 0;

  fMenuD_MSp_SpNb    = 0;
  fMenuBarD_MSp_SpNb = 0;
  fMenuD_MSp_SpDs    = 0;
  fMenuBarD_MSp_SpDs = 0;

  fMenuD_SSp_SpNb    = 0;
  fMenuBarD_SSp_SpNb = 0;
  fMenuD_SSp_SpDs    = 0;
  fMenuBarD_SSp_SpDs = 0;

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
  fMenuHistory    = 0;
  fMenuBarHistory = 0;

  //++++++++++++++++++++++++++++++++++++ Menu Event Distribution
  fMenuAdcProj          = 0;
  fMenuBarAdcProj       = 0;
  fLayoutMenuBarAdcProj = 0;

  //++++++++++++++++++++++++++++++++++++ LinLog + Color Palette Frame
  fLinLogFrame = 0;  

  //---------------------------------- Lin/Log X
  fButLogx         = 0;
  fLayoutLogxBut   = 0;
  //---------------------------------- Lin/Log Y
  fButLogy         = 0;
  fLayoutLogyBut   = 0;
  //---------------------------------- Proj Y
  fButProjy        = 0;
  fLayoutProjyBut  = 0;

  //++++++++++++++++++++++++++++++++++++ Frame: General Title (Gent) (Button + EntryField)
  fGentFrame            = 0;
  fGentBut              = 0;
  fLayoutGentBut        = 0;
  fGentText             = 0;
  fEntryGentNumber      = 0;
  fLayoutGentFieldText  = 0;
  fLayoutGentFieldFrame = 0;

  //++++++++++++++++++++++++++++++++++++ Color + Exit
  //---------------------------------- Color Palette
  fButColPal       = 0;
  fLayoutColPalBut = 0;
  //---------------------------------- Exit
  fButExit       = 0;   
  fLayoutExitBut = 0;

  //++++++++++++++++++++++++++++++++++++ Last Frame
  fLastFrame = 0;   

  //--------------------------------- Clone Last Canvas (Button)
  fButClone       = 0;
  fLayoutCloneBut = 0;

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

  //.................. Init codes Menu bars (all the numbers must be different)

  fMenuSubmit8nmC  = 2011;
  fMenuSubmit1nhC  = 2012;
  fMenuSubmit8nhC  = 2013;
  fMenuSubmit1ndC  = 2014;
  fMenuSubmit1nwC  = 2015;

  fMenuCleanSubC  = 3011;
  fMenuCleanJobC  = 3012;
  fMenuCleanPythC = 3013;
  fMenuCleanAllC  = 3014;

  fMenuComputStdC = 3111;
  fMenuComputSccC = 3112;
  fMenuComputSttC = 3113;

  fMenuD_NOE_ChNbFullC      = 600101;
  fMenuD_NOE_ChNbSameC      = 600102;
  fMenuD_NOE_ChNbAsciiFileC = 600104;

  fMenuD_Ped_ChNbFullC      = 123051;
  fMenuD_Ped_ChNbSameC      = 123052;
  fMenuD_Ped_ChNbAsciiFileC = 123054;

  fMenuD_TNo_ChNbFullC      = 123061;
  fMenuD_TNo_ChNbSameC      = 123062;
  fMenuD_TNo_ChNbSamePC     = 123063;
  fMenuD_TNo_ChNbAsciiFileC = 123064;

  fMenuD_MCs_ChNbFullC      = 123071;
  fMenuD_MCs_ChNbSameC      = 123072;
  fMenuD_MCs_ChNbSamePC     = 123073;
  fMenuD_MCs_ChNbAsciiFileC = 123074;

  fMenuD_LFN_ChNbFullC      = 800051;
  fMenuD_LFN_ChNbSameC      = 800052;
  fMenuD_LFN_ChNbSamePC     = 800053;
  fMenuD_LFN_ChNbAsciiFileC = 800054;

  fMenuD_HFN_ChNbFullC      = 800061;
  fMenuD_HFN_ChNbSameC      = 800062;
  fMenuD_HFN_ChNbSamePC     = 800063;
  fMenuD_HFN_ChNbAsciiFileC = 800064;

  fMenuD_SCs_ChNbFullC      = 800071;
  fMenuD_SCs_ChNbSameC      = 800072;
  fMenuD_SCs_ChNbSamePC     = 800073;
  fMenuD_SCs_ChNbAsciiFileC = 800074;

  fMenuLFccColzC = 70010;
  fMenuLFccLegoC = 70011;
  fMenuHFccColzC = 70020;
  fMenuHFccLegoC = 70021;

  fMenuLFccMosColzC = 70110;
  fMenuLFccMosLegoC = 70111;
  fMenuHFccMosColzC = 70120;
  fMenuHFccMosLegoC = 70121;

  fMenuD_NOE_ChNbHocoVecoC = 524051;
  fMenuD_Ped_ChNbHocoVecoC = 524052;
  fMenuD_TNo_ChNbHocoVecoC = 524053;
  fMenuD_MCs_ChNbHocoVecoC = 524054;
  fMenuD_LFN_ChNbHocoVecoC = 524055;
  fMenuD_HFN_ChNbHocoVecoC = 524056;
  fMenuD_SCs_ChNbHocoVecoC = 524057;

  fStinAButC = 90009; 
  fStinBButC = 90010;

  fChanButC = 6;
  fSampButC = 7;

  fMenuCorssAllColzC = 10;
  fMenuCovssAllColzC = 11;
 
  fMenuCorssColzC      = 221;
  fMenuCorssBoxC       = 222;
  fMenuCorssTextC      = 223;
  fMenuCorssContzC     = 224;
  fMenuCorssLegoC      = 225;
  fMenuCorssSurf1C     = 226;
  fMenuCorssSurf2C     = 227;
  fMenuCorssSurf3C     = 228;
  fMenuCorssSurf4C     = 229;
  fMenuCorssAsciiFileC = 220;

  fMenuCovssColzC      = 231;
  fMenuCovssBoxC       = 232;
  fMenuCovssTextC      = 233;
  fMenuCovssContzC     = 234;
  fMenuCovssLegoC      = 235;
  fMenuCovssSurf1C     = 236;
  fMenuCovssSurf2C     = 237;
  fMenuCovssSurf3C     = 238;
  fMenuCovssSurf4C     = 239;
  fMenuCovssAsciiFileC = 230;
 
  fMenuD_MSp_SpNbLineFullC    = 411;
  fMenuD_MSp_SpNbLineSameC    = 412;
  fMenuD_MSp_SpNbLineAllStinC = 413;
  fMenuD_MSp_SpDsLineFullC    = 414;
  fMenuD_MSp_SpDsLineSameC    = 415;
  fMenuD_MSp_SpDsLineAllStinC = 416;

  fMenuD_SSp_SpNbLineFullC    = 421;
  fMenuD_SSp_SpNbLineSameC    = 422;
  fMenuD_SSp_SpNbLineAllStinC = 423;
  fMenuD_SSp_SpDsLineFullC    = 424;
  fMenuD_SSp_SpDsLineSameC    = 425;
  fMenuD_SSp_SpDsLineAllStinC = 426;

  fMenuLFCorccColzC = 51;
  fMenuLFCorccLegoC = 52;

  fMenuHFCorccColzC = 61;
  fMenuHFCorccLegoC = 62;
  
  fMenuAdcProjSampLineFullC = 711;
  fMenuAdcProjSampLineSameC = 712;
  fMenuAdcProjLineLinyFullC = 713;
  fMenuAdcProjLineLinySameC = 714;

  fMenuH_Ped_DatePolmFullC = 811;
  fMenuH_Ped_DatePolmSameC = 812;

  fMenuH_TNo_DatePolmFullC  = 821;
  fMenuH_TNo_DatePolmSameC  = 822;
  fMenuH_TNo_DatePolmSamePC = 823;

  fMenuH_LFN_DatePolmFullC  = 824;
  fMenuH_LFN_DatePolmSameC  = 825;
  fMenuH_LFN_DatePolmSamePC = 826;

  fMenuH_HFN_DatePolmFullC  = 827;
  fMenuH_HFN_DatePolmSameC  = 828;
  fMenuH_HFN_DatePolmSamePC = 829;

  fMenuH_MCs_DatePolmFullC  = 831;
  fMenuH_MCs_DatePolmSameC  = 832;
  fMenuH_MCs_DatePolmSamePC = 833;

  fMenuH_SCs_DatePolmFullC  = 841;
  fMenuH_SCs_DatePolmSameC  = 842;
  fMenuH_SCs_DatePolmSamePC = 843;

  //...................... Init Button codes: Root version, Help, Exit
  fButStexNbC = 90;
  fButChNbC   = 91;
  fButCloneC  = 95;
  fButRootC   = 96;
  fButHelpC   = 97;
  fButExitC   = 98;

  //=================================== LIN/LOG + Y proj + Color palette flags
  Int_t MaxCar = fgMaxCar;
  fMemoScaleX.Resize(MaxCar);   
  fMemoScaleX = "LIN";

  MaxCar = fgMaxCar;
  fMemoScaleY.Resize(MaxCar); 
  fMemoScaleY = "LIN";

  MaxCar = fgMaxCar;
  fMemoProjY.Resize(MaxCar); 
  fMemoProjY = "normal";

  MaxCar = fgMaxCar;
  fMemoColPal.Resize(MaxCar); 
  fMemoColPal = "ECCNAColor";

  //=================================== Init option codes =================================

  MaxCar = fgMaxCar;
  fOptPlotFull.Resize(MaxCar);
  fOptPlotFull = "ONLYONE";

  MaxCar = fgMaxCar;
  fOptPlotSame.Resize(MaxCar);
  fOptPlotSame = "SAME";

  MaxCar = fgMaxCar;
  fOptPlotSameP.Resize(MaxCar);
  fOptPlotSameP = "SAME n";

  MaxCar = fgMaxCar;
  fOptPlotSameInStin.Resize(MaxCar);
  fOptPlotSameInStin = "SAME in Stin";

  MaxCar = fgMaxCar;
  fOptAscii.Resize(MaxCar);
  fOptAscii = "ASCII";

}  // end of Init()



//================================================================================================

//-------------------------------------------------------------------------
//
//
//                      B O X     M A K I N G
//
//
//-------------------------------------------------------------------------

void TEcnaGui::DialogBox()
{
  // Gui box making

  //fCnaP = (TGWindow *)p;
  //fCnaW = w;
  //fCnaH = h;

  //......................... Background colors

  //TColor* my_color = new TColor();
  //Color_t orange  = (Color_t)my_color->GetColor("#FF6611");  // orange

  Pixel_t SubDetColor = GetBackground();

  if( fSubDet == "EB" ){SubDetColor = GetBackground();}
  if( fSubDet == "EE" ){SubDetColor = GetBackground();}

  // Bool_t GlobFont = kFALSE;

  //Pixel_t BkgColMainWindow  = (Pixel_t)SubDetColor;
  //Pixel_t BkgColChSpFrame   = (Pixel_t)SubDetColor;
  //Pixel_t BkgColStexUpFrame = (Pixel_t)SubDetColor;

  //  Pixel_t BkgColButExit     = (Pixel_t)555888;

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

  fLayoutGeneral      = new TGLayoutHints (kLHintsCenterX | kLHintsCenterY);   fCnew++;
  fLayoutBottLeft     = new TGLayoutHints (kLHintsLeft    | kLHintsBottom);    fCnew++;
  fLayoutTopLeft      = new TGLayoutHints (kLHintsLeft    | kLHintsTop);       fCnew++;
  fLayoutBottRight    = new TGLayoutHints (kLHintsRight   | kLHintsBottom);    fCnew++;
  fLayoutTopRight     = new TGLayoutHints (kLHintsRight   | kLHintsTop);       fCnew++;
  fLayoutCenterYLeft  = new TGLayoutHints (kLHintsLeft    | kLHintsCenterY);   fCnew++;  
  fLayoutCenterYRight = new TGLayoutHints (kLHintsRight   | kLHintsCenterY);   fCnew++;  
  fLayoutCenterXTop   = new TGLayoutHints (kLHintsCenterX | kLHintsTop);       fCnew++;  

  fVoidFrame = new TGCompositeFrame(this,60,20, kVerticalFrame, kSunkenFrame); fCnew++;
  AddFrame(fVoidFrame, fLayoutGeneral);

  //......................... Pads border
  Int_t xB1 = 0;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  //
  //                 SECTOR 1:  Submit, File Parameters, Calculations, ...
  //
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  //
  //             Horizontal frame Analysis + First requested evt number + Run number
  //
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  fAnaNorsRunFrame = new TGCompositeFrame(this,60,20,kHorizontalFrame,
					  GetDefaultFrameBackground());   fCnew++;

  //=================================== ANALYSIS NAME (type of analysis)
  TString xAnaButText  = " Analysis ";
  Int_t typ_of_ana_buf_lenght = 80;
  fAnaFrame =  new TGCompositeFrame(fAnaNorsRunFrame,60,20, kHorizontalFrame,
				    kSunkenFrame);                    fCnew++;
  //..................... Button  
  fAnaBut = new TGTextButton(fAnaFrame, xAnaButText, fAnaButC);       fCnew++;
  fAnaBut->Connect("Clicked()","TEcnaGui", this, "DoButtonAna()");
  // fAnaBut->Resize(typ_of_ana_buf_lenght, fAnaBut->GetDefaultHeight());
  fAnaBut->SetToolTipText("Click here to register the analysis name written on the right");
  fAnaBut->SetBackgroundColor(SubDetColor);
  //fAnaBut->SetFont("courier", GlobFont);
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
  fAnaText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonAna()");
  fLayoutAnaField =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);     fCnew++;
  fAnaFrame->AddFrame(fAnaText, fLayoutAnaField);

  //=================================== FIRST REQUESTED EVENT NUMBER
  TString xFirstReqEvtNumberButText = " 1st event#  ";
  Int_t first_evt_buf_lenght   = 65;
  fFevFrame =
    new TGCompositeFrame(fAnaNorsRunFrame,60,20, kHorizontalFrame, kSunkenFrame); fCnew++;
  
  fFevBut= new TGTextButton(fFevFrame, xFirstReqEvtNumberButText);                fCnew++;
  fFevBut->Connect("Clicked()","TEcnaGui", this, "DoButtonFev()");
  fFevBut->SetToolTipText
    ("Click here to register the number of the first requested event number");
  fFevBut->SetBackgroundColor(SubDetColor);
  fLayoutFevBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);     fCnew++;
  fFevFrame->AddFrame(fFevBut,  fLayoutFevBut);

  fEntryFevNumber = new TGTextBuffer();                               fCnew++;
  fFevText = new TGTextEntry(fFevFrame, fEntryFevNumber);             fCnew++;
  fFevText->SetToolTipText("Click and enter the first requested event number");
  fFevText->Resize(first_evt_buf_lenght, fFevText->GetDefaultHeight());
  DisplayInEntryField(fFevText,fKeyFirstReqEvtNumber);
  fFevText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonFev()");
  fLayoutFevFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsLeft, xB1,xB1,xB1,xB1);  fCnew++;
  fFevFrame->AddFrame(fFevText, fLayoutFevFieldText);

  //=================================== RUN
  TString xRunButText  = " Run ";
  Int_t run_buf_lenght = 65;
  fRunFrame = new TGCompositeFrame(fAnaNorsRunFrame,0,0,
				   kHorizontalFrame, kSunkenFrame);   fCnew++;
  fRunBut = new TGTextButton(fRunFrame, xRunButText, fRunButC);       fCnew++;
  fRunBut->Connect("Clicked()","TEcnaGui", this, "DoButtonRun()");
  fRunBut->SetToolTipText("Click here to register the run number");
  fRunBut->SetBackgroundColor(SubDetColor);
  // fRunBut->SetFont("helvetica", GlobFont);
  fLayoutRunBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);  fCnew++;
  fRunFrame->AddFrame(fRunBut,  fLayoutRunBut);
  fEntryRunNumber = new TGTextBuffer();                               fCnew++;
  fRunText = new TGTextEntry(fRunFrame, fEntryRunNumber);             fCnew++;
  fRunText->SetToolTipText("Click and enter the run number");
  fRunText->Resize(run_buf_lenght, fRunText->GetDefaultHeight());
  DisplayInEntryField(fRunText,fKeyRunNumber);
  fRunText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonRun()");
  fLayoutRunField =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);    fCnew++;
  fRunFrame->AddFrame(fRunText, fLayoutRunField);

  //-------------------------- display frame ana + Fev + Run
  fAnaNorsRunFrame->AddFrame(fAnaFrame, fLayoutTopLeft);
  fAnaNorsRunFrame->AddFrame(fFevFrame, fLayoutTopLeft);
  fAnaNorsRunFrame->AddFrame(fRunFrame,fLayoutTopRight);
  fLayoutAnaNorsRunFrame =  new TGLayoutHints(kLHintsTop | kLHintsExpandX,
					      xB1, xB1, xB1, xB1);    fCnew++;

  AddFrame(fAnaNorsRunFrame, fLayoutAnaNorsRunFrame);

  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  //
  //    Horizontal frame Nb Of Samples + last requested evt number + Clean + Submit       
  //
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  fFevLevStexFrame = new TGCompositeFrame(this,60,20,kHorizontalFrame,
					  GetDefaultFrameBackground());            fCnew++;

  //=================================== Number Of Requested Samples
  TString xNorsButText  = "Nb Samp in File";
  Int_t nors_buf_lenght = 45;
  fNorsFrame = new TGCompositeFrame(fFevLevStexFrame,0,0, kHorizontalFrame,
				    kSunkenFrame);                                 fCnew++;  
  //..................... Button  
  fNorsBut = new TGTextButton(fNorsFrame, xNorsButText, fNorsButC);                fCnew++;
  fNorsBut->Connect("Clicked()","TEcnaGui", this, "DoButtonNors()");
  //fNorsBut->Resize(nors_buf_lenght, fNorsBut->GetDefaultHeight());
  fNorsBut->SetToolTipText("Click here to register the value written on the right");
  fNorsBut->SetBackgroundColor(SubDetColor);
  //fNorsBut->SetFont("courier", GlobFont);
  fLayoutNorsBut =
    new TGLayoutHints(kLHintsLeft | kLHintsTop, xB1,xB1,xB1,xB1);                 fCnew++;
  fNorsFrame->AddFrame(fNorsBut,  fLayoutNorsBut);
  //...................... Entry field
  fEntryNorsNumber = new TGTextBuffer();                                          fCnew++;
  fNorsText = new TGTextEntry(fNorsFrame, fEntryNorsNumber);                      fCnew++;
  fNorsText->SetToolTipText("Click and enter the number of required samples");
  fNorsText->Resize(nors_buf_lenght, fNorsText->GetDefaultHeight());
  DisplayInEntryField(fNorsText,fKeyNbOfSamples);
  fNorsText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonNors()");
  fLayoutNorsField =
    new TGLayoutHints(kLHintsTop | kLHintsCenterX, xB1,xB1,xB1,xB1);              fCnew++;
  fNorsFrame->AddFrame(fNorsText, fLayoutNorsField);

  //=================================== LAST REQUESTED EVENT NUMBER
  TString xLastReqEvtButText  = " Last event# ";
  Int_t last_evt_buf_lenght = 65;
  fLevFrame =
    new TGCompositeFrame(fFevLevStexFrame,60,20, kHorizontalFrame, kSunkenFrame); fCnew++;
  
  fLevBut = new TGTextButton(fLevFrame, xLastReqEvtButText);                      fCnew++;
  fLevBut->Connect("Clicked()","TEcnaGui", this, "DoButtonLev()");
  fLevBut->SetToolTipText("Click here to register the last requested event number");
  fLevBut->SetBackgroundColor(SubDetColor);
  fLayoutLevBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                 fCnew++;
  fLevFrame->AddFrame(fLevBut,  fLayoutLevBut);

  fEntryLevNumber = new TGTextBuffer();                                           fCnew++;
  fLevText = new TGTextEntry(fLevFrame, fEntryLevNumber);                         fCnew++;
  fLevText->SetToolTipText("Click and enter the last requested event number");
  fLevText->Resize(last_evt_buf_lenght, fLevText->GetDefaultHeight());
  DisplayInEntryField(fLevText,fKeyLastReqEvtNumber);
  fLevText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonLev()");
  fLayoutLevFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);             fCnew++;
  fLevFrame->AddFrame(fLevText, fLayoutLevFieldText);

  //----------------------------------- Clean
  TString xMenuBarClean  = "Clean  ";

  fMenuClean = new TGPopupMenu(gClient->GetRoot());                                fCnew++;
  fMenuClean->AddEntry("Submission scripts",fMenuCleanSubC);
  fMenuClean->AddEntry("LSFJOB reports",fMenuCleanJobC);
  fMenuClean->AddEntry("Python files",fMenuCleanPythC);
  fMenuClean->AddEntry("All",fMenuCleanAllC);

  fMenuClean->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarClean = new TGMenuBar(fFevLevStexFrame , 1, 1, kHorizontalFrame);        fCnew++;
  fMenuBarClean->AddPopup(xMenuBarClean, fMenuClean, fLayoutTopLeft);

  //--------------------------------- SUBMIT IN BATCH MODE
  TString xMenuBarSubmit  = " Submit ";
  fMenuSubmit = new TGPopupMenu(gClient->GetRoot());                               fCnew++;

  fMenuSubmit->AddEntry(" -q 8nm ",fMenuSubmit8nmC);
  fMenuSubmit->AddEntry(" -q 1nh ",fMenuSubmit1nhC);
  fMenuSubmit->AddEntry(" -q 8nh ",fMenuSubmit8nhC);
  fMenuSubmit->AddEntry(" -q 1nd ",fMenuSubmit1ndC);
  fMenuSubmit->AddEntry(" -q 1nw ",fMenuSubmit1nwC);

  fMenuSubmit->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarSubmit = new TGMenuBar(fFevLevStexFrame, 1, 1, kHorizontalFrame);        fCnew++;
  fMenuBarSubmit->AddPopup(xMenuBarSubmit, fMenuSubmit, fLayoutTopLeft);

  //-------------------------- display frame Nors + Lev + Clean + Submit
  fFevLevStexFrame->AddFrame(fNorsFrame,fLayoutTopLeft);
  fFevLevStexFrame->AddFrame(fLevFrame, fLayoutTopLeft);
  fFevLevStexFrame->AddFrame(fMenuBarSubmit, fLayoutTopRight);
  fFevLevStexFrame->AddFrame(fMenuBarClean, fLayoutTopRight);

  fLayoutFevLevStexFrame =  new TGLayoutHints(kLHintsTop | kLHintsExpandX,
					      xB1, xB1, xB1, xB1);                 fCnew++;

  AddFrame(fFevLevStexFrame, fLayoutFevLevStexFrame);


  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  //
  //      Horizontal Frame: StexNumber + Nb of Requested events
  //
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  fCompStRqFrame = new TGCompositeFrame(this,60,20,kHorizontalFrame,
					GetDefaultFrameBackground());              fCnew++;

  //----------------------------------- STEX NUMBER
  TString xSumoButText;
  if( fSubDet == "EB" ){xSumoButText = "  SM#   (0=EB)  ";}
  if( fSubDet == "EE" ){xSumoButText = " Dee#  (0=EE)  ";} 

  Int_t stex_number_buf_lenght = 36;
  fStexFrame =
    new TGCompositeFrame(fCompStRqFrame,60,20, kHorizontalFrame, kSunkenFrame);   fCnew++;

  fStexBut = new TGTextButton(fStexFrame, xSumoButText);                          fCnew++;
  fStexBut->Connect("Clicked()","TEcnaGui", this, "DoButtonStex()");
  fStexBut->SetToolTipText("Click here to register the number written on the right");
  fStexBut->SetBackgroundColor(SubDetColor);
  //fStexBut->SetFont("courier", GlobFont);
  fLayoutStexBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);  fCnew++;
  fStexFrame->AddFrame(fStexBut,  fLayoutStexBut);

  fEntryStexNumber = new TGTextBuffer();                                          fCnew++;
  fStexText = new TGTextEntry(fStexFrame, fEntryStexNumber);                      fCnew++;

  TString xStexNumber;
  if( fSubDet == "EB" ){xStexNumber = "Click and enter the SM number";}
  if( fSubDet == "EE" ){xStexNumber = "Click and enter the Dee number";}
  fStexText->SetToolTipText(xStexNumber);
  fStexText->Resize(stex_number_buf_lenght, fStexText->GetDefaultHeight());
  DisplayInEntryField(fStexText, fKeyStexNumber);
  fStexText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonStex()");
  
  fLayoutStexFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);             fCnew++;
  fStexFrame->AddFrame(fStexText, fLayoutStexFieldText);
  
  //=================================== NUMBER OF REQUESTED EVENTS
  TString xNbOfReqEvtButText  = " Nb of events ";
  Int_t nbof_evt_buf_lenght = 65;
  fRevFrame =
    new TGCompositeFrame(fCompStRqFrame,60,20, kHorizontalFrame, kSunkenFrame); fCnew++;
  
  fRevBut = new TGTextButton(fRevFrame, xNbOfReqEvtButText);                      fCnew++;
  fRevBut->Connect("Clicked()","TEcnaGui", this, "DoButtonRev()");
  fRevBut->SetToolTipText("Click here to register the requested number of events");
  fRevBut->SetBackgroundColor(SubDetColor);
  fLayoutRevBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                 fCnew++;
  fRevFrame->AddFrame(fRevBut,  fLayoutRevBut);

  fEntryRevNumber = new TGTextBuffer();                                           fCnew++;
  fRevText = new TGTextEntry(fRevFrame, fEntryRevNumber);                         fCnew++;
  fRevText->SetToolTipText("Click and enter the requested number of events");
  fRevText->Resize(nbof_evt_buf_lenght, fRevText->GetDefaultHeight());
  DisplayInEntryField(fRevText,fKeyReqNbOfEvts);
  fRevText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonRev()");
  fLayoutRevFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);             fCnew++;
  fRevFrame->AddFrame(fRevText, fLayoutRevFieldText);

  //-------------------------- display frame stex number + Nb of req evts
  fCompStRqFrame->AddFrame(fStexFrame,fLayoutTopLeft);
  fCompStRqFrame->AddFrame(fRevFrame,fLayoutTopLeft);

  fLayoutCompStRqFrame =  new TGLayoutHints(kLHintsTop | kLHintsExpandX,
					    xB1, xB1, xB1, xB1);                  fCnew++;
  AddFrame(fCompStRqFrame, fLayoutCompStRqFrame);
  AddFrame(fVoidFrame, fLayoutGeneral);


  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  //
  //      Horizontal Frame: StexStin numbering + NbSampForCalc + Calculations
  //
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  fCompStnbFrame = new TGCompositeFrame(this,60,20,kHorizontalFrame,
					GetDefaultFrameBackground());              fCnew++;


  // ---------------------------------STEX STIN NUMBERING VIEW BUTTON
  //............ Button texts and lenghts of the input widgets
  TString xStexNbButText;
  if( fSubDet == "EB" ){xStexNbButText = "SM Tower Numbering";}
  if( fSubDet == "EE" ){xStexNbButText = "Dee SC Numbering";}
  fButStexNb = new TGTextButton(fCompStnbFrame, xStexNbButText, fButStexNbC);     fCnew++;
  fButStexNb->Connect("Clicked()","TEcnaGui", this, "DoButtonStexNb()");
  fButStexNb->SetBackgroundColor(SubDetColor); 

  //----------------------------------- Nb Of Samples For Calculations
  TString xNbSampForCalcButText = "Nb Samp Calc";
  Int_t nb_of_samp_calc_buf_lenght = 28;
  fNbSampForCalcFrame =
    new TGCompositeFrame(fCompStnbFrame,60,20, kHorizontalFrame, kSunkenFrame);   fCnew++;

  fNbSampForCalcBut = new TGTextButton(fNbSampForCalcFrame, xNbSampForCalcButText);                          fCnew++;
  fNbSampForCalcBut->Connect("Clicked()","TEcnaGui", this, "DoButtonNbSampForCalc()");
  fNbSampForCalcBut->SetToolTipText("Click here to register the number written on the right");
  fNbSampForCalcBut->SetBackgroundColor(SubDetColor);
  //fNbSampForCalcBut->SetFont("courier", GlobFont);
  fLayoutNbSampForCalcBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);  fCnew++;
  fNbSampForCalcFrame->AddFrame(fNbSampForCalcBut,  fLayoutNbSampForCalcBut);

  fEntryNbSampForCalcNumber = new TGTextBuffer();                                          fCnew++;
  fNbSampForCalcText = new TGTextEntry(fNbSampForCalcFrame, fEntryNbSampForCalcNumber);                      fCnew++;

  TString xNbSampForCalcNumber = "Click and enter the nb of samples for calculations";
  fNbSampForCalcText->SetToolTipText(xNbSampForCalcNumber);
  fNbSampForCalcText->Resize(nb_of_samp_calc_buf_lenght, fNbSampForCalcText->GetDefaultHeight());
  DisplayInEntryField(fNbSampForCalcText, fKeyNbOfSampForCalc);
  fNbSampForCalcText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonNbSampForCalc()");
  
  fLayoutNbSampForCalcFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);             fCnew++;
  fNbSampForCalcFrame->AddFrame(fNbSampForCalcText, fLayoutNbSampForCalcFieldText);
  
  //--------------------------------- Calculations Menu
  TString xMenuBarComput  = "Calculations  ";
  fMenuComput = new TGPopupMenu(gClient->GetRoot());                              fCnew++;
  fMenuComput->AddEntry("Standard ( Pedestals, Noises, Cor(s,s') )",fMenuComputStdC);
  fMenuComput->AddEntry("Standard + |Cor(t,t')|  (long time)",fMenuComputSttC);
  fMenuComput->AddEntry("Standard + |Cor(t,t')| + |Cor(c,c')|  (long time + big file)",fMenuComputSccC);

  fMenuComput->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarComput = new TGMenuBar(fCompStnbFrame , 1, 1, kHorizontalFrame);        fCnew++;
  fMenuBarComput->AddPopup(xMenuBarComput, fMenuComput, fLayoutTopLeft);

  //-------------------------- display frame stexstin numbering + Nb samp for calc + Calculations
  fCompStnbFrame->AddFrame(fButStexNb,fLayoutTopLeft);
  fCompStnbFrame->AddFrame(fMenuBarComput,fLayoutTopRight);
  fCompStnbFrame->AddFrame(fNbSampForCalcFrame,fLayoutTopRight);

  fLayoutCompStnbFrame =  new TGLayoutHints(kLHintsTop | kLHintsExpandX,
					    xB1, xB1, xB1, xB1);                  fCnew++;
  AddFrame(fCompStnbFrame, fLayoutCompStnbFrame);
  AddFrame(fVoidFrame, fLayoutGeneral);

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  //
  //                             SECTOR 2: Stex's if SM # 0 or Stas's if SM =0 
  //
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Int_t minmax_buf_lenght = 45;

  fStexUpFrame = new TGCompositeFrame
    (this,60,20,kVerticalFrame, GetDefaultFrameBackground());                    fCnew++;
  TString xYminButText = " Ymin ";
  TString xYmaxButText = " Ymax ";
  //########################################### Composite frame number of events found in the data
  fVmmD_NOE_ChNbFrame = new TGCompositeFrame
    (fStexUpFrame,60,20, kHorizontalFrame, kSunkenFrame);                        fCnew++;
  //...................................... Menu number of events found in the data

  //...................................... Frame for Ymax
  fVmaxD_NOE_ChNbFrame = new TGCompositeFrame
    (fVmmD_NOE_ChNbFrame,60,20, kHorizontalFrame, kSunkenFrame);                 fCnew++;
  //...................................... Button Max + Entry field 
  fVmaxD_NOE_ChNbBut = new TGTextButton(fVmaxD_NOE_ChNbFrame, xYmaxButText);     fCnew++;
  fVmaxD_NOE_ChNbBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVmaxD_NOE_ChNb()");
  fVmaxD_NOE_ChNbBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fVmaxD_NOE_ChNbBut->SetBackgroundColor(SubDetColor); 
  fLayoutVmaxD_NOE_ChNbBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                fCnew++;
  fVmaxD_NOE_ChNbFrame->AddFrame(fVmaxD_NOE_ChNbBut,  fLayoutVmaxD_NOE_ChNbBut);
  fEntryVmaxD_NOE_ChNbNumber = new TGTextBuffer();                               fCnew++;
  fVmaxD_NOE_ChNbText =
    new TGTextEntry(fVmaxD_NOE_ChNbFrame, fEntryVmaxD_NOE_ChNbNumber);           fCnew++;
  fVmaxD_NOE_ChNbText->SetToolTipText("Click and enter ymax");
  fVmaxD_NOE_ChNbText->Resize(minmax_buf_lenght, fVmaxD_NOE_ChNbText->GetDefaultHeight());
  DisplayInEntryField(fVmaxD_NOE_ChNbText, fKeyVmaxD_NOE_ChNb);
  fVmaxD_NOE_ChNbText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVmaxD_NOE_ChNb()");

  fLayoutVmaxD_NOE_ChNbFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);            fCnew++;
  fVmaxD_NOE_ChNbFrame->AddFrame(fVmaxD_NOE_ChNbText, fLayoutVmaxD_NOE_ChNbFieldText);
  fLayoutVmaxD_NOE_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);               fCnew++;
  fVmmD_NOE_ChNbFrame->AddFrame(fVmaxD_NOE_ChNbFrame, fLayoutVmaxD_NOE_ChNbFrame);

  //...................................... Frame for Ymin
  fVminD_NOE_ChNbFrame = new TGCompositeFrame       
    (fVmmD_NOE_ChNbFrame,60,20, kHorizontalFrame, kSunkenFrame);                 fCnew++;
  //...................................... Button Min + Entry field 
  fVminD_NOE_ChNbBut = new TGTextButton(fVminD_NOE_ChNbFrame, xYminButText);     fCnew++;
  fVminD_NOE_ChNbBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVminD_NOE_ChNb()");
  fVminD_NOE_ChNbBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fVminD_NOE_ChNbBut->SetBackgroundColor(SubDetColor);
  fLayoutVminD_NOE_ChNbBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);               fCnew++;
  fVminD_NOE_ChNbFrame->AddFrame(fVminD_NOE_ChNbBut,  fLayoutVminD_NOE_ChNbBut);
  fEntryVminD_NOE_ChNbNumber = new TGTextBuffer();                              fCnew++;
  fVminD_NOE_ChNbText =
    new TGTextEntry(fVminD_NOE_ChNbFrame, fEntryVminD_NOE_ChNbNumber);          fCnew++;
  fVminD_NOE_ChNbText->SetToolTipText("Click and enter ymin");
  fVminD_NOE_ChNbText->Resize(minmax_buf_lenght, fVminD_NOE_ChNbText->GetDefaultHeight());
  DisplayInEntryField(fVminD_NOE_ChNbText,fKeyVminD_NOE_ChNb);
  fVminD_NOE_ChNbText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVminD_NOE_ChNb()");
  fLayoutVminD_NOE_ChNbFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);           fCnew++;
  fVminD_NOE_ChNbFrame->AddFrame(fVminD_NOE_ChNbText, fLayoutVminD_NOE_ChNbFieldText);
  fLayoutVminD_NOE_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);              fCnew++;
  fVmmD_NOE_ChNbFrame->AddFrame(fVminD_NOE_ChNbFrame, fLayoutVminD_NOE_ChNbFrame);

  //...................................... MenuBar strings
  TString xHistoChannels = "1D Histo";
  TString xHistoChannelsSame = "1D Histo SAME" ;
  TString xHistoChannelsSameP = "1D Histo SAME n";
  TString xHistoProjection = "1D Histo Projection";
  TString xHistoProjectionSame = "1D Histo Projection SAME";
  TString xHistoProjectionSameP = "1D Histo Projection SAME n";
  TString xHocoVecoViewSorS = "2D, Histo";
  if( fSubDet == "EB" ){xHocoVecoViewSorS = "2D, Histo (eta,phi)";}
  if( fSubDet == "EE" ){xHocoVecoViewSorS = "2D, Histo (IX,IY)";}
  TString xAsciiFileStex = "1D Histo, write in ASCII file";

  //...................................... Frame
  TString xMenuD_NOE_ChNb = "Numbers of events ";
  fMenuD_NOE_ChNb = new TGPopupMenu(gClient->GetRoot());                                   fCnew++;
  fMenuD_NOE_ChNb->AddEntry(xHistoChannels,fMenuD_NOE_ChNbFullC);
  fMenuD_NOE_ChNb->AddEntry(xHistoChannelsSame,fMenuD_NOE_ChNbSameC);
  fMenuD_NOE_ChNb->AddSeparator();
  fMenuD_NOE_ChNb->AddEntry(xHocoVecoViewSorS,fMenuD_NOE_ChNbHocoVecoC);
  fMenuD_NOE_ChNb->AddSeparator();
  fMenuD_NOE_ChNb->AddEntry(xAsciiFileStex,fMenuD_NOE_ChNbAsciiFileC);
  fMenuD_NOE_ChNb->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarD_NOE_ChNb = new TGMenuBar(fVmmD_NOE_ChNbFrame, 1, 1, kHorizontalFrame);         fCnew++;

  //fMenuBarD_NOE_ChNb->SetMinWidth(200);    // <= N'A STRICTEMENT AUCUN EFFET.

  fMenuBarD_NOE_ChNb->AddPopup(xMenuD_NOE_ChNb, fMenuD_NOE_ChNb, fLayoutGeneral);
  fLayoutMenuBarD_NOE_ChNb = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);             fCnew++;
  fVmmD_NOE_ChNbFrame->AddFrame(fMenuBarD_NOE_ChNb, fLayoutMenuBarD_NOE_ChNb);
  fLayoutVmmD_NOE_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                         fCnew++;
  fStexUpFrame->AddFrame(fVmmD_NOE_ChNbFrame, fLayoutVmmD_NOE_ChNbFrame);

  //............................. Expectation values + Sigmas Vertical frame
  fStexHozFrame =
    new TGCompositeFrame(fStexUpFrame,60,20,kVerticalFrame,
			 GetDefaultFrameBackground());                   fCnew++;

  //########################################### Composite frame ev of ev (pedestals)
  fVmmD_Ped_ChNbFrame = new TGCompositeFrame
    (fStexHozFrame,60,20, kHorizontalFrame, kSunkenFrame);          fCnew++;

  //...................................... Menu ev of ev

  //...................................... Frame for Ymax
  fVmaxD_Ped_ChNbFrame = new TGCompositeFrame
    (fVmmD_Ped_ChNbFrame,60,20, kHorizontalFrame, kSunkenFrame);                           fCnew++;
  //...................................... Button Max + Entry field 
  fVmaxD_Ped_ChNbBut = new TGTextButton(fVmaxD_Ped_ChNbFrame, xYmaxButText);               fCnew++;
  fVmaxD_Ped_ChNbBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVmaxD_Ped_ChNb()");
  fVmaxD_Ped_ChNbBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fLayoutVmaxD_Ped_ChNbBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1); fCnew++;
  fVmaxD_Ped_ChNbBut->SetBackgroundColor(SubDetColor);
  fVmaxD_Ped_ChNbFrame->AddFrame(fVmaxD_Ped_ChNbBut,  fLayoutVmaxD_Ped_ChNbBut);
  fEntryVmaxD_Ped_ChNbNumber = new TGTextBuffer();                                         fCnew++;
  fVmaxD_Ped_ChNbText = new TGTextEntry(fVmaxD_Ped_ChNbFrame, fEntryVmaxD_Ped_ChNbNumber); fCnew++;
  fVmaxD_Ped_ChNbText->SetToolTipText("Click and enter ymax");
  fVmaxD_Ped_ChNbText->Resize(minmax_buf_lenght, fVmaxD_Ped_ChNbText->GetDefaultHeight());
  DisplayInEntryField(fVmaxD_Ped_ChNbText,fKeyVmaxD_Ped_ChNb);
  fVmaxD_Ped_ChNbText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVmaxD_Ped_ChNb()");
  fLayoutVmaxD_Ped_ChNbFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                      fCnew++;
  fVmaxD_Ped_ChNbFrame->AddFrame(fVmaxD_Ped_ChNbText, fLayoutVmaxD_Ped_ChNbFieldText);
  fLayoutVmaxD_Ped_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                         fCnew++;
  fVmmD_Ped_ChNbFrame->AddFrame(fVmaxD_Ped_ChNbFrame, fLayoutVmaxD_Ped_ChNbFrame);

  //...................................... Frame for Ymin
  fVminD_Ped_ChNbFrame = new TGCompositeFrame
    (fVmmD_Ped_ChNbFrame,60,20, kHorizontalFrame, kSunkenFrame);                           fCnew++;
  //...................................... Button Min + Entry field 
  fVminD_Ped_ChNbBut = new TGTextButton(fVminD_Ped_ChNbFrame, xYminButText);               fCnew++;
  fVminD_Ped_ChNbBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVminD_Ped_ChNb()");
  fVminD_Ped_ChNbBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fVminD_Ped_ChNbBut->SetBackgroundColor(SubDetColor);
  fLayoutVminD_Ped_ChNbBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                          fCnew++;
  fVminD_Ped_ChNbFrame->AddFrame(fVminD_Ped_ChNbBut,  fLayoutVminD_Ped_ChNbBut);

  fEntryVminD_Ped_ChNbNumber = new TGTextBuffer();                                         fCnew++;
  fVminD_Ped_ChNbText = new TGTextEntry(fVminD_Ped_ChNbFrame, fEntryVminD_Ped_ChNbNumber); fCnew++;
  fVminD_Ped_ChNbText->SetToolTipText("Click and enter ymin");
  fVminD_Ped_ChNbText->Resize(minmax_buf_lenght, fVminD_Ped_ChNbText->GetDefaultHeight());
  DisplayInEntryField(fVminD_Ped_ChNbText,fKeyVminD_Ped_ChNb);
  fVminD_Ped_ChNbText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVminD_Ped_ChNb()");
  fLayoutVminD_Ped_ChNbFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                  fCnew++;
  fVminD_Ped_ChNbFrame->AddFrame(fVminD_Ped_ChNbText, fLayoutVminD_Ped_ChNbFieldText);
  fLayoutVminD_Ped_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                     fCnew++;
  fVmmD_Ped_ChNbFrame->AddFrame(fVminD_Ped_ChNbFrame, fLayoutVminD_Ped_ChNbFrame);

  //...................................... Frame
  TString xMenuD_Ped_ChNb = "       Pedestals ";
  fMenuD_Ped_ChNb = new TGPopupMenu(gClient->GetRoot());                               fCnew++;
  fMenuD_Ped_ChNb->AddEntry(xHistoChannels,fMenuD_Ped_ChNbFullC);
  fMenuD_Ped_ChNb->AddEntry(xHistoChannelsSame,fMenuD_Ped_ChNbSameC);
  fMenuD_Ped_ChNb->AddSeparator();
  fMenuD_Ped_ChNb->AddEntry(xHocoVecoViewSorS,fMenuD_Ped_ChNbHocoVecoC);
  fMenuD_Ped_ChNb->AddSeparator();
  fMenuD_Ped_ChNb->AddEntry(xAsciiFileStex,fMenuD_Ped_ChNbAsciiFileC);
  fMenuD_Ped_ChNb->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarD_Ped_ChNb = new TGMenuBar(fVmmD_Ped_ChNbFrame, 1, 1, kHorizontalFrame);     fCnew++;
  fMenuBarD_Ped_ChNb->AddPopup(xMenuD_Ped_ChNb, fMenuD_Ped_ChNb, fLayoutGeneral);
  fLayoutMenuBarD_Ped_ChNb = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);         fCnew++;
  fVmmD_Ped_ChNbFrame->AddFrame(fMenuBarD_Ped_ChNb, fLayoutMenuBarD_Ped_ChNb);  

  fLayoutVmmD_Ped_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                     fCnew++;
  fStexHozFrame->AddFrame(fVmmD_Ped_ChNbFrame, fLayoutVmmD_Ped_ChNbFrame);

  //########################################### Composite frame for TOTAL NOISE
  fVmmD_TNo_ChNbFrame = new TGCompositeFrame
    (fStexHozFrame,60,20, kHorizontalFrame, kSunkenFrame);                                  fCnew++;

  //...................................... Menu ev of sig 
  //...................................... Frame for Ymax
  fVmaxD_TNo_ChNbFrame = new TGCompositeFrame
    (fVmmD_TNo_ChNbFrame,60,20, kHorizontalFrame, kSunkenFrame);                            fCnew++;
  //...................................... Button Max + Entry field 
  fVmaxD_TNo_ChNbBut = new TGTextButton(fVmaxD_TNo_ChNbFrame, xYmaxButText);                fCnew++;
  fVmaxD_TNo_ChNbBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVmaxD_TNo_ChNb()");
  fVmaxD_TNo_ChNbBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fVmaxD_TNo_ChNbBut->SetBackgroundColor(SubDetColor);
  fLayoutVmaxD_TNo_ChNbBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                           fCnew++;
  fVmaxD_TNo_ChNbFrame->AddFrame(fVmaxD_TNo_ChNbBut,  fLayoutVmaxD_TNo_ChNbBut);
  fEntryVmaxD_TNo_ChNbNumber = new TGTextBuffer();                                          fCnew++;
  fVmaxD_TNo_ChNbText = new TGTextEntry(fVmaxD_TNo_ChNbFrame, fEntryVmaxD_TNo_ChNbNumber);  fCnew++;
  fVmaxD_TNo_ChNbText->SetToolTipText("Click and enter ymax");
  fVmaxD_TNo_ChNbText->Resize(minmax_buf_lenght, fVmaxD_TNo_ChNbText->GetDefaultHeight());
  DisplayInEntryField(fVmaxD_TNo_ChNbText,fKeyVmaxD_TNo_ChNb);
  fVmaxD_TNo_ChNbText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVmaxD_TNo_ChNb()");
  fLayoutVmaxD_TNo_ChNbFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                       fCnew++;
  fVmaxD_TNo_ChNbFrame->AddFrame(fVmaxD_TNo_ChNbText, fLayoutVmaxD_TNo_ChNbFieldText);
  fLayoutVmaxD_TNo_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                          fCnew++;
  fVmmD_TNo_ChNbFrame->AddFrame(fVmaxD_TNo_ChNbFrame, fLayoutVmaxD_TNo_ChNbFrame);

  //...................................... Frame for Ymin
  fVminD_TNo_ChNbFrame = new TGCompositeFrame
    (fVmmD_TNo_ChNbFrame,60,20, kHorizontalFrame, kSunkenFrame);                            fCnew++;
  //...................................... Button Min + Entry field 
  fVminD_TNo_ChNbBut = new TGTextButton(fVminD_TNo_ChNbFrame, xYminButText);                fCnew++;
  fVminD_TNo_ChNbBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVminD_TNo_ChNb()");
  fVminD_TNo_ChNbBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fVminD_TNo_ChNbBut->SetBackgroundColor(SubDetColor);
  fLayoutVminD_TNo_ChNbBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                           fCnew++;
  fVminD_TNo_ChNbFrame->AddFrame(fVminD_TNo_ChNbBut,  fLayoutVminD_TNo_ChNbBut);

  fEntryVminD_TNo_ChNbNumber = new TGTextBuffer();                                          fCnew++;
  fVminD_TNo_ChNbText = new TGTextEntry(fVminD_TNo_ChNbFrame, fEntryVminD_TNo_ChNbNumber);  fCnew++;
  fVminD_TNo_ChNbText->SetToolTipText("Click and enter ymin");
  fVminD_TNo_ChNbText->Resize(minmax_buf_lenght, fVminD_TNo_ChNbText->GetDefaultHeight());
  DisplayInEntryField(fVminD_TNo_ChNbText,fKeyVminD_TNo_ChNb);
  fVminD_TNo_ChNbText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVminD_TNo_ChNb()");
  fLayoutVminD_TNo_ChNbFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                 fCnew++;
  fVminD_TNo_ChNbFrame->AddFrame(fVminD_TNo_ChNbText, fLayoutVminD_TNo_ChNbFieldText);
  fLayoutVminD_TNo_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                    fCnew++;
  fVmmD_TNo_ChNbFrame->AddFrame(fVminD_TNo_ChNbFrame, fLayoutVminD_TNo_ChNbFrame);

  //...................................... Frame
  TString xMenuD_TNo_ChNb =  "        Total Noise ";
  fMenuD_TNo_ChNb = new TGPopupMenu(gClient->GetRoot());                              fCnew++;
  fMenuD_TNo_ChNb->AddEntry(xHistoChannels,fMenuD_TNo_ChNbFullC);
  fMenuD_TNo_ChNb->AddEntry(xHistoChannelsSame,fMenuD_TNo_ChNbSameC);
  fMenuD_TNo_ChNb->AddEntry(xHistoChannelsSameP,fMenuD_TNo_ChNbSamePC);
  fMenuD_TNo_ChNb->AddSeparator();
  fMenuD_TNo_ChNb->AddEntry(xHocoVecoViewSorS,fMenuD_TNo_ChNbHocoVecoC);
  fMenuD_TNo_ChNb->AddSeparator();
  fMenuD_TNo_ChNb->AddEntry(xAsciiFileStex,fMenuD_TNo_ChNbAsciiFileC);
  fMenuD_TNo_ChNb->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarD_TNo_ChNb = new TGMenuBar(fVmmD_TNo_ChNbFrame, 1, 1, kHorizontalFrame);    fCnew++;
  fMenuBarD_TNo_ChNb->AddPopup(xMenuD_TNo_ChNb, fMenuD_TNo_ChNb, fLayoutGeneral);
  fLayoutMenuBarD_TNo_ChNb = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);        fCnew++;
  fVmmD_TNo_ChNbFrame->AddFrame(fMenuBarD_TNo_ChNb, fLayoutMenuBarD_TNo_ChNb);

  fLayoutVmmD_TNo_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                     fCnew++;
  fStexHozFrame->AddFrame(fVmmD_TNo_ChNbFrame, fLayoutVmmD_TNo_ChNbFrame);

  //########################################### Composite frame for LOW FREQUENCY NOISE
  fVmmD_LFN_ChNbFrame = new TGCompositeFrame
    (fStexHozFrame,60,20, kHorizontalFrame, kSunkenFrame);                                     fCnew++;

  //...................................... Menu sig of ev (LOW FREQUENCY NOISE)
  //...................................... Frame for Ymax
  fVmaxD_LFN_ChNbFrame = new TGCompositeFrame
    (fVmmD_LFN_ChNbFrame,60,20, kHorizontalFrame, kSunkenFrame);                               fCnew++;
  //...................................... Button Max + Entry field 
  fVmaxD_LFN_ChNbBut = new TGTextButton(fVmaxD_LFN_ChNbFrame, xYmaxButText);                   fCnew++;
  fVmaxD_LFN_ChNbBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVmaxD_LFN_ChNb()");
  fVmaxD_LFN_ChNbBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fVmaxD_LFN_ChNbBut->SetBackgroundColor(SubDetColor);
  fLayoutVmaxD_LFN_ChNbBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                              fCnew++;
  fVmaxD_LFN_ChNbFrame->AddFrame(fVmaxD_LFN_ChNbBut,  fLayoutVmaxD_LFN_ChNbBut);
  fEntryVmaxD_LFN_ChNbNumber = new TGTextBuffer();                                             fCnew++;
  fVmaxD_LFN_ChNbText = new TGTextEntry(fVmaxD_LFN_ChNbFrame, fEntryVmaxD_LFN_ChNbNumber);     fCnew++;
  fVmaxD_LFN_ChNbText->SetToolTipText("Click and enter ymax");
  fVmaxD_LFN_ChNbText->Resize(minmax_buf_lenght, fVmaxD_LFN_ChNbText->GetDefaultHeight());
  DisplayInEntryField(fVmaxD_LFN_ChNbText,fKeyVmaxD_LFN_ChNb);
  fVmaxD_LFN_ChNbText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVmaxD_LFN_ChNb()");
  fLayoutVmaxD_LFN_ChNbFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                          fCnew++;
  fVmaxD_LFN_ChNbFrame->AddFrame(fVmaxD_LFN_ChNbText, fLayoutVmaxD_LFN_ChNbFieldText);
  fLayoutVmaxD_LFN_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                             fCnew++;
  fVmmD_LFN_ChNbFrame->AddFrame(fVmaxD_LFN_ChNbFrame, fLayoutVmaxD_LFN_ChNbFrame);

  //...................................... Frame for Ymin
  fVminD_LFN_ChNbFrame = new TGCompositeFrame
    (fVmmD_LFN_ChNbFrame,60,20, kHorizontalFrame, kSunkenFrame);                               fCnew++;
  //...................................... Button Min + Entry field 
  fVminD_LFN_ChNbBut = new TGTextButton(fVminD_LFN_ChNbFrame, xYminButText);                   fCnew++;
  fVminD_LFN_ChNbBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVminD_LFN_ChNb()");
  fVminD_LFN_ChNbBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fVminD_LFN_ChNbBut->SetBackgroundColor(SubDetColor);
  fLayoutVminD_LFN_ChNbBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                             fCnew++;
  fVminD_LFN_ChNbFrame->AddFrame(fVminD_LFN_ChNbBut,  fLayoutVminD_LFN_ChNbBut);
  fEntryVminD_LFN_ChNbNumber = new TGTextBuffer();                                            fCnew++;
  fVminD_LFN_ChNbText = new TGTextEntry(fVminD_LFN_ChNbFrame, fEntryVminD_LFN_ChNbNumber);    fCnew++;
  fVminD_LFN_ChNbText->SetToolTipText("Click and enter ymin");
  fVminD_LFN_ChNbText->Resize(minmax_buf_lenght, fVminD_LFN_ChNbText->GetDefaultHeight());
  DisplayInEntryField(fVminD_LFN_ChNbText,fKeyVminD_LFN_ChNb);
  fVminD_LFN_ChNbText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVminD_LFN_ChNb()");
  fLayoutVminD_LFN_ChNbFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                         fCnew++;
  fVminD_LFN_ChNbFrame->AddFrame(fVminD_LFN_ChNbText, fLayoutVminD_LFN_ChNbFieldText);
  fLayoutVminD_LFN_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                            fCnew++;
  fVmmD_LFN_ChNbFrame->AddFrame(fVminD_LFN_ChNbFrame, fLayoutVminD_LFN_ChNbFrame);

  //...................................... Frame
  TString xMenuD_LFN_ChNb =      " Low Frequency Noise ";
  fMenuD_LFN_ChNb = new TGPopupMenu(gClient->GetRoot());                                      fCnew++;
  fMenuD_LFN_ChNb->AddEntry(xHistoChannels,fMenuD_LFN_ChNbFullC);
  fMenuD_LFN_ChNb->AddEntry(xHistoChannelsSame,fMenuD_LFN_ChNbSameC);
  fMenuD_LFN_ChNb->AddEntry(xHistoChannelsSameP,fMenuD_LFN_ChNbSamePC);
  fMenuD_LFN_ChNb->AddSeparator();
  fMenuD_LFN_ChNb->AddEntry(xHocoVecoViewSorS,fMenuD_LFN_ChNbHocoVecoC);
  fMenuD_LFN_ChNb->AddSeparator();
  fMenuD_LFN_ChNb->AddEntry(xAsciiFileStex,fMenuD_LFN_ChNbAsciiFileC);
  fMenuD_LFN_ChNb->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarD_LFN_ChNb = new TGMenuBar(fVmmD_LFN_ChNbFrame, 1, 1, kHorizontalFrame);            fCnew++;
  fMenuBarD_LFN_ChNb->AddPopup(xMenuD_LFN_ChNb, fMenuD_LFN_ChNb, fLayoutGeneral);
  fLayoutMenuBarD_LFN_ChNb = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVmmD_LFN_ChNbFrame->AddFrame(fMenuBarD_LFN_ChNb, fLayoutMenuBarD_LFN_ChNb);
  fLayoutVmmD_LFN_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                            fCnew++;
  fStexHozFrame->AddFrame(fVmmD_LFN_ChNbFrame, fLayoutVmmD_LFN_ChNbFrame);

  //########################################### Composite frame for HIGH FREQUENCY NOISE
  fVmmD_HFN_ChNbFrame = new TGCompositeFrame
    (fStexHozFrame,60,20, kHorizontalFrame, kSunkenFrame);                                    fCnew++;

  //...................................... Menu sig of sig (HIGH FREQUENCY NOISE)
  //...................................... Frame for Ymax
  fVmaxD_HFN_ChNbFrame = new TGCompositeFrame
    (fVmmD_HFN_ChNbFrame,60,20, kHorizontalFrame, kSunkenFrame);                              fCnew++;
  //...................................... Button Max + Entry field
  fVmaxD_HFN_ChNbBut = new TGTextButton(fVmaxD_HFN_ChNbFrame, xYmaxButText);                  fCnew++; 
  fVmaxD_HFN_ChNbBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVmaxD_HFN_ChNb()");
  fVmaxD_HFN_ChNbBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fVmaxD_HFN_ChNbBut->SetBackgroundColor(SubDetColor);
  fLayoutVmaxD_HFN_ChNbBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                             fCnew++;
  fVmaxD_HFN_ChNbFrame->AddFrame(fVmaxD_HFN_ChNbBut,  fLayoutVmaxD_HFN_ChNbBut);
  fEntryVmaxD_HFN_ChNbNumber = new TGTextBuffer();                                            fCnew++;
  fVmaxD_HFN_ChNbText = new TGTextEntry(fVmaxD_HFN_ChNbFrame, fEntryVmaxD_HFN_ChNbNumber);    fCnew++;
  fVmaxD_HFN_ChNbText->SetToolTipText("Click and enter ymax");
  fVmaxD_HFN_ChNbText->Resize(minmax_buf_lenght, fVmaxD_HFN_ChNbText->GetDefaultHeight());
  DisplayInEntryField(fVmaxD_HFN_ChNbText,fKeyVmaxD_HFN_ChNb);
  fVmaxD_HFN_ChNbText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVmaxD_HFN_ChNb()");
  fLayoutVmaxD_HFN_ChNbFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                         fCnew++;
  fVmaxD_HFN_ChNbFrame->AddFrame(fVmaxD_HFN_ChNbText, fLayoutVmaxD_HFN_ChNbFieldText);
  fLayoutVmaxD_HFN_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                            fCnew++;
  fVmmD_HFN_ChNbFrame->AddFrame(fVmaxD_HFN_ChNbFrame, fLayoutVmaxD_HFN_ChNbFrame);

  //...................................... Frame for Ymin
  fVminD_HFN_ChNbFrame = new TGCompositeFrame
    (fVmmD_HFN_ChNbFrame,60,20, kHorizontalFrame, kSunkenFrame);                              fCnew++;
  //...................................... Button Min + Entry field
  fVminD_HFN_ChNbBut = new TGTextButton(fVminD_HFN_ChNbFrame, xYminButText);                  fCnew++;
  fVminD_HFN_ChNbBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVminD_HFN_ChNb()");
  fVminD_HFN_ChNbBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fVminD_HFN_ChNbBut->SetBackgroundColor(SubDetColor);
  fLayoutVminD_HFN_ChNbBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                          fCnew++;
  fVminD_HFN_ChNbFrame->AddFrame(fVminD_HFN_ChNbBut,  fLayoutVminD_HFN_ChNbBut);
  fEntryVminD_HFN_ChNbNumber = new TGTextBuffer();                                         fCnew++;
  fVminD_HFN_ChNbText = new TGTextEntry(fVminD_HFN_ChNbFrame, fEntryVminD_HFN_ChNbNumber); fCnew++;
  fVminD_HFN_ChNbText->SetToolTipText("Click and enter ymin");
  fVminD_HFN_ChNbText->Resize(minmax_buf_lenght, fVminD_HFN_ChNbText->GetDefaultHeight());
  DisplayInEntryField(fVminD_HFN_ChNbText,fKeyVminD_HFN_ChNb);
  fVminD_HFN_ChNbText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVminD_HFN_ChNb()");
  fLayoutVminD_HFN_ChNbFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                      fCnew++;
  fVminD_HFN_ChNbFrame->AddFrame(fVminD_HFN_ChNbText, fLayoutVminD_HFN_ChNbFieldText);
  fLayoutVminD_HFN_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                         fCnew++;
  fVmmD_HFN_ChNbFrame->AddFrame(fVminD_HFN_ChNbFrame, fLayoutVminD_HFN_ChNbFrame);

  //...................................... Frame
  TString xMenuD_HFN_ChNb =  " High Frequency Noise ";
  fMenuD_HFN_ChNb = new TGPopupMenu(gClient->GetRoot());                              fCnew++;
  fMenuD_HFN_ChNb->AddEntry(xHistoChannels,fMenuD_HFN_ChNbFullC);
  fMenuD_HFN_ChNb->AddEntry(xHistoChannelsSame,fMenuD_HFN_ChNbSameC);
  fMenuD_HFN_ChNb->AddEntry(xHistoChannelsSameP,fMenuD_HFN_ChNbSamePC);
  fMenuD_HFN_ChNb->AddSeparator();
  fMenuD_HFN_ChNb->AddEntry(xHocoVecoViewSorS,fMenuD_HFN_ChNbHocoVecoC);
  fMenuD_HFN_ChNb->AddSeparator();
  fMenuD_HFN_ChNb->AddEntry(xAsciiFileStex,fMenuD_HFN_ChNbAsciiFileC);
  fMenuD_HFN_ChNb->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarD_HFN_ChNb = new TGMenuBar(fVmmD_HFN_ChNbFrame, 1, 1, kHorizontalFrame);    fCnew++;
  fMenuBarD_HFN_ChNb->AddPopup(xMenuD_HFN_ChNb, fMenuD_HFN_ChNb, fLayoutGeneral);
  fLayoutMenuBarD_HFN_ChNb = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);        fCnew++;
  fVmmD_HFN_ChNbFrame->AddFrame(fMenuBarD_HFN_ChNb, fLayoutMenuBarD_HFN_ChNb);

  fLayoutVmmD_HFN_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                    fCnew++;
  fStexHozFrame->AddFrame(fVmmD_HFN_ChNbFrame, fLayoutVmmD_HFN_ChNbFrame);

  //########################################### Composite frame for MEAN COR(s,s')
  fVmmD_MCs_ChNbFrame = new TGCompositeFrame
    (fStexHozFrame,60,20, kHorizontalFrame, kSunkenFrame);                            fCnew++;

  //...................................... Menu ev of Corss

  //...................................... Frame
  fVmaxD_MCs_ChNbFrame = new TGCompositeFrame
    (fVmmD_MCs_ChNbFrame,60,20, kHorizontalFrame, kSunkenFrame);                              fCnew++;
  //...................................... Button Max + Entry field 
  fVmaxD_MCs_ChNbBut = new TGTextButton(fVmaxD_MCs_ChNbFrame, xYmaxButText);                  fCnew++;
  fVmaxD_MCs_ChNbBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVmaxD_MCs_ChNb()");
  fVmaxD_MCs_ChNbBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fVmaxD_MCs_ChNbBut->SetBackgroundColor(SubDetColor);
  fLayoutVmaxD_MCs_ChNbBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                             fCnew++;
  fVmaxD_MCs_ChNbFrame->AddFrame(fVmaxD_MCs_ChNbBut,  fLayoutVmaxD_MCs_ChNbBut);
  fEntryVmaxD_MCs_ChNbNumber = new TGTextBuffer();                                            fCnew++;
  fVmaxD_MCs_ChNbText = new TGTextEntry(fVmaxD_MCs_ChNbFrame, fEntryVmaxD_MCs_ChNbNumber);    fCnew++;
  fVmaxD_MCs_ChNbText->SetToolTipText("Click and enter ymax");
  fVmaxD_MCs_ChNbText->Resize(minmax_buf_lenght, fVmaxD_MCs_ChNbText->GetDefaultHeight());
  DisplayInEntryField(fVmaxD_MCs_ChNbText, fKeyVmaxD_MCs_ChNb);
  fVmaxD_MCs_ChNbText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVmaxD_MCs_ChNb()");
  fLayoutVmaxD_MCs_ChNbFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                         fCnew++;
  fVmaxD_MCs_ChNbFrame->AddFrame(fVmaxD_MCs_ChNbText, fLayoutVmaxD_MCs_ChNbFieldText);
  fLayoutVmaxD_MCs_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                            fCnew++;
  fVmmD_MCs_ChNbFrame->AddFrame(fVmaxD_MCs_ChNbFrame, fLayoutVmaxD_MCs_ChNbFrame);

  //...................................... Frame for Ymin
  fVminD_MCs_ChNbFrame = new TGCompositeFrame
    (fVmmD_MCs_ChNbFrame,60,20, kHorizontalFrame, kSunkenFrame);                              fCnew++;
  //...................................... Button Min + Entry field 
  fVminD_MCs_ChNbBut = new TGTextButton(fVminD_MCs_ChNbFrame, xYminButText);                  fCnew++;
  fVminD_MCs_ChNbBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVminD_MCs_ChNb()");
  fVminD_MCs_ChNbBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fVminD_MCs_ChNbBut->SetBackgroundColor(SubDetColor);
  fLayoutVminD_MCs_ChNbBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                             fCnew++;
  fVminD_MCs_ChNbFrame->AddFrame(fVminD_MCs_ChNbBut,  fLayoutVminD_MCs_ChNbBut);
  fEntryVminD_MCs_ChNbNumber = new TGTextBuffer();                                            fCnew++;
  fVminD_MCs_ChNbText = new TGTextEntry(fVminD_MCs_ChNbFrame, fEntryVminD_MCs_ChNbNumber);    fCnew++;
  fVminD_MCs_ChNbText->SetToolTipText("Click and enter ymin");
  fVminD_MCs_ChNbText->Resize(minmax_buf_lenght, fVminD_MCs_ChNbText->GetDefaultHeight());
  DisplayInEntryField(fVminD_MCs_ChNbText,fKeyVminD_MCs_ChNb);
  fVminD_MCs_ChNbText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVminD_MCs_ChNb()");
  fLayoutVminD_MCs_ChNbFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                         fCnew++;
  fVminD_MCs_ChNbFrame->AddFrame(fVminD_MCs_ChNbText, fLayoutVminD_MCs_ChNbFieldText);
  fLayoutVminD_MCs_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                            fCnew++;
  fVmmD_MCs_ChNbFrame->AddFrame(fVminD_MCs_ChNbFrame, fLayoutVminD_MCs_ChNbFrame);

  //...................................... Frame for Mean cor(s,s')
  TString xMenuD_MCs_ChNb = "     Mean cor(s,s') ";
  fMenuD_MCs_ChNb = new TGPopupMenu(gClient->GetRoot());                                   fCnew++;
  fMenuD_MCs_ChNb->AddEntry(xHistoChannels,fMenuD_MCs_ChNbFullC);
  fMenuD_MCs_ChNb->AddEntry(xHistoChannelsSame,fMenuD_MCs_ChNbSameC);
  fMenuD_MCs_ChNb->AddEntry(xHistoChannelsSameP,fMenuD_MCs_ChNbSamePC);
  fMenuD_MCs_ChNb->AddSeparator();
  fMenuD_MCs_ChNb->AddEntry(xHocoVecoViewSorS,fMenuD_MCs_ChNbHocoVecoC);
  fMenuD_MCs_ChNb->AddSeparator();
  fMenuD_MCs_ChNb->AddEntry(xAsciiFileStex,fMenuD_MCs_ChNbAsciiFileC);
  fMenuD_MCs_ChNb->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarD_MCs_ChNb = new TGMenuBar(fVmmD_MCs_ChNbFrame, 1, 1, kHorizontalFrame);         fCnew++;
  fMenuBarD_MCs_ChNb->AddPopup(xMenuD_MCs_ChNb, fMenuD_MCs_ChNb, fLayoutGeneral);
  fLayoutMenuBarD_MCs_ChNb = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);             fCnew++;
  fVmmD_MCs_ChNbFrame->AddFrame(fMenuBarD_MCs_ChNb, fLayoutMenuBarD_MCs_ChNb);

  fLayoutVmmD_MCs_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                         fCnew++;
  fStexHozFrame->AddFrame(fVmmD_MCs_ChNbFrame, fLayoutVmmD_MCs_ChNbFrame);

  //########################################### Composite frame for SIG OF COR(s,s')
  fVmmD_SCs_ChNbFrame = new TGCompositeFrame 
    (fStexHozFrame,60,20, kHorizontalFrame, kSunkenFrame);                                 fCnew++;

  //...................................... Menu sig of Corss
  //...................................... Frame for Ymax
  fVmaxD_SCs_ChNbFrame = new TGCompositeFrame
    (fVmmD_SCs_ChNbFrame,60,20, kHorizontalFrame, kSunkenFrame);                           fCnew++;
  //...................................... Button Max + Entry field
  fVmaxD_SCs_ChNbBut = new TGTextButton(fVmaxD_SCs_ChNbFrame, xYmaxButText);               fCnew++;
  fVmaxD_SCs_ChNbBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVmaxD_SCs_ChNb()");
  fVmaxD_SCs_ChNbBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fVmaxD_SCs_ChNbBut->SetBackgroundColor(SubDetColor);
  fLayoutVmaxD_SCs_ChNbBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                          fCnew++;
  fVmaxD_SCs_ChNbFrame->AddFrame(fVmaxD_SCs_ChNbBut,  fLayoutVmaxD_SCs_ChNbBut);
  fEntryVmaxD_SCs_ChNbNumber = new TGTextBuffer();                                         fCnew++;
  fVmaxD_SCs_ChNbText = new TGTextEntry(fVmaxD_SCs_ChNbFrame, fEntryVmaxD_SCs_ChNbNumber); fCnew++;
  fVmaxD_SCs_ChNbText->SetToolTipText("Click and enter ymax");
  fVmaxD_SCs_ChNbText->Resize(minmax_buf_lenght, fVmaxD_SCs_ChNbText->GetDefaultHeight());
  DisplayInEntryField(fVmaxD_SCs_ChNbText,fKeyVmaxD_SCs_ChNb);
  fVmaxD_SCs_ChNbText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVmaxD_SCs_ChNb()");
  fLayoutVmaxD_SCs_ChNbFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                      fCnew++;
  fVmaxD_SCs_ChNbFrame->AddFrame(fVmaxD_SCs_ChNbText, fLayoutVmaxD_SCs_ChNbFieldText);
  fLayoutVmaxD_SCs_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                         fCnew++;
  fVmmD_SCs_ChNbFrame->AddFrame(fVmaxD_SCs_ChNbFrame, fLayoutVmaxD_SCs_ChNbFrame);

  //...................................... Frame for Ymin
  fVminD_SCs_ChNbFrame = new TGCompositeFrame
    (fVmmD_SCs_ChNbFrame,60,20, kHorizontalFrame, kSunkenFrame);                           fCnew++;
  //...................................... Button Min + Entry field 
  fVminD_SCs_ChNbBut = new TGTextButton(fVminD_SCs_ChNbFrame, xYminButText);               fCnew++;
  fVminD_SCs_ChNbBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVminD_SCs_ChNb()");
  fVminD_SCs_ChNbBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fVminD_SCs_ChNbBut->SetBackgroundColor(SubDetColor);
  fLayoutVminD_SCs_ChNbBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                          fCnew++;
  fVminD_SCs_ChNbFrame->AddFrame(fVminD_SCs_ChNbBut,  fLayoutVminD_SCs_ChNbBut);
  fEntryVminD_SCs_ChNbNumber = new TGTextBuffer();                                         fCnew++;
  fVminD_SCs_ChNbText = new TGTextEntry(fVminD_SCs_ChNbFrame, fEntryVminD_SCs_ChNbNumber); fCnew++;
  fVminD_SCs_ChNbText->SetToolTipText("Click and enter ymin");
  fVminD_SCs_ChNbText->Resize(minmax_buf_lenght, fVminD_SCs_ChNbText->GetDefaultHeight());
  DisplayInEntryField(fVminD_SCs_ChNbText,fKeyVminD_SCs_ChNb);
  fVminD_SCs_ChNbText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVminD_SCs_ChNb()");
  fLayoutVminD_SCs_ChNbFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                 fCnew++;
  fVminD_SCs_ChNbFrame->AddFrame(fVminD_SCs_ChNbText, fLayoutVminD_SCs_ChNbFieldText);
  fLayoutVminD_SCs_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                    fCnew++;
  fVmmD_SCs_ChNbFrame->AddFrame(fVminD_SCs_ChNbFrame, fLayoutVminD_SCs_ChNbFrame);

  //...................................... Frame
  TString xMenuD_SCs_ChNb = "   Sigma of cor(s,s') ";
  fMenuD_SCs_ChNb = new TGPopupMenu(gClient->GetRoot());                              fCnew++;
  fMenuD_SCs_ChNb->AddEntry(xHistoChannels,fMenuD_SCs_ChNbFullC);
  fMenuD_SCs_ChNb->AddEntry(xHistoChannelsSame,fMenuD_SCs_ChNbSameC);
  fMenuD_SCs_ChNb->AddEntry(xHistoChannelsSameP,fMenuD_SCs_ChNbSamePC);
  fMenuD_SCs_ChNb->AddSeparator();
  fMenuD_SCs_ChNb->AddEntry(xHocoVecoViewSorS,fMenuD_SCs_ChNbHocoVecoC);
  fMenuD_SCs_ChNb->AddSeparator();
  fMenuD_SCs_ChNb->AddEntry(xAsciiFileStex,fMenuD_SCs_ChNbAsciiFileC);
  fMenuD_SCs_ChNb->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarD_SCs_ChNb = new TGMenuBar(fVmmD_SCs_ChNbFrame, 1, 1, kHorizontalFrame);    fCnew++;
  fMenuBarD_SCs_ChNb->AddPopup(xMenuD_SCs_ChNb, fMenuD_SCs_ChNb, fLayoutGeneral);
  fLayoutMenuBarD_SCs_ChNb = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);        fCnew++;
  fVmmD_SCs_ChNbFrame->AddFrame(fMenuBarD_SCs_ChNb, fLayoutMenuBarD_SCs_ChNb);

  fLayoutVmmD_SCs_ChNbFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                    fCnew++;
  fStexHozFrame->AddFrame(fVmmD_SCs_ChNbFrame, fLayoutVmmD_SCs_ChNbFrame);

  //######################################################################################################"

  //------------------------------------------- subframe
  fLayoutStexHozFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1, xB1, xB1, xB1);                fCnew++;
  fStexUpFrame->AddFrame(fStexHozFrame, fLayoutStexHozFrame);
  AddFrame(fVoidFrame, fLayoutGeneral);

  //########################################### Composite frame corcc in Stins
  fVmmLHFccFrame = new TGCompositeFrame
    (fStexUpFrame,60,20, kHorizontalFrame, kSunkenFrame);                            fCnew++;
 
  //............ Menu Low and High Frequency correlations between channels for each Stin of Stex
  //...................................... Frame for Ymax
  fVmaxLHFccFrame = new TGCompositeFrame
    (fVmmLHFccFrame,60,20, kHorizontalFrame, kSunkenFrame);                          fCnew++;
  //...................................... Button Max + Entry field 
  fVmaxLHFccBut = new TGTextButton(fVmaxLHFccFrame, xYmaxButText);                   fCnew++;
  fVmaxLHFccBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVmaxLHFcc()");
  fVmaxLHFccBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fVmaxLHFccBut->SetBackgroundColor(SubDetColor);
  fLayoutVmaxLHFccBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVmaxLHFccFrame->AddFrame(fVmaxLHFccBut,  fLayoutVmaxLHFccBut);
  fEntryVmaxLHFccNumber = new TGTextBuffer();                                        fCnew++;
  fVmaxLHFccText = new TGTextEntry(fVmaxLHFccFrame, fEntryVmaxLHFccNumber);          fCnew++;
  fVmaxLHFccText->SetToolTipText("Click and enter ymax");
  fVmaxLHFccText->Resize(minmax_buf_lenght, fVmaxLHFccText->GetDefaultHeight());
  DisplayInEntryField(fVmaxLHFccText, fKeyVmaxLHFcc);
  fVmaxLHFccText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVmaxLHFcc()");

  fLayoutVmaxLHFccFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVmaxLHFccFrame->AddFrame(fVmaxLHFccText, fLayoutVmaxLHFccFieldText);
  fLayoutVmaxLHFccFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmLHFccFrame->AddFrame(fVmaxLHFccFrame, fLayoutVmaxLHFccFrame);

  //...................................... Frame for Ymin
  fVminLHFccFrame = new TGCompositeFrame
    (fVmmLHFccFrame,60,20, kHorizontalFrame, kSunkenFrame);                          fCnew++;
  //...................................... Button Min + Entry field
  fVminLHFccBut = new TGTextButton(fVminLHFccFrame, xYminButText);                   fCnew++;
  fVminLHFccBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVminLHFcc()");
  fVminLHFccBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fVminLHFccBut->SetBackgroundColor(SubDetColor);
  fLayoutVminLHFccBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVminLHFccFrame->AddFrame(fVminLHFccBut,  fLayoutVminLHFccBut);
  fEntryVminLHFccNumber = new TGTextBuffer();                                        fCnew++;
  fVminLHFccText = new TGTextEntry(fVminLHFccFrame, fEntryVminLHFccNumber);          fCnew++;
  fVminLHFccText->SetToolTipText("Click and enter ymin");
  fVminLHFccText->Resize(minmax_buf_lenght, fVminLHFccText->GetDefaultHeight());
  DisplayInEntryField(fVminLHFccText,fKeyVminLHFcc);
  fVminLHFccText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVminLHFcc()");
  fLayoutVminLHFccFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVminLHFccFrame->AddFrame(fVminLHFccText, fLayoutVminLHFccFieldText);
  fLayoutVminLHFccFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmLHFccFrame->AddFrame(fVminLHFccFrame, fLayoutVminLHFccFrame);

  //........................................... Frame
  TString xMenuLHFcc = "GeoView LF,HF Cor(c,c') (expert)";
  TString xLFccViewSorS;
  if( fSubDet == "EB" ){xLFccViewSorS = "Low Frequency Cor(c,c'), tower place -> Cor matrix";}
  if( fSubDet == "EE" ){xLFccViewSorS = "Low Frequency Cor(c,c'), SC place -> Cor matrix";}
  TString xHFccViewSorS;
  if( fSubDet == "EB" ){xHFccViewSorS = "High Frequency Cor(c,c'), tower place -> Cor matrix";}
  if( fSubDet == "EE" ){xHFccViewSorS = "High Frequency Cor(c,c'), SC place -> Cor matrix";}

  fMenuLHFcc = new TGPopupMenu(gClient->GetRoot());                                  fCnew++;
  fMenuLHFcc->AddEntry(xLFccViewSorS,fMenuLFccColzC);
  fMenuLHFcc->AddEntry(xHFccViewSorS,fMenuHFccColzC);
  fMenuLHFcc->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarLHFcc = new TGMenuBar(fVmmLHFccFrame, 1, 1, kHorizontalFrame);             fCnew++;
  fMenuBarLHFcc->AddPopup(xMenuLHFcc, fMenuLHFcc, fLayoutGeneral);
  fLayoutMenuBarLHFcc = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);            fCnew++;
  fVmmLHFccFrame->AddFrame(fMenuBarLHFcc, fLayoutMenuBarLHFcc);
  fLayoutVmmLHFccFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fStexUpFrame->AddFrame(fVmmLHFccFrame, fLayoutVmmLHFccFrame);

  //################################# Composite frame Low Freq Cor(c,c') for each pair of Stins
  fVmmLFccMosFrame = new TGCompositeFrame
    (fStexUpFrame,60,20, kHorizontalFrame, kSunkenFrame);                            fCnew++;
 
  //...................................... Menu correlations between Stins 
  //...................................... Frame
  fVmaxLFccMosFrame = new TGCompositeFrame
    (fVmmLFccMosFrame,60,20, kHorizontalFrame, kSunkenFrame);                        fCnew++;
  //...................................... Button Max + Entry field 
  fVmaxLFccMosBut = new TGTextButton(fVmaxLFccMosFrame, xYmaxButText);               fCnew++;
  fVmaxLFccMosBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVmaxLFccMos()");
  fVmaxLFccMosBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fVmaxLFccMosBut->SetBackgroundColor(SubDetColor);
  fLayoutVmaxLFccMosBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVmaxLFccMosFrame->AddFrame(fVmaxLFccMosBut,  fLayoutVmaxLFccMosBut);
  fEntryVmaxLFccMosNumber = new TGTextBuffer();                                      fCnew++;
  fVmaxLFccMosText = new TGTextEntry(fVmaxLFccMosFrame, fEntryVmaxLFccMosNumber);    fCnew++;
  fVmaxLFccMosText->SetToolTipText("Click and enter ymax");
  fVmaxLFccMosText->Resize(minmax_buf_lenght, fVmaxLFccMosText->GetDefaultHeight());
  DisplayInEntryField(fVmaxLFccMosText, fKeyVmaxLFccMos);
  fVmaxLFccMosText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVmaxLFccMos()");

  fLayoutVmaxLFccMosFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVmaxLFccMosFrame->AddFrame(fVmaxLFccMosText, fLayoutVmaxLFccMosFieldText);
  fLayoutVmaxLFccMosFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmLFccMosFrame->AddFrame(fVmaxLFccMosFrame, fLayoutVmaxLFccMosFrame);

  //...................................... Frame for Ymin
  fVminLFccMosFrame = new TGCompositeFrame
    (fVmmLFccMosFrame,60,20, kHorizontalFrame, kSunkenFrame);                        fCnew++;
  //...................................... Button Min + Entry field 
  fVminLFccMosBut = new TGTextButton(fVminLFccMosFrame, xYminButText);               fCnew++;
  fVminLFccMosBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVminLFccMos()");
  fVminLFccMosBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fVminLFccMosBut->SetBackgroundColor(SubDetColor);
  fLayoutVminLFccMosBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVminLFccMosFrame->AddFrame(fVminLFccMosBut,  fLayoutVminLFccMosBut);
  fEntryVminLFccMosNumber = new TGTextBuffer();                                      fCnew++;
  fVminLFccMosText = new TGTextEntry(fVminLFccMosFrame, fEntryVminLFccMosNumber);    fCnew++;
  fVminLFccMosText->SetToolTipText("Click and enter ymin");
  fVminLFccMosText->Resize(minmax_buf_lenght, fVminLFccMosText->GetDefaultHeight());
  DisplayInEntryField(fVminLFccMosText,fKeyVminLFccMos);
  fVminLFccMosText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVminLFccMos()");
  fLayoutVminLFccMosFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVminLFccMosFrame->AddFrame(fVminLFccMosText, fLayoutVminLFccMosFieldText);
  fLayoutVminLFccMosFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmLFccMosFrame->AddFrame(fVminLFccMosFrame, fLayoutVminLFccMosFrame);

  //...................................... Frame
  TString xMenuLFccMos;
  if( fSubDet == "EB" ){xMenuLFccMos = "Mean LF |Cor(c,c')| in (tow,tow')";}
  if( fSubDet == "EE" ){xMenuLFccMos = "Mean LF |Cor(c,c')| in (SC,SC')";}

  fMenuLFccMos = new TGPopupMenu(gClient->GetRoot());                                fCnew++;
  fMenuLFccMos->AddEntry("2D, COLZ ",fMenuLFccMosColzC);
  fMenuLFccMos->AddEntry("3D, LEGO2Z" ,fMenuLFccMosLegoC);
  fMenuLFccMos->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarLFccMos = new TGMenuBar(fVmmLFccMosFrame, 1, 1, kHorizontalFrame);         fCnew++;
  fMenuBarLFccMos->AddPopup(xMenuLFccMos, fMenuLFccMos, fLayoutGeneral);
  fLayoutMenuBarLFccMos = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);          fCnew++;
  fVmmLFccMosFrame->AddFrame(fMenuBarLFccMos, fLayoutMenuBarLFccMos);
  fLayoutVmmLFccMosFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fStexUpFrame->AddFrame(fVmmLFccMosFrame, fLayoutVmmLFccMosFrame);
 
  //################################# Composite frame High Freq Cor(c,c') for each pair of Stins
  fVmmHFccMosFrame = new TGCompositeFrame
    (fStexUpFrame,60,20, kHorizontalFrame, kSunkenFrame);                            fCnew++;
 
  //...................................... Menu correlations between Stins 
  //...................................... Frame
  fVmaxHFccMosFrame = new TGCompositeFrame
    (fVmmHFccMosFrame,60,20, kHorizontalFrame, kSunkenFrame);                        fCnew++;
  //...................................... Button Max + Entry field 
  fVmaxHFccMosBut = new TGTextButton(fVmaxHFccMosFrame, xYmaxButText);               fCnew++;
  fVmaxHFccMosBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVmaxHFccMos()");
  fVmaxHFccMosBut->SetToolTipText("Click here to register ymax for the display of the quantity");
  fVmaxHFccMosBut->SetBackgroundColor(SubDetColor);
  fLayoutVmaxHFccMosBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVmaxHFccMosFrame->AddFrame(fVmaxHFccMosBut,  fLayoutVmaxHFccMosBut);
  fEntryVmaxHFccMosNumber = new TGTextBuffer();                                      fCnew++;
  fVmaxHFccMosText = new TGTextEntry(fVmaxHFccMosFrame, fEntryVmaxHFccMosNumber);    fCnew++;
  fVmaxHFccMosText->SetToolTipText("Click and enter ymax");
  fVmaxHFccMosText->Resize(minmax_buf_lenght, fVmaxHFccMosText->GetDefaultHeight());
  DisplayInEntryField(fVmaxHFccMosText, fKeyVmaxHFccMos);
  fVmaxHFccMosText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVmaxHFccMos()");

  fLayoutVmaxHFccMosFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVmaxHFccMosFrame->AddFrame(fVmaxHFccMosText, fLayoutVmaxHFccMosFieldText);
  fLayoutVmaxHFccMosFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmHFccMosFrame->AddFrame(fVmaxHFccMosFrame, fLayoutVmaxHFccMosFrame);

  //...................................... Frame for Ymin
  fVminHFccMosFrame = new TGCompositeFrame
    (fVmmHFccMosFrame,60,20, kHorizontalFrame, kSunkenFrame);                        fCnew++;
  //...................................... Button Min + Entry field 
  fVminHFccMosBut = new TGTextButton(fVminHFccMosFrame, xYminButText);               fCnew++;
  fVminHFccMosBut->Connect("Clicked()","TEcnaGui", this, "DoButtonVminHFccMos()");
  fVminHFccMosBut->SetToolTipText("Click here to register ymin for the display of the quantity");
  fVminHFccMosBut->SetBackgroundColor(SubDetColor);
  fLayoutVminHFccMosBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);                    fCnew++;
  fVminHFccMosFrame->AddFrame(fVminHFccMosBut,  fLayoutVminHFccMosBut);
  fEntryVminHFccMosNumber = new TGTextBuffer();                                      fCnew++;
  fVminHFccMosText = new TGTextEntry(fVminHFccMosFrame, fEntryVminHFccMosNumber);    fCnew++;
  fVminHFccMosText->SetToolTipText("Click and enter ymin");
  fVminHFccMosText->Resize(minmax_buf_lenght, fVminHFccMosText->GetDefaultHeight());
  DisplayInEntryField(fVminHFccMosText,fKeyVminHFccMos);
  fVminHFccMosText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonVminHFccMos()");
  fLayoutVminHFccMosFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsRight, xB1,xB1,xB1,xB1);                fCnew++;
  fVminHFccMosFrame->AddFrame(fVminHFccMosText, fLayoutVminHFccMosFieldText);
  fLayoutVminHFccMosFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fVmmHFccMosFrame->AddFrame(fVminHFccMosFrame, fLayoutVminHFccMosFrame);

  //...................................... Frame
  TString xMenuHFccMos;
  if( fSubDet == "EB" ){xMenuHFccMos = "Mean HF |Cor(c,c')| in (tow,tow')";}
  if( fSubDet == "EE" ){xMenuHFccMos = "Mean HF |Cor(c,c')| in (SC,SC')";}

  fMenuHFccMos = new TGPopupMenu(gClient->GetRoot());                                fCnew++;
  fMenuHFccMos->AddEntry("2D, COLZ ",fMenuHFccMosColzC);
  fMenuHFccMos->AddEntry("3D, LEGO2Z" ,fMenuHFccMosLegoC);
  fMenuHFccMos->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarHFccMos = new TGMenuBar(fVmmHFccMosFrame, 1, 1, kHorizontalFrame);         fCnew++;
  fMenuBarHFccMos->AddPopup(xMenuHFccMos, fMenuHFccMos, fLayoutGeneral);
  fLayoutMenuBarHFccMos = new TGLayoutHints(kLHintsRight, xB1,xB1,xB1,xB1);          fCnew++;
  fVmmHFccMosFrame->AddFrame(fMenuBarHFccMos, fLayoutMenuBarHFccMos);
  fLayoutVmmHFccMosFrame =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1);                   fCnew++;
  fStexUpFrame->AddFrame(fVmmHFccMosFrame, fLayoutVmmHFccMosFrame);
 

  //======================================= "Stex" frame =====================================
  fLayoutStexUpFrame =
    new TGLayoutHints(kLHintsTop | kLHintsCenterX, xB1, xB1, xB1, xB1);              fCnew++;
  AddFrame(fStexUpFrame, fLayoutStexUpFrame);

  AddFrame(fVoidFrame, fLayoutGeneral);

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  //
  //                             SECTOR 3: Stin's
  //
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  //-------------------------------- Stin A & Stin B FRAME
  fStinSpFrame =
    new TGCompositeFrame(this,60,20,kHorizontalFrame,
			 GetDefaultFrameBackground());                               fCnew++;

  TString xStinAButText = "?";
  TString xStinBButText = "?";
  Int_t Stin_buf_lenght = 10;

  if ( fSubDet == "EB" )
    {xStinAButText = "      Tower# [1,68]     "; xStinBButText  = "     Tower'# [1,68]      "; Stin_buf_lenght =  50;}
  if ( fSubDet == "EE" && ( fKeyStexNumber == 1 || fKeyStexNumber == 3 ) )
    {xStinAButText = "SC# for const. [150,298] "; xStinBButText  = "SC'# for const. [150,298] "; Stin_buf_lenght =  50;}
  if ( fSubDet == "EE" && ( fKeyStexNumber == 2 || fKeyStexNumber == 4 ) )
    {xStinAButText = "SC# for const. [  1,149] "; xStinBButText  = "SC'# for const. [  1,149] "; Stin_buf_lenght =  50;}

  //============================= STIN A =====================================
  TString xStinNumberText;
  if ( fSubDet == "EB" )
    {xStinNumberText = "Click here to register the tower number written on the right";}
  if ( fSubDet == "EE" )
    {xStinNumberText = "Click here to register the SC number written on the right";}

  TString xStinNumberValue;
  if ( fSubDet == "EB" )
    {xStinNumberValue = "Click and enter the tower number";}
  if ( fSubDet == "EE" )
    {xStinNumberValue = "Click and enter the SC number";}

  fTxSubFrame = new TGCompositeFrame
    (fStinSpFrame,60,20,kVerticalFrame, GetDefaultFrameBackground());     fCnew++;

  fStinAFrame = new TGCompositeFrame
    (fTxSubFrame,60,20,kHorizontalFrame,kSunkenFrame);                    fCnew++;

  fStinABut = new TGTextButton(fStinAFrame, xStinAButText, fStinAButC);   fCnew++;
  fStinABut->Connect("Clicked()","TEcnaGui", this, "DoButtonStinA()");
  fStinABut->SetToolTipText(xStinNumberText);
  fStinABut->Resize(Stin_buf_lenght, fStinABut->GetDefaultHeight());
  fStinABut->SetBackgroundColor(SubDetColor);
  fLayoutStinABut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);         fCnew++;
  fStinAFrame->AddFrame(fStinABut,  fLayoutStinABut);

  fEntryStinANumber = new TGTextBuffer();                                 fCnew++;
  fStinAText = new TGTextEntry(fStinAFrame, fEntryStinANumber);           fCnew++;
  fStinAText->SetToolTipText(xStinNumberValue);
  fStinAText->Resize(Stin_buf_lenght, fStinAText->GetDefaultHeight());

  Int_t StinAValue = 0;
  if( fSubDet == "EB"){StinAValue = fKeyStinANumber;}
  if( fSubDet == "EE" && fKeyStexNumber != 0 )
    {StinAValue = fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fKeyStexNumber, fKeyStinANumber);}
  DisplayInEntryField(fStinAText,StinAValue);

  fStinAText->Connect("ReturnPressed()", "TEcnaGui",this, "DoButtonStinA()");
  fLayoutStinAField =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1 );        fCnew++;
  fStinAFrame->AddFrame(fStinAText, fLayoutStinAField);
  fTxSubFrame->AddFrame(fStinAFrame, fLayoutGeneral);

  //========================== STIN A CRYSTAL NUMBERING VIEW
  TString xChNbButText;
  if ( fSubDet == "EB" ){xChNbButText = "Tower Xtal Numbering ";}
  if ( fSubDet == "EE" ){xChNbButText = "   SC Xtal Numbering  ";}

  fButChNb = new TGTextButton(fTxSubFrame, xChNbButText, fButChNbC);      fCnew++;
  fButChNb->Connect("Clicked()","TEcnaGui", this, "DoButtonChNb()");
  fButChNb->SetBackgroundColor(SubDetColor);
  fLayoutChNbBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);      fCnew++;
  fTxSubFrame->AddFrame(fButChNb, fLayoutChNbBut); 

  //---------------- menus relative to the Stin A subframe 

  //===================== Menus relative to the Stin A ======================
  TString xMenuBarCorGlob;
  if ( fSubDet == "EB" ){xMenuBarCorGlob = " GeoView Cor(s,s') (expert)";}
  if ( fSubDet == "EE" ){xMenuBarCorGlob = " GeoView Cor(s,s') (expert)";}

  TString xMenuBarCovGlob;
  if ( fSubDet == "EB" ){xMenuBarCovGlob = " GeoView Cov(s,s') (expert)";}
  if ( fSubDet == "EE" ){xMenuBarCovGlob = " GeoView Cov(s,s') (expert)";}

  //................. Menu correlations between samples for all the channels. Global view
  fMenuCorssAll = new TGPopupMenu(gClient->GetRoot());                      fCnew++;
  fMenuCorssAll->AddEntry(" Cor(s,s'), Xtal place -> Cor matrix",fMenuCorssAllColzC);
  fMenuCorssAll->AddEntry(" Cov(s,s'), Xtal place -> Cov matrix",fMenuCovssAllColzC);
  fMenuCorssAll->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarCorssAll =  new TGMenuBar(fTxSubFrame, 1, 1, kHorizontalFrame);   fCnew++;
  fMenuBarCorssAll->AddPopup(xMenuBarCorGlob, fMenuCorssAll, fLayoutGeneral);
  fTxSubFrame->AddFrame(fMenuBarCorssAll, fLayoutTopLeft);

  //................. Menu covariances between samples for all the channels. Global view
  //fMenuCovssAll = new TGPopupMenu(gClient->GetRoot());                      fCnew++;
  //fMenuCovssAll->AddEntry(" Cov(s,s'), Xtal place -> Cov matrix",fMenuCovssAllColzC);
  //fMenuCovssAll->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  //fMenuBarCovssAll =  new TGMenuBar(fTxSubFrame, 1, 1, kHorizontalFrame);   fCnew++;
  //fMenuBarCovssAll->AddPopup(xMenuBarCovGlob, fMenuCovssAll, fLayoutGeneral);
  //fTxSubFrame->AddFrame(fMenuBarCovssAll, fLayoutTopLeft);

  //------------------ Add Stin A frame to the subframe 
  fLayoutTxSubFrame = 
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);        fCnew++;
  fStinSpFrame->AddFrame(fTxSubFrame, fLayoutTxSubFrame);

  //============================= STIN B =====================================
  fTySubFrame = new TGCompositeFrame
    (fStinSpFrame,60,20,kVerticalFrame, GetDefaultFrameBackground());  fCnew++;

  fStinBFrame = new TGCompositeFrame
    (fTySubFrame,60,20,kHorizontalFrame,kSunkenFrame);                 fCnew++;

  fStinBBut =
    new TGTextButton(fStinBFrame, xStinBButText, fStinBButC);          fCnew++;
  fStinBBut->Connect("Clicked()","TEcnaGui", this, "DoButtonStinB()");
  fStinBBut->SetToolTipText(xStinNumberText);
  fStinBBut->Resize(Stin_buf_lenght, fStinBBut->GetDefaultHeight());
  fStinBBut->SetBackgroundColor(SubDetColor);
  fLayoutStinBBut = new TGLayoutHints(kLHintsLeft, xB1,xB1,xB1,xB1);   fCnew++;
  fStinBFrame->AddFrame(fStinBBut,  fLayoutStinBBut);

  fEntryStinBNumber = new TGTextBuffer();                              fCnew++;
  fStinBText = new TGTextEntry(fStinBFrame, fEntryStinBNumber);        fCnew++;
  fStinBText->SetToolTipText(xStinNumberValue);
  fStinBText->Resize(Stin_buf_lenght, fStinBText->GetDefaultHeight());

  Int_t StinBValue = 0;
  if( fSubDet == "EB"){StinBValue = fKeyStinBNumber;}
  if( fSubDet == "EE" && fKeyStexNumber != 0 )
    {StinBValue = fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fKeyStexNumber, fKeyStinBNumber);}
  DisplayInEntryField(fStinBText, StinBValue);

  fStinBText->Connect("ReturnPressed()", "TEcnaGui",this, "DoButtonStinB()");
  fLayoutStinBField =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1 );    fCnew++;
  fStinBFrame->AddFrame(fStinBText, fLayoutStinBField);
  fTySubFrame->AddFrame(fStinBFrame, fLayoutGeneral);

  //---------------- menus relative to the Stin B subframe 

  //                    (no such menus )

  //------------------ Add Stin B subframe to the frame 
  fLayoutTySubFrame =  new TGLayoutHints
    (kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);                   fCnew++;

  fStinSpFrame->AddFrame(fTySubFrame, fLayoutTySubFrame);

  //---------------------- composite frame (Stin X, Stin Y)
  fLayoutStinSpFrame =  new TGLayoutHints
    (kLHintsTop | kLHintsCenterX, xB1, xB1, xB1, xB1);                fCnew++;
  AddFrame(fStinSpFrame, fLayoutStinSpFrame);

  //------------------ menus relatives to the Horizontal frame (Stin_A + Stin_B)
  TString xMenuBarLFCorcc;
  if ( fSubDet == "EB" ){xMenuBarLFCorcc = " Low Frequency Cor(Xtal tower, Xtal tower')";}
  if ( fSubDet == "EE" ){xMenuBarLFCorcc = " Low Frequency Cor(Xtal SC, Xtal SC')";}

  TString xMenuBarHFCorcc;
  if ( fSubDet == "EB" ){xMenuBarHFCorcc = " High Frequency Cor(Xtal tower, Xtal tower')";}
  if ( fSubDet == "EE" ){xMenuBarHFCorcc = " High Frequency Cor(Xtal SC, Xtal SC')";}

  //...................... Menu LF correlations between channels
  fMenuLFCorcc = new TGPopupMenu(gClient->GetRoot());                   fCnew++;
  fMenuLFCorcc->AddEntry("2D, COLZ",fMenuLFCorccColzC);
  fMenuLFCorcc->AddSeparator();
  fMenuLFCorcc->AddEntry("3D, LEGO2Z",fMenuLFCorccLegoC);
  fMenuLFCorcc->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarLFCorcc = new TGMenuBar(this, 1, 1, kHorizontalFrame);        fCnew++;
  fMenuBarLFCorcc->AddPopup(xMenuBarLFCorcc, fMenuLFCorcc, fLayoutTopRight);
  AddFrame(fMenuBarLFCorcc, fLayoutGeneral);

  //...................... Menu HF correlations between channels
  fMenuHFCorcc = new TGPopupMenu(gClient->GetRoot());                   fCnew++;
  fMenuHFCorcc->AddEntry("2D, COLZ",fMenuHFCorccColzC);
  fMenuHFCorcc->AddSeparator();
  fMenuHFCorcc->AddEntry("3D, LEGO2Z",fMenuHFCorccLegoC);
  fMenuHFCorcc->Connect("Activated(Int_t)", "TEcnaGui", this,"HandleMenu(Int_t)");
  fMenuBarHFCorcc = new TGMenuBar(this, 1, 1, kHorizontalFrame);        fCnew++;
  fMenuBarHFCorcc->AddPopup(xMenuBarHFCorcc, fMenuHFCorcc, fLayoutTopRight);
  AddFrame(fMenuBarHFCorcc, fLayoutGeneral);

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  //
  //                             SECTOR 4: Channels, Samples
  //
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  fChSpFrame = new TGCompositeFrame(this,60,20,kHorizontalFrame,
				    GetDefaultFrameBackground());     fCnew++;

  TString xChanButText;
  if ( fSubDet == "EB" ){xChanButText = " Channel# in tower [0,24] ";}
  if ( fSubDet == "EE" ){xChanButText = " Crystal# in SC [1,25] ";}

  TString xSampButText  = " Sample# [1,10] ";

  Int_t chan_buf_lenght =  50;
  Int_t samp_buf_lenght =  50;

  TString xMenuBarCorss    = " Correlations between samples";
  TString xMenuBarCovss    = " Covariances between samples";
  TString xMenuBarEvs      = " Sample means";
  TString xMenuBarSigs     = " Sample sigmas";

  //=================================== CHANNEL (CRYSTAL)
  fChSubFrame = new TGCompositeFrame
    (fChSpFrame,60,20,kVerticalFrame, GetDefaultFrameBackground());   fCnew++;

  fChanFrame = new TGCompositeFrame
    (fChSubFrame,60,20,kHorizontalFrame,kSunkenFrame);                fCnew++;

  fChanBut =
    new TGTextButton(fChanFrame, xChanButText, fChanButC);            fCnew++;
  fChanBut->Connect("Clicked()","TEcnaGui", this, "DoButtonChan()");
  fChanBut->SetToolTipText("Click here to register the channel number written to the right");
  fChanBut->Resize(chan_buf_lenght, fChanBut->GetDefaultHeight());
  fChanBut->SetBackgroundColor(SubDetColor);
  fLayoutChanBut = new TGLayoutHints(kLHintsLeft, xB1,xB1,xB1,xB1);   fCnew++;
  fChanFrame->AddFrame(fChanBut,  fLayoutChanBut);

  fEntryChanNumber = new TGTextBuffer();                              fCnew++;
  fChanText = new TGTextEntry(fChanFrame, fEntryChanNumber);          fCnew++;
  fChanText->SetToolTipText("Click and enter the channel number");
  fChanText->Resize(chan_buf_lenght, fChanText->GetDefaultHeight());

  Int_t xReadChanNumber = 0;
  if( fSubDet == "EB" ){xReadChanNumber = 0;}         // offset =  0 (EB: electronic channel number)
  if( fSubDet == "EE" ){xReadChanNumber = 1;}         // offset = +1 (EE: xtal number for construction)
  DisplayInEntryField(fChanText, xReadChanNumber);

  fChanText->Connect("ReturnPressed()", "TEcnaGui",this, "DoButtonChan()");
  fLayoutChanField =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1 );    fCnew++;
  fChanFrame->AddFrame(fChanText, fLayoutChanField);
  fChSubFrame->AddFrame(fChanFrame, fLayoutGeneral);

  //--------------------- Menus relative to the channel SubFrame -------------
  //...................... Menu correlations between samples

  fMenuCorss = new TGPopupMenu(gClient->GetRoot());                   fCnew++;
  fMenuCorss->AddEntry("2D, COLZ", fMenuCorssColzC);
  fMenuCorss->AddEntry("2D, BOX",  fMenuCorssBoxC);
  fMenuCorss->AddEntry("2D, TEXT", fMenuCorssTextC);
  fMenuCorss->AddEntry("2D, CONTZ",fMenuCorssContzC);
  fMenuCorss->AddSeparator();
  fMenuCorss->AddEntry("3D, LEGO2Z",fMenuCorssLegoC);
  fMenuCorss->AddEntry("3D, SURF1Z",fMenuCorssSurf1C);
  fMenuCorss->AddEntry("3D, SURF2Z",fMenuCorssSurf2C);
  fMenuCorss->AddEntry("3D, SURF3Z",fMenuCorssSurf3C);
  fMenuCorss->AddEntry("3D, SURF4" ,fMenuCorssSurf4C);
  fMenuCorss->AddSeparator();
  fMenuCorss->AddEntry("2D, Write in ASCII file",fMenuCorssAsciiFileC);
  fMenuCorss->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarCorss = new TGMenuBar(fChSubFrame, 1, 1, kHorizontalFrame); fCnew++;
  fMenuBarCorss->AddPopup(xMenuBarCorss, fMenuCorss, fLayoutTopLeft);
  fChSubFrame->AddFrame(fMenuBarCorss, fLayoutTopLeft);

  //...................... Menu covariances between samples

  fMenuCovss = new TGPopupMenu(gClient->GetRoot());                   fCnew++;
  fMenuCovss->AddEntry("2D, COLZ", fMenuCovssColzC);
  fMenuCovss->AddEntry("2D, BOX",  fMenuCovssBoxC);
  fMenuCovss->AddEntry("2D, TEXT", fMenuCovssTextC);
  fMenuCovss->AddEntry("2D, CONTZ",fMenuCovssContzC);
  fMenuCovss->AddSeparator();
  fMenuCovss->AddEntry("3D, LEGO2Z",fMenuCovssLegoC);
  fMenuCovss->AddEntry("3D, SURF1Z",fMenuCovssSurf1C);
  fMenuCovss->AddEntry("3D, SURF2Z",fMenuCovssSurf2C);
  fMenuCovss->AddEntry("3D, SURF3Z",fMenuCovssSurf3C);
  fMenuCovss->AddEntry("3D, SURF4" ,fMenuCovssSurf4C);
  fMenuCovss->AddSeparator();
  fMenuCovss->AddEntry("2D, Write in ASCII file",fMenuCovssAsciiFileC);
  fMenuCovss->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarCovss = new TGMenuBar(fChSubFrame, 1, 1, kHorizontalFrame); fCnew++;
  fMenuBarCovss->AddPopup(xMenuBarCovss, fMenuCovss, fLayoutTopLeft);
  fChSubFrame->AddFrame(fMenuBarCovss, fLayoutTopLeft);

  //...................... Menu expectation values of the samples

  fMenuD_MSp_SpNb = new TGPopupMenu(gClient->GetRoot());                      fCnew++;
  fMenuD_MSp_SpNb->AddEntry("1D Histo ",fMenuD_MSp_SpNbLineFullC);
  fMenuD_MSp_SpNb->AddEntry("1D Histo SAME",fMenuD_MSp_SpNbLineSameC);
  fMenuD_MSp_SpNb->AddEntry("1D Histo 25 Xtals",fMenuD_MSp_SpNbLineAllStinC);
  fMenuD_MSp_SpNb->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarD_MSp_SpNb = new TGMenuBar(fChSubFrame, 1, 1, kHorizontalFrame);    fCnew++;
  fMenuBarD_MSp_SpNb->AddPopup(xMenuBarEvs, fMenuD_MSp_SpNb, fLayoutTopLeft);
  fChSubFrame->AddFrame(fMenuBarD_MSp_SpNb, fLayoutTopLeft);

  //...................... Menu sigmas/variances of the samples

  fMenuD_SSp_SpNb = new TGPopupMenu(gClient->GetRoot());                     fCnew++;
  fMenuD_SSp_SpNb->AddEntry("1D Histo ",fMenuD_SSp_SpNbLineFullC);
  fMenuD_SSp_SpNb->AddEntry("1D Histo SAME",fMenuD_SSp_SpNbLineSameC);
  fMenuD_SSp_SpNb->AddEntry("1D Histo 25 Xtals",fMenuD_SSp_SpNbLineAllStinC);
  fMenuD_SSp_SpNb->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarD_SSp_SpNb = new TGMenuBar(fChSubFrame, 1, 1, kHorizontalFrame);   fCnew++;
  fMenuBarD_SSp_SpNb->AddPopup(xMenuBarSigs, fMenuD_SSp_SpNb, fLayoutTopLeft);
  fChSubFrame->AddFrame(fMenuBarD_SSp_SpNb, fLayoutTopLeft);

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
  fSampBut->Connect("Clicked()","TEcnaGui", this, "DoButtonSamp()");
  fSampBut->SetToolTipText("Click here to register the sample number written to the right");
  fSampBut->Resize(samp_buf_lenght, fSampBut->GetDefaultHeight());
  fSampBut->SetBackgroundColor(SubDetColor);
  fLayoutSampBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);     fCnew++;
  fSampFrame->AddFrame(fSampBut, fLayoutSampBut);

  fEntrySampNumber = new TGTextBuffer();                              fCnew++;
  fSampText = new TGTextEntry(fSampFrame, fEntrySampNumber);          fCnew++;
  fSampText->SetToolTipText("Click and enter the sample number");
  fSampText->Resize(samp_buf_lenght, fSampText->GetDefaultHeight());
  Int_t xKeySampNumber = fKeySampNumber+1;
  DisplayInEntryField(fSampText, xKeySampNumber);
  fSampText->Connect("ReturnPressed()", "TEcnaGui",this, "DoButtonSamp()");
  fLayoutSampField =
    new TGLayoutHints(kLHintsTop | kLHintsRight, xB1,xB1,xB1,xB1 );   fCnew++;
  fSampFrame->AddFrame(fSampText, fLayoutSampField);

  fSpSubFrame->AddFrame(fSampFrame,fLayoutGeneral);

  fLayoutSpSubFrame =  new TGLayoutHints
    (kLHintsTop | kLHintsRight, xB1, xB1, xB1, xB1);                  fCnew++;
  fChSpFrame->AddFrame(fSpSubFrame, fLayoutSpSubFrame);

  //---------------------- composite frame (channel/sample+menus)
  fLayoutChSpFrame =  new TGLayoutHints
    (kLHintsTop | kLHintsCenterX, xB1, xB1, xB1, xB1);                fCnew++;
  AddFrame(fChSpFrame, fLayoutChSpFrame);

  //====================== Menu histogram of the distribution
  //                       for a given (channel, sample)
  fMenuAdcProj = new TGPopupMenu(gClient->GetRoot());                 fCnew++;
  fMenuAdcProj->AddEntry("1D Histo ",fMenuAdcProjSampLineFullC);
  fMenuAdcProj->AddEntry("1D Histo SAME",fMenuAdcProjSampLineSameC);
  fMenuAdcProj->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");

  fMenuBarAdcProj = new TGMenuBar(this, 1, 1, kHorizontalFrame);      fCnew++;

  TString xEvtDistrib;
  xEvtDistrib = "ADC sample values for (Xtal, Sample)";

  fMenuBarAdcProj->AddPopup(xEvtDistrib, fMenuAdcProj, fLayoutGeneral);

  fLayoutMenuBarAdcProj =
    new TGLayoutHints(kLHintsCenterX, xB1,xB1,xB1,xB1);               fCnew++;
  AddFrame(fMenuBarAdcProj, fLayoutMenuBarAdcProj);

  AddFrame(fVoidFrame, fLayoutGeneral);

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  //
  //                             SECTOR 5: Time Evolution / history plots
  //
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  TString xRunListButText = " List of run file name for history plots ";
  Int_t run_list_buf_lenght = 170;

  fRulFrame = new TGCompositeFrame(this,60,20, kHorizontalFrame, kSunkenFrame); fCnew++;
  
  fRulBut= new TGTextButton(fRulFrame, xRunListButText);                        fCnew++;
  fRulBut->Connect("Clicked()","TEcnaGui", this, "DoButtonRul()");
  fRulBut->SetToolTipText
    ("Click here to register the name of the file \n containing the run list (written on the right)");
  fRulBut->SetBackgroundColor(SubDetColor);
  fLayoutRulBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);     fCnew++;
  fRulFrame->AddFrame(fRulBut,  fLayoutRulBut);

  fEntryRulNumber = new TGTextBuffer();                               fCnew++;
  fRulText = new TGTextEntry(fRulFrame, fEntryRulNumber);             fCnew++;
  fRulText->SetToolTipText("Click and enter the name of the file \n containing the run list");
  fRulText->Resize(run_list_buf_lenght, fRulText->GetDefaultHeight());
  fRulText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonRul()");
  fLayoutRulFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsLeft, xB1,xB1,xB1,xB1);  fCnew++;
  fRulFrame->AddFrame(fRulText, fLayoutRulFieldText);

  fLayoutRulFieldFrame =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);     fCnew++;
  AddFrame(fRulFrame, fLayoutRulFieldFrame);

  //...................... Menu history plot (evolution in time)
  TString xMenuBarHistory = " Menu for history plots";
  fMenuHistory = new TGPopupMenu(gClient->GetRoot());                 fCnew++;
  fMenuHistory->AddEntry("1D, Pedestals ",fMenuH_Ped_DatePolmFullC);
  fMenuHistory->AddEntry("1D, Pedestals SAME",fMenuH_Ped_DatePolmSameC);
  fMenuHistory->AddSeparator();
  fMenuHistory->AddEntry("1D, Total Noise ",fMenuH_TNo_DatePolmFullC);
  fMenuHistory->AddEntry("1D, Total Noise SAME",fMenuH_TNo_DatePolmSameC);
  fMenuHistory->AddEntry("1D, Total Noise SAME n",fMenuH_TNo_DatePolmSamePC);
  fMenuHistory->AddSeparator();
  fMenuHistory->AddEntry("1D, Low Frequency Noise ",fMenuH_LFN_DatePolmFullC);
  fMenuHistory->AddEntry("1D, Low Frequency Noise SAME",fMenuH_LFN_DatePolmSameC);
  fMenuHistory->AddEntry("1D, Low Frequency Noise SAME n",fMenuH_LFN_DatePolmSamePC);
  fMenuHistory->AddSeparator();
  fMenuHistory->AddEntry("1D, High Frequency Noise ",fMenuH_HFN_DatePolmFullC);
  fMenuHistory->AddEntry("1D, High Frequency Noise SAME",fMenuH_HFN_DatePolmSameC);
  fMenuHistory->AddEntry("1D, High Frequency Noise SAME n",fMenuH_HFN_DatePolmSamePC);
  fMenuHistory->AddSeparator();
  fMenuHistory->AddEntry("1D, Mean cor(s,s') ",fMenuH_MCs_DatePolmFullC);
  fMenuHistory->AddEntry("1D, Mean cor(s,s') SAME",fMenuH_MCs_DatePolmSameC);
  fMenuHistory->AddEntry("1D, Mean cor(s,s') SAME n",fMenuH_MCs_DatePolmSamePC);
  fMenuHistory->AddSeparator();
  fMenuHistory->AddEntry("1D, Sigma of cor(s,s') ",fMenuH_SCs_DatePolmFullC);
  fMenuHistory->AddEntry("1D, Sigma of cor(s,s') SAME",fMenuH_SCs_DatePolmSameC);
  fMenuHistory->AddEntry("1D, Sigma of cor(s,s') SAME n",fMenuH_SCs_DatePolmSamePC);

  fMenuHistory->Connect("Activated(Int_t)", "TEcnaGui", this, "HandleMenu(Int_t)");
  fMenuBarHistory = new TGMenuBar(this , 1, 1, kHorizontalFrame);        fCnew++;
  fMenuBarHistory->AddPopup(xMenuBarHistory, fMenuHistory, fLayoutTopLeft);
  AddFrame(fMenuBarHistory, fLayoutTopLeft);

  AddFrame(fVoidFrame, fLayoutGeneral);

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  //
  //                             SECTOR 6: Last Buttons
  //
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  //GContext_t   norm = GetDefaultGC()();
  //FontStruct_t font = GetDefaultFontStruct();

  //========================== LinLog frame: buttons: LinX/LogX, LinY/LogY, Projection on Y Axis

  fLinLogFrame = new TGCompositeFrame(this,60,20, kHorizontalFrame, kSunkenFrame);    fCnew++;


  //-------------------------- Lin X <-> Log X
  TString xLogxButText     = " LOG X ";
  fButLogx = new TGCheckButton(fLinLogFrame, xLogxButText, fButLogxC);             fCnew++;
  fButLogx->Connect("Clicked()","TEcnaGui", this, "DoButtonLogx()");
  fButLogx->SetBackgroundColor(SubDetColor);
  fLayoutLogxBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);   fCnew++;
  fLinLogFrame->AddFrame(fButLogx, fLayoutLogxBut);

  //-------------------------- Lin Y <-> Log Y
  TString xLogyButText     = " LOG Y ";
  fButLogy = new TGCheckButton(fLinLogFrame, xLogyButText, fButLogyC);             fCnew++;
  fButLogy->Connect("Clicked()","TEcnaGui", this, "DoButtonLogy()");
  fButLogy->SetBackgroundColor(SubDetColor);
  fLayoutLogyBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);   fCnew++;
  fLinLogFrame->AddFrame(fButLogy, fLayoutLogyBut);

  //-------------------------- Projection
  TString xProjyButText     = " Y projection ";
  fButProjy = new TGCheckButton(fLinLogFrame, xProjyButText, fButProjyC);           fCnew++;
  fButProjy->Connect("Clicked()","TEcnaGui", this, "DoButtonProjy()");
  fButProjy->SetBackgroundColor(SubDetColor);
  fLayoutProjyBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);  fCnew++;
  fLinLogFrame->AddFrame(fButProjy, fLayoutProjyBut);

  AddFrame(fVoidFrame, fLayoutBottRight);
  AddFrame(fLinLogFrame, fLayoutGeneral);

  //======================================== GENERAL TITLE FOR THE PLOTS
  TString xGenTitleButText  = " General title for plots ";
  Int_t gen_title_buf_lenght  = 220;

  fGentFrame = new TGCompositeFrame(this,60,20, kHorizontalFrame, kSunkenFrame); fCnew++;
  
  fGentBut= new TGTextButton(fGentFrame, xGenTitleButText);                      fCnew++;
  fGentBut->Connect("Clicked()","TEcnaGui", this, "DoButtonGent()");
  fGentBut->SetToolTipText
    ("Click here to register the general title (written on the right)");
  fGentBut->SetBackgroundColor(SubDetColor);
  fLayoutGentBut =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);       fCnew++;
  fGentFrame->AddFrame(fGentBut,  fLayoutGentBut);

  fEntryGentNumber = new TGTextBuffer();                                fCnew++;
  fGentText = new TGTextEntry(fGentFrame, fEntryGentNumber);            fCnew++;
  fGentText->SetToolTipText("Click and enter the general title");
  fGentText->Resize(gen_title_buf_lenght, fGentText->GetDefaultHeight());
  DisplayInEntryField(fGentText, fKeyGeneralTitle);
  fGentText->Connect("ReturnPressed()", "TEcnaGui", this, "DoButtonGent()");
  fLayoutGentFieldText =
    new TGLayoutHints(kLHintsBottom | kLHintsLeft, xB1,xB1,xB1,xB1);    fCnew++;
  fGentFrame->AddFrame(fGentText, fLayoutGentFieldText);

  fLayoutGentFieldFrame =
    new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1,xB1,xB1,xB1);       fCnew++;
  AddFrame(fGentFrame, fLayoutGentFieldFrame);
  AddFrame(fVoidFrame);

  //========================== Color Palette + EXIT
  fColorExitFrame = new TGCompositeFrame(this,60,20, kHorizontalFrame,
					 GetDefaultFrameBackground());  fCnew++;

  //-------------------------- Black/Red/Blue <-> Rainbow
  TString xColPalButText     = " Colors ";
  fButColPal = new TGCheckButton(fColorExitFrame, xColPalButText, fButColPalC);          fCnew++;
  fButColPal->Connect("Clicked()","TEcnaGui", this, "DoButtonColPal()");
  fButColPal->SetBackgroundColor(SubDetColor);
  fLayoutColPalBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1); fCnew++;
  fColorExitFrame->AddFrame(fButColPal, fLayoutColPalBut);

  //-------------------------- Exit
  TString xExitButText     = " Exit ";
  fButExit = new TGTextButton(fColorExitFrame, xExitButText, fButExitC);              fCnew++;
  fButExit->Connect("Clicked()","TEcnaGui", this, "DoButtonExit()");
  fButExit->SetBackgroundColor(SubDetColor);
  fLayoutExitBut = new TGLayoutHints(kLHintsTop | kLHintsRight, xB1, xB1, xB1, xB1);  fCnew++;
  fColorExitFrame->AddFrame(fButExit, fLayoutExitBut);
 
  fLayoutColorExitFrame =  new TGLayoutHints(kLHintsTop | kLHintsExpandX,
					     xB1, xB1, xB1, xB1);     fCnew++;

  //AddFrame(fVoidFrame, fLayoutBottRight);
  AddFrame(fColorExitFrame, fLayoutColorExitFrame);

  //========================== Last frame: buttons: ROOT version, Help

  fLastFrame = new TGCompositeFrame(this,60,20, kHorizontalFrame, kSunkenFrame);      fCnew++;

  //-------------------------- Clone Last Canvas
  TString xCloneButText     = " Clone Last Canvas ";
  fButClone = new TGTextButton(fLastFrame, xCloneButText, fButCloneC);                fCnew++;
  fButClone->Connect("Clicked()","TEcnaGui", this, "DoButtonClone()");
  fButClone->SetBackgroundColor(SubDetColor);
  fLayoutCloneBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);  fCnew++;
  fLastFrame->AddFrame(fButClone, fLayoutCloneBut);

  //-------------------------- ROOT version
  TString xRootButText     = " ROOT Version ";
  fButRoot = new TGTextButton(fLastFrame, xRootButText, fButRootC);                   fCnew++;
  fButRoot->Connect("Clicked()","TEcnaGui", this, "DoButtonRoot()");
  fButRoot->SetBackgroundColor(SubDetColor);
  fLayoutRootBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);   fCnew++;
  fLastFrame->AddFrame(fButRoot, fLayoutRootBut);

  //-------------------------- HELP
  TString xHelpButText     = " Help ";
  fButHelp = new TGTextButton(fLastFrame, xHelpButText, fButHelpC);                   fCnew++;
  fButHelp->Connect("Clicked()","TEcnaGui", this, "DoButtonHelp()");
  fButHelp->SetBackgroundColor(SubDetColor);
  fLayoutHelpBut = new TGLayoutHints(kLHintsTop | kLHintsLeft, xB1, xB1, xB1, xB1);   fCnew++;
  fLastFrame->AddFrame(fButHelp, fLayoutHelpBut);

  AddFrame(fLastFrame, fLayoutGeneral);

  //................................. Window

  MapSubwindows();
  Layout();

  if( fSubDet == "EB" ){SetWindowName("CMS  Ecal Correlated Noise Analysis  <EB>");}
  if( fSubDet == "EE" ){SetWindowName("CMS  Ecal Correlated Noise Analysis  <EE>");}

  SetBackgroundColor(SubDetColor);
  SetIconName("CNA");
  MapWindow();
  // } // end of if( fCnaParPaths-GetPaths() == kTRUE )
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

//----------------------------------------------------------------------
//void TEcnaGui::DoButtonPyf()
//{
////Register the name of the file containing the data file name(s)
//// which are in the "source" sector of the python file
  
//  //........................... get info from the entry field
//  const char* listchain = fPyfText->GetBuffer()->GetString();  
//  fKeyPyf = listchain;
  
//  fCnaCommand++;
//  cout << "   *TEcnaGui [" << fCnaCommand
//       << "]> Registration of file name for python file source sector -> "
//       << fKeyPyf.Data() << endl;
//}
//----------------------------------------------------------------------
void TEcnaGui::DoButtonAna()
{
//Registration of the type of the analysis

  const char *bufferchain = fAnaText->GetBuffer()->GetString();

  fKeyAnaType = bufferchain;
  
  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of analysis name -> "
       << fKeyAnaType << endl;
}

//----------------------------------------------------------------------
void TEcnaGui::DoButtonNors()
{
//Registration of the number of samples (ROOT file)

  const char *bufferchain = fNorsText->GetBuffer()->GetString();
  fKeyNbOfSamplesString = bufferchain;
  fKeyNbOfSamples = atoi(bufferchain);

  if ( !(fKeyNbOfSamples >= 1 && fKeyNbOfSamples <= fEcal->MaxSampADC()) )
    {
      fCnaError++;
      cout << "   !TEcnaGui (" << fCnaError << ") *** ERROR *** ===>"
	   << " Number of required samples for reading ROOT file = " << fKeyNbOfSamples
	   << ": OUT OF RANGE, " << endl 
	   << "                                        forced to the default value (="
	   << fEcal->MaxSampADC() << ")." << fTTBELL << endl;
      fKeyNbOfSamples = fEcal->MaxSampADC();
      DisplayInEntryField(fNorsText,fKeyNbOfSamples);
    }

  //................... Update of Sample Button Text according to the number of sample
  TString xSampButText = " Sample [?,?] ";
  
  if( fKeyNbOfSamples ==  1 ){xSampButText  = " Sample [1,1] ";}
  if( fKeyNbOfSamples ==  2 ){xSampButText  = " Sample [1,2] ";}
  if( fKeyNbOfSamples ==  3 ){xSampButText  = " Sample [1,3] ";}
  if( fKeyNbOfSamples ==  4 ){xSampButText  = " Sample [1,4] ";}
  if( fKeyNbOfSamples ==  5 ){xSampButText  = " Sample [1,5] ";}
  if( fKeyNbOfSamples ==  6 ){xSampButText  = " Sample [1,6] ";}
  if( fKeyNbOfSamples ==  7 ){xSampButText  = " Sample [1,7] ";}
  if( fKeyNbOfSamples ==  8 ){xSampButText  = " Sample [1,8] ";}
  if( fKeyNbOfSamples ==  9 ){xSampButText  = " Sample [1,9] ";}
  if( fKeyNbOfSamples == 10 ){xSampButText  = " Sample [1,10] ";}
  
  fSampBut->SetText(xSampButText);
  
  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of number of samples in ROOT file -> "
       << fKeyNbOfSamples << endl;
}
//----------------------------------------------------------------------
void TEcnaGui::DoButtonNbSampForCalc()
{
//Registration of the number of samples (ROOT file)

  const char *bufferchain = fNbSampForCalcText->GetBuffer()->GetString();
  fKeyNbOfSampForCalcString = bufferchain;
  fKeyNbOfSampForCalc = atoi(bufferchain);

  if ( !(fKeyNbOfSampForCalc >= 1 && fKeyNbOfSampForCalc <= fKeyNbOfSamples) )
    {
      fCnaError++;
      cout << "   !TEcnaGui (" << fCnaError << ") *** ERROR *** ===>"
	   << " Number of required samples for calculations = " << fKeyNbOfSampForCalc
	   << ": OUT OF RANGE, " << endl 
	   << "                                        forced to the default value (="
	   << fKeyNbOfSamples << ")." << fTTBELL << endl;
      fKeyNbOfSampForCalc = fKeyNbOfSamples;
      DisplayInEntryField(fNbSampForCalcText,fKeyNbOfSampForCalc);
    }
  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of number of samples for calculations -> "
       << fKeyNbOfSampForCalc << endl;
}

//----------------------------------------------------------------------
void TEcnaGui::DoButtonRun()
{
//Register run number
  
  //........................... get info from the entry field
  const char* bufferchain = fRunText->GetBuffer()->GetString();
  fKeyRunNumberString = bufferchain;
  fKeyRunNumber = atoi(bufferchain);
  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of run number -> "
       << fKeyRunNumber << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonFev()
{
//Registration of the first requested event number

  const char *bufferchain = fFevText->GetBuffer()->GetString();
  fKeyFirstReqEvtNumberString = bufferchain;
  fKeyFirstReqEvtNumber = atoi(bufferchain);

  if ( fKeyFirstReqEvtNumber <= 0)
    {
      fCnaError++;
      cout << "   !TEcnaGui (" << fCnaError << ") *ERROR* ===> "
	   << " First event number = " << fKeyFirstReqEvtNumber
	   << ": negative. " << endl 
	   << fTTBELL << endl;
    }

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of first requested event number -> "
       << fKeyFirstReqEvtNumber << endl;
}
//-------------------------------------------------------------------
void TEcnaGui::DoButtonLev()
{
//Registration of the last requested event number

  const char *bufferchain = fLevText->GetBuffer()->GetString();
  fKeyLastReqEvtNumberString = bufferchain;
  fKeyLastReqEvtNumber = atoi(bufferchain);

  if ( fKeyLastReqEvtNumber <= fKeyFirstReqEvtNumber )
    {
      fCnaError++;
      cout << "   !TEcnaGui (" << fCnaError << ") *WARNING* ===> "
	   << " Last requested event number = " << fKeyLastReqEvtNumber
	   << ": less than first requested event number (= " << fKeyFirstReqEvtNumber << ")." 
	   << endl;
    }

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of last requested event number -> "
       << fKeyLastReqEvtNumber << endl;
}
//-------------------------------------------------------------------
void TEcnaGui::DoButtonRev()
{
//Registration of the nb ofrequested events

  const char *bufferchain = fRevText->GetBuffer()->GetString();
  fKeyReqNbOfEvtsString = bufferchain;
  fKeyReqNbOfEvts = atoi(bufferchain);

  Int_t nb_range_evts = fKeyLastReqEvtNumber - fKeyFirstReqEvtNumber + 1;

  if( fKeyLastReqEvtNumber < fKeyFirstReqEvtNumber)
    {
      fCnaError++;
      cout << "   !TEcnaGui (" << fCnaError << ") *WARNING* ===> "
	   << " Last requested event number = " << fKeyLastReqEvtNumber
	   << " less than first requested event number = " << fKeyFirstReqEvtNumber
	   << endl;
    }

  if ( fKeyLastReqEvtNumber >= fKeyFirstReqEvtNumber && fKeyReqNbOfEvts > nb_range_evts )
    {
      fCnaError++;
      cout << "   !TEcnaGui (" << fCnaError << ") *WARNING* ===> "
	   << " Nb of requested events = " << fKeyReqNbOfEvts
	   << ": out of range (range = " << fKeyFirstReqEvtNumber << ","
	   << fKeyLastReqEvtNumber << ") => " << nb_range_evts << " events."
	   << endl;
    }

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of requested number of events -> "
       << fKeyReqNbOfEvts << endl;
}

//-------------------------------------------------------------------
void TEcnaGui::DoButtonStex()
{
//Registration of the Stex number

  const char *bufferchain = fStexText->GetBuffer()->GetString();
  fKeyStexNumberString = bufferchain;
  fKeyStexNumber = atoi(bufferchain);

  if( fSubDet == "EB" )
    {
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Registration of SuperModule number -> "
	   << fKeyStexNumber << endl;

      //.......... Positive number for EB- [-1,-18] -> [19,36]  
      if( fKeyStexNumber < 0 ){fKeyStexNumber = - fKeyStexNumber + fEcal->MaxSMInEB()/2;}

      if( (fKeyStexNumber < 0) || (fKeyStexNumber > fEcal->MaxSMInEB() )  )
	{
	  fCnaError++;
	  cout << "   !TEcnaGui (" << fCnaError << ") *** ERROR *** ===> "
	       << " EB / SM number = " << fKeyStexNumber
	       << ": out of range. Range = 0 (EB) or [ 1 ," << fEcal->MaxSMInEB() << " ] (SM)"
	       << " or [ -" << fEcal->MaxSMInEBMinus() << ", +" <<  fEcal->MaxSMInEBPlus() << "] (SM)"
	       << fTTBELL << endl;
	}
    }

  if( fSubDet == "EE" )
    {
      //................... Update of SC Button Text according to the Dee Number
      TString xStinAButText = "?";
      TString xStinBButText = "?";
      if ( fSubDet == "EE" && ( fKeyStexNumber == 1 || fKeyStexNumber == 3 ) )
	{xStinAButText = "SC for const. [150,298] "; xStinBButText  = "SC' for const. [150,298] ";
	fStinABut->SetText(xStinAButText); fStinBBut->SetText(xStinBButText);}
      if ( fSubDet == "EE" && ( fKeyStexNumber == 2 || fKeyStexNumber == 4 ) )
	{xStinAButText = "SC for const. [  1,149] "; xStinBButText  = "SC' for const. [  1,149] ";
	fStinABut->SetText(xStinAButText); fStinBBut->SetText(xStinBButText);}

      if ( fSubDet == "EE" && ( fKeyStexNumber == 0 ) )
	{xStinAButText = "SC for const.           "; xStinBButText  = "SC' for const.           ";
	fStinABut->SetText(xStinAButText); fStinBBut->SetText(xStinBButText);}
      
      //................... Update of SC widget according to the Dee Number
      if( fKeyStexNumber > 0 )
	{
	  Int_t StinAValue = fKeyStinANumber;
	  if( fSubDet == "EE" )
	    {StinAValue = fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fKeyStexNumber, fKeyStinANumber);}
	  DisplayInEntryField(fStinAText,StinAValue);
	  Int_t StinBValue = fKeyStinBNumber;
	  if( fSubDet == "EE" )
	    {StinBValue = fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fKeyStexNumber, fKeyStinBNumber);}
	  DisplayInEntryField(fStinBText,StinBValue);
	}
      //............................................ Command message
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Registration of Dee number -> "
	   << fKeyStexNumber << endl;
      
      if ( (fKeyStexNumber < 0) || (fKeyStexNumber > fEcal->MaxDeeInEE() )  )
	{
	  fCnaError++;
	  cout << "   !TEcnaGui (" << fCnaError << ") *** ERROR *** ===> "
	       << " EE / Dee number = " << fKeyStexNumber
	       << ": out of range. Range = 0 (EE) or [ 1 ," << fEcal->MaxDeeInEE() << " ] (Dee)"
	       << fTTBELL << endl;
	}
    }  // -- end of if( fSubDet == "EE" ) -------
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVminD_NOE_ChNb()
{
//Registration of Ymin for number of events

  const char *bufferchain = fVminD_NOE_ChNbText->GetBuffer()->GetString();

  fKeyVminD_NOE_ChNb = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymin for plot 'number of events' -> "
       << fKeyVminD_NOE_ChNb << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVmaxD_NOE_ChNb()
{
//Registration of Ymax for number of events

  const char *bufferchain = fVmaxD_NOE_ChNbText->GetBuffer()->GetString();

  fKeyVmaxD_NOE_ChNb = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymax for plot 'number of events' -> "
       << fKeyVmaxD_NOE_ChNb << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVminD_Ped_ChNb()
{
//Registration of Ymin for pedestals

  const char *bufferchain = fVminD_Ped_ChNbText->GetBuffer()->GetString();

  fKeyVminD_Ped_ChNb = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymin for plot 'pedestal' -> "
       << fKeyVminD_Ped_ChNb << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVmaxD_Ped_ChNb()
{
//Registration of Ymax for pedestals

  const char *bufferchain = fVmaxD_Ped_ChNbText->GetBuffer()->GetString();

  fKeyVmaxD_Ped_ChNb = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymax for plot 'pedestal' -> "
       << fKeyVmaxD_Ped_ChNb << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVminD_TNo_ChNb()
{
//Registration of Ymin for mean sample sigmas (noise)

  const char *bufferchain = fVminD_TNo_ChNbText->GetBuffer()->GetString();

  fKeyVminD_TNo_ChNb = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymin for plot 'total noise' -> "
       << fKeyVminD_TNo_ChNb << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVmaxD_TNo_ChNb()
{
//Registration of Ymax for mean sample sigmas (noise)

  const char *bufferchain = fVmaxD_TNo_ChNbText->GetBuffer()->GetString();

  fKeyVmaxD_TNo_ChNb = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymax for plot 'total noise' -> "
       << fKeyVmaxD_TNo_ChNb << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVminD_MCs_ChNb()
{
//Registration of Ymin for mean cor(s,s')

  const char *bufferchain = fVminD_MCs_ChNbText->GetBuffer()->GetString();

  fKeyVminD_MCs_ChNb = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymin for plot 'mean cor(s,s')' -> "
       << fKeyVminD_MCs_ChNb << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVmaxD_MCs_ChNb()
{
//Registration of Ymax for mean cor(s,s')

  const char *bufferchain = fVmaxD_MCs_ChNbText->GetBuffer()->GetString();

  fKeyVmaxD_MCs_ChNb = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymax for plot 'mean cor(s,s')' -> "
       << fKeyVmaxD_MCs_ChNb << endl;
}
//-------------------------------------------------------------------


void TEcnaGui::DoButtonVminD_LFN_ChNb()
{
//Registration of Ymin for sigmas of sample means

  const char *bufferchain = fVminD_LFN_ChNbText->GetBuffer()->GetString();

  fKeyVminD_LFN_ChNb = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymin for plot 'low frequency noise' -> "
       << fKeyVminD_LFN_ChNb << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVmaxD_LFN_ChNb()
{
//Registration of Ymax for sigmas of sample means 

  const char *bufferchain = fVmaxD_LFN_ChNbText->GetBuffer()->GetString();

  fKeyVmaxD_LFN_ChNb = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymax for plot 'low frequency noise' -> "
       << fKeyVmaxD_LFN_ChNb << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVminD_HFN_ChNb()
{
//Registration of Ymin for sigmas of sample sigmas

  const char *bufferchain = fVminD_HFN_ChNbText->GetBuffer()->GetString();

  fKeyVminD_HFN_ChNb = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymin for plot 'high frequency noise' -> "
       << fKeyVminD_HFN_ChNb << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVmaxD_HFN_ChNb()
{
//Registration of Ymax for sigmas of sample sigmas

  const char *bufferchain = fVmaxD_HFN_ChNbText->GetBuffer()->GetString();

  fKeyVmaxD_HFN_ChNb = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymax for plot 'high frequency noise' -> "
       << fKeyVmaxD_HFN_ChNb << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVminD_SCs_ChNb()
{
//Registration of Ymin for sigmas of cor(s,s')

  const char *bufferchain = fVminD_SCs_ChNbText->GetBuffer()->GetString();

  fKeyVminD_SCs_ChNb = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymin for plot 'sigma of cor(s,s')' -> "
       << fKeyVminD_SCs_ChNb << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVmaxD_SCs_ChNb()
{
//Registration of Ymax for sigmas of cor(s,s')

  const char *bufferchain = (char*)fVmaxD_SCs_ChNbText->GetBuffer()->GetString();

  fKeyVmaxD_SCs_ChNb = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymax for plot 'sigma of cor(s,s')' -> "
       << fKeyVmaxD_SCs_ChNb << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVminLFccMos()
{
//Registration of Ymin for LF Mean Cor(c,c')

  const char *bufferchain = fVminLFccMosText->GetBuffer()->GetString();

  fKeyVminLFccMos = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymin for plot 'mean LF |cor(c,c')|' -> "
       << fKeyVminLFccMos << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVmaxLFccMos()
{
//Registration of Ymax for LF Mean Cor(c,c')

  const char *bufferchain = fVmaxLFccMosText->GetBuffer()->GetString();

  fKeyVmaxLFccMos = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymax for plot 'mean LF |cor(c,c')|' -> "
       << fKeyVmaxLFccMos << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVminHFccMos()
{
//Registration of Ymin for HF Mean Cor(c,c')

  const char *bufferchain = fVminHFccMosText->GetBuffer()->GetString();

  fKeyVminHFccMos = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymin for plot 'mean HF |cor(c,c')|' -> "
       << fKeyVminHFccMos << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVmaxHFccMos()
{
//Registration of Ymax for HF Mean Cor(c,c')

  const char *bufferchain = fVmaxHFccMosText->GetBuffer()->GetString();

  fKeyVmaxHFccMos = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymax for plot 'mean HF |cor(c,c')|' -> "
       << fKeyVmaxHFccMos << endl;
}

//-------------------------------------------------------------------

void TEcnaGui::DoButtonVminLHFcc()
{
//Registration of Ymin for cov(c,c') in Stins

  const char *bufferchain = fVminLHFccText->GetBuffer()->GetString();

  fKeyVminLHFcc = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymin for plot 'cor(c,c') in "
       << fStinName.Data() << "s' -> " << fKeyVminLHFcc << endl;
}
//-------------------------------------------------------------------

void TEcnaGui::DoButtonVmaxLHFcc()
{
//Registration of Ymax for cov(c,c') in Stins

  const char *bufferchain = fVmaxLHFccText->GetBuffer()->GetString();

  fKeyVmaxLHFcc = (Double_t)atof(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of Ymax for plot 'cor(c,c') in "
       << fStinName.Data() << "s' -> " << fKeyVmaxLHFcc << endl;
}

//-------------------------------------------------------------------

void TEcnaGui::DoButtonStexNb()
{
  ViewStexStinNumbering();  // message in the method
}
//-------------------------------------------------------------------
void TEcnaGui::DoButtonStinA()
{
//Registration of the Stin A number (A = X coordinate for cor(c,c') plots)

  const char *bufferchain = (char*)fStinAText->GetBuffer()->GetString();

  Int_t xReadStinANumberForCons = atoi(bufferchain);

  if( fSubDet == "EB" ){fKeyStinANumber = xReadStinANumberForCons;}
  if( fSubDet == "EE" )
    {fKeyStinANumber = fEcalNumbering->Get1DeeSCEcnaFromDeeSCCons(fKeyStexNumber, xReadStinANumberForCons);}

  if( fSubDet == "EB" )
    {
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Registration of " << fStinName.Data() << " number -> "
	   << xReadStinANumberForCons << endl;

      if ( (fKeyStinANumber < 1) || (fKeyStinANumber > fEcal->MaxStinEcnaInStex())  )
	{
	  fCnaError++;
	  cout << "   !TEcnaGui (" << fCnaError << ") *** ERROR *** ===> " << fStinName.Data()
	       << " number = " << fKeyStinANumber
	       << ": out of range ( range = [ 1 ," << fEcal->MaxStinEcnaInStex() << " ] ) "
	       << fTTBELL << endl;
	}
    }
  
  if( fSubDet == "EE" )
    {
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Registration of " << fStinName.Data() << " number for construction -> "
	   << xReadStinANumberForCons << endl;

      if( fKeyStexNumber > 0 && fKeyStexNumber <= fEcal->MaxDeeInEE() )
	{
	  Int_t off_set_cons = 0;
	  if( fKeyStexNumber == 1 || fKeyStexNumber == 3 ){off_set_cons = fEcal->MaxSCForConsInDee();}
	  
	  if( xReadStinANumberForCons <= off_set_cons ||
	      xReadStinANumberForCons > fEcal->MaxSCForConsInDee()+off_set_cons )
	    {
	      fCnaError++;
	      cout << "   !TEcnaGui (" << fCnaError << ") *** ERROR *** ===> SC nb for construction = "
		   << xReadStinANumberForCons << ". Out of range ( range = [ " << off_set_cons+1
		   << "," << fEcal->MaxSCForConsInDee()+off_set_cons << "] )"
		   << fTTBELL << endl;
	    }
	}
      else
	{
	  fCnaError++;
	  cout << "   !TEcnaGui (" << fCnaError << ") *** ERROR *** ===> DeeNumber = " <<  fKeyStexNumber
	       << ". Out of range ( range = [1," << fEcal->MaxDeeInEE() << "] )"
	       << fTTBELL << endl;
	}      
    }
}

//-------------------------------------------------------------------
void TEcnaGui::DoButtonStinB()
{
//Registration of the Stin B number (B = Y coordinate for cor(c,c') plots)

  const char *bufferchain = fStinBText->GetBuffer()->GetString();

  Int_t xReadStinBNumberForCons = atoi(bufferchain);

  if( fSubDet == "EB" ){fKeyStinBNumber = xReadStinBNumberForCons;}
  if( fSubDet == "EE" )
    {fKeyStinBNumber = fEcalNumbering->Get1DeeSCEcnaFromDeeSCCons(fKeyStexNumber, xReadStinBNumberForCons);}

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of " << fStinName.Data() << "' number -> "
       << xReadStinBNumberForCons << endl;

  if( fSubDet == "EB" )
    {
      if ( (fKeyStinBNumber < 1) || (fKeyStinBNumber > fEcal->MaxStinEcnaInStex())  )
	{
	  fCnaError++;
	  cout << "   !TEcnaGui (" << fCnaError << ") *** ERROR *** ===> " << fStinName.Data()
	       << "' number = " << fKeyStinBNumber
	       << ": out of range ( range = [ 1 ," << fEcal->MaxStinEcnaInStex() << " ] ) "
	       << fTTBELL << endl;
	}
    }
  
  if( fSubDet == "EE" )
    {
      if( fKeyStexNumber > 0 && fKeyStexNumber <= fEcal->MaxDeeInEE() )
	{
	  Int_t off_set_cons = 0;
	  if( fKeyStexNumber == 1 || fKeyStexNumber == 3 ){off_set_cons = fEcal->MaxSCForConsInDee();}
	  
	  if( xReadStinBNumberForCons < off_set_cons ||
	      xReadStinBNumberForCons > fEcal->MaxSCForConsInDee()+off_set_cons )
	    {
	      fCnaError++;
	      cout << "   !TEcnaGui (" << fCnaError << ") *** ERROR *** ===> SC nb for construction = "
		   << xReadStinBNumberForCons << ". Out of range ( range = [ " << off_set_cons+1
		   << "," << fEcal->MaxSCForConsInDee()+off_set_cons << "] )"
		   << fTTBELL << endl;
	    }
	}
      else
	{
	  fCnaError++;
	  cout << "   !TEcnaGui (" << fCnaError << ") *** ERROR *** ===> DeeNumber = " <<  fKeyStexNumber
	       << ". Out of range ( range = [1," << fEcal->MaxDeeInEE() << "] )"
	       << fTTBELL << endl;
	}      
    }
}
//-------------------------------------------------------------------
void TEcnaGui::DoButtonChNb()
{
//Display of StinA Channel numbering
  ViewStinCrystalNumbering(fKeyStinANumber);
}

//-------------------------------------------------------------------
void TEcnaGui::DoButtonChan()
{
//Registration of the channel number

  const char *bufferchain = fChanText->GetBuffer()->GetString();
  Int_t xReadNumber = atoi(bufferchain);

  Int_t Choffset = -1;
  TString ChString = "?";

  if( fSubDet == "EB"){Choffset = 0; ChString = "channel";}
  if( fSubDet == "EE"){Choffset = 1; ChString = "crystal";}

  fKeyChanNumber = xReadNumber-Choffset;   // fKeyChanNumber : range = [0,25]
  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of " << ChString.Data() << " number -> "
       << xReadNumber << endl;
  
  if ( (fKeyChanNumber < 0) || (fKeyChanNumber > fEcal->MaxCrysInStin()-1 )  )
    {
      fCnaError++;
      cout << "   !TEcnaGui (" << fCnaError << ") *** ERROR *** ===> "
	   << ChString.Data() << " number in " << fStinName.Data() << " = " << xReadNumber
	   << ": out of range ( range = [" << Choffset << ","
	   << fEcal->MaxCrysInStin()-1+Choffset << "] )"
	   << fTTBELL << endl;
    } 
}
//-------------------------------------------------------------------
void TEcnaGui::DoButtonSamp()
{
//Registration of the sample number

  const char *bufferchain = fSampText->GetBuffer()->GetString();
  Int_t xKeySampNumber = atoi(bufferchain);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of sample number -> "
       << xKeySampNumber << endl;

  if ( (xKeySampNumber < 1) || (xKeySampNumber > fKeyNbOfSamples )  )
    {
      fCnaError++;
      cout << "   !TEcnaGui (" << fCnaError << ") *** ERROR *** ===> "
     	   << " Sample number = " << xKeySampNumber
     	   << ": out of range ( range = [ 1 ," << fKeyNbOfSamples << " ] )"
     	   << fTTBELL << endl;
    }

  fKeySampNumber = xKeySampNumber-1;
}

//----------------------------------------------------------------------
void TEcnaGui::DoButtonRul()
{
//Register the name of the file containing the list of run parameters

  //........................... get info from the entry field
  char* listchain = (char*)fRulText->GetBuffer()->GetString();
  if( listchain[0] == '\0' )
    {
      fCnaError++;
      cout << "   !TEcnaGui (" << fCnaError << ") *ERROR* ===> "
	   << " Empty file name in entry for TIME EVOLUTION plots."
	   << fTTBELL << endl;
      fKeyFileNameRunList = fKeyRunListInitCode;
    }
  else
    {
      char tchiffr[10] = {'0', '1', '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9' };
      
      //............. test of the first character (figure => run number, letter => file name)
      if( listchain[0] == tchiffr [0] || listchain[0] == tchiffr [1] ||
	  listchain[0] == tchiffr [2] || listchain[0] == tchiffr [3] ||
	  listchain[0] == tchiffr [4] || listchain[0] == tchiffr [5] ||
	  listchain[0] == tchiffr [6] || listchain[0] == tchiffr [7] ||
	  listchain[0] == tchiffr [8] || listchain[0] == tchiffr [9] )
	{
	  fCnaError++;
	  cout << "   !TEcnaGui (" << fCnaError << ") *ERROR* ===> "
	       << " Please, enter a file name beginning with an alphabetic letter."
	       << fTTBELL << endl;
	}
      else
	{
	  fKeyFileNameRunList = listchain;
	  
	  fCnaCommand++;
	  cout << "   *TEcnaGui [" << fCnaCommand
	       << "]> Registration of run list file name for history plots -> "
	       << fKeyFileNameRunList.Data() << endl;
	}
    }
}

//----------------------------------------------------------------------
void TEcnaGui::DoButtonGent()
{
//Register the general title
  //........................... get info from the entry field
  char* listchain = (char*)fGentText->GetBuffer()->GetString();  
  fKeyGeneralTitle = listchain;
  
  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Registration of general title -> "
       << fKeyGeneralTitle.Data() << endl;
}
//-------------------------------------------------------------------
//
//                        Last buttons methods
//
//-------------------------------------------------------------------
//======================= LIN/LOG + Projy FRAME

void TEcnaGui::DoButtonLogx()
{
  if( fMemoScaleX == "LOG"){fKeyScaleX = "LIN";}
  if( fMemoScaleX == "LIN"){fKeyScaleX = "LOG";}
  fMemoScaleX = fKeyScaleX;

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> X axis -> " << fKeyScaleX << " scale " << endl;
}
void TEcnaGui::DoButtonLogy()
{
  if( fMemoScaleY == "LOG" ){fKeyScaleY = "LIN";}
  if( fMemoScaleY == "LIN" ){fKeyScaleY = "LOG";}
  fMemoScaleY = fKeyScaleY;

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Y axis -> " << fKeyScaleY << " scale " << endl;
}

void TEcnaGui::DoButtonProjy()
{
  if( fMemoProjY == "Y projection" ){fKeyProjY = "normal";}
  if( fMemoProjY == "normal"       ){fKeyProjY = "Y projection";}
  fMemoProjY = fKeyProjY;

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> 1D Histo display -> " << fKeyProjY << " mode " << endl;
}

//------------------------------------------------------------------- Colors + Exit

void TEcnaGui::DoButtonColPal()
{
  if( fMemoColPal == "ECCNAColor"   ){fKeyColPal = "Rainbow";}
  if( fMemoColPal == "Rainbow" ){fKeyColPal = "ECCNAColor";}
  fMemoColPal = fKeyColPal;

  TString sColPalComment = "?";
  if( fKeyColPal == "ECCNAColor" )
    {sColPalComment = "ECNAColor option: black-red-blue-green-brown-purple (default)";}
  if( fKeyColPal == "Rainbow"    )
    {sColPalComment = "Rainbow option:   red-orange-yellow-green-blue-indigo-purple";}

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Color palette -> " << sColPalComment << endl;
}

void TEcnaGui::DoButtonExit()
{
  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Exit CNA session."
       << endl;
  //............................ Quit the ROOT session
  fButExit->SetCommand(".q");
}

//======================= LAST FRAME
//-------------------------------------------------------------------
void TEcnaGui::DoButtonClone()
{
  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Clone last canvas. " << endl;

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());      /*fCnew++*/ ;}
  fHistos->PlotCloneOfCurrentCanvas();
}
//-------------------------------------------------------------------
void TEcnaGui::DoButtonRoot()
{
  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> This is ROOT version " << gROOT->GetVersion()
       << endl;
}
//-------------------------------------------------------------------
void TEcnaGui::DoButtonHelp()
{
  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> HELP: for documentation, see the ECNA web page: " << endl
       << "    http://cms-fabbro.web.cern.ch/cms-fabbro/cna_new/Correlated_Noise_analysis/ECNA_main_page.htm"
       << endl;
}

//===================================================================
//
//                       HandleMenu
//
//===================================================================
void TEcnaGui::HandleMenu(Int_t id)
{
  //HandleMenu
  //.................... SUBMIT on batch system

  if( id == fMenuSubmit8nmC ){SubmitOnBatchSystem("8nm");}
  if( id == fMenuSubmit1nhC ){SubmitOnBatchSystem("1nh");}
  if( id == fMenuSubmit8nhC ){SubmitOnBatchSystem("8nh");}
  if( id == fMenuSubmit1ndC ){SubmitOnBatchSystem("1nd");}
  if( id == fMenuSubmit1nwC ){SubmitOnBatchSystem("1nw");}

  //.................... Clean
  if( id == fMenuCleanSubC  ){CleanBatchFiles("Sub");}
  if( id == fMenuCleanJobC  ){CleanBatchFiles("Job");}
  if( id == fMenuCleanPythC ){CleanBatchFiles("Pyth");}
  if( id == fMenuCleanAllC  ){CleanBatchFiles("All");}

  //.................... Calculations
  if( id == fMenuComputStdC ){Calculations("Std");}
  if( id == fMenuComputSttC ){Calculations("Stt");}
  if( id == fMenuComputSccC ){Calculations("Scc");}

  //.................... Nb of events in Stex
  if( id == fMenuD_NOE_ChNbFullC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSNumberOfEventsOfCrystals(fOptPlotFull);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSNumberOfEventsDistribution(fOptPlotFull);}
    }
  if( id == fMenuD_NOE_ChNbSameC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSNumberOfEventsOfCrystals(fOptPlotSame);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSNumberOfEventsDistribution(fOptPlotSame);}
    }
  if( id == fMenuD_NOE_ChNbHocoVecoC   ){ViewSorSNumberOfEvents();}
  if( id == fMenuD_NOE_ChNbAsciiFileC  ){ViewHistoSorSNumberOfEventsOfCrystals(fOptAscii);}

  //.................... Pedestal in Stex                  (HandleMenu)
  if( id == fMenuD_Ped_ChNbFullC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSPedestalsOfCrystals(fOptPlotFull);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSPedestalsDistribution(fOptPlotFull);}
    }
  if( id == fMenuD_Ped_ChNbSameC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSPedestalsOfCrystals(fOptPlotSame);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSPedestalsDistribution(fOptPlotSame);}
    }
  if( id == fMenuD_Ped_ChNbHocoVecoC   ){ViewSorSPedestals();}
  if( id == fMenuD_Ped_ChNbAsciiFileC  ){ViewHistoSorSPedestalsOfCrystals(fOptAscii);}

  //.................... Total noise in Stex                 (HandleMenu)
  if( id == fMenuD_TNo_ChNbFullC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSTotalNoiseOfCrystals(fOptPlotFull);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSTotalNoiseDistribution(fOptPlotFull);}
    }
  if( id == fMenuD_TNo_ChNbSameC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSTotalNoiseOfCrystals(fOptPlotSame);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSTotalNoiseDistribution(fOptPlotSame);}
    }
  if( id == fMenuD_TNo_ChNbSamePC)
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSTotalNoiseOfCrystals(fOptPlotSameP);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSTotalNoiseDistribution(fOptPlotSameP);}
    }
  if( id == fMenuD_TNo_ChNbHocoVecoC   ){ViewSorSTotalNoise();}
  if( id == fMenuD_TNo_ChNbAsciiFileC  ){ViewHistoSorSTotalNoiseOfCrystals(fOptAscii);}

  //.................... Low Frequency noise in Stex                 (HandleMenu)
  if( id == fMenuD_LFN_ChNbFullC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSLowFrequencyNoiseOfCrystals(fOptPlotFull);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSLowFrequencyNoiseDistribution(fOptPlotFull);}
    }
  if( id == fMenuD_LFN_ChNbSameC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSLowFrequencyNoiseOfCrystals(fOptPlotSame);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSLowFrequencyNoiseDistribution(fOptPlotSame);}
    }
  if( id == fMenuD_LFN_ChNbSamePC)
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSLowFrequencyNoiseOfCrystals(fOptPlotSameP);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSLowFrequencyNoiseDistribution(fOptPlotSameP);}
    }
  if( id == fMenuD_LFN_ChNbHocoVecoC   ){ViewSorSLowFrequencyNoise();}
  if( id == fMenuD_LFN_ChNbAsciiFileC  ){ViewHistoSorSLowFrequencyNoiseOfCrystals(fOptAscii);}

  //.................... High Frequency noise in Stex                 (HandleMenu)
  if( id == fMenuD_HFN_ChNbFullC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSHighFrequencyNoiseOfCrystals(fOptPlotFull);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSHighFrequencyNoiseDistribution(fOptPlotFull);}
    }
  if( id == fMenuD_HFN_ChNbSameC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSHighFrequencyNoiseOfCrystals(fOptPlotSame);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSHighFrequencyNoiseDistribution(fOptPlotSame);}
    }
  if( id == fMenuD_HFN_ChNbSamePC)
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSHighFrequencyNoiseOfCrystals(fOptPlotSameP);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSHighFrequencyNoiseDistribution(fOptPlotSameP);}
    }
  if( id == fMenuD_HFN_ChNbHocoVecoC   ){ViewSorSHighFrequencyNoise();}
  if( id == fMenuD_HFN_ChNbAsciiFileC  ){ViewHistoSorSHighFrequencyNoiseOfCrystals(fOptAscii);}

  //.................... Mean Corss in Stex                 (HandleMenu)
  if( id == fMenuD_MCs_ChNbFullC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSMeanCorssOfCrystals(fOptPlotFull);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSMeanCorssDistribution(fOptPlotFull);}
    }
  if( id == fMenuD_MCs_ChNbSameC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSMeanCorssOfCrystals(fOptPlotSame);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSMeanCorssDistribution(fOptPlotSame);}
    }
  if( id == fMenuD_MCs_ChNbSamePC)
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSMeanCorssOfCrystals(fOptPlotSameP);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSMeanCorssDistribution(fOptPlotSameP);}
    }
  if( id == fMenuD_MCs_ChNbHocoVecoC   ){ViewSorSMeanCorss();}
  if( id == fMenuD_MCs_ChNbAsciiFileC  ){ViewHistoSorSMeanCorssOfCrystals(fOptAscii);}

  //.................... Sigma of Corss in the Stex                 (HandleMenu)
  if( id == fMenuD_SCs_ChNbFullC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSSigmaOfCorssOfCrystals(fOptPlotFull);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSSigmaOfCorssDistribution(fOptPlotFull);}
    }
  if( id == fMenuD_SCs_ChNbSameC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSSigmaOfCorssOfCrystals(fOptPlotSame);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSSigmaOfCorssDistribution(fOptPlotSame);}
    }
  if( id == fMenuD_SCs_ChNbSamePC)
    {
      if( fMemoProjY == "normal"      ){ViewHistoSorSSigmaOfCorssOfCrystals(fOptPlotSameP);}
      if( fMemoProjY == "Y projection"){ViewHistoSorSSigmaOfCorssDistribution(fOptPlotSameP);}
    }
  if( id == fMenuD_SCs_ChNbHocoVecoC   ){ViewSorSSigmaOfCorss();}
  if( id == fMenuD_SCs_ChNbAsciiFileC  ){ViewHistoSorSSigmaOfCorssOfCrystals(fOptAscii);}

  //............................... Low Freq Mean Cor(c,c') for each pair of Stins                 (HandleMenu)
  if( id == fMenuLFccMosColzC ){ViewMatrixLowFrequencyMeanCorrelationsBetweenStins("COLZ");}
  if( id == fMenuLFccMosLegoC ){ViewMatrixLowFrequencyMeanCorrelationsBetweenStins("LEGO2Z");}
  //............................... High Freq Mean Cor(c,c') for each pair of Stins
  if( id == fMenuHFccMosColzC ){ViewMatrixHighFrequencyMeanCorrelationsBetweenStins("COLZ");}
  if( id == fMenuHFccMosLegoC ){ViewMatrixHighFrequencyMeanCorrelationsBetweenStins("LEGO2Z");}

  //............................... Corcc for each Stin in the Stex
  if( id == fMenuLFccColzC ){ViewStexLowFrequencyCorcc();}
  if( id == fMenuHFccColzC ){ViewStexHighFrequencyCorcc();}

  //--------> Nb for Cons for Stin numbers in case of EE
  Int_t cKeyStinANumber = fKeyStinANumber;
  if( fSubDet == "EE" && fKeyStexNumber != 0 )
    {cKeyStinANumber = fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fKeyStexNumber, fKeyStinANumber);}
  Int_t cKeyStinBNumber = fKeyStinBNumber;
  if( fSubDet == "EE" && fKeyStexNumber != 0 )
    {cKeyStinBNumber = fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fKeyStexNumber, fKeyStinBNumber);}

  //............................... Low Frequency Correlations and covariances between channels
  if( id == fMenuLFCorccColzC )
    {ViewMatrixLowFrequencyCorrelationsBetweenChannels(cKeyStinANumber, cKeyStinBNumber, "COLZ");}
  if( id == fMenuLFCorccLegoC )
    {ViewMatrixLowFrequencyCorrelationsBetweenChannels(cKeyStinANumber, cKeyStinBNumber, "LEGO2Z");}

  if( id == fMenuHFCorccColzC )
    {ViewMatrixHighFrequencyCorrelationsBetweenChannels(cKeyStinANumber, cKeyStinBNumber, "COLZ");}
  if( id == fMenuHFCorccLegoC )
    {ViewMatrixHighFrequencyCorrelationsBetweenChannels(cKeyStinANumber, cKeyStinBNumber, "LEGO2Z");}

  //.................................... Correlations and covariances between samples     (HandleMenu)
  if( id == fMenuCorssColzC      ){ViewMatrixCorrelationSamples(cKeyStinANumber, fKeyChanNumber, "COLZ");}
  if( id == fMenuCorssBoxC       ){ViewMatrixCorrelationSamples(cKeyStinANumber, fKeyChanNumber, "BOX");}
  if( id == fMenuCorssTextC      ){ViewMatrixCorrelationSamples(cKeyStinANumber, fKeyChanNumber, "TEXT");}
  if( id == fMenuCorssContzC     ){ViewMatrixCorrelationSamples(cKeyStinANumber, fKeyChanNumber, "CONTZ");}
  if( id == fMenuCorssLegoC      ){ViewMatrixCorrelationSamples(cKeyStinANumber, fKeyChanNumber, "LEGO2Z");}
  if( id == fMenuCorssSurf1C     ){ViewMatrixCorrelationSamples(cKeyStinANumber, fKeyChanNumber, "SURF1Z");}
  if( id == fMenuCorssSurf2C     ){ViewMatrixCorrelationSamples(cKeyStinANumber, fKeyChanNumber, "SURF2Z");}
  if( id == fMenuCorssSurf3C     ){ViewMatrixCorrelationSamples(cKeyStinANumber, fKeyChanNumber, "SURF3Z");}
  if( id == fMenuCorssSurf4C     ){ViewMatrixCorrelationSamples(cKeyStinANumber, fKeyChanNumber, "SURF4");}
  if( id == fMenuCorssAsciiFileC ){ViewMatrixCorrelationSamples(cKeyStinANumber, fKeyChanNumber, "ASCII");}

  if( id == fMenuCovssColzC      ){ViewMatrixCovarianceSamples(cKeyStinANumber, fKeyChanNumber, "COLZ");}
  if( id == fMenuCovssBoxC       ){ViewMatrixCovarianceSamples(cKeyStinANumber, fKeyChanNumber, "BOX");}
  if( id == fMenuCovssTextC      ){ViewMatrixCovarianceSamples(cKeyStinANumber, fKeyChanNumber, "TEXT");}
  if( id == fMenuCovssContzC     ){ViewMatrixCovarianceSamples(cKeyStinANumber, fKeyChanNumber, "CONTZ");}
  if( id == fMenuCovssLegoC      ){ViewMatrixCovarianceSamples(cKeyStinANumber, fKeyChanNumber, "LEGO2Z");}
  if( id == fMenuCovssSurf1C     ){ViewMatrixCovarianceSamples(cKeyStinANumber, fKeyChanNumber, "SURF1Z");}
  if( id == fMenuCovssSurf2C     ){ViewMatrixCovarianceSamples(cKeyStinANumber, fKeyChanNumber, "SURF2Z");}
  if( id == fMenuCovssSurf3C     ){ViewMatrixCovarianceSamples(cKeyStinANumber, fKeyChanNumber, "SURF3Z");}
  if( id == fMenuCovssSurf4C     ){ViewMatrixCovarianceSamples(cKeyStinANumber, fKeyChanNumber, "SURF4");}
  if( id == fMenuCovssAsciiFileC ){ViewMatrixCovarianceSamples(cKeyStinANumber, fKeyChanNumber, "ASCII");}

  //.................... Correlations and covariances between samples for all channels of a Stin
  if( id == fMenuCorssAllColzC ){ViewStinCorrelationSamples(cKeyStinANumber);}
  if( id == fMenuCovssAllColzC ){ViewStinCovarianceSamples(cKeyStinANumber);}
     
  //..................................... Sample means (pedestals)       (HandleMenu) 
  if( id == fMenuD_MSp_SpNbLineFullC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoCrystalSampleMeans(cKeyStinANumber, fKeyChanNumber, fOptPlotFull);}
      if( fMemoProjY == "Y projection"){ViewHistoCrystalSampleMeansDistribution(cKeyStinANumber, fKeyChanNumber, fOptPlotFull);}
    }

  if( id == fMenuD_MSp_SpNbLineSameC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoCrystalSampleMeans(cKeyStinANumber, fKeyChanNumber, fOptPlotSame);}
      if( fMemoProjY == "Y projection"){ViewHistoCrystalSampleMeansDistribution(cKeyStinANumber, fKeyChanNumber, fOptPlotSame);}
    }
  if( id == fMenuD_MSp_SpNbLineAllStinC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoCrystalSampleMeans(cKeyStinANumber, fKeyChanNumber, fOptPlotSameInStin);}
      if( fMemoProjY == "Y projection"){ViewHistoCrystalSampleMeansDistribution(cKeyStinANumber, fKeyChanNumber, fOptPlotSameInStin);}
    }
    
  //..................................... Sample sigmas
  if( id == fMenuD_SSp_SpNbLineFullC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoCrystalSampleSigmas(cKeyStinANumber, fKeyChanNumber, fOptPlotFull);}
      if( fMemoProjY == "Y projection"){ViewHistoCrystalSampleSigmasDistribution(cKeyStinANumber, fKeyChanNumber, fOptPlotFull);}
    }
  if( id == fMenuD_SSp_SpNbLineSameC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoCrystalSampleSigmas(cKeyStinANumber, fKeyChanNumber, fOptPlotSame);}
      if( fMemoProjY == "Y projection"){ViewHistoCrystalSampleSigmasDistribution(cKeyStinANumber, fKeyChanNumber, fOptPlotSame);}
    }
   if( id == fMenuD_SSp_SpNbLineAllStinC )
    {
      if( fMemoProjY == "normal"      ){ViewHistoCrystalSampleSigmas(cKeyStinANumber, fKeyChanNumber, fOptPlotSameInStin);}
      if( fMemoProjY == "Y projection"){ViewHistoCrystalSampleSigmasDistribution(cKeyStinANumber, fKeyChanNumber, fOptPlotSameInStin);}
    }
   
  //..................................... Evolution in time (ViewHistime, except EvolSamp -> Viewhisto)
  if(   id == fMenuH_Ped_DatePolmFullC || id == fMenuH_Ped_DatePolmSameC  
     || id == fMenuH_TNo_DatePolmFullC || id == fMenuH_TNo_DatePolmSameC || id == fMenuH_TNo_DatePolmSamePC
     || id == fMenuH_LFN_DatePolmFullC || id == fMenuH_LFN_DatePolmSameC || id == fMenuH_LFN_DatePolmSamePC
     || id == fMenuH_HFN_DatePolmFullC || id == fMenuH_HFN_DatePolmSameC || id == fMenuH_HFN_DatePolmSamePC
     || id == fMenuH_MCs_DatePolmFullC || id == fMenuH_MCs_DatePolmSameC || id == fMenuH_MCs_DatePolmSamePC
     || id == fMenuH_SCs_DatePolmFullC || id == fMenuH_SCs_DatePolmSameC || id == fMenuH_SCs_DatePolmSamePC)
    {
      if(fKeyFileNameRunList == fKeyRunListInitCode )
	{fCnaError++;
	  cout << "   !TEcnaGui (" << fCnaError << ") *** ERROR *** ===> "
	       << " EMPTY STRING for list of run file name (TIME EVOLUTION plots)." << fTTBELL << endl;}
      else
	{
	  //........................................ Pedestals                 (HandleMenu / ViewHistime)
	  if( id == fMenuH_Ped_DatePolmFullC )
	    {
	      if( fMemoProjY == "normal"      )
		{ViewHistimeCrystalPedestals(fKeyFileNameRunList.Data(),
					     cKeyStinANumber, fKeyChanNumber, fOptPlotFull);}
	      if( fMemoProjY == "Y projection")
		{ViewHistimeCrystalPedestalsRuns(fKeyFileNameRunList.Data(),
						 cKeyStinANumber, fKeyChanNumber, fOptPlotFull);}
	    }
	  if( id == fMenuH_Ped_DatePolmSameC )
	    { if( fMemoProjY == "normal"      )
	      {ViewHistimeCrystalPedestals(fKeyFileNameRunList.Data(),
					   cKeyStinANumber, fKeyChanNumber, fOptPlotSame);}
	    if( fMemoProjY == "Y projection")
	      {ViewHistimeCrystalPedestalsRuns(fKeyFileNameRunList.Data(),
					       cKeyStinANumber, fKeyChanNumber, fOptPlotSame);}
	    }

	  //........................................ Total noise
	  if( id == fMenuH_TNo_DatePolmFullC )
	    {
	      if( fMemoProjY == "normal"      )
		{ViewHistimeCrystalTotalNoise(fKeyFileNameRunList.Data(),
					      cKeyStinANumber, fKeyChanNumber, fOptPlotFull);}
	      if( fMemoProjY == "Y projection")
		{ViewHistimeCrystalTotalNoiseRuns(fKeyFileNameRunList.Data(),
						  cKeyStinANumber, fKeyChanNumber, fOptPlotFull);}
	    }
	  if( id == fMenuH_TNo_DatePolmSameC )
	    {
	      if( fMemoProjY == "normal"      )
		{ViewHistimeCrystalTotalNoise(fKeyFileNameRunList.Data(),
					      cKeyStinANumber, fKeyChanNumber, fOptPlotSame);}
	      if( fMemoProjY == "Y projection")
		{ViewHistimeCrystalTotalNoiseRuns(fKeyFileNameRunList.Data(),
						  cKeyStinANumber, fKeyChanNumber, fOptPlotSame);}
	    }
	  if( id == fMenuH_TNo_DatePolmSamePC )
	    {
	      if( fMemoProjY == "normal"      )
		{ViewHistimeCrystalTotalNoise(fKeyFileNameRunList.Data(),
					      cKeyStinANumber, fKeyChanNumber, fOptPlotSameP);}
	      if( fMemoProjY == "Y projection")
		{ViewHistimeCrystalTotalNoiseRuns(fKeyFileNameRunList.Data(),
						  cKeyStinANumber, fKeyChanNumber, fOptPlotSameP);}
	    }

	  //........................................ Low frequency noise                 (HandleMenu / ViewHistime)
	  if( id == fMenuH_LFN_DatePolmFullC )
	    {
	      if( fMemoProjY == "normal"      )
		{ViewHistimeCrystalLowFrequencyNoise(fKeyFileNameRunList.Data(),
						     cKeyStinANumber, fKeyChanNumber, fOptPlotFull);}
	      if( fMemoProjY == "Y projection")
		{ViewHistimeCrystalLowFrequencyNoiseRuns(fKeyFileNameRunList.Data(),
							 cKeyStinANumber, fKeyChanNumber, fOptPlotFull);}
	    }
	  if( id == fMenuH_LFN_DatePolmSameC )
	    {
	      if( fMemoProjY == "normal"      )
		{ViewHistimeCrystalLowFrequencyNoise(fKeyFileNameRunList.Data(),
						     cKeyStinANumber, fKeyChanNumber, fOptPlotSame);}
	      if( fMemoProjY == "Y projection")
		{ViewHistimeCrystalLowFrequencyNoiseRuns(fKeyFileNameRunList.Data(),
							 cKeyStinANumber, fKeyChanNumber, fOptPlotSame);}
	    }
	  if( id == fMenuH_LFN_DatePolmSamePC )
	    {
	      if( fMemoProjY == "normal"      )
		{ViewHistimeCrystalLowFrequencyNoise(fKeyFileNameRunList.Data(),
						     cKeyStinANumber, fKeyChanNumber, fOptPlotSameP);}
	      if( fMemoProjY == "Y projection")
		{ViewHistimeCrystalLowFrequencyNoiseRuns(fKeyFileNameRunList.Data(),
							 cKeyStinANumber, fKeyChanNumber, fOptPlotSameP);}
	    }
	  
	  //........................................ High frequency noise
	  if( id == fMenuH_HFN_DatePolmFullC )
	    {
	      if( fMemoProjY == "normal"      )
		{ViewHistimeCrystalHighFrequencyNoise(fKeyFileNameRunList.Data(),
						      cKeyStinANumber, fKeyChanNumber, fOptPlotFull);}
	      if( fMemoProjY == "Y projection")
		{ViewHistimeCrystalHighFrequencyNoiseRuns(fKeyFileNameRunList.Data(),
							  cKeyStinANumber, fKeyChanNumber, fOptPlotFull);}
	    }
	  if( id == fMenuH_HFN_DatePolmSameC )
	    {
	      if( fMemoProjY == "normal"      )
		{ViewHistimeCrystalHighFrequencyNoise(fKeyFileNameRunList.Data(),
						      cKeyStinANumber, fKeyChanNumber, fOptPlotSame);}
	      if( fMemoProjY == "Y projection")
		{ViewHistimeCrystalHighFrequencyNoiseRuns(fKeyFileNameRunList.Data(),
							  cKeyStinANumber, fKeyChanNumber, fOptPlotSame);}
	    }
	  if( id == fMenuH_HFN_DatePolmSamePC )
	    {
	      if( fMemoProjY == "normal"      )
		{ViewHistimeCrystalHighFrequencyNoise(fKeyFileNameRunList.Data(),
						      cKeyStinANumber, fKeyChanNumber, fOptPlotSameP);}
	      if( fMemoProjY == "Y projection")
		{ViewHistimeCrystalHighFrequencyNoiseRuns(fKeyFileNameRunList.Data(),
							  cKeyStinANumber, fKeyChanNumber, fOptPlotSameP);}
	    }
	  
	  //........................................ Mean Corss                 (HandleMenu / ViewHistime)
	  if( id == fMenuH_MCs_DatePolmFullC )
	    {
	      if( fMemoProjY == "normal"      )
		{ViewHistimeCrystalMeanCorss(fKeyFileNameRunList.Data(),
					       cKeyStinANumber, fKeyChanNumber, fOptPlotFull);}
	      if( fMemoProjY == "Y projection")
		{ViewHistimeCrystalMeanCorssRuns(fKeyFileNameRunList.Data(),
						   cKeyStinANumber, fKeyChanNumber, fOptPlotFull);}
	    }
	  if( id == fMenuH_MCs_DatePolmSameC )
	    {
	      if( fMemoProjY == "normal"      )
		{ViewHistimeCrystalMeanCorss(fKeyFileNameRunList.Data(),
					       cKeyStinANumber, fKeyChanNumber, fOptPlotSame);}
	      if( fMemoProjY == "Y projection")
		{ViewHistimeCrystalMeanCorssRuns(fKeyFileNameRunList.Data(),
						   cKeyStinANumber, fKeyChanNumber, fOptPlotSame);}
	    }
	  if( id == fMenuH_MCs_DatePolmSamePC )
	    {
	      if( fMemoProjY == "normal"      )
		{ViewHistimeCrystalMeanCorss(fKeyFileNameRunList.Data(),
					       cKeyStinANumber, fKeyChanNumber, fOptPlotSameP);}
	      if( fMemoProjY == "Y projection")
		{ViewHistimeCrystalMeanCorssRuns(fKeyFileNameRunList.Data(),
						   cKeyStinANumber, fKeyChanNumber, fOptPlotSameP);}
	    }
	  
	  //........................................ Sigmas of Corss
	  if( id == fMenuH_SCs_DatePolmFullC )
	    {
	      if( fMemoProjY == "normal"      )
		{ViewHistimeCrystalSigmaOfCorss(fKeyFileNameRunList.Data(),
						cKeyStinANumber, fKeyChanNumber, fOptPlotFull);}
	      if( fMemoProjY == "Y projection")
		{ViewHistimeCrystalSigmaOfCorssRuns(fKeyFileNameRunList.Data(),
						    cKeyStinANumber, fKeyChanNumber, fOptPlotFull);}
	    }
	  if( id == fMenuH_SCs_DatePolmSameC )
	    {
	      if( fMemoProjY == "normal"      )
		{ViewHistimeCrystalSigmaOfCorss(fKeyFileNameRunList.Data(),
						cKeyStinANumber, fKeyChanNumber, fOptPlotSame);}
	      if( fMemoProjY == "Y projection")
		{ViewHistimeCrystalSigmaOfCorssRuns(fKeyFileNameRunList.Data(),
						    cKeyStinANumber, fKeyChanNumber, fOptPlotSame);}
	    }
	  if( id == fMenuH_SCs_DatePolmSamePC )
	    {
	      if( fMemoProjY == "normal"      )
		{ViewHistimeCrystalSigmaOfCorss(fKeyFileNameRunList.Data(),
						cKeyStinANumber, fKeyChanNumber, fOptPlotSameP);}
	      if( fMemoProjY == "Y projection")
		{ViewHistimeCrystalSigmaOfCorssRuns(fKeyFileNameRunList.Data(),
						    cKeyStinANumber, fKeyChanNumber, fOptPlotSameP);}
	    }
	}
    }

  //...................................... SampTime                               (HandleMenu / ADC)
  if( id == fMenuAdcProjSampLineFullC )
    {
      if( fMemoProjY == "normal"      )
	{ViewHistoCrystalSampleValues(cKeyStinANumber, fKeyChanNumber, fKeySampNumber, fOptPlotFull);}
      if( fMemoProjY == "Y projection")
	{ViewHistoSampleEventDistribution(cKeyStinANumber, fKeyChanNumber, fKeySampNumber, fOptPlotFull);}
    }
  if( id == fMenuAdcProjSampLineSameC )
    {
      if( fMemoProjY == "normal"      )
	{ViewHistoCrystalSampleValues(cKeyStinANumber, fKeyChanNumber, fKeySampNumber, fOptPlotSame);}
      if( fMemoProjY == "Y projection")
	{ViewHistoSampleEventDistribution(cKeyStinANumber, fKeyChanNumber, fKeySampNumber, fOptPlotSame);}
    }
}
// ------------- ( end of HandleMenu(...) ) -------------

//==========================================================================
//
//             SubmitOnBatchSystem()   M E T H O D 
//
//==========================================================================
void TEcnaGui::SubmitOnBatchSystem(const TString& QueueCode)
{
  //Submit job in batch mode
  
  if( (fConfirmSubmit == 1) && (fConfirmRunNumber == fKeyRunNumber) )
    {
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Submitting job in batch mode for run " << fConfirmRunNumber << endl;

      //.......................... get the path "modules/data"
      // /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/src/CalibCalorimetry/EcalCorrelatedNoiseAnalysisModules/data/
      TString ModulesdataPath = fCnaParPaths->PathModulesData();

      //---------------------------------- python file building script: command text
      TString CnaPythonCommand = ModulesdataPath;

      //......................................... Script for python file building: script name
      TString PythonScriptName = "EcnaSystemScriptPython";
      const Text_t *t_PythonScriptName = (const Text_t *)PythonScriptName.Data();
      CnaPythonCommand.Append(t_PythonScriptName);

      //......................................... Script for python file building: arguments
      //  In the calling command, TString arguments must be of the form: \"STRING1\"  \"STRING2\"  etc...

      //......................................... arguments -> Run number
      //.......... ${1}
      CnaPythonCommand.Append(' ');
      CnaPythonCommand.Append(fConfirmRunNumberString);

      //......................................... arguments -> Analyzer parameters
      //.......... ${2}
      CnaPythonCommand.Append(' ');
      CnaPythonCommand.Append('\\'); CnaPythonCommand.Append('\"');
      const Text_t *t_StringAnaType = (const Text_t *)fKeyAnaType.Data();
      CnaPythonCommand.Append(t_StringAnaType);
      CnaPythonCommand.Append('\\'); CnaPythonCommand.Append('\"');

      //.......... ${3}
      CnaPythonCommand.Append(' ');
      CnaPythonCommand.Append('\\'); CnaPythonCommand.Append('\"');
      const Text_t *t_NbOfSamplesString = (const Text_t *)fKeyNbOfSamplesString.Data();
      CnaPythonCommand.Append(t_NbOfSamplesString);
      CnaPythonCommand.Append('\\'); CnaPythonCommand.Append('\"');

      //.......... ${4}
      CnaPythonCommand.Append(' ');
      CnaPythonCommand.Append('\\'); CnaPythonCommand.Append('\"');
      const Text_t *t_FirstReqEvtNumberString = (const Text_t *)fKeyFirstReqEvtNumberString.Data();
      CnaPythonCommand.Append(t_FirstReqEvtNumberString);
      CnaPythonCommand.Append('\\'); CnaPythonCommand.Append('\"');

      //.......... ${5}
      CnaPythonCommand.Append(' ');
      CnaPythonCommand.Append('\\'); CnaPythonCommand.Append('\"');
      const Text_t *t_LastReqEvtNumberString = (const Text_t *)fKeyLastReqEvtNumberString.Data();
      CnaPythonCommand.Append(t_LastReqEvtNumberString);
      CnaPythonCommand.Append('\\'); CnaPythonCommand.Append('\"');

      //.......... ${6}
      CnaPythonCommand.Append(' ');
      CnaPythonCommand.Append('\\'); CnaPythonCommand.Append('\"');
      const Text_t *t_ReqNbOfEvtsString = (const Text_t *)fKeyReqNbOfEvtsString.Data();
      CnaPythonCommand.Append(t_ReqNbOfEvtsString);
      CnaPythonCommand.Append('\\'); CnaPythonCommand.Append('\"');

      //.......... ${7}
      CnaPythonCommand.Append(' ');
      CnaPythonCommand.Append('\\'); CnaPythonCommand.Append('\"');
      const Text_t *t_fStexName = (const Text_t *)fStexName.Data();
      CnaPythonCommand.Append(t_fStexName);
      CnaPythonCommand.Append('\\'); CnaPythonCommand.Append('\"');

      //.......... ${8}
      CnaPythonCommand.Append(' ');
      CnaPythonCommand.Append('\\'); CnaPythonCommand.Append('\"');
      const Text_t *t_StexNumberString = (const Text_t *)fKeyStexNumberString.Data();
      CnaPythonCommand.Append(t_StexNumberString);
      CnaPythonCommand.Append('\\'); CnaPythonCommand.Append('\"');

      //......................................... arguments -> python file name
      //.......... ${9}
      // - - - - - - - - - - - - - - - - - Build the name
      fPythonFileName = "EcnaPython_";

      fPythonFileName.Append(t_StringAnaType);
      fPythonFileName.Append('_');

      fPythonFileName.Append('S');
      fPythonFileName.Append('1');
      fPythonFileName.Append('_');
      fPythonFileName.Append(t_NbOfSamplesString);
      fPythonFileName.Append('_');
      fPythonFileName.Append('R');

      const Text_t *t_fConfirmRunNumberString = (const Text_t *)fConfirmRunNumberString.Data();
      fPythonFileName.Append(t_fConfirmRunNumberString);
      fPythonFileName.Append('_');

      fPythonFileName.Append(t_FirstReqEvtNumberString);
      fPythonFileName.Append('_');

      fPythonFileName.Append(t_LastReqEvtNumberString);
      fPythonFileName.Append('_');

      fPythonFileName.Append(t_ReqNbOfEvtsString);
      fPythonFileName.Append('_');

      fPythonFileName.Append(t_fStexName);

      fPythonFileName.Append(t_StexNumberString);      //  <== (python file name without extension .py)
      // - - - - - - - - - - - - - - - - - 

      CnaPythonCommand.Append(' ');
      const Text_t *t_fPythonFileName = (const Text_t *)fPythonFileName.Data();
      CnaPythonCommand.Append(t_fPythonFileName);

      //......................................... arguments -> modules:data path
      //.......... ${9}
      CnaPythonCommand.Append(' ');
      const Text_t *t_modules_data_path = (const Text_t *)ModulesdataPath.Data();
      CnaPythonCommand.Append(t_modules_data_path);

      //......................................... arguments -> last evt number (without "")
      //.......... ${10}
      //CnaPythonCommand.Append(' ');
      //CnaPythonCommand.Append(t_LastReqEvtNumberString);

      //......................................... arguments -> SourceForPythonFileName
      //.......... ${11}
      //CnaPythonCommand.Append(' ');
      //const Text_t *t_Pyf = (const Text_t *)fKeyPyf.Data();
      //CnaPythonCommand.Append(t_Pyf);

      //---------------------------------- Exec python file building command (csh before command text)  
      const Text_t *t_cnapythoncommand = (const Text_t *)CnaPythonCommand.Data();
      TString CnaExecPythonCommand = "csh ";
      CnaExecPythonCommand.Append(t_cnapythoncommand);

      Int_t i_exec_python = gSystem->Exec(CnaExecPythonCommand.Data());

      if( i_exec_python != 0 )
	{
	  cout << "*TEcnaGui> Script for python file building was executed with error code = "
	       << i_exec_python << "." << endl
	       << "           python file: " << fPythonFileName.Data() << ".py" << endl
	       << "           Command: " << CnaExecPythonCommand.Data() << endl
	       << fTTBELL << endl;
	}
      else
	{
	  cout << "*TEcnaGui> Script for python file building was successfully executed." << endl
	       << "           python file: " << fPythonFileName.Data() << ".py" << endl
	       << "           (Command: " << CnaExecPythonCommand.Data() << ")" << endl;

	  //========================================================== Job submission script
	  TString CnaSubmitCommand = ModulesdataPath;

	  //......................................... Script for job submission: command name
	  TString SubmitScriptName = "EcnaSystemScriptSubmit";
	  const Text_t *t_SubmitScriptName= (const Text_t *)SubmitScriptName.Data();
	  CnaSubmitCommand.Append(t_SubmitScriptName);
	  CnaSubmitCommand.Append(' ');

	  //......................................... Script for job submission: arguments
	  const Text_t *t_cmssw_base = (const Text_t *)fCnaParPaths->CMSSWBase().Data();
	  CnaSubmitCommand.Append(t_cmssw_base);
	  CnaSubmitCommand.Append(' ');

	  const Text_t *t_cmssw_subsystem = (const Text_t *)fCnaParPaths->CMSSWSubsystem().Data();
	  CnaSubmitCommand.Append(t_cmssw_subsystem);
	  CnaSubmitCommand.Append(' ');

	  const Text_t *t_cfgp_file = (const Text_t *)fPythonFileName.Data();
	  CnaSubmitCommand.Append(t_cfgp_file);
	  CnaSubmitCommand.Append(' ');
      
	  const Text_t *t_QueueCode = (const Text_t *)QueueCode.Data();
	  CnaSubmitCommand.Append(t_QueueCode);

	  //----------------------------------------- Exec Submit Command (csh before command text)
	  const Text_t *t_cnasubmitcommand = (const Text_t *)CnaSubmitCommand.Data();
	  TString CnaExecSubmitCommand = "csh ";
	  CnaExecSubmitCommand.Append(t_cnasubmitcommand);

	  Int_t i_exec_submit = gSystem->Exec(CnaExecSubmitCommand.Data());
      
	  if( i_exec_submit != 0 )
	    {
	      cout << "*TEcnaGui> Script for job submission was executed with error code = "
		   << i_exec_submit << "."  << endl
		   << "          Command: " << CnaExecSubmitCommand.Data() << endl
		   << fTTBELL << endl;
	    }
	  else
	    {
	      cout << "*TEcnaGui> Job with configuration file: " << fPythonFileName.Data()
		   << " was successfully submitted." << endl
		   << "          (Command: " << CnaExecSubmitCommand.Data() << ")" << endl;	  
	    }
      
	  fConfirmSubmit          = 0;
	  fConfirmRunNumber       = 0;
	  fConfirmRunNumberString = "0";
	}
    }
  else
    {
      if( fKeyAnaType.BeginsWith("Adc") )
	{
	  fCnaCommand++;
	  cout << "   *TEcnaGui [" << fCnaCommand
	       << "]> Request for submitting job in batch mode for run " << fKeyRunNumber
	       << ". Syntax OK. Please, click again to confirm."
	       << fTTBELL << endl;
	  
	  fConfirmSubmit          = 1;
	  fConfirmRunNumber       = fKeyRunNumber;
	  fConfirmRunNumberString = fKeyRunNumberString;
	}
      else
	{
	  fCnaError++;
	  cout << "   !TEcnaGui (" << fCnaError << ") *** ERROR *** ===>"
	       << " Analysis name = " << fKeyAnaType
	       << ": should begin with 'Adc'."
	       << " Please, change the analysis name." << fTTBELL << endl;
	  
	  fConfirmSubmit          = 0;
	  fConfirmRunNumber       = 0;
	  fConfirmRunNumberString = "0";
	}
    }
}
//------------------------------------------- end of SubmitOnBatchSystem() ------------------------

//==========================================================================
//
//             CleanBatchFiles()   M E T H O D 
//
//==========================================================================
void TEcnaGui::CleanBatchFiles(const TString& clean_code)
{
  //Clean python files, submission scripts,...

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Clean requested with code: " << clean_code
       << endl;

  //================================ CLEAN SUBMISSION SCRIPTS ===================================
  if( clean_code == "Sub"  || clean_code == "All")
    {
      //.......................... get the path "modules/data"
      // /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/src/CalibCalorimetry/EcalCorrelatedNoiseAnalysisModules/data/
      TString ModulesdataPath = fCnaParPaths->PathModulesData();

      //----------------------------------------- submission clean script: command text
      //......................................... submission clean script: script name
      TString CnaCleanSubmissionCommand = ModulesdataPath;
      TString CleanSubmissionScriptName = "EcnaSystemScriptCleanSubmissionScripts";
      const Text_t *t_CleanSubmissionScriptName = (const Text_t *)CleanSubmissionScriptName.Data();
      CnaCleanSubmissionCommand.Append(t_CleanSubmissionScriptName);

      //......................................... arguments -> test/slc... path
      //.......................... get the path "test/slc4_ia32_gcc345"
      // /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/test/slc4_ia32_gcc345/
      TString TestslcPath = fCnaParPaths->PathTestScramArch();
      CnaCleanSubmissionCommand.Append(' ');
      const Text_t *t_modules_data_path = (const Text_t *)TestslcPath.Data();
      CnaCleanSubmissionCommand.Append(t_modules_data_path);

      //----------------------------------------- Exec CleanSubmission Command (csh before command text)
      const Text_t *t_cnacleansubmissioncommand = (const Text_t *)CnaCleanSubmissionCommand.Data();
      TString CnaExecCleanSubmissionCommand = "csh ";
      CnaExecCleanSubmissionCommand.Append(t_cnacleansubmissioncommand);

      Int_t i_exec_cleansubmission = gSystem->Exec(CnaExecCleanSubmissionCommand.Data());

      if( i_exec_cleansubmission != 0 )
	{
	  cout << "*TEcnaGui> Script for submission script clean was executed with error code = "
	       << i_exec_cleansubmission << "."  << endl
	       << "          Command: " << CnaExecCleanSubmissionCommand.Data() << endl
	       << fTTBELL << endl;
	}
      else
	{
	  cout << "*TEcnaGui> Script for submission script clean"
	       << " was successfully executed." << endl
	       << "          (Command: " << CnaExecCleanSubmissionCommand.Data() << ")" << endl;	  
	}

    }

  //================================= CLEAN LSFJOB REPORTS ======================================
  if( clean_code == "Job"  || clean_code == "All")
    {
      //.......................... get the path "modules/data"
      // /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/src/CalibCalorimetry/EcalCorrelatedNoiseAnalysisModules/data/
      TString ModulesdataPath = fCnaParPaths->PathModulesData();

      //----------------------------------------- jobreport clean script: command text
      //......................................... jobreport clean script: script name
      TString CnaCleanJobreportCommand = ModulesdataPath;
      TString CleanJobreportScriptName = "EcnaSystemScriptCleanLSFJOBReports";
      const Text_t *t_CleanJobreportScriptName = (const Text_t *)CleanJobreportScriptName.Data();
      CnaCleanJobreportCommand.Append(t_CleanJobreportScriptName);

      //......................................... arguments -> test/slc... path
      //.......................... get the path "test/slc4_ia32_gcc345"
      // /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/test/slc4_ia32_gcc345/
      TString TestslcPath = fCnaParPaths->PathTestScramArch();
      CnaCleanJobreportCommand.Append(' ');
      const Text_t *t_modules_data_path = (const Text_t *)TestslcPath.Data();
      CnaCleanJobreportCommand.Append(t_modules_data_path);

      //----------------------------------------- Exec CleanJobreport Command (csh before command text)
      const Text_t *t_cnacleanjobreportcommand = (const Text_t *)CnaCleanJobreportCommand.Data();
      TString CnaExecCleanJobreportCommand = "csh ";
      CnaExecCleanJobreportCommand.Append(t_cnacleanjobreportcommand);

      Int_t i_exec_cleanjobreport = gSystem->Exec(CnaExecCleanJobreportCommand.Data());

      if( i_exec_cleanjobreport != 0 )
	{
	  cout << "*TEcnaGui> Script for LSFJOB report clean was executed with error code = "
	       << i_exec_cleanjobreport << "."  << endl
	       << "          Command: " << CnaExecCleanJobreportCommand.Data() << endl
	       << fTTBELL << endl;
	}
      else
	{
	  cout << "*TEcnaGui> Script for LSFJOB report clean"
	       << " was successfully executed." << endl
	       << "          (Command: " << CnaExecCleanJobreportCommand.Data() << ")" << endl;	  
	}
    }

  //==================================== CLEAN PYTHON FILES =====================================
  if( clean_code == "Pyth" || clean_code == "All")
    {
      //.......................... get the path "modules/data"
      // /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/src/CalibCalorimetry/EcalCorrelatedNoiseAnalysisModules/data/
      TString ModulesdataPath = fCnaParPaths->PathModulesData();

      //----------------------------------------- python file clean script: command text
      //......................................... python file clean script: script name
      TString CnaCleanPythonCommand = ModulesdataPath;
      TString CleanPythonScriptName = "EcnaSystemScriptCleanPythonFiles";
      const Text_t *t_CleanPythonScriptName = (const Text_t *)CleanPythonScriptName.Data();
      CnaCleanPythonCommand.Append(t_CleanPythonScriptName);

      //......................................... arguments -> modules:data path
      CnaCleanPythonCommand.Append(' ');
      const Text_t *t_modules_data_path = (const Text_t *)ModulesdataPath.Data();
      CnaCleanPythonCommand.Append(t_modules_data_path);

      //----------------------------------------- Exec CleanPython Command (csh before command text)
      const Text_t *t_cnacleanpythoncommand = (const Text_t *)CnaCleanPythonCommand.Data();
      TString CnaExecCleanPythonCommand = "csh ";
      CnaExecCleanPythonCommand.Append(t_cnacleanpythoncommand);

      Int_t i_exec_cleanpython = gSystem->Exec(CnaExecCleanPythonCommand.Data());

      if( i_exec_cleanpython != 0 )
	{
	  cout << "*TEcnaGui> Script for python file clean was executed with error code = "
	       << i_exec_cleanpython << "."  << endl
	       << "          Command: " << CnaExecCleanPythonCommand.Data() << endl
	       << fTTBELL << endl;
	}
      else
	{
	  cout << "*TEcnaGui> Script for python file clean"
	       << " was successfully executed." << endl
	       << "          (Command: " << CnaExecCleanPythonCommand.Data() << ")" << endl;	  
	}
    }
}
//------------------------------------------- end of CleanBatchFiles() -----------------------

//==========================================================================
//
//             Calculations()   M E T H O D 
//
//==========================================================================
void TEcnaGui::Calculations(const TString& calc_code)
{
  //Calculations of quantities (Pedestals, correlations, ... )

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Calculations requested with code: " << calc_code
       << endl;

  //===== Check if Analysis name is right 
  if( fKeyAnaType == "AdcPed1"  || fKeyAnaType == "AdcSPed1"  ||   
      fKeyAnaType == "AdcPed6"  || fKeyAnaType == "AdcSPed6"  || 
      fKeyAnaType == "AdcPed12" || fKeyAnaType == "AdcSPed12" || 
      fKeyAnaType == "AdcPeg12" || fKeyAnaType == "AdcSPeg12" ||
      fKeyAnaType == "AdcLaser" || fKeyAnaType == "AdcSLaser" || 
      fKeyAnaType == "AdcPes12" || fKeyAnaType == "AdcSPes12" || 
      fKeyAnaType == "AdcPhys"  || fKeyAnaType == "AdcAny" )
    {
      //------------ Check if Std or (Scc or Stt)-Confirmed
      if( calc_code == "Std" || ( ( calc_code == "Scc" || calc_code == "Stt" ) && fConfirmCalcScc == 1 ) )
	{
	  if( fKeyNbOfSamples >= fKeyNbOfSampForCalc )
	    {
	      Int_t nStexMin = fKeyStexNumber;
	      Int_t nStexMax = fKeyStexNumber;
	      if( fKeyStexNumber == 0 ){nStexMin = 1; nStexMax = fEcal->MaxStexInStas();}

	      for( Int_t nStex = nStexMin; nStex<= nStexMax; nStex++)
		{
		  Int_t n_samp_fic = fKeyNbOfSamples;

		  //....................... READ the "ADC" (AdcPed.., AdcLaser..., ...) file
		  TEcnaRun* MyRun = 0; 
		  if ( MyRun == 0 ){MyRun = new TEcnaRun(fObjectManager, fSubDet.Data(), n_samp_fic);  fCnew++;}

		  MyRun->GetReadyToReadData(fKeyAnaType.Data(),    fKeyRunNumber,
					    fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, nStex);

		  if( MyRun->ReadSampleAdcValues(fKeyNbOfSampForCalc) == kTRUE )
		    {
		      cout << "*TEcnaGui::Calculations> File "
			   << MyRun->GetRootFileNameShort() << " found. Starting calculations."
			   << endl;

		      MyRun->GetReadyToCompute();

		      //............... Calculations
		      if( calc_code == "Std" ||
			  ( (calc_code == "Scc" || calc_code == "Stt") && fConfirmCalcScc == 1 ) )
			{
			  //-------------- Standard calculations: pedestals, noises, correlations between samples
			  MyRun->StandardCalculations();
			}
		      if( (calc_code == "Scc" || calc_code == "Stt") && fConfirmCalcScc == 1 )
			{
			  if( calc_code == "Scc" )
			    {
			      //------ Additional calculations:
			      //       "correlations" between Xtals and Stins (long time, big file)
			      cout << "*TEcnaGui::Calculations> Please, wait." << endl;
			  
			      MyRun->Expert1Calculations();    //   (long time, big file)
			      // <=> MyRun->LowFrequencyCorrelationsBetweenChannels();     //   (big file)
			      //     MyRun->HighFrequencyCorrelationsBetweenChannels();    //   (big file) 
			  
			      MyRun->Expert2Calculations();    //   (fast because expert1 has been called)
			      // <=> MyRun->LowFrequencyMeanCorrelationsBetweenStins();
			      //     MyRun->HighFrequencyMeanCorrelationsBetweenStins();
			    }
		      
			  if( calc_code == "Stt" )
			    {
			      //---Additional calculations:
			      //   "correlations" between Stins (long time, "normal" size file)
			      cout << "*TEcnaGui::Calculations> Please, wait." << endl;

			      MyRun->Expert2Calculations();    //  (long time but not big file)

			      // Explanation/example: if MyRun->LowFrequencyCorrelationsBetweenChannels() (expert1)
			      // has not been called by the user, it is automatically called by
			      // MyRun->LowFrequencyMeanCorrelationsBetweenStins()
			      // but the corresponding file is not written (idem for "HighFrequency")
			    }
			}
		      //.......................... WRITE results in file 
		      TString calc_file_name = "?";
		      if( calc_code == "Std" )
			{
			  if( fKeyAnaType == "AdcPed1"   ){calc_file_name = "StdPed1";}
			  if( fKeyAnaType == "AdcPed6"   ){calc_file_name = "StdPed6";}
			  if( fKeyAnaType == "AdcPed12"  ){calc_file_name = "StdPed12";}
			  if( fKeyAnaType == "AdcPeg12"  ){calc_file_name = "StdPeg12";}
			  if( fKeyAnaType == "AdcLaser"  ){calc_file_name = "StdLaser";}
			  if( fKeyAnaType == "AdcPes12"  ){calc_file_name = "StdPes12";}

			  if( fKeyAnaType == "AdcSPed1"  ){calc_file_name = "StdSPed1";}
			  if( fKeyAnaType == "AdcSPed6"  ){calc_file_name = "StdSPed6";}
			  if( fKeyAnaType == "AdcSPed12" ){calc_file_name = "StdSPed12";}
			  if( fKeyAnaType == "AdcSPeg12" ){calc_file_name = "StdSPeg12";}
			  if( fKeyAnaType == "AdcSLaser" ){calc_file_name = "StdSLaser";}
			  if( fKeyAnaType == "AdcSPes12" ){calc_file_name = "StdSPes12";}

			  if( fKeyAnaType == "AdcPhys"   ){calc_file_name = "StdPhys";}
			  if( fKeyAnaType == "AdcAny"    ){calc_file_name = "StdAny";}

			}
		      if( calc_code == "Scc" )
			{
			  if( fKeyAnaType == "AdcPed1"   ){calc_file_name = "SccPed1";}
			  if( fKeyAnaType == "AdcPed6"   ){calc_file_name = "SccPed6";}
			  if( fKeyAnaType == "AdcPed12"  ){calc_file_name = "SccPed12";}
			  if( fKeyAnaType == "AdcPeg12"  ){calc_file_name = "SccPeg12";}
			  if( fKeyAnaType == "AdcLaser"  ){calc_file_name = "SccLaser";}
			  if( fKeyAnaType == "AdcPes12"  ){calc_file_name = "SccPes12" ;}

			  if( fKeyAnaType == "AdcSPed1"  ){calc_file_name = "SccSPed1";}
			  if( fKeyAnaType == "AdcSPed6"  ){calc_file_name = "SccSPed6";}
			  if( fKeyAnaType == "AdcSPed12" ){calc_file_name = "SccSPed12";}
			  if( fKeyAnaType == "AdcSPeg12" ){calc_file_name = "SccSPeg12";}
			  if( fKeyAnaType == "AdcSLaser" ){calc_file_name = "SccSLaser";}
			  if( fKeyAnaType == "AdcSPes12" ){calc_file_name = "SccSPes12";}

			  if( fKeyAnaType == "AdcPhys"   ){calc_file_name = "SccPhys";}
			  if( fKeyAnaType == "AdcAny"    ){calc_file_name = "SccAny";}
			}

		      if( calc_code == "Stt" )
			{
			  if( fKeyAnaType == "AdcPed1"   ){calc_file_name = "SttPed1";}
			  if( fKeyAnaType == "AdcPed6"   ){calc_file_name = "SttPed6";}
			  if( fKeyAnaType == "AdcPed12"  ){calc_file_name = "SttPed12";}
			  if( fKeyAnaType == "AdcPeg12"  ){calc_file_name = "SttPeg12";}
			  if( fKeyAnaType == "AdcLaser"  ){calc_file_name = "SttLaser";}
			  if( fKeyAnaType == "AdcPes12"  ){calc_file_name = "SttPes12" ;}

			  if( fKeyAnaType == "AdcSPed1"  ){calc_file_name = "SttSPed1";}
			  if( fKeyAnaType == "AdcSPed6"  ){calc_file_name = "SttSPed6";}
			  if( fKeyAnaType == "AdcSPed12" ){calc_file_name = "SttSPed12";}
			  if( fKeyAnaType == "AdcSPeg12" ){calc_file_name = "SttSPeg12";}
			  if( fKeyAnaType == "AdcSLaser" ){calc_file_name = "SttSLaser";}
			  if( fKeyAnaType == "AdcSPes12" ){calc_file_name = "SttSPes12";}

			  if( fKeyAnaType == "AdcPhys"   ){calc_file_name = "SttPhys";}
			  if( fKeyAnaType == "AdcAny"    ){calc_file_name = "SttAny";}
			}

		      if( MyRun->WriteNewRootFile(calc_file_name.Data()) == kTRUE )
			{
			  cout << "*TEcnaGui::Calculations> Done. Write ROOT file: "
			       << MyRun->GetNewRootFileNameShort() << " OK" << endl << endl;
			}
		      else 
			{
			  cout << "!TEcnaGui::Calculations> Writing ROOT file failure for file "
			       << MyRun->GetNewRootFileNameShort()
			       << fTTBELL << endl << endl;
			}
		    }
		  else
		    {
		      cout << "!TEcnaGui::Calculations> " << MyRun->GetRootFileNameShort() << ": file not found."
			   << fTTBELL << endl << endl;
		    }
		  //.......................................................................
		  delete MyRun; MyRun = 0;                    fCdelete++;
		} // end of for( Int_t nStex = nStexMin; nStex<= nStexMax; nStex++)
	      fConfirmCalcScc = 0;
	    } // end of if( fKeyNbOfSamples >= fKeyNbOfSampForCalc )
	  else
	    {
	      cout << "!TEcnaGui::Calculations> *** ERROR *** Number of samples in file (=" << fKeyNbOfSamples
		   << ") less than number of samples for calculations (= " << fKeyNbOfSampForCalc << "). " << endl;
	    }
	} // end of if( calc_code == "Std" || ( ( calc_code == "Scc" || calc_code == "Stt" ) && fConfirmCalcScc == 1 ) )
      else
	{
	  cout << "   *TEcnaGui [" << fCnaCommand
	       << "]> Calculation requested with option " << calc_code
	       << ". This can last more than 5 minutes. Please, click again to confirm."
	       << fTTBELL << endl;
	  fConfirmCalcScc = 1;
	}
    }
  else
    {
      cout << "!TEcnaGui::Calculations> fKeyAnaType = " << fKeyAnaType
	   << "  : wrong code in analysis name." << endl
	   << "                        List of available standard analysis names for calculations: " << endl
	   << "                        AdcPed1,  AdcPed6,  AdcPed12,  AdcPeg12,  AdcLaser,  AdcPes12," << endl
	   << "                        AdcSPed1, AdcSPed6, AdcSPed12, AdcSPeg12, AdcSLaser, AdcSPes12," << endl
	   << "                        AdcPhys,  AdcAny (all names must begin with 'Adc')."
	   << fTTBELL << endl; 
    }
}
//==========================================================================
//
//                       "View"    M E T H O D S
//
//==========================================================================
//---------- common messages

void TEcnaGui::MessageCnaCommandReplyA(const TString& first_same_plot)
{
  // reply message of the Cna command

  cout << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st req. evt#: " << fKeyFirstReqEvtNumber
       << ", last req. evt#: " << fKeyLastReqEvtNumber << endl;
  
  if( first_same_plot == "ASCII" )
    {
      cout  << "                   " << fStexName.Data() << ": " << fKeyStexNumber
	    << ", option: " << first_same_plot << endl;
    }
}

void TEcnaGui::MessageCnaCommandReplyB(const TString& first_same_plot)
{
  // reply message of the Cna command

  if( first_same_plot == "ASCII" )
    {
      if(fKeyStexNumber > 0)
	{
	  if( fHistos->StatusFileFound() == kTRUE && fHistos->StatusDataExist() == kTRUE )
	    {
	      TString xAsciiFileName = fHistos->AsciiFileName();
	      if( xAsciiFileName != "?" )
		{cout  << "               Histo written in ASCII file: " << xAsciiFileName.Data();}
	    }
	}
      else
	{
	  cout  << "               No writing in ASCII file since "
		<< fStexName.Data() << " number = " << fKeyStexNumber;
	}
      cout << endl;
    }
}

//==========================================================================
//
//                  View  Matrix
//
//==========================================================================
//---------------------------- Cortt
void TEcnaGui::ViewMatrixLowFrequencyMeanCorrelationsBetweenStins(const TString& option_plot)
{
  // Plot of Low Frequency Mean Cor(c,c') for each pair of Stins

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Low Frequency Mean Cor(c,c') for each pair of " << fStinName.Data()
       << "s. Option: "
       << option_plot;
  MessageCnaCommandReplyA(option_plot);

  fHistos->SetHistoMin(fKeyVminLFccMos);
  fHistos->SetHistoMax(fKeyVmaxLFccMos);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->PlotMatrix("Cor", "MttLF", option_plot);

  MessageCnaCommandReplyB(option_plot);
}
void TEcnaGui::ViewMatrixHighFrequencyMeanCorrelationsBetweenStins(const TString& option_plot)
{
  // Plot of Low Frequency Mean Cor(c,c') for each pair of Stins

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> High Frequency Mean Cor(c,c') for each pair of " << fStinName.Data()
       << "s. Option: "
       << option_plot;
  MessageCnaCommandReplyA(option_plot);

  fHistos->SetHistoMin(fKeyVminHFccMos);
  fHistos->SetHistoMax(fKeyVmaxHFccMos);
  fHistos->GeneralTitle(fKeyGeneralTitle); 
  fHistos->PlotMatrix("Cor", "MttHF", option_plot);

  MessageCnaCommandReplyB(option_plot);
}
//---------------------------------------------- Corcc
void TEcnaGui::ViewMatrixLowFrequencyCorrelationsBetweenChannels(const Int_t&  cStexStin_A,
								 const Int_t&  cStexStin_B,
								 const TString& option_plot)
{
  // Low Frequency Correlation matrix (crystal of Stin X, crystal of Stin X) for each Stin

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  
  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Low Frequency Correlation matrix between channels. "
       << fStinName.Data() << " A: " << cStexStin_A
       << ", " << fStinName.Data() << " B: " << cStexStin_B
       << ", option: " << option_plot;
  MessageCnaCommandReplyA(option_plot);

  fHistos->SetHistoMin(fKeyVminLHFcc);
  fHistos->SetHistoMax(fKeyVmaxLHFcc); 
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->PlotMatrix("Cor", "MccLF", cStexStin_A, cStexStin_B, option_plot);
  MessageCnaCommandReplyB(option_plot);
}

void TEcnaGui::ViewMatrixHighFrequencyCorrelationsBetweenChannels(const Int_t&  cStexStin_A, const Int_t& cStexStin_B,
								  const TString& option_plot)
{
// High Frequency Correlation matrix (crystal of Stin X, crystal of Stin X) for each Stin

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  
  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> High Frequency Correlation matrix between channels. "
       << fStinName.Data() << " A: " << cStexStin_A
       << ", " << fStinName.Data() << " B: " << cStexStin_B
       << ", option: " << option_plot;
  MessageCnaCommandReplyA(option_plot);

  fHistos->SetHistoMin(fKeyVminLHFcc);
  fHistos->SetHistoMax(fKeyVmaxLHFcc); 
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->PlotMatrix("Cor", "MccHF", cStexStin_A, cStexStin_B, option_plot);
  
  MessageCnaCommandReplyB(option_plot);
}

void TEcnaGui::ViewStexLowFrequencyCorcc()
{
  //===> big matrix

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> LF Correlations between channels for each " << fStinName.Data()
       << " in " << fStexName.Data() << ". 2D histo. "
       << fStexName.Data() << ": " << fKeyStexNumber;
  MessageCnaCommandReplyA("DUMMY");

  fHistos->SetHistoMin(fKeyVminLHFcc);
  fHistos->SetHistoMax(fKeyVmaxLHFcc);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->PlotMatrix("Cor", "MccLF", "COLZ");

  MessageCnaCommandReplyB("DUMMY");
}

void TEcnaGui::ViewStexHighFrequencyCorcc()
{
  //===> big matrix

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;} 
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> HF Correlations between channels for each " << fStinName.Data()
       << " in " << fStexName.Data() << ". 2D histo. "
       << fStexName.Data() << ": " << fKeyStexNumber;
  MessageCnaCommandReplyA("DUMMY");

  fHistos->SetHistoMin(fKeyVminLHFcc);
  fHistos->SetHistoMax(fKeyVmaxLHFcc);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->PlotMatrix("Cor", "MccHF", "COLZ");

  MessageCnaCommandReplyB("DUMMY");
}

//---------------------------- Corss, Covss
void TEcnaGui::ViewMatrixCorrelationSamples(const Int_t&  cStexStin_A, const Int_t& i0StinEcha,
					   const TString& option_plot)
{
// Plot of correlation matrix between samples for a given channel

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  
  Int_t ChOffset = 0;
  if(fSubDet == "EE"){ChOffset = 1;}

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Correlation matrix between samples. "
       << fStinName.Data() << ": " << cStexStin_A  << ", channel " << i0StinEcha + ChOffset
       << ", option: " << option_plot;
  MessageCnaCommandReplyA(option_plot);

  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->SetHistoMin(fKeyVminD_MCs_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_MCs_ChNb);
  fHistos->PlotMatrix("Cor", "Mss", cStexStin_A, i0StinEcha, option_plot);


  MessageCnaCommandReplyB(option_plot);
}

void TEcnaGui::ViewMatrixCovarianceSamples(const Int_t&  cStexStin_A, const Int_t& i0StinEcha,
					  const TString& option_plot)
{
// Plot of covariance matrix between samples for a given channel

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);

  Int_t ChOffset = 0;
  if(fSubDet == "EE"){ChOffset = 1;}
  
  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Covariance matrix between samples. "
       << fStinName.Data() << ": " << cStexStin_A  << ", channel " << i0StinEcha + ChOffset
       << ", option: " << option_plot;
  MessageCnaCommandReplyA(option_plot);

  fHistos->SetHistoMin(fKeyVminD_TNo_ChNb);  // same as mean sample sigmas
  fHistos->SetHistoMax(fKeyVmaxD_TNo_ChNb);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->PlotMatrix("Cov", "Mss", cStexStin_A, i0StinEcha, option_plot);

  MessageCnaCommandReplyB(option_plot);
}

//==========================================================================
//
//                         ViewStin...     
//
//     StexStin ==> (sample,sample) matrices for all the crystal of cStexStin              
//
//==========================================================================
void TEcnaGui::ViewStinCorrelationSamples(const Int_t& cStexStin)
{
  // Plot of (sample,sample) correlation matrices for all the crystal of a given Stin  

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  
  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Correlation matrices between samples for each channel of "
       << fStinName.Data() << " " << cStexStin;
  MessageCnaCommandReplyA("DUMMY"); 

  fHistos->SetHistoMin(fKeyVminD_MCs_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_MCs_ChNb);
  fHistos->GeneralTitle(fKeyGeneralTitle);  
  fHistos->CorrelationsBetweenSamples(cStexStin);
  
  MessageCnaCommandReplyB("DUMMY");
}

void TEcnaGui::ViewStinCovarianceSamples(const Int_t& cStexStin)
{
  // Plot of (sample,sample) covariance matrices for all the crystal of a given Stin  

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  
  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Covariance matrices between samples for each channel of "
       << fStinName.Data() << " " << cStexStin;
  MessageCnaCommandReplyA("DUMMY");

  fHistos->SetHistoMin(fKeyVminD_TNo_ChNb);   // same as mean sample sigmas
  fHistos->SetHistoMax(fKeyVmaxD_TNo_ChNb);
  fHistos->GeneralTitle(fKeyGeneralTitle);    
  fHistos->CovariancesBetweenSamples(cStexStin);
  
  MessageCnaCommandReplyB("DUMMY");
}
//==========================================================================
//
//                         ViewSorS (eta,phi)
//
//==========================================================================

void TEcnaGui::ViewSorSNumberOfEvents()
{
  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;} 

  if( fKeyStexNumber > 0 && fKeyStexNumber <= fEcal->MaxStexInStas() )
    {
      fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			      fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);      
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Number of Events. 2D histo. "
	   << fStexName.Data() << ": " << fKeyStexNumber;
    }
  if( fKeyStexNumber == 0 )
    {
      fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			      fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, 0);
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Average Number of Events. 2D histo for "
	   << fSubDet.Data();
    }

  MessageCnaCommandReplyA("DUMMY");

  fHistos->SetHistoMin(fKeyVminD_NOE_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_NOE_ChNb); 
  fHistos->GeneralTitle(fKeyGeneralTitle);    
  if( fKeyStexNumber > 0 && fKeyStexNumber <= fEcal->MaxStexInStas() )
    {fHistos->PlotDetector("NOE", "SM");}
  if( fKeyStexNumber == 0 )
    {fHistos->PlotDetector("NOE", "EB");}

  MessageCnaCommandReplyB("DUMMY");
}

void TEcnaGui::ViewSorSPedestals()
{
  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;} 

  if( fKeyStexNumber > 0 && fKeyStexNumber <= fEcal->MaxStexInStas() )
    {
      fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			      fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);      
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Pedestals. 2D histo. "
	   << fStexName.Data() << ": " << fKeyStexNumber;
    }
  if( fKeyStexNumber == 0 )
    {
      fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			      fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, 0);
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Pedestals. 2D histo for "
	   << fSubDet.Data();
    }
  MessageCnaCommandReplyA("DUMMY");

  fHistos->SetHistoMin(fKeyVminD_Ped_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_Ped_ChNb); 
  fHistos->GeneralTitle(fKeyGeneralTitle);
     
  if( fKeyStexNumber > 0 && fKeyStexNumber <= fEcal->MaxStexInStas() )
    {fHistos->PlotDetector("Ped", "SM");}
  if( fKeyStexNumber == 0 )
    {fHistos->PlotDetector("Ped", "EB");}

  MessageCnaCommandReplyB("DUMMY");
}

void TEcnaGui::ViewSorSTotalNoise()
{
  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;} 

  if( fKeyStexNumber > 0 && fKeyStexNumber <= fEcal->MaxStexInStas() )
    {
      fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			      fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);      
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Total noise. 2D histo. "
	   << fStexName.Data() << ": " << fKeyStexNumber;
    }
  if( fKeyStexNumber == 0 )
    {
      fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			      fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, 0);
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Average total noise. 2D histo for "
	   << fSubDet.Data();
    }
  MessageCnaCommandReplyA("DUMMY");

  fHistos->SetHistoMin(fKeyVminD_TNo_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_TNo_ChNb); 
  fHistos->GeneralTitle(fKeyGeneralTitle);      
  if( fKeyStexNumber > 0 && fKeyStexNumber <= fEcal->MaxStexInStas() )
    {fHistos->PlotDetector("TNo", "SM");}
  if( fKeyStexNumber == 0 )
    {fHistos->PlotDetector("TNo", "EB");}
  
  MessageCnaCommandReplyB("DUMMY");
}


void TEcnaGui::ViewSorSLowFrequencyNoise()
{
  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}

  if( fKeyStexNumber > 0 && fKeyStexNumber <= fEcal->MaxStexInStas() )
    {
      fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			      fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);      
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Low frequency noise. 2D histo. "
	   << fStexName.Data() << ": " << fKeyStexNumber;
    }
  if( fKeyStexNumber == 0 )
    {
      fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			      fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, 0);
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Average low frequency noise. 2D histo for "
	   << fSubDet.Data();
    }
  MessageCnaCommandReplyA("DUMMY");

  fHistos->SetHistoMin(fKeyVminD_LFN_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_LFN_ChNb);
  fHistos->GeneralTitle(fKeyGeneralTitle);     
  if( fKeyStexNumber > 0 && fKeyStexNumber <= fEcal->MaxStexInStas() )
    {fHistos->PlotDetector("LFN", "SM");}
  if( fKeyStexNumber == 0 )
    {fHistos->PlotDetector("LFN", "EB");}
  
  MessageCnaCommandReplyB("DUMMY");
}

void TEcnaGui::ViewSorSHighFrequencyNoise()
{
  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}

  if( fKeyStexNumber > 0 && fKeyStexNumber <= fEcal->MaxStexInStas() )
    {
      fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			      fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);      
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> High frequency noise. 2D histo. "
	   << fStexName.Data() << ": " << fKeyStexNumber;
    }
  if( fKeyStexNumber == 0 )
    {
      fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			      fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, 0);
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Average high frequency noise. 2D histo for "
	   << fSubDet.Data();
    }
  MessageCnaCommandReplyA("DUMMY");

  fHistos->SetHistoMin(fKeyVminD_HFN_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_HFN_ChNb); 
  fHistos->GeneralTitle(fKeyGeneralTitle);       
  if( fKeyStexNumber > 0 && fKeyStexNumber <= fEcal->MaxStexInStas() )
    {
      fHistos->PlotDetector("HFN", "SM");
    }
  if( fKeyStexNumber == 0 )
    {
      fHistos->PlotDetector("HFN", "EB");
    }
  
  MessageCnaCommandReplyB("DUMMY");
}

void TEcnaGui::ViewSorSMeanCorss()
{
  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;} 

  if( fKeyStexNumber > 0 && fKeyStexNumber <= fEcal->MaxStexInStas() )
    {
      fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			      fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);      
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Mean cor(s,s'). 2D histo. "
	   << fStexName.Data() << ": " << fKeyStexNumber;
    }
  if( fKeyStexNumber == 0 )
    {
      fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			      fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, 0);
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Average mean cor(s,s'). 2D histo for "
	   << fSubDet.Data();
    }
  MessageCnaCommandReplyA("DUMMY");

  fHistos->SetHistoMin(fKeyVminD_MCs_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_MCs_ChNb);
  fHistos->GeneralTitle(fKeyGeneralTitle);     
  if( fKeyStexNumber > 0 && fKeyStexNumber <= fEcal->MaxStexInStas() )
    {
      fHistos->PlotDetector("MCs", "SM");
    }
  if( fKeyStexNumber == 0 )
    {
      fHistos->PlotDetector("MCs", "EB");
    }
  
  MessageCnaCommandReplyB("DUMMY");
}

void TEcnaGui::ViewSorSSigmaOfCorss()
{
  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}

  if( fKeyStexNumber > 0 && fKeyStexNumber <= fEcal->MaxStexInStas() )
    {
      fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			      fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);      
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Sigma of Cor(s,s'). 2D histo. "
	   << fStexName.Data() << ": " << fKeyStexNumber;
    }
  if( fKeyStexNumber == 0 )
    {
      fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			      fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, 0);
      fCnaCommand++;
      cout << "   *TEcnaGui [" << fCnaCommand
	   << "]> Average sigma of Cor(s,s'). 2D histo for "
	   << fSubDet.Data();
    }
  MessageCnaCommandReplyA("DUMMY");

  fHistos->SetHistoMin(fKeyVminD_SCs_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_SCs_ChNb);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  if( fKeyStexNumber > 0 && fKeyStexNumber <= fEcal->MaxStexInStas() )
    {
      fHistos->PlotDetector("SCs", "SM");
    }
  if( fKeyStexNumber == 0 )
    {
      fHistos->PlotDetector("SCs", "EB");
    }

  MessageCnaCommandReplyB("DUMMY");
}

//=======================================================================================
//
//                        ViewStinCrystalNumbering
//
//=======================================================================================  
void TEcnaGui::ViewStinCrystalNumbering(const Int_t& StexStinEcna)
{
  // Plot the crystal numbering of one Stin

  Int_t StinNumber = -1;
  if( fSubDet == "EB" ){StinNumber = StexStinEcna;}
  if( fSubDet == "EE"  && fKeyStexNumber != 0 )
    {StinNumber = fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fKeyStexNumber,StexStinEcna);}

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Crystal numbering for " << " " << fStexName.Data() << " "
       << fKeyStexNumber << ", " << fStinName.Data() << " " << StinNumber << endl;

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;} 
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->StinCrystalNumbering(fKeyStexNumber, StinNumber);
}
//---------------->  end of ViewStinCrystalNumbering()

//===========================================================================
//
//                        ViewStexStinNumbering
//
//===========================================================================  
void TEcnaGui::ViewStexStinNumbering()
{
  // Plot the Stin numbering of one Stex.
  // No argument here since the Stex number is a part of the ROOT file name
  // and is in the entry field of the Stex button (fKeyStexNumber)

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> " << fStinName.Data() << " numbering for " << fStexName.Data()
       << " " << fKeyStexNumber << endl;

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->StexStinNumbering(fKeyStexNumber);
}
//---------------->  end of ViewStexStinNumbering()

//===============================================================================
//
//                         ViewHisto...
//
//===============================================================================
//......................... Nb of evts
void TEcnaGui::ViewHistoSorSNumberOfEventsOfCrystals(const TString& first_same_plot)
{
// Plot the 1D histogram of the number of events (found in the data)
// as a function of crystals (grouped by Stins)

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Number of events for crystals";
  MessageCnaCommandReplyA(first_same_plot);

  fHistos->SetHistoMin(fKeyVminD_NOE_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_NOE_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);  fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("Xtal", "NOE", "SM", first_same_plot);  // "SM" not active since fFapStexNumber is defined "outside"

  MessageCnaCommandReplyB(first_same_plot);
}

void TEcnaGui::ViewHistoSorSNumberOfEventsDistribution(const TString& first_same_plot)
{
// Plot the 1D histogram of the number of events distribution for a Stex

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Number of events distribution";
  MessageCnaCommandReplyA(first_same_plot);

  fHistos->SetHistoMin(fKeyVminD_NOE_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_NOE_ChNb); 
  fHistos->SetHistoScaleY(fKeyScaleY);  fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("NOE", "NOX", "SM", first_same_plot);

  MessageCnaCommandReplyB(first_same_plot);
}

//........................... Pedestals
void TEcnaGui::ViewHistoSorSPedestalsOfCrystals(const TString& first_same_plot)
{
// Plot the 1D histogram of the pedestals as a function of crystals (grouped by Stins)

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Pedestals";
  MessageCnaCommandReplyA(first_same_plot);

  fHistos->SetHistoMin(fKeyVminD_Ped_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_Ped_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);    fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("Xtal", "Ped", "SM", first_same_plot);

  MessageCnaCommandReplyB(first_same_plot);
}

void TEcnaGui::ViewHistoSorSPedestalsDistribution(const TString& first_same_plot)
{
// Plot the 1D histogram of the pedestals distribution for a Stex

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Pedestals distribution";
  MessageCnaCommandReplyA(first_same_plot);

  fHistos->SetHistoMin(fKeyVminD_Ped_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_Ped_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);    fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("Ped", "NOX", "SM", first_same_plot);

  MessageCnaCommandReplyB(first_same_plot);
}

//............................... Total noise
void TEcnaGui::ViewHistoSorSTotalNoiseOfCrystals(const TString& first_same_plot)
{
// Plot the 1D histogram of the mean sample sigmas as a function of crystals (grouped by Stins)

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Total noise";
  MessageCnaCommandReplyA(first_same_plot);

  fHistos->SetHistoMin(fKeyVminD_TNo_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_TNo_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);    fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("Xtal", "TNo", "SM", first_same_plot);

  MessageCnaCommandReplyB(first_same_plot);
}

void TEcnaGui::ViewHistoSorSTotalNoiseDistribution(const TString& first_same_plot)
{
// Plot the 1D histogram of the mean sample sigmas distribution for a Stex

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);  

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Total noise distribution";
  MessageCnaCommandReplyA(first_same_plot);

  fHistos->SetHistoMin(fKeyVminD_TNo_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_TNo_ChNb);  
  fHistos->SetHistoScaleY(fKeyScaleY);    fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("TNo", "NOX", "SM", first_same_plot);

  MessageCnaCommandReplyB(first_same_plot);
}


//............................ Low frequency noise
void TEcnaGui::ViewHistoSorSLowFrequencyNoiseOfCrystals(const TString& first_same_plot)
{
// Plot the 1D histogram of the pedestals as a function of crystals (grouped by Stins)

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber); 

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Low frequency noise";
  MessageCnaCommandReplyA(first_same_plot);

  fHistos->SetHistoMin(fKeyVminD_LFN_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_LFN_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);    fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("Xtal", "LFN", "SM", first_same_plot);

  MessageCnaCommandReplyB(first_same_plot);
}

void TEcnaGui::ViewHistoSorSLowFrequencyNoiseDistribution(const TString& first_same_plot)
{
// Plot the 1D histogram of the pedestals distribution for a Stex

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber); 

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Low frequency noise distribution";
  MessageCnaCommandReplyA(first_same_plot);

  fHistos->SetHistoMin(fKeyVminD_LFN_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_LFN_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);    fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("LFN", "NOX", "SM", first_same_plot);

  MessageCnaCommandReplyB(first_same_plot);
}

//............................ High frequency noise
void TEcnaGui::ViewHistoSorSHighFrequencyNoiseOfCrystals(const TString& first_same_plot)
{
// Plot the 1D histogram of the mean sample sigmas as a function of crystals (grouped by Stins)

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber); 

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> High frequency noise";
  MessageCnaCommandReplyA(first_same_plot);

  fHistos->SetHistoMin(fKeyVminD_HFN_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_HFN_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);    fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("Xtal", "HFN", "SM", first_same_plot);

  MessageCnaCommandReplyB(first_same_plot);
}

void TEcnaGui::ViewHistoSorSHighFrequencyNoiseDistribution(const TString& first_same_plot)
{
// Plot the 1D histogram of the mean sample sigmas distribution for a Stex

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber); 

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> High frequency noise distribution";
  MessageCnaCommandReplyA(first_same_plot);

  fHistos->SetHistoMin(fKeyVminD_HFN_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_HFN_ChNb);  
  fHistos->SetHistoScaleY(fKeyScaleY);  fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("HFN", "NOX", "SM", first_same_plot);

  MessageCnaCommandReplyB(first_same_plot);
}

//............................ Correlations between samples
void TEcnaGui::ViewHistoSorSMeanCorssOfCrystals(const TString& first_same_plot)
{
// Plot the 1D histogram of the mean cor(s,s') as a function of crystals (grouped by Stins)

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber); 

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Mean cor(s,s')";
  MessageCnaCommandReplyA(first_same_plot);

  fHistos->SetHistoMin(fKeyVminD_MCs_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_MCs_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);  fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("Xtal", "MCs", "SM", first_same_plot);

  MessageCnaCommandReplyB(first_same_plot);
}

void TEcnaGui::ViewHistoSorSMeanCorssDistribution(const TString& first_same_plot)
{
// Plot the 1D histogram of the mean cor(s,s') sigmas distribution for a Stex

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber); 

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Mean cor(s,s') distribution";
  MessageCnaCommandReplyA(first_same_plot);

  fHistos->SetHistoMin(fKeyVminD_MCs_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_MCs_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);    fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("MCs", "NOX", "SM", first_same_plot);

  MessageCnaCommandReplyB(first_same_plot);
}

void TEcnaGui::ViewHistoSorSSigmaOfCorssOfCrystals(const TString& first_same_plot)
{
// Plot the 1D histogram of the mean cor(s,s') as a function of crystals (grouped by Stins)

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber); 

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Sigma of cor(s,s')";
  MessageCnaCommandReplyA(first_same_plot);

  fHistos->SetHistoMin(fKeyVminD_SCs_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_SCs_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);    fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("Xtal", "SCs", "SM", first_same_plot);

  MessageCnaCommandReplyB(first_same_plot);
}

void TEcnaGui::ViewHistoSorSSigmaOfCorssDistribution(const TString& first_same_plot)
{
// Plot the 1D histogram of the mean cor(s,s') sigmas distribution for a Stex

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber); 

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Sigma of cor(s,s') distribution";
  MessageCnaCommandReplyA(first_same_plot);

  fHistos->SetHistoMin(fKeyVminD_SCs_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_SCs_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);    fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("SCs", "NOX", "SM", first_same_plot);

  MessageCnaCommandReplyB(first_same_plot);
}

//........................................................................................................
void TEcnaGui::ViewHistoCrystalSampleMeans(const Int_t&  cStexStin_A, const Int_t& crystal,
					   const TString& first_same_plot)
{
// Plot the 1D histogram of the mean sample ADC for a crystal

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber); 

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Sample means"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st req. evt#: " << fKeyFirstReqEvtNumber << ", last req. evt#: " << fKeyLastReqEvtNumber
       << ", Stex: " << fKeyStexNumber << ", " << fStinName.Data() << ": " << cStexStin_A << ", crystal" << crystal
       << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_Ped_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_Ped_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);    fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("Sample#", "SampleMean", cStexStin_A, crystal, first_same_plot);
}

//........................................................................................................
void TEcnaGui::ViewHistoCrystalSampleMeansDistribution(const Int_t&  cStexStin_A, const Int_t& crystal,
						       const TString& first_same_plot)
{
// Plot the 1D histogram distribution of the mean sample ADC for a crystal

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber); 

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Sample means"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st req. evt#: " << fKeyFirstReqEvtNumber << ", last req. evt#: " << fKeyLastReqEvtNumber
       << ", Stex: " << fKeyStexNumber << ", " << fStinName.Data() << ": " << cStexStin_A << ", crystal" << crystal
       << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_Ped_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_Ped_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);    fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("SampleMean", "NbOfSamples", cStexStin_A, crystal, first_same_plot);
}

void TEcnaGui::ViewHistoCrystalSampleSigmas(const Int_t&  cStexStin_A, const Int_t& crystal,
					    const TString& first_same_plot)
{
// Plot the 1D histogram of the sigmas of the sample ADC for a crystal

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber); 

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Sample sigmas"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st req. evt#: " << fKeyFirstReqEvtNumber << ", last req. evt#: " << fKeyLastReqEvtNumber
       << ", Stex: " << fKeyStexNumber << ", " << fStinName.Data() << ": " << cStexStin_A << ", crystal:" << crystal
       << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_TNo_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_TNo_ChNb); 
  fHistos->SetHistoScaleY(fKeyScaleY);    fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("Sample#", "SampleSigma", cStexStin_A, crystal, first_same_plot);
}

void TEcnaGui::ViewHistoCrystalSampleSigmasDistribution(const Int_t&  cStexStin_A, const Int_t& crystal,
							const TString& first_same_plot)
{
// Plot the 1D histogram distribution of the sigmas of the sample ADC for a crystal

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber); 

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Sample sigmas"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st req. evt#: " << fKeyFirstReqEvtNumber << ", last req. evt#: " << fKeyLastReqEvtNumber
       << ", Stex: " << fKeyStexNumber << ", " << fStinName.Data() << ": " << cStexStin_A << ", crystal:" << crystal
       << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_TNo_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_TNo_ChNb); 
  fHistos->SetHistoScaleY(fKeyScaleY);    fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("SampleSigma", "NbOfSamples", cStexStin_A, crystal, first_same_plot);
}

//............................ Sample values
void TEcnaGui::ViewHistoCrystalSampleValues(const Int_t& cStexStin_A, const Int_t& crystal,
					    const Int_t& sample,     const TString& first_same_plot)
{
// Plot the 1D histogram of the pedestals as a function of the event number for a crystal

  Int_t n1Sample = sample+1;
  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber); 

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> ADC sample values"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st req. evt#: " << fKeyFirstReqEvtNumber << ", last req. evt#: " << fKeyLastReqEvtNumber
       << ", Stex: " << fKeyStexNumber << ", " << fStinName.Data() << ": " << cStexStin_A << ", crystal: " << crystal
       << ", sample: " << n1Sample << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_Ped_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_Ped_ChNb); 
  fHistos->SetHistoScaleY(fKeyScaleY);    fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("Event#", "AdcValue", cStexStin_A, crystal, n1Sample, first_same_plot);
}

void TEcnaGui::ViewHistoSampleEventDistribution(const Int_t& cStexStin_A, const Int_t& crystal,
						const Int_t& sample,      const TString& first_same_plot)
{
// Plot the 1D histogram of the ADC event distribution for a sample

  Int_t n1Sample = sample+1;
  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber); 

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> ADC event distribution"
       << ". Analysis: " << fKeyAnaType << ", Run: " << fKeyRunNumber
       << ", 1st req. evt#: " << fKeyFirstReqEvtNumber << ", last req. evt#: " << fKeyLastReqEvtNumber
       << ", Stex: " << fKeyStexNumber << ", " << fStinName.Data() << ": " << cStexStin_A << ", crystal: " << crystal
       << ", sample " << n1Sample << ", option: " << first_same_plot << endl;
 
  fHistos->SetHistoMin(fKeyVminD_Ped_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_Ped_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);    fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->Plot1DHisto("AdcValue", "NbOfEvts", cStexStin_A, crystal, n1Sample, first_same_plot);
}

//------------------------------------------------------- Evolution in time (as a function of run date)
void TEcnaGui::ViewHistimeCrystalPedestals(const TString&  run_par_file_name,
					   const Int_t&   cStexStin_A, const Int_t& i0StinEcha,
					   const TString&  first_same_plot)
{
// Plot the graph of Pedestals evolution for a given channel

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Pedestal history"
       << ". Run parameters file name: " << run_par_file_name
       << ", " << fStinName.Data() << ": " << cStexStin_A << ", channel: " << i0StinEcha
       << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_Ped_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_Ped_ChNb); 
  fHistos->SetHistoScaleY(fKeyScaleY);  fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, 0,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  fHistos->PlotHistory("Time", "Ped", run_par_file_name, cStexStin_A, i0StinEcha, first_same_plot);
}

void TEcnaGui::ViewHistimeCrystalPedestalsRuns(const TString&  run_par_file_name,
					       const Int_t&   cStexStin_A, const Int_t& i0StinEcha,
					       const TString&  first_same_plot)
{
// Plot the graph of Pedestals evolution for a given channel

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Pedestal history distribution"
       << ". Run parameters file name: " << run_par_file_name
       << ", " << fStinName.Data() << ": " << cStexStin_A << ", channel: " << i0StinEcha
       << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_Ped_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_Ped_ChNb); 
  fHistos->SetHistoScaleY(fKeyScaleY);  fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, 0,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  fHistos->PlotHistory("Ped", "NOR", run_par_file_name, cStexStin_A, i0StinEcha, first_same_plot);
}

//....................................................................................................
void TEcnaGui::ViewHistimeCrystalTotalNoise(const TString&  run_par_file_name,
					    const Int_t&   cStexStin_A,    const Int_t&  i0StinEcha,
					    const TString&  first_same_plot)
{
// Plot the graph of total noise evolution for a given channel
  
  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Total noise history"
       << ". Run parameters file name: " << run_par_file_name
       << ", " << fStinName.Data() << ": " << cStexStin_A << ", channel: " << i0StinEcha
       << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_TNo_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_TNo_ChNb); 
  fHistos->SetHistoScaleY(fKeyScaleY);  fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, 0,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  fHistos->PlotHistory("Time", "TNo", run_par_file_name, cStexStin_A, i0StinEcha, first_same_plot);
}

void TEcnaGui::ViewHistimeCrystalTotalNoiseRuns(const TString&  run_par_file_name,
						const Int_t&   cStexStin_A,    const Int_t&  i0StinEcha,
						const TString&  first_same_plot)
{
// Plot the graph of total noise evolution for a given channel
  
  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Total noise history distribution"
       << ". Run parameters file name: " << run_par_file_name
       << ", " << fStinName.Data() << ": " << cStexStin_A << ", channel: " << i0StinEcha
       << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_TNo_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_TNo_ChNb); 
  fHistos->SetHistoScaleY(fKeyScaleY);  fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, 0,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  fHistos->PlotHistory("TNo", "NOR", run_par_file_name, cStexStin_A, i0StinEcha, first_same_plot);
}
//....................................................................................................
void TEcnaGui::ViewHistimeCrystalLowFrequencyNoise(const TString&  run_par_file_name,
						   const Int_t&   cStexStin_A,    const Int_t&  i0StinEcha,
						   const TString&  first_same_plot)
{
// Plot the graph of Low Frequency Noise evolution for a given channel
  
  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Low frequency noise history"
       << ". Run parameters file name: " << run_par_file_name
       << ", " << fStinName.Data() << ": " << cStexStin_A << ", channel: " << i0StinEcha
       << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_LFN_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_LFN_ChNb); 
  fHistos->SetHistoScaleY(fKeyScaleY);  fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, 0,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  fHistos->PlotHistory("Time", "LFN", run_par_file_name, cStexStin_A, i0StinEcha, first_same_plot);
}

void TEcnaGui::ViewHistimeCrystalLowFrequencyNoiseRuns(const TString&  run_par_file_name,
						       const Int_t&   cStexStin_A,    const Int_t&  i0StinEcha,
						       const TString&  first_same_plot)
{
// Plot the graph of Low Frequency Noise evolution for a given channel
  
  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Low frequency noise history distribution"
       << ". Run parameters file name: " << run_par_file_name
       << ", " << fStinName.Data() << ": " << cStexStin_A << ", channel: " << i0StinEcha
       << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_LFN_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_LFN_ChNb); 
  fHistos->SetHistoScaleY(fKeyScaleY);  fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, 0,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  fHistos->PlotHistory("LFN", "NOR", run_par_file_name, cStexStin_A, i0StinEcha, first_same_plot);
}
//....................................................................................................
void TEcnaGui::ViewHistimeCrystalHighFrequencyNoise(const TString&  run_par_file_name,
						    const Int_t&   cStexStin_A,    const Int_t&  i0StinEcha,
						    const TString&  first_same_plot)
{
// Plot the graph of High Frequency Noise evolution for a given channel
  
  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> High frequency noise history"
       << ". Run parameters file name: " << run_par_file_name
       << ", " << fStinName.Data() << ": " << cStexStin_A << ", channel: " << i0StinEcha
       << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_HFN_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_HFN_ChNb); 
  fHistos->SetHistoScaleY(fKeyScaleY);  fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, 0,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  fHistos->PlotHistory("Time", "HFN", run_par_file_name, cStexStin_A, i0StinEcha, first_same_plot);
}

void TEcnaGui::ViewHistimeCrystalHighFrequencyNoiseRuns(const TString&  run_par_file_name,
							const Int_t&   cStexStin_A,    const Int_t&  i0StinEcha,
							const TString&  first_same_plot)
{
// Plot the graph of High Frequency Noise evolution for a given channel
  
  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> High frequency noise history distribution"
       << ". Run parameters file name: " << run_par_file_name
       << ", " << fStinName.Data() << ": " << cStexStin_A << ", channel: " << i0StinEcha
       << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_HFN_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_HFN_ChNb); 
  fHistos->SetHistoScaleY(fKeyScaleY);  fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, 0,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  fHistos->PlotHistory("HFN", "NOR", run_par_file_name, cStexStin_A, i0StinEcha, first_same_plot);
}
//....................................................................................................
void TEcnaGui::ViewHistimeCrystalMeanCorss(const TString&  run_par_file_name,
					   const Int_t&   cStexStin_A,    const Int_t& i0StinEcha,
					   const TString&  first_same_plot)
{
// Plot the graph for Mean Corss evolution for a given channel

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Mean corss history"
       << ". Run parameters file name: " << run_par_file_name
       << ", " << fStinName.Data() << ": " << cStexStin_A << ", channel: " << i0StinEcha
       << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_MCs_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_MCs_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);  fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, 0,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  fHistos->PlotHistory("Time", "MCs", run_par_file_name, cStexStin_A, i0StinEcha, first_same_plot);
}

void TEcnaGui::ViewHistimeCrystalMeanCorssRuns(const TString&  run_par_file_name,
					       const Int_t&   cStexStin_A,    const Int_t& i0StinEcha,
					       const TString&  first_same_plot)
{
// Plot the graph for Mean Corss evolution for a given channel

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/ ;}

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Mean corss history distribution"
       << ". Run parameters file name: " << run_par_file_name
       << ", " << fStinName.Data() << ": " << cStexStin_A << ", channel: " << i0StinEcha
       << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_MCs_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_MCs_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);  fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, 0,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  fHistos->PlotHistory("MCs", "NOR", run_par_file_name, cStexStin_A, i0StinEcha, first_same_plot);
}
//....................................................................................................
void TEcnaGui::ViewHistimeCrystalSigmaOfCorss(const TString& run_par_file_name,
					      const Int_t&  cStexStin_A, const Int_t& i0StinEcha,
					      const TString& first_same_plot)
{
// Plot the graph of Mean Corss evolution for a given channel

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/;}

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Sigma of corss history"
       << ". Run parameters file name: " << run_par_file_name
       << ", " << fStinName.Data() << ": " << cStexStin_A << ", channel: " << i0StinEcha
       << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_SCs_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_SCs_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);  fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, 0,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  fHistos->PlotHistory("Time", "SCs", run_par_file_name, cStexStin_A, i0StinEcha, first_same_plot);
}

void TEcnaGui::ViewHistimeCrystalSigmaOfCorssRuns(const TString& run_par_file_name,
						  const Int_t&  cStexStin_A, const Int_t& i0StinEcha,
						  const TString& first_same_plot)
{
// Plot the graph of Mean Corss evolution for a given channel

  if( fHistos == 0 ){fHistos = new TEcnaHistos(fObjectManager, fSubDet.Data());       /*fCnew++*/;}

  fCnaCommand++;
  cout << "   *TEcnaGui [" << fCnaCommand
       << "]> Sigma of corss history distribution"
       << ". Run parameters file name: " << run_par_file_name
       << ", " << fStinName.Data() << ": " << cStexStin_A << ", channel: " << i0StinEcha
       << ", option: " << first_same_plot << endl;

  fHistos->SetHistoMin(fKeyVminD_SCs_ChNb);
  fHistos->SetHistoMax(fKeyVmaxD_SCs_ChNb);
  fHistos->SetHistoScaleY(fKeyScaleY);  fHistos->SetHistoScaleX(fKeyScaleX);
  fHistos->SetHistoColorPalette(fKeyColPal);
  fHistos->GeneralTitle(fKeyGeneralTitle);
  fHistos->FileParameters(fKeyAnaType, fKeyNbOfSamples, 0,
			  fKeyFirstReqEvtNumber, fKeyLastReqEvtNumber, fKeyReqNbOfEvts, fKeyStexNumber);
  fHistos->PlotHistory("SCs", "NOR", run_par_file_name, cStexStin_A, i0StinEcha, first_same_plot);
}

//====================================================================================================

void TEcnaGui::InitKeys()
{
  //.....Input widgets for: analysis, run, channel, sample,
  //                        number of events, first event number, etc...
  
  //fKeyPyf = "";

  fKeyAnaType = "StdPeg12";
  Int_t MaxCar = fgMaxCar;
  fKeyRunListInitCode.Resize(MaxCar);
  fKeyRunListInitCode = "0123";

  MaxCar = fgMaxCar;
  fKeyFileNameRunList.Resize(MaxCar);
  fKeyFileNameRunList = fKeyRunListInitCode.Data();

  fKeyNbOfSamples = fEcal->MaxSampADC();
  fKeyNbOfSamplesString = "10";  // ! THE NUMBER IN STRING MUST BE EQUAL TO fKeyNbOfSamples VALUE

  fKeyNbOfSampForCalc = fEcal->MaxSampADC();
  fKeyNbOfSampForCalcString = "10";  // ! THE NUMBER IN STRING MUST BE EQUAL TO fKeyNbOfSampForCalc VALUE

  fKeyRunNumber  = 0;

  fKeyFirstReqEvtNumber = 1;
  fKeyFirstReqEvtNumberString = "1";  // ! THE NUMBER IN STRING MUST BE EQUAL TO fKeyFirstReqEvtNumber VALUE

  fKeyLastReqEvtNumber = 0;
  fKeyLastReqEvtNumberString = "0";  // ! THE NUMBER IN STRING MUST BE EQUAL TO fKeyLastReqEvtNumber VALUE

  fKeyReqNbOfEvts = 150;
  fKeyReqNbOfEvtsString = "150";  // ! THE NUMBER IN STRING MUST BE EQUAL TO fKeyReqNbOfEvts VALUE

  fKeyStexNumber = 1;
  fKeyStexNumberString = "1";  // ! THE NUMBER IN STRING MUST BE EQUAL TO fKeyStexNumber VALUE

  fKeyChanNumber = 0;
  fKeySampNumber = 0;

  fKeyStinANumber = 1;
  fKeyStinBNumber = 1;
  if( fSubDet == "EE" )
    {if( fKeyStexNumber == 1 || fKeyStexNumber == 3 )
      {
	fKeyStinANumber = fEcalNumbering->Get1DeeSCEcnaFromDeeSCCons(fKeyStexNumber, 150);
	fKeyStinBNumber = fEcalNumbering->Get1DeeSCEcnaFromDeeSCCons(fKeyStexNumber, 150);
      }
    }

  MaxCar = fgMaxCar;
  fKeyScaleX.Resize(MaxCar); 
  fKeyScaleX = "LIN";
  MaxCar = fgMaxCar;
  fKeyScaleY.Resize(MaxCar); 
  fKeyScaleY = "LIN";
  fKeyGeneralTitle = "Ecal Correlated Noise Analysis";

  fKeyColPal = "ECCNAColor";

  //.... ymin and ymax values => values which are displayed on the dialog box

  fKeyVminD_NOE_ChNb = (Double_t)0.; 
  fKeyVmaxD_NOE_ChNb = fKeyReqNbOfEvts + fKeyReqNbOfEvts/3;
 
  fKeyVminD_Ped_ChNb = (Double_t)0.; 
  fKeyVmaxD_Ped_ChNb = (Double_t)0.;

  fKeyVminD_TNo_ChNb = (Double_t)0.; 
  fKeyVmaxD_TNo_ChNb = (Double_t)0.;

  fKeyVminD_LFN_ChNb = (Double_t)0.; 
  fKeyVmaxD_LFN_ChNb = (Double_t)0.;

  fKeyVminD_HFN_ChNb = (Double_t)0.;
  fKeyVmaxD_HFN_ChNb = (Double_t)0.;

  fKeyVminD_MCs_ChNb = (Double_t)(-1.); 
  fKeyVmaxD_MCs_ChNb = (Double_t)1.;

  fKeyVminD_SCs_ChNb = (Double_t)0.; 
  fKeyVmaxD_SCs_ChNb = (Double_t)0.; 

  fKeyVminLHFcc = fKeyVminD_MCs_ChNb;
  fKeyVmaxLHFcc = fKeyVmaxD_MCs_ChNb;

  fKeyVminLFccMos = (Double_t)-1.; 
  fKeyVmaxLFccMos = (Double_t)1.;
  fKeyVminHFccMos = (Double_t)0.; 
  fKeyVmaxHFccMos = (Double_t)1.;

  fKeyFileNameRunList = "";
}

void  TEcnaGui::DisplayInEntryField(TGTextEntry* StringOfField, Int_t& value)
{
  char* f_in = new char[20];          fCnew++;
  sprintf( f_in, "%d", value );
  StringOfField->SetText(f_in);
  delete [] f_in;                     fCdelete++;
}

void  TEcnaGui::DisplayInEntryField(TGTextEntry* StringOfField, Double_t& value)
{
  char* f_in = new char[20];          fCnew++;
  sprintf( f_in, "%g", value );
  StringOfField->SetText(f_in);
  delete [] f_in;                     fCdelete++;
}
void  TEcnaGui::DisplayInEntryField(TGTextEntry* StringOfField, const TString& value)
{
  //StringOfField->Insert(value);
  StringOfField->SetText(value);
}
