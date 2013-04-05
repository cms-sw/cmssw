//---------Author's Name: B.Fabbro DSM/IRFU/SPP CEA-Saclay
//---------Copyright: Those valid for CEA sofware
//---------Modified: 04/07/2011

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaHistos.h"

//--------------------------------------
//  TEcnaHistos.cc
//  Class creation: 18 April 2005
//  Documentation: see TEcnaHistos.h
//--------------------------------------

ClassImp(TEcnaHistos)
//______________________________________________________________________________
//

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//       (attributes) ===> TEcnaParPaths --->|
//                         TEcnaParEcal --->|
//                         TEcnaWrite ---> TEcnaParPaths --->| 
//                                         TEcnaParCout  --->| 
//                                         TEcnaParEcal --->|
//                                         TEcnaNumbering ---> TEcnaParEcal --->|
//                         TEcnaParHistos ---> TEcnaParEcal --->|
//                                             TEcnaNumbering ---> TEcnaParEcal --->|
//                         TEcnaNumbering ---> TEcnaParEcal --->|
//
//                         TEcnaRead ---> TEcnaParCout --->|  
//                                        TEcnaParPaths --->|
//                                        TEcnaHeader --->|
//                                        TEcnaParEcal --->|
//                                        TEcnaWrite ---> TEcnaParPaths --->|
//                                                        TEcnaParCout --->| 
//                                                        TEcnaParEcal --->|
//                                                        TEcnaNumbering ---> TEcnaParEcal --->|
//                                        TEcnaNumbering ---> TEcnaParEcal --->|
//
//
//          Terminal classes: TEcnaParPaths, TEcnaParEcal, TEcnaParCout, TEcnaHeader, TEcnaNArrayD,
//                            TEcnaObject, TEcnaResultType, TEcnaRootFile
//      Non terminal classes: TEcnaGui, TEcnaHistos, TEcnaParHistos, TEcnaNumbering, TEcnaRead,
//                            TEcnaRun, TEcnaWrite
//
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

TEcnaHistos::~TEcnaHistos()
{
  //destructor

  if (fT1DRunNumber != 0){delete [] fT1DRunNumber; fCdelete++;}
  
  //if (fCnaParHistos  != 0){delete fCnaParHistos;  fCdelete++;}
  //if (fCnaParPaths   != 0){delete fCnaParPaths;   fCdelete++;}
  //if (fCnaParCout    != 0){delete fCnaParCout;    fCdelete++;}
  //if (fCnaWrite      != 0){delete fCnaWrite;      fCdelete++;}
  //if (fEcal          != 0){delete fEcal;          fCdelete++;}
  //if (fEcalNumbering != 0){delete fEcalNumbering; fCdelete++;}

  //if (fMyRootFile     != 0){delete fMyRootFile;     fCdelete++;}
  //if (fReadHistoDummy != 0){delete fReadHistoDummy; fCdelete++;}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if ( fCnew != fCdelete )
    {
      cout << "*TEcnaHistos> WRONG MANAGEMENT OF ALLOCATIONS: fCnew = "
	   << fCnew << ", fCdelete = " << fCdelete << fTTBELL << endl;
    }
  else
    {
     //  cout << "*TEcnaHistos> BRAVO! GOOD MANAGEMENT OF ALLOCATIONS: fCnew = "
     //	      << fCnew << ", fCdelete = " << fCdelete << endl;
    }

#define MGRA
#ifndef MGRA
  if ( fCnewRoot != fCdeleteRoot )
    {
      cout << "*TEcnaHistos> WRONG MANAGEMENT OF ROOT ALLOCATIONS: fCnewRoot = "
	   << fCnewRoot << ", fCdeleteRoot = " << fCdeleteRoot << endl;
    }
  else
    {
      cout << "*TEcnaHistos> BRAVO! GOOD MANAGEMENT OF ROOT ALLOCATIONS:"
	   << " fCnewRoot = " << fCnewRoot <<", fCdeleteRoot = "
	   << fCdeleteRoot << endl;
    }
#endif // MGRA

  // cout << "TEcnaHistos> Leaving destructor" << endl;
  // cout << "            fCnew = " << fCnew << ", fCdelete = " << fCdelete << endl;

 // cout << "[Info Management] CLASS: TEcnaHistos.        DESTROY OBJECT: this = " << this << endl;

}

//===================================================================
//
//                   Constructors 
//
//===================================================================
TEcnaHistos::TEcnaHistos(){
// Constructor without argument. Call to Init() 

 // cout << "[Info Management] CLASS: TEcnaHistos.        CREATE OBJECT: this = " << this << endl;

  Init();
}

TEcnaHistos::TEcnaHistos(TEcnaObject* pObjectManager, const TString& SubDet)
{
 // cout << "[Info Management] CLASS: TEcnaHistos.        CREATE OBJECT: this = " << this << endl;


  Long_t i_this = (Long_t)this;
  pObjectManager->RegisterPointer("TEcnaHistos", i_this);

  Init();

  //----------------------- Object management

  //............................ fCnaParCout
  fCnaParCout = 0;
  Long_t iCnaParCout = pObjectManager->GetPointerValue("TEcnaParCout");
  if( iCnaParCout == 0 )
    {fCnaParCout = new TEcnaParCout(pObjectManager); /*fCnew++*/}
  else
    {fCnaParCout = (TEcnaParCout*)iCnaParCout;}

  //............................ fCnaParPaths
  fCnaParPaths = 0;
  Long_t iCnaParPaths = pObjectManager->GetPointerValue("TEcnaParPaths");
  if( iCnaParPaths == 0 )
    {fCnaParPaths = new TEcnaParPaths(pObjectManager); /*fCnew++*/}
  else
    {fCnaParPaths = (TEcnaParPaths*)iCnaParPaths;}

  fCfgResultsRootFilePath    = fCnaParPaths->ResultsRootFilePath();
  fCfgHistoryRunListFilePath = fCnaParPaths->HistoryRunListFilePath();

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

  //............................ fMyRootFile
  fMyRootFile = 0;
  Long_t iMyRootFile = pObjectManager->GetPointerValue("TEcnaRead");
  if( iMyRootFile == 0 )
    {fMyRootFile = new TEcnaRead(pObjectManager, SubDet.Data()); /*fCnew++*/}
  else
    {fMyRootFile = (TEcnaRead*)iMyRootFile;}

  fMyRootFile->PrintNoComment();

  //------------------- creation objet TEcnaRead fMyRootFile (a reprendre plus clairement)
  //fFileHeader = 0;
  //fMyRootFile = new TEcnaRead(fFlagSubDet.Data(), fCnaParPaths, fCnaParCout,
  //			      fFileHeader, fEcalNumbering, fCnaWrite);           fCnew++;
  //fMyRootFile->PrintNoComment();


  SetEcalSubDetector(SubDet.Data());
  //......... init ymin,ymax histos -> Default values for Ymin and Ymax
  SetAllYminYmaxMemoFromDefaultValues();
}


void TEcnaHistos::Init()
{
//========================= GENERAL INITIALISATION 
  fCnew        = 0;
  fCdelete     = 0;
  fCnewRoot    = 0;
  fCdeleteRoot = 0;

  fCnaCommand  = 0;
  fCnaError    = 0;

  fgMaxCar = 512;
  Int_t MaxCar = fgMaxCar;

  //------------------------------ initialisations ----------------------
  fTTBELL = '\007';

  fT1DRunNumber = 0;

  //.......... init flags Same plot
  fMemoPlotH1SamePlus = 0;
  fMemoPlotD_NOE_ChNb = 0; fMemoPlotD_NOE_ChDs = 0;
  fMemoPlotD_Ped_ChNb = 0; fMemoPlotD_Ped_ChDs = 0;
  fMemoPlotD_TNo_ChNb = 0; fMemoPlotD_TNo_ChDs = 0; 
  fMemoPlotD_MCs_ChNb = 0; fMemoPlotD_MCs_ChDs = 0;
  fMemoPlotD_LFN_ChNb = 0; fMemoPlotD_LFN_ChDs = 0; 
  fMemoPlotD_HFN_ChNb = 0; fMemoPlotD_HFN_ChDs = 0; 
  fMemoPlotD_SCs_ChNb = 0; fMemoPlotD_SCs_ChDs = 0; 
  fMemoPlotD_MSp_SpNb = 0; fMemoPlotD_SSp_SpNb = 0; 
  fMemoPlotD_MSp_SpDs = 0; fMemoPlotD_SSp_SpDs = 0;
  fMemoPlotD_Adc_EvDs = 0; fMemoPlotD_Adc_EvNb = 0;
  fMemoPlotH_Ped_Date = 0; fMemoPlotH_TNo_Date = 0;
  fMemoPlotH_MCs_Date = 0; fMemoPlotH_LFN_Date = 0;
  fMemoPlotH_HFN_Date = 0; fMemoPlotH_SCs_Date = 0;
  fMemoPlotH_Ped_RuDs = 0; fMemoPlotH_TNo_RuDs = 0;
  fMemoPlotH_MCs_RuDs = 0; fMemoPlotH_LFN_RuDs = 0;
  fMemoPlotH_HFN_RuDs = 0; fMemoPlotH_SCs_RuDs = 0;
  //.......... init flags colors                                       (Init)
  fMemoColorH1SamePlus = 0;
  fMemoColorD_NOE_ChNb = 0; fMemoColorD_NOE_ChDs = 0;
  fMemoColorD_Ped_ChNb = 0; fMemoColorD_Ped_ChDs = 0;
  fMemoColorD_TNo_ChNb = 0; fMemoColorD_TNo_ChDs = 0; 
  fMemoColorD_MCs_ChNb = 0; fMemoColorD_MCs_ChDs = 0;
  fMemoColorD_LFN_ChNb = 0; fMemoColorD_LFN_ChDs = 0; 
  fMemoColorD_HFN_ChNb = 0; fMemoColorD_HFN_ChDs = 0; 
  fMemoColorD_SCs_ChNb = 0; fMemoColorD_SCs_ChDs = 0; 
  fMemoColorD_MSp_SpNb = 0; fMemoColorD_SSp_SpNb = 0; 
  fMemoColorD_MSp_SpDs = 0; fMemoColorD_SSp_SpDs = 0;
  fMemoColorD_Adc_EvDs = 0; fMemoColorD_Adc_EvNb = 0;
  fMemoColorH_Ped_Date = 0; fMemoColorH_TNo_Date = 0;
  fMemoColorH_MCs_Date = 0; fMemoColorH_LFN_Date = 0;
  fMemoColorH_HFN_Date = 0; fMemoColorH_SCs_Date = 0;
  fMemoColorH_Ped_RuDs = 0; fMemoColorH_TNo_RuDs = 0;
  fMemoColorH_MCs_RuDs = 0; fMemoColorH_LFN_RuDs = 0;
  fMemoColorH_HFN_RuDs = 0; fMemoColorH_SCs_RuDs = 0;

  //.......... init counter Same canvas
  fCanvSameH1SamePlus = 0;
  fCanvSameD_NOE_ChNb = 0; fCanvSameD_NOE_ChDs = 0;
  fCanvSameD_Ped_ChNb = 0; fCanvSameD_Ped_ChDs = 0;
  fCanvSameD_TNo_ChNb = 0; fCanvSameD_TNo_ChDs = 0; 
  fCanvSameD_MCs_ChNb = 0; fCanvSameD_MCs_ChDs = 0;
  fCanvSameD_LFN_ChNb = 0; fCanvSameD_LFN_ChDs = 0; 
  fCanvSameD_HFN_ChNb = 0; fCanvSameD_HFN_ChDs = 0; 
  fCanvSameD_SCs_ChNb = 0; fCanvSameD_SCs_ChDs = 0; 
  fCanvSameD_MSp_SpNb = 0; fCanvSameD_SSp_SpNb = 0;
  fCanvSameD_MSp_SpDs = 0; fCanvSameD_SSp_SpDs = 0;
  fCanvSameD_Adc_EvDs = 0; fCanvSameD_Adc_EvNb = 0;
  fCanvSameH_Ped_Date = 0; fCanvSameH_TNo_Date = 0;
  fCanvSameH_MCs_Date = 0; fCanvSameH_LFN_Date = 0;
  fCanvSameH_HFN_Date = 0; fCanvSameH_SCs_Date = 0;
  fCanvSameH_Ped_RuDs = 0; fCanvSameH_TNo_RuDs = 0;
  fCanvSameH_MCs_RuDs = 0; fCanvSameH_LFN_RuDs = 0;
  fCanvSameH_HFN_RuDs = 0; fCanvSameH_SCs_RuDs = 0;
  //................. Flag Scale X anf Y set to "LIN" and flag color palete set to "Black/Red/Blue"

  MaxCar = fgMaxCar;
  fFlagScaleX.Resize(MaxCar);
  fFlagScaleX = "LIN";

  MaxCar = fgMaxCar;                                     //   (Init)
  fFlagScaleY.Resize(MaxCar);
  fFlagScaleY = "LIN";

  MaxCar = fgMaxCar;
  fFlagColPal.Resize(MaxCar);
  fFlagColPal = "Black/Red/Blue";

  //................. Flag General Title set to empty string
  MaxCar = fgMaxCar;
  fFlagGeneralTitle.Resize(MaxCar);
  fFlagGeneralTitle = "";

  //................. Init codes Options
  fOptScaleLinx = 31400;
  fOptScaleLogx = 31401;
  fOptScaleLiny = 31402;
  fOptScaleLogy = 31403;

  fOptVisLine = 1101; 
  fOptVisPolm = 1102;

  //............................                                       (Init)
  MaxCar = fgMaxCar;
  fCovarianceMatrix.Resize(MaxCar);
  fCovarianceMatrix = "Cov";
  MaxCar = fgMaxCar;
  fCorrelationMatrix.Resize(MaxCar);
  fCorrelationMatrix = "Cor";

  MaxCar = fgMaxCar;
  fLFBetweenStins.Resize(MaxCar);
  fLFBetweenStins = "MttLF";
  MaxCar = fgMaxCar;
  fHFBetweenStins.Resize(MaxCar);
  fHFBetweenStins = "MttHF";

  MaxCar = fgMaxCar;
  fLFBetweenChannels.Resize(MaxCar);
  fLFBetweenChannels = "MccLF";
  MaxCar = fgMaxCar;
  fHFBetweenChannels.Resize(MaxCar);
  fHFBetweenChannels = "MccHF";

  MaxCar = fgMaxCar;
  fBetweenSamples.Resize(MaxCar);
  fBetweenSamples = "Mss";

  //.................................. text pave alignement for pave "SeveralChanging" (HistimePlot)
  fTextPaveAlign  = 12;              // 1 = left adjusted, 2 = vertically centered
  fTextPaveFont   = 100;             // 10*10 = 10*(ID10 = Courier New)
  fTextPaveSize   = (Float_t)0.025;  // 0.0xxx = xxx% of the pave size
  fTextBorderSize = 1;               // Pave Border (=>Shadow)

  //................................. Init Xvar, Yvar, NbBins management for options SAME and SAME n
  fXMemoH1SamePlus = "";
  fXMemoD_NOE_ChNb = "";
  fXMemoD_NOE_ChDs = "";
  fXMemoD_Ped_ChNb = "";
  fXMemoD_Ped_ChDs = "";
  fXMemoD_TNo_ChNb = "";   
  fXMemoD_TNo_ChDs = ""; 
  fXMemoD_MCs_ChNb = ""; 
  fXMemoD_MCs_ChDs = "";
  fXMemoD_LFN_ChNb = "";
  fXMemoD_LFN_ChDs = ""; 
  fXMemoD_HFN_ChNb = "";   
  fXMemoD_HFN_ChDs = ""; 
  fXMemoD_SCs_ChNb = ""; 
  fXMemoD_SCs_ChDs = ""; 
  fXMemoD_MSp_SpNb = "";
  fXMemoD_MSp_SpDs = "";
  fXMemoD_SSp_SpNb = "";  
  fXMemoD_SSp_SpDs = "";  
  fXMemoD_Adc_EvDs = "";     
  fXMemoD_Adc_EvNb = "";
  fXMemoH_Ped_Date = "";
  fXMemoH_TNo_Date = "";
  fXMemoH_MCs_Date = "";
  fXMemoH_LFN_Date = "";
  fXMemoH_HFN_Date = "";
  fXMemoH_SCs_Date = "";
  fXMemoH_Ped_RuDs = "";
  fXMemoH_TNo_RuDs = "";
  fXMemoH_MCs_RuDs = "";
  fXMemoH_LFN_RuDs = "";
  fXMemoH_HFN_RuDs = "";
  fXMemoH_SCs_RuDs = "";

  fYMemoH1SamePlus = "";
  fYMemoD_NOE_ChNb = "";
  fYMemoD_NOE_ChDs = "";
  fYMemoD_Ped_ChNb = "";
  fYMemoD_Ped_ChDs = "";
  fYMemoD_TNo_ChNb = "";
  fYMemoD_TNo_ChDs = "";
  fYMemoD_MCs_ChNb = "";
  fYMemoD_MCs_ChDs = "";
  fYMemoD_LFN_ChNb = "";
  fYMemoD_LFN_ChDs = "";
  fYMemoD_HFN_ChNb = "";
  fYMemoD_HFN_ChDs = "";
  fYMemoD_SCs_ChNb = "";
  fYMemoD_SCs_ChDs = "";
  fYMemoD_MSp_SpNb = "";
  fYMemoD_MSp_SpDs = "";
  fYMemoD_SSp_SpNb = ""; 
  fYMemoD_SSp_SpDs = "";  
  fYMemoD_Adc_EvDs = "";     
  fYMemoD_Adc_EvNb = "";
  fYMemoH_Ped_Date = "";
  fYMemoH_TNo_Date = "";
  fYMemoH_MCs_Date = "";
  fYMemoH_LFN_Date = "";
  fYMemoH_HFN_Date = "";
  fYMemoH_SCs_Date = "";
  fYMemoH_Ped_RuDs = "";
  fYMemoH_TNo_RuDs = "";
  fYMemoH_MCs_RuDs = "";
  fYMemoH_LFN_RuDs = "";
  fYMemoH_HFN_RuDs = "";
  fYMemoH_SCs_RuDs = "";

  fNbBinsMemoH1SamePlus = 0;
  fNbBinsMemoD_NOE_ChNb = 0;
  fNbBinsMemoD_NOE_ChDs = 0;
  fNbBinsMemoD_Ped_ChNb = 0;
  fNbBinsMemoD_Ped_ChDs = 0;
  fNbBinsMemoD_TNo_ChNb = 0;   
  fNbBinsMemoD_TNo_ChDs = 0; 
  fNbBinsMemoD_MCs_ChNb = 0; 
  fNbBinsMemoD_MCs_ChDs = 0;
  fNbBinsMemoD_LFN_ChNb = 0;
  fNbBinsMemoD_LFN_ChDs = 0; 
  fNbBinsMemoD_HFN_ChNb = 0;   
  fNbBinsMemoD_HFN_ChDs = 0; 
  fNbBinsMemoD_SCs_ChNb = 0; 
  fNbBinsMemoD_SCs_ChDs = 0; 
  fNbBinsMemoD_MSp_SpNb = 0; 
  fNbBinsMemoD_MSp_SpDs = 0;
  fNbBinsMemoD_SSp_SpNb = 0; 
  fNbBinsMemoD_SSp_SpDs = 0;  
  fNbBinsMemoD_Adc_EvDs = 0;     
  fNbBinsMemoD_Adc_EvNb = 0;
  fNbBinsMemoH_Ped_Date = 0;
  fNbBinsMemoH_TNo_Date = 0;
  fNbBinsMemoH_MCs_Date = 0;
  fNbBinsMemoH_LFN_Date = 0;
  fNbBinsMemoH_HFN_Date = 0;
  fNbBinsMemoH_SCs_Date = 0;
  fNbBinsMemoH_Ped_RuDs = 0;
  fNbBinsMemoH_TNo_RuDs = 0;
  fNbBinsMemoH_MCs_RuDs = 0;
  fNbBinsMemoH_LFN_RuDs = 0;
  fNbBinsMemoH_HFN_RuDs = 0;
  fNbBinsMemoH_SCs_RuDs = 0;

  //.................................. Init canvas/pad pointers                (Init)
  fCurrentCanvas         = 0;

  fCurrentCanvasName     = "?";

  fCanvH1SamePlus = 0; 
  fCanvD_NOE_ChNb = 0;
  fCanvD_NOE_ChDs = 0;
  fCanvD_Ped_ChNb = 0;
  fCanvD_Ped_ChDs = 0;
  fCanvD_TNo_ChNb = 0;   
  fCanvD_TNo_ChDs = 0; 
  fCanvD_MCs_ChNb = 0; 
  fCanvD_MCs_ChDs = 0;
  fCanvD_LFN_ChNb = 0;
  fCanvD_LFN_ChDs = 0; 
  fCanvD_HFN_ChNb = 0;   
  fCanvD_HFN_ChDs = 0; 
  fCanvD_SCs_ChNb = 0; 
  fCanvD_SCs_ChDs = 0; 
  fCanvD_MSp_SpNb = 0; 
  fCanvD_MSp_SpDs = 0;
  fCanvD_SSp_SpNb = 0;
  fCanvD_SSp_SpDs = 0;  
  fCanvD_Adc_EvDs = 0;     
  fCanvD_Adc_EvNb = 0;
  fCanvH_Ped_Date = 0;
  fCanvH_TNo_Date = 0;
  fCanvH_MCs_Date = 0;
  fCanvH_LFN_Date = 0;
  fCanvH_HFN_Date = 0;
  fCanvH_SCs_Date = 0;
  fCanvH_Ped_RuDs = 0;
  fCanvH_TNo_RuDs = 0;
  fCanvH_MCs_RuDs = 0;
  fCanvH_LFN_RuDs = 0;
  fCanvH_HFN_RuDs = 0;
  fCanvH_SCs_RuDs = 0;
  

  fClosedH1SamePlus = kFALSE;    // (Canvas Closed SIGNAL)
  fClosedD_NOE_ChNb = kFALSE;
  fClosedD_NOE_ChDs = kFALSE;
  fClosedD_Ped_ChNb = kFALSE;
  fClosedD_Ped_ChDs = kFALSE;
  fClosedD_TNo_ChNb = kFALSE; 
  fClosedD_TNo_ChDs = kFALSE; 
  fClosedD_MCs_ChNb = kFALSE; 
  fClosedD_MCs_ChDs = kFALSE;
  fClosedD_LFN_ChNb = kFALSE;
  fClosedD_LFN_ChDs = kFALSE; 
  fClosedD_HFN_ChNb = kFALSE; 
  fClosedD_HFN_ChDs = kFALSE; 
  fClosedD_SCs_ChNb = kFALSE; 
  fClosedD_SCs_ChDs = kFALSE; 
  fClosedD_MSp_SpNb = kFALSE; 
  fClosedD_MSp_SpDs = kFALSE;
  fClosedD_SSp_SpNb = kFALSE;
  fClosedD_SSp_SpDs = kFALSE;  
  fClosedD_Adc_EvDs = kFALSE;   
  fClosedD_Adc_EvNb = kFALSE;
  fClosedH_Ped_Date = kFALSE;
  fClosedH_TNo_Date = kFALSE;
  fClosedH_MCs_Date = kFALSE;
  fClosedH_LFN_Date = kFALSE;
  fClosedH_HFN_Date = kFALSE;
  fClosedH_SCs_Date = kFALSE;
  fClosedH_Ped_RuDs = kFALSE;
  fClosedH_TNo_RuDs = kFALSE;
  fClosedH_MCs_RuDs = kFALSE;
  fClosedH_LFN_RuDs = kFALSE;
  fClosedH_HFN_RuDs = kFALSE;
  fClosedH_SCs_RuDs = kFALSE;

  fCurrentPad = 0;                                    //   (Init)

  fPadH1SamePlus = 0;
  fPadD_NOE_ChNb = 0;
  fPadD_NOE_ChDs = 0;
  fPadD_Ped_ChNb = 0;
  fPadD_Ped_ChDs = 0;
  fPadD_TNo_ChNb = 0;   
  fPadD_TNo_ChDs = 0; 
  fPadD_MCs_ChNb = 0; 
  fPadD_MCs_ChDs = 0;
  fPadD_LFN_ChNb = 0;
  fPadD_LFN_ChDs = 0; 
  fPadD_HFN_ChNb = 0;   
  fPadD_HFN_ChDs = 0; 
  fPadD_SCs_ChNb = 0; 
  fPadD_SCs_ChDs = 0; 
  fPadD_MSp_SpNb = 0;
  fPadD_MSp_SpDs = 0;
  fPadD_SSp_SpNb = 0;
  fPadD_SSp_SpDs = 0;
  fPadD_Adc_EvDs = 0;
  fPadD_Adc_EvNb = 0;
  fPadH_Ped_Date = 0;
  fPadH_TNo_Date = 0;
  fPadH_MCs_Date = 0;
  fPadH_LFN_Date = 0;
  fPadH_HFN_Date = 0;
  fPadH_SCs_Date = 0;
  fPadH_Ped_RuDs = 0;
  fPadH_TNo_RuDs = 0;
  fPadH_MCs_RuDs = 0;
  fPadH_LFN_RuDs = 0;
  fPadH_HFN_RuDs = 0;
  fPadH_SCs_RuDs = 0;

  fPavTxtH1SamePlus = 0;                                    //   (Init)
  fPavTxtD_NOE_ChNb = 0;
  fPavTxtD_NOE_ChDs = 0;
  fPavTxtD_Ped_ChNb = 0;
  fPavTxtD_Ped_ChDs = 0;
  fPavTxtD_TNo_ChNb = 0;   
  fPavTxtD_TNo_ChDs = 0; 
  fPavTxtD_MCs_ChNb = 0; 
  fPavTxtD_MCs_ChDs = 0;
  fPavTxtD_LFN_ChNb = 0;
  fPavTxtD_LFN_ChDs = 0; 
  fPavTxtD_HFN_ChNb = 0;   
  fPavTxtD_HFN_ChDs = 0; 
  fPavTxtD_SCs_ChNb = 0; 
  fPavTxtD_SCs_ChDs = 0; 
  fPavTxtD_MSp_SpNb = 0; 
  fPavTxtD_MSp_SpDs = 0;
  fPavTxtD_SSp_SpNb = 0;
  fPavTxtD_SSp_SpDs = 0;
  fPavTxtD_Adc_EvDs = 0;
  fPavTxtD_Adc_EvNb = 0;
  fPavTxtH_Ped_Date = 0;
  fPavTxtH_TNo_Date = 0;
  fPavTxtH_MCs_Date = 0;
  fPavTxtH_LFN_Date = 0;
  fPavTxtH_HFN_Date = 0;
  fPavTxtH_SCs_Date = 0;
  fPavTxtH_Ped_RuDs = 0;
  fPavTxtH_TNo_RuDs = 0;
  fPavTxtH_MCs_RuDs = 0;
  fPavTxtH_LFN_RuDs = 0;
  fPavTxtH_HFN_RuDs = 0;
  fPavTxtH_SCs_RuDs = 0;

  fImpH1SamePlus = 0;                                    //   (Init)
  fImpD_NOE_ChNb = 0;
  fImpD_NOE_ChDs = 0;
  fImpD_Ped_ChNb = 0;
  fImpD_Ped_ChDs = 0;
  fImpD_TNo_ChNb = 0;   
  fImpD_TNo_ChDs = 0; 
  fImpD_MCs_ChNb = 0; 
  fImpD_MCs_ChDs = 0;
  fImpD_LFN_ChNb = 0;
  fImpD_LFN_ChDs = 0; 
  fImpD_HFN_ChNb = 0;   
  fImpD_HFN_ChDs = 0; 
  fImpD_SCs_ChNb = 0; 
  fImpD_SCs_ChDs = 0; 
  fImpD_MSp_SpNb = 0; 
  fImpD_MSp_SpDs = 0;
  fImpD_SSp_SpNb = 0; 
  fImpD_SSp_SpDs = 0;  
  fImpD_Adc_EvDs = 0;     
  fImpD_Adc_EvNb = 0;
  fImpH_Ped_Date = 0;
  fImpH_TNo_Date = 0;
  fImpH_MCs_Date = 0;
  fImpH_LFN_Date = 0;
  fImpH_HFN_Date = 0;
  fImpH_SCs_Date = 0;
  fImpH_Ped_RuDs = 0;
  fImpH_TNo_RuDs = 0;
  fImpH_MCs_RuDs = 0;
  fImpH_LFN_RuDs = 0;
  fImpH_HFN_RuDs = 0;
  fImpH_SCs_RuDs = 0;

  fNbBinsProj = 100;       // number of bins for histos in option Projection

  //.................................... Miscellaneous parameters                (Init)

  fNbOfListFileH_Ped_Date = 0;
  fNbOfListFileH_TNo_Date = 0;
  fNbOfListFileH_MCs_Date = 0;
  fNbOfListFileH_LFN_Date = 0;
  fNbOfListFileH_HFN_Date = 0;
  fNbOfListFileH_SCs_Date = 0;

  fNbOfListFileH_Ped_RuDs = 0;
  fNbOfListFileH_TNo_RuDs = 0;
  fNbOfListFileH_MCs_RuDs = 0;
  fNbOfListFileH_LFN_RuDs = 0;
  fNbOfListFileH_HFN_RuDs = 0;
  fNbOfListFileH_SCs_RuDs = 0;

  fNbOfExistingRuns = 0;

  fFapNbOfRuns    = -1;      // INIT NUMBER OF RUNS: set to -1
  fFapMaxNbOfRuns = -1;      // INIT MAXIMUM NUMBER OF RUNS: set to -1 

  MaxCar = fgMaxCar;
  fFapFileRuns.Resize(MaxCar);
  fFapFileRuns = "(file with list of runs parameters: no info)";

  fStartEvolTime = 0;
  fStopEvolTime  = 0;
  fStartEvolDate = "Start date: not known";
  fStopEvolDate  = "Stop date:  not known";

  fStartEvolRun  = 0;
  fStopEvolRun   = 0;

  fRunType       = "Run type: not known";

  fFapNbOfEvts = 0;

  MaxCar = fgMaxCar;
  fMyRootFileName.Resize(MaxCar);
  fMyRootFileName = "No ROOT file name available (fMyRootFileName).";

  fFapAnaType           = "Analysis name: not known"; // Init Type of analysis
  fFapNbOfSamples       = 0; // Init Nb of required samples
  fFapRunNumber         = 0; // Init Run number
  fFapFirstReqEvtNumber = 0; // Init First requested event number
  fFapLastReqEvtNumber  = 0; // Init Last requested event number
  fFapReqNbOfEvts       = 0; // Init Requested number of events
  fFapStexNumber        = 0; // Init Stex number

  //------------------ Init read file flags
  fAlreadyRead = 1;
  fMemoAlreadyRead = 0;
  fTobeRead = 0;
  fZerv    =   0;
  fUnev    =   1;
  TVectorD fReadHistoDummy(fUnev);
  TMatrixD fReadMatrixDummy(fUnev, fUnev);

  //------------------ Init fAsciiFileName
  fAsciiFileName = "?";

} // end of Init()

//----------------------------------------------------------------------------------------
void TEcnaHistos::SetEcalSubDetector(const TString& SubDet)
{
 // Set Subdetector (EB or EE)

  Int_t MaxCar = fgMaxCar;
  fFlagSubDet.Resize(MaxCar);
  fFlagSubDet = fEcal->GetEcalSubDetector();      // fFlagSubDet = "EB" or "EE"

  //.................................. Init specific EB/EE parameters ( SetEcalSubDetector(...) )
  MaxCar = fgMaxCar;
  fFapStexName.Resize(MaxCar);
  fFapStexName = "no info for Stex";
  MaxCar = fgMaxCar;
  fFapStinName.Resize(MaxCar);
  fFapStinName = "no info for Stin";
  MaxCar = fgMaxCar;
  fFapXtalName.Resize(MaxCar);
  fFapXtalName = "no info for Xtal";
  MaxCar = fgMaxCar;
  fFapEchaName.Resize(MaxCar);
  fFapEchaName = "no info for Echa";

  if( fFlagSubDet == "EB" )
    {
      fFapStexName   = "SM";
      fFapStinName   = "Tower";
      fFapXtalName   = "Xtal";
      fFapEchaName   = "Chan";
      fFapStexBarrel = fEcalNumbering->GetSMHalfBarrel(fFapStexNumber);
    }

  if( fFlagSubDet == "EE" )
    {
      fFapStexName     = "Dee";
      fFapStinName     = "SC";
      fFapXtalName     = "Xtal";
      fFapEchaName     = "Chan";
      fFapStexType     = fEcalNumbering->GetEEDeeType(fFapStexNumber);
      fFapStexDir      = "right";
      fFapStinQuadType = "top";
    }

  //........................ init code plot type                     (SetEcalSubDetector)
  MaxCar = fgMaxCar;
  fOnlyOnePlot.Resize(MaxCar);
  fOnlyOnePlot = fCnaParHistos->GetCodeOnlyOnePlot();  // "ONLYONE"

  MaxCar = fgMaxCar;
  fSeveralPlot.Resize(MaxCar);
  fSeveralPlot = fCnaParHistos->GetCodeSeveralPlot();  // "SEVERAL"

  MaxCar = fgMaxCar;
  fSameOnePlot.Resize(MaxCar);
  fSameOnePlot = fCnaParHistos->GetCodeSameOnePlot();  // "SAME n";

  MaxCar = fgMaxCar;
  fAllXtalsInStinPlot.Resize(MaxCar);
  fAllXtalsInStinPlot = fCnaParHistos->GetCodeAllXtalsInStinPlot();  // "SAME in Stin";

  fPlotAllXtalsInStin = fCnaParHistos->GetCodePlotAllXtalsInStin();  //  0

} // ---------------- end of  SetEcalSubDetector(...) ----------------

//--------------------------------------------------------------------------------------------
//
//          FileParameters(s)(...) 
//
//--------------------------------------------------------------------------------------------

//===> DON'T SUPPRESS: THESE METHODS ARE CALLED BY TEcnaGui and can be called by any other program
void TEcnaHistos::FileParameters(const TString& xArgAnaType,          const Int_t&  xArgNbOfSamples, 
				 const Int_t&  xArgRunNumber,        const Int_t&  xArgFirstReqEvtNumber,
				 const Int_t&  xArgLastReqEvtNumber, const Int_t&  xArgReqNbOfEvts, 
				 const Int_t&  xArgStexNumber)
{
// Set parameters for reading the right ECNA results file

  fFapAnaType           = xArgAnaType;
  fFapNbOfSamples       = xArgNbOfSamples;
  fFapRunNumber         = xArgRunNumber;
  fFapFirstReqEvtNumber = xArgFirstReqEvtNumber;
  fFapLastReqEvtNumber  = xArgLastReqEvtNumber;
  fFapReqNbOfEvts       = xArgReqNbOfEvts;
  fFapStexNumber        = xArgStexNumber;

  InitSpecParBeforeFileReading();   // SpecPar = Special Parameters (dates, times, run types)
}

void TEcnaHistos::FileParameters(TEcnaRead* MyRootFile)
{
// Set parameters for reading the right ECNA results file

  InitSpecParBeforeFileReading();   // SpecPar = Special Parameters (dates, times, run types)

  //............... Filename parameter values
  fFapAnaType           = MyRootFile->GetAnalysisName();
  fFapNbOfSamples       = MyRootFile->GetNbOfSamples();
  fFapRunNumber         = MyRootFile->GetRunNumber();
  fFapFirstReqEvtNumber = MyRootFile->GetFirstReqEvtNumber();
  fFapLastReqEvtNumber  = MyRootFile->GetLastReqEvtNumber();
  fFapReqNbOfEvts       = MyRootFile->GetReqNbOfEvts();
  fFapStexNumber        = MyRootFile->GetStexNumber();

  //............... parameter values from file contents
  fStartDate = MyRootFile->GetStartDate();
  fStopDate  = MyRootFile->GetStopDate();
  fRunType   = MyRootFile->GetRunType();

  fFapNbOfEvts = MyRootFile->GetNumberOfEvents(fFapReqNbOfEvts, fFapStexNumber);
}

//=============================================================================================
//                 Set general title
//                 Set lin or log scale on X or Y axis
//                 Set color palette
//                 Set start and stop date
//                 Set run type
//=============================================================================================
//............................................................................................ 
void TEcnaHistos::GeneralTitle(const TString& title)
{
  fFlagGeneralTitle = title.Data();
}
void TEcnaHistos::SetHistoScaleX(const TString&  option_scale)
{
  fFlagScaleX = "LIN";
  if ( option_scale == "LOG" ){fFlagScaleX = "LOG";}
}
void TEcnaHistos::SetHistoScaleY(const TString&  option_scale)
{
  fFlagScaleY = "LIN";
  if ( option_scale == "LOG" ){fFlagScaleY = "LOG";}
}
void TEcnaHistos::SetHistoColorPalette (const TString&  option_palette)
{
  fFlagColPal = "Black/Red/Blue";
  if ( !(option_palette == "Rainbow" || option_palette == "rainbow") ){fFlagColPal = "Black/Red/Blue";}
  if (   option_palette == "Rainbow" || option_palette == "rainbow"  ){fFlagColPal = "Rainbow";}
}
void TEcnaHistos::StartStopDate(const TString& start_date, const TString& stop_date)
{
  fStartDate = start_date.Data();
  fStopDate  = stop_date.Data();
}
void TEcnaHistos::RunType(const TString& run_type)
{
  fRunType = run_type.Data();
}
void TEcnaHistos::NumberOfEvents(const Int_t& nb_of_evts)
{
  fFapNbOfEvts = nb_of_evts;
}
//====================== return status for root file and data existence
Bool_t TEcnaHistos::StatusFileFound(){return fStatusFileFound;}
Bool_t TEcnaHistos::StatusDataExist(){return fStatusDataExist;}

//=======================================================================================
//
//                       ( R e a d A n d ) P l o t    (1D , 2D , History)
//
//=======================================================================================
//---------------------------------------------------------------------------------------------
// TechHistoCode list modification (06/10/09)
//
//    D = Detector Plot    ChNb = Channel Number
//                         ChDs = Channel Distribution (Y projection)
//
//    H = History  Plot    Date = date in format YYMMJJ hhmmss
//                         RuDs = Run distribution
//
//      old code             new code    std code X  std code Y   (std = standard)
//                     
// *  1 H1NbOfEvtsGlobal     D_NOE_ChNb   Xtal        NOE            NOE = Number Of Events
// *  2 H1NbOfEvtsProj       D_NOE_ChDs   NOE         NOX            NOX = Number Of Xtals
// *  3 H1EvEvGlobal         D_Ped_ChNb   Xtal        Ped            Ped = Pedestal
// *  4 H1EvEvProj           D_Ped_ChDs   Ped         NOX
// *  5 H1EvSigGlobal        D_TNo_ChNb   Xtal        TNo            TNo = Total Noise
// *  6 H1EvSigProj          D_TNo_ChDs   TNo         NOX
// *  7 H1SigEvGlobal        D_LFN_ChNb   Xtal        LFN            LFN = Low Frequency noise
// *  8 H1SigEvProj          D_LFN_ChDs   LFN         NOX
// *  9 H1SigSigGlobal       D_HFN_ChNb   Xtal        HFN            HFN = High Frequency noise
// * 10 H1SigSigProj         D_HFN_ChDs   HFN         NOX
// * 11 H1EvCorssGlobal      D_MCs_ChNb   Xtal        MCs            MCs = Mean correlations between samples
// * 12 H1EvCorssProj        D_MCs_ChDs   MCs         NOX
// * 13 H1SigCorssGlobal     D_SCs_ChNb   Xtal        SCs            SCs = Sigma of the correlations between samples
// * 14 H1SigCorssProj       D_SCs_ChDs   SCs         NOX
// * 15 Ev                   D_MSp_SpNb   Sample      MSp            MSp = Means  of the samples
// * 16 EvProj               D_MSp_SpDs   MSp         NOS            NOS = Number of samples
// * 17 Sigma                D_SSp_SpNb   Sample      SSp            SSp = Sigmas of the samples
// * 18 SigmaProj            D_SSp_SpDs   SSp         NOS 
// * 19 SampTime             D_Adc_EvNb   Event       Adc            Adc = ADC count as a function of Event number
// * 20 AdcProj              D_Adc_EvDs   Adc         NOE            EvDs = Event distribution
// * 21 EvolEvEv             H_Ped_Date   Time        Ped            Time = date YY/MM/DD hh:mm:ss
// * 22 EvolEvEvProj         H_Ped_RuDs   Ped         NOR            NOR  = Number Of Runs
// * 23 EvolEvSig            H_TNo_Date   Time        TNo
// * 24 EvolEvSigProj        H_TNo_RuDs   TNo         NOR
// * 25 EvolSigEv            H_LFN_Date   Time        LFN
// * 26 EvolSigEvProj        H_LFN_RuDs   LFN         NOR
// * 27 EvolSigSig           H_HFN_Date   Time        HFN
// * 28 EvolSigSigProj       H_HFN_RuDs   HFN         NOR
// * 29 EvolEvCorss          H_MCs_Date   Time        MCs
// * 30 EvolEvCorssProj      H_MCs_RuDs   MCs         NOR
// * 31 EvolSigCorss         H_SCs_Date   Time        SCs
// * 32 EvolSigCorssProj     H_SCs_RuDs   SCs         NOR
//
//---------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------
//
//                              (ReadAnd)PlotMatrix
//
//---------------------------------------------------------------------------------------
//................................ Corcc[for 1 Stex] (big matrix), Cortt[for 1 Stex]
void TEcnaHistos::PlotMatrix(const TMatrixD& read_matrix_corcc,
			     const TString&   UserCorOrCov, const TString& UserBetweenWhat)
{PlotMatrix(read_matrix_corcc, UserCorOrCov, UserBetweenWhat, "");}

void TEcnaHistos::PlotMatrix(const TMatrixD& read_matrix_corcc,
			     const TString&   UserCorOrCov, const TString& UserBetweenWhat,
			     const TString&   UserPlotOption)
{
  TString CallingMethod = "2D";

  TString StandardPlotOption = fCnaParHistos->BuildStandardPlotOption(CallingMethod, UserPlotOption); 

  TString BetweenWhat = fCnaParHistos->BuildStandardBetweenWhatCode(CallingMethod, UserBetweenWhat);
  TString CorOrCov = fCnaParHistos->BuildStandardCovOrCorCode(CallingMethod, UserCorOrCov);
  
  if( BetweenWhat != "?" && CorOrCov != "?" )
    {
      if( BetweenWhat == "MttLF" ||  BetweenWhat == "MttHF" )
	{
	  fAlreadyRead = 1;
	  ViewMatrix(read_matrix_corcc, fAlreadyRead,
		     fZerv, fZerv, fZerv, CorOrCov, BetweenWhat, StandardPlotOption); 
	}
      if( BetweenWhat == "MccLF" ){StexHocoVecoLHFCorcc("LF");} // forced to Read file and Plot
      if( BetweenWhat == "MccHF" ){StexHocoVecoLHFCorcc("HF");} // forced to Read file and Plot
    }
  else
    {fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
      cout << "!TEcnaHistos::PlotMatrix(...)> Histo cannot be reached." << fTTBELL << endl;}
}

void TEcnaHistos::PlotMatrix(const TString& UserCorOrCov, const TString& UserBetweenWhat)
{PlotMatrix(UserCorOrCov, UserBetweenWhat, "");}

void TEcnaHistos::PlotMatrix(const TString& UserCorOrCov, const TString& UserBetweenWhat,
			     const TString& UserPlotOption)
{
  TString CallingMethod = "2D";

  TString StandardPlotOption = fCnaParHistos->BuildStandardPlotOption(CallingMethod, UserPlotOption); 

  TString BetweenWhat = fCnaParHistos->BuildStandardBetweenWhatCode(CallingMethod, UserBetweenWhat);
  TString CorOrCov = fCnaParHistos->BuildStandardCovOrCorCode(CallingMethod, UserCorOrCov);

  if( BetweenWhat != "?" && CorOrCov != "?" )
    {
      if( BetweenWhat == "MttLF" ||  BetweenWhat == "MttHF" )
	{
	  ViewMatrix(fReadMatrixDummy, fTobeRead,
		     fZerv, fZerv, fZerv, CorOrCov, BetweenWhat, StandardPlotOption);
	}
      if( BetweenWhat == "MccLF" ){StexHocoVecoLHFCorcc("LF");} // Plot  only
      if( BetweenWhat == "MccHF" ){StexHocoVecoLHFCorcc("HF");} // Plot  only
    }
  else
    {fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
      cout << "!TEcnaHistos::PlotMatrix(...)> Histo cannot be reached." << fTTBELL << endl;}
}

//....................................... Corcc for channels (cStexStin_A, cStexStin_B)
//                                        Corss, Covss for one channel (-> i0StinEcha)
void TEcnaHistos::PlotMatrix(const TMatrixD& read_matrix,
			     const TString&   UserCorOrCov, const TString& UserBetweenWhat,
			     const Int_t&    arg_n1,       const Int_t& arg_n2)
{PlotMatrix(read_matrix, UserCorOrCov, UserBetweenWhat, arg_n1, arg_n2, "");}

void TEcnaHistos::PlotMatrix(const TMatrixD& read_matrix,
			     const TString&   UserCorOrCov, const TString& UserBetweenWhat,
			     const Int_t&    arg_n1,       const Int_t& arg_n2,
			     const TString&   UserPlotOption)
{
  TString CallingMethod = "2D";

  TString StandardPlotOption = fCnaParHistos->BuildStandardPlotOption(CallingMethod, UserPlotOption); 

  TString BetweenWhat = fCnaParHistos->BuildStandardBetweenWhatCode(CallingMethod, UserBetweenWhat);
  TString CorOrCov    = fCnaParHistos->BuildStandardCovOrCorCode(CallingMethod, UserCorOrCov);

  if( BetweenWhat != "?" && CorOrCov != "?" )
    {
      if( BetweenWhat == "MccLF" || BetweenWhat == "MccHF" )
	{
	  Int_t cStexStin_A = arg_n1;
	  Int_t cStexStin_B = arg_n2;
	  fAlreadyRead = 1;
	  ViewMatrix(read_matrix, fAlreadyRead,
		     cStexStin_A, cStexStin_B, fZerv, CorOrCov, BetweenWhat, StandardPlotOption); 
	}
      
      if( BetweenWhat == "Mss" )
	{
	  Int_t n1StexStin = arg_n1;
	  Int_t i0StinEcha = arg_n2; 
	  if( fFlagSubDet == "EE" ){n1StexStin = fEcalNumbering->Get1DeeSCEcnaFromDeeSCCons(fFapStexNumber, arg_n1);}
	  fAlreadyRead = 1;
	  ViewMatrix(read_matrix, fAlreadyRead,
		     n1StexStin, fZerv,  i0StinEcha, CorOrCov, BetweenWhat, StandardPlotOption); 
	}
    }
  else
    {fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
      cout << "!TEcnaHistos::PlotMatrix(...)> Histo cannot be reached." << fTTBELL << endl;}
}

void TEcnaHistos::PlotMatrix(const TString& UserCorOrCov, const TString& UserBetweenWhat,
			     const Int_t&  arg_n1,       const Int_t&  arg_n2)
{PlotMatrix(UserCorOrCov, UserBetweenWhat, arg_n1, arg_n2, "");}

void TEcnaHistos::PlotMatrix(const TString& UserCorOrCov, const TString& UserBetweenWhat,
			     const Int_t&  arg_n1,       const Int_t&  arg_n2,
			     const TString& UserPlotOption)
{
  TString CallingMethod = "2D";

  TString StandardPlotOption = fCnaParHistos->BuildStandardPlotOption(CallingMethod, UserPlotOption); 

  TString StandardBetweenWhat = fCnaParHistos->BuildStandardBetweenWhatCode(CallingMethod, UserBetweenWhat);
  TString StandardCorOrCov = fCnaParHistos->BuildStandardCovOrCorCode(CallingMethod, UserCorOrCov);

  if( StandardBetweenWhat != "?" && StandardCorOrCov != "?" )
    {
      if( StandardBetweenWhat == "MccLF" ||  StandardBetweenWhat == "MccHF" )
	{
	  Int_t cStexStin_A = arg_n1;
	  Int_t cStexStin_B = arg_n2;
	  ViewMatrix(fReadMatrixDummy, fTobeRead,
		     cStexStin_A, cStexStin_B, fZerv, StandardCorOrCov, StandardBetweenWhat, StandardPlotOption); 
	}
      
      if( StandardBetweenWhat == "Mss" )
	{
	  Int_t n1StexStin = arg_n1;
	  Int_t i0StinEcha = arg_n2;
	  if( fFlagSubDet == "EE" ){n1StexStin = fEcalNumbering->Get1DeeSCEcnaFromDeeSCCons(fFapStexNumber, arg_n1);}
	  
	  ViewMatrix(fReadMatrixDummy,  fTobeRead,
		     n1StexStin, fZerv, i0StinEcha, StandardCorOrCov, StandardBetweenWhat, StandardPlotOption); 
	}
    }
  else
    {fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
      cout << "!TEcnaHistos::PlotMatrix(...)> Histo cannot be reached." << fTTBELL << endl;}
}

//---------------------------------------------------------------------------------------
//
//                              (ReadAnd)PlotDetector
//
//---------------------------------------------------------------------------------------
//.................................... 2D plots for Stex OR Stas
void TEcnaHistos::PlotDetector(const TString& UserHistoCode, const TString& UserDetector)
{
  TString CallingMethod = "2DS";

  TString StandardHistoCode = fCnaParHistos->BuildStandard1DHistoCodeY(CallingMethod, UserHistoCode);
  if( StandardHistoCode != "?" )
    {
      TString TechHistoCode = fCnaParHistos->GetTechHistoCode(StandardHistoCode);
      TString StandardDetectorCode = fCnaParHistos->BuildStandardDetectorCode(UserDetector);
      if( StandardDetectorCode != "?" )
	{
	  //if( StandardDetectorCode == "SM" || StandardDetectorCode == "EB" )
	  //  {fEcal->SetEcalSubDetector("EB");}
	  //if( StandardDetectorCode == "Dee" || StandardDetectorCode == "EE" )
	  //  {fEcal->SetEcalSubDetector("EE");}

	  if( StandardDetectorCode == "SM" || StandardDetectorCode == "Dee" )
	    {ViewStex(fReadHistoDummy, fTobeRead, TechHistoCode);}
	  if( StandardDetectorCode == "EB" || StandardDetectorCode == "EE"  )
	    {ViewStas(fReadHistoDummy, fTobeRead, TechHistoCode);}
	}
      else
	{fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
	  cout << "!TEcnaHistos::PlotDetector(...)> Histo cannot be reached." << fTTBELL << endl;}
    }
  else
    {fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
      cout << "!TEcnaHistos::PlotDetector(...)> Histo cannot be reached." << fTTBELL << endl;}
}

void TEcnaHistos::PlotDetector(const TVectorD& read_histo, const TString& UserHistoCode, const TString& UserDetector)
{
  TString CallingMethod = "2DS";

  TString StandardHistoCode = fCnaParHistos->BuildStandard1DHistoCodeY(CallingMethod, UserHistoCode);
  if( StandardHistoCode != "?" )
    {
      TString TechHistoCode = fCnaParHistos->GetTechHistoCode(StandardHistoCode);
      TString StandardDetectorCode = fCnaParHistos->BuildStandardDetectorCode(UserDetector);
      if( StandardDetectorCode != "?" )
	{
	  fAlreadyRead = 1;

	  //if( StandardDetectorCode == "SM" || StandardDetectorCode == "EB" )
	  //  {fEcal->SetEcalSubDetector("EB");}
	  //if( StandardDetectorCode == "Dee" || StandardDetectorCode == "EE" )
	  //  {fEcal->SetEcalSubDetector("EE");}

	  if( StandardDetectorCode == "SM" || StandardDetectorCode == "Dee" )
	    {ViewStex(read_histo, fAlreadyRead, TechHistoCode);}
	  if( StandardDetectorCode == "EB" || StandardDetectorCode == "EE"  )
	    {ViewStas(read_histo, fAlreadyRead, TechHistoCode);}
	}
      else
	{fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
	  cout << "!TEcnaHistos::PlotDetector(...)> Histo cannot be reached." << fTTBELL << endl;}
    }
  else
    {fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
      cout << "!TEcnaHistos::PlotDetector(...)> Histo cannot be reached." << fTTBELL << endl;}
}

//---------------------------------------------------------------------------------------
//
//                              (ReadAnd)Plot1DHisto
//
//---------------------------------------------------------------------------------------
void TEcnaHistos::Plot1DHisto(const TVectorD& InputHisto,
			      const TString&   User_X_Quantity, const TString& User_Y_Quantity,
			      const TString&   UserDetector)
{Plot1DHisto(InputHisto, User_X_Quantity, User_Y_Quantity, UserDetector, "");}

void TEcnaHistos::Plot1DHisto(const TVectorD& InputHisto,
			      const TString&   User_X_Quantity, const TString& User_Y_Quantity,
			      const TString&   UserDetector,
			      const TString&   UserPlotOption)
{
  TString CallingMethod = "1D";

  TString StandardPlotOption = fCnaParHistos->BuildStandardPlotOption(CallingMethod, UserPlotOption); 

  TString Standard_X_Quantity = fCnaParHistos->BuildStandard1DHistoCodeX(CallingMethod, User_X_Quantity);
  TString Standard_Y_Quantity = fCnaParHistos->BuildStandard1DHistoCodeY(CallingMethod, User_Y_Quantity);
  
  if( Standard_X_Quantity != "?" && Standard_Y_Quantity != "?" )
    {  
      TString TechHistoCode = fCnaParHistos->GetTechHistoCode(Standard_X_Quantity, Standard_Y_Quantity);
      if( fAlreadyRead > 1 ){fAlreadyRead = 1;}
      TString StandardDetectorCode = fCnaParHistos->BuildStandardDetectorCode(UserDetector);
      if( StandardDetectorCode != "?" )
	{
	  if( StandardDetectorCode == "EB" || StandardDetectorCode == "EE" ){fFapStexNumber = 0;}
	  ViewHisto(InputHisto, fAlreadyRead, fZerv, fZerv, fZerv, TechHistoCode, StandardPlotOption);
	}
      else
	{fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
	  cout << "!TEcnaHistos::Plot1DHisto(...)> Histo cannot be reached." << fTTBELL << endl;}
    }
  else
    {fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
      cout << "!TEcnaHistos::Plot1DHisto(...)> Histo cannot be reached." << fTTBELL << endl;}
}

void TEcnaHistos::Plot1DHisto(const TString& User_X_Quantity, const TString& User_Y_Quantity,
			      const TString& UserDetector)
{Plot1DHisto(User_X_Quantity, User_Y_Quantity, UserDetector, "");}

void TEcnaHistos::Plot1DHisto(const TString& User_X_Quantity, const TString& User_Y_Quantity,
			      const TString& UserDetector,    const TString& UserPlotOption)
{
  TString CallingMethod = "1D";

  TString StandardPlotOption = fCnaParHistos->BuildStandardPlotOption(CallingMethod, UserPlotOption); 

  TString Standard_X_Quantity = fCnaParHistos->BuildStandard1DHistoCodeX(CallingMethod, User_X_Quantity);
  TString Standard_Y_Quantity = fCnaParHistos->BuildStandard1DHistoCodeY(CallingMethod, User_Y_Quantity);
  
  if( Standard_X_Quantity != "?" && Standard_Y_Quantity != "?" )
    {
      TString TechHistoCode = fCnaParHistos->GetTechHistoCode(Standard_X_Quantity, Standard_Y_Quantity);
      TString StandardDetectorCode = fCnaParHistos->BuildStandardDetectorCode(UserDetector);
      if( StandardDetectorCode != "?" )
	{
	  if( StandardDetectorCode == "EB" || StandardDetectorCode == "EE" ){fFapStexNumber = 0;}
	  ViewHisto(fReadHistoDummy, fTobeRead, fZerv, fZerv, fZerv, TechHistoCode, StandardPlotOption);
	}
      else
	{fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
	  cout << "!TEcnaHistos::Plot1DHisto(...)> Histo cannot be reached." << fTTBELL << endl;}
    }
  else
    {fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
      cout << "!TEcnaHistos::Plot1DHisto(...)> Histo cannot be reached." << fTTBELL << endl;}
}



//=> BUG SCRAM? Si on enleve la methode ci-dessous, ca passe a la compilation de test/EcnaHistosExample2.cc 
//   (qui appelle cette methode) et ca se plante a l'execution (voir test/TEcnaHistosExample2.cc).
#define PLUD
#ifdef PLUD
void TEcnaHistos::Plot1DHisto(const TVectorD& InputHisto,
			      const TString&   User_X_Quantity, const TString& User_Y_Quantity,
			      const Int_t&    n1StexStin)
{Plot1DHisto(InputHisto, User_X_Quantity, User_Y_Quantity, n1StexStin, "");}

void TEcnaHistos::Plot1DHisto(const TVectorD& InputHisto,
			      const TString&   User_X_Quantity, const TString& User_Y_Quantity,
			      const Int_t&    n1StexStin,   
			      const TString&   UserPlotOption)
{
  TString CallingMethod = "1DX";
  TString StandardPlotOption = fCnaParHistos->BuildStandardPlotOption(CallingMethod, UserPlotOption);
  Int_t i0StinEcha = 0;
  Plot1DHisto(InputHisto, User_X_Quantity, User_Y_Quantity, n1StexStin, i0StinEcha, StandardPlotOption);
}
#endif // PLUD

void TEcnaHistos::Plot1DHisto(const TVectorD& InputHisto,
			      const TString&   User_X_Quantity, const TString& User_Y_Quantity,
			      const Int_t&    n1StexStin,      const Int_t&  i0StinEcha)
{Plot1DHisto(InputHisto, User_X_Quantity, User_Y_Quantity, n1StexStin, i0StinEcha, "");}

void TEcnaHistos::Plot1DHisto(const TVectorD& InputHisto,
			      const TString&   User_X_Quantity, const TString& User_Y_Quantity,
			      const Int_t&    n1StexStin,      const Int_t&  i0StinEcha,   
			      const TString&   UserPlotOption)
{
  TString CallingMethod = "1D";
  TString StandardPlotOption  = fCnaParHistos->BuildStandardPlotOption(CallingMethod, UserPlotOption); 
  TString Standard_X_Quantity = fCnaParHistos->BuildStandard1DHistoCodeX(CallingMethod, User_X_Quantity);
  TString Standard_Y_Quantity = fCnaParHistos->BuildStandard1DHistoCodeY(CallingMethod, User_Y_Quantity);
  
  TString TechHistoCode = fCnaParHistos->GetTechHistoCode(Standard_X_Quantity, Standard_Y_Quantity);
  
  if( Standard_X_Quantity != "?" && Standard_Y_Quantity != "?" )
    {
      fAlreadyRead = 1;
      if( StandardPlotOption != fAllXtalsInStinPlot )
	{
	  ViewHisto(InputHisto, fAlreadyRead, n1StexStin, i0StinEcha, fZerv, TechHistoCode, StandardPlotOption);
	}
      
      if( StandardPlotOption == fAllXtalsInStinPlot && fAlreadyRead >= 1 && fAlreadyRead <= fEcal->MaxCrysInStin() )
	{
	  if( Standard_X_Quantity == "Smp" && Standard_Y_Quantity == "MSp" )
	    {XtalSamplesEv(InputHisto, fAlreadyRead, n1StexStin, i0StinEcha, StandardPlotOption);}
	  if( Standard_X_Quantity == "MSp" && Standard_Y_Quantity == "NOS" )
	    {EvSamplesXtals(InputHisto, fAlreadyRead, n1StexStin, i0StinEcha, StandardPlotOption);}
	  if( Standard_X_Quantity == "Smp" && Standard_Y_Quantity == "SSp" )
	    {XtalSamplesSigma(InputHisto, fAlreadyRead, n1StexStin, i0StinEcha, StandardPlotOption);}
	  if( Standard_X_Quantity == "SSp" && Standard_Y_Quantity == "NOS" )
	    {SigmaSamplesXtals(InputHisto, fAlreadyRead, n1StexStin, i0StinEcha, StandardPlotOption);}
	}   
    }
  else
    {fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
      cout << "!TEcnaHistos::Plot1DHisto(...)> Histo cannot be reached." << fTTBELL << endl;}
}

void TEcnaHistos::Plot1DHisto(const TString& User_X_Quantity, const TString& User_Y_Quantity,
			      const Int_t&  n1StexStin,      const Int_t&  i0StinEcha)
{Plot1DHisto(User_X_Quantity, User_Y_Quantity, n1StexStin, i0StinEcha, "");}

void TEcnaHistos::Plot1DHisto(const TString& User_X_Quantity, const TString& User_Y_Quantity,
			      const Int_t&  n1StexStin,      const Int_t&  i0StinEcha,
			      const TString& UserPlotOption)
{
  TString CallingMethod = "1D";
  
  TString StandardPlotOption = fCnaParHistos->BuildStandardPlotOption(CallingMethod, UserPlotOption); 
  
  TString Standard_X_Quantity = fCnaParHistos->BuildStandard1DHistoCodeX(CallingMethod, User_X_Quantity);
  TString Standard_Y_Quantity = fCnaParHistos->BuildStandard1DHistoCodeY(CallingMethod, User_Y_Quantity);
  
  if( Standard_X_Quantity != "?" && Standard_Y_Quantity != "?" )
    {  
      if( StandardPlotOption != fAllXtalsInStinPlot )
	{
	  TString TechHistoCode = fCnaParHistos->GetTechHistoCode(Standard_X_Quantity, Standard_Y_Quantity);
	  ViewHisto(fReadHistoDummy, fTobeRead, n1StexStin, i0StinEcha, fZerv, TechHistoCode, StandardPlotOption);
	}
      if( StandardPlotOption == fAllXtalsInStinPlot && fAlreadyRead >= 1 && fAlreadyRead <= fEcal->MaxCrysInStin() )
	{
	  if( Standard_X_Quantity == "Smp" && Standard_Y_Quantity == "MSp" )
	    {XtalSamplesEv(fReadHistoDummy, fTobeRead, n1StexStin, i0StinEcha, StandardPlotOption);}
	  if( Standard_X_Quantity == "MSp" && Standard_Y_Quantity == "NOS" )
	    {EvSamplesXtals(fReadHistoDummy, fTobeRead, n1StexStin, i0StinEcha, StandardPlotOption);}
	  if( Standard_X_Quantity == "Smp" && Standard_Y_Quantity == "SSp" )
	    {XtalSamplesSigma(fReadHistoDummy, fTobeRead, n1StexStin, i0StinEcha, StandardPlotOption);}
	  if( Standard_X_Quantity == "SSp" && Standard_Y_Quantity == "NOS" )
    	    {SigmaSamplesXtals(fReadHistoDummy, fTobeRead, n1StexStin, i0StinEcha, StandardPlotOption);}
	}
    }
  else
    {fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
      cout << "!TEcnaHistos::Plot1DHisto(...)> Histo cannot be reached." << fTTBELL << endl;}
}

void TEcnaHistos::Plot1DHisto(const TVectorD& InputHisto,
			      const TString&   User_X_Quantity, const TString& User_Y_Quantity,
			      const Int_t&    n1StexStin,      const Int_t&  i0StinEcha, const Int_t& n1Sample)
{Plot1DHisto(InputHisto, User_X_Quantity, User_Y_Quantity, n1StexStin, i0StinEcha, n1Sample, "");}

void TEcnaHistos::Plot1DHisto(const TVectorD& InputHisto,
			      const TString&   User_X_Quantity, const TString& User_Y_Quantity,
			      const Int_t&    n1StexStin,      const Int_t&  i0StinEcha, const Int_t& n1Sample,  
			      const TString&   UserPlotOption)
{
  TString CallingMethod = "1D";

  TString StandardPlotOption = fCnaParHistos->BuildStandardPlotOption(CallingMethod, UserPlotOption); 

  TString Standard_X_Quantity = fCnaParHistos->BuildStandard1DHistoCodeX(CallingMethod, User_X_Quantity);
  TString Standard_Y_Quantity = fCnaParHistos->BuildStandard1DHistoCodeY(CallingMethod, User_Y_Quantity);
  
  TString TechHistoCode = fCnaParHistos->GetTechHistoCode(Standard_X_Quantity, Standard_Y_Quantity);
  
  if( Standard_X_Quantity != "?" && Standard_Y_Quantity != "?" )
    {
      Int_t i0Sample = n1Sample-1;
      fAlreadyRead = 1;
      ViewHisto(InputHisto, fAlreadyRead, n1StexStin, i0StinEcha, i0Sample, TechHistoCode, StandardPlotOption); 
    }
  else
    {fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
      cout << "!TEcnaHistos::Plot1DHisto(...)> Histo cannot be reached." << fTTBELL << endl;}
}

void TEcnaHistos::Plot1DHisto(const TString& User_X_Quantity, const TString& User_Y_Quantity,
			      const Int_t&  n1StexStin,      const Int_t&  i0StinEcha, const Int_t& n1Sample)
{Plot1DHisto(User_X_Quantity, User_Y_Quantity, n1StexStin, i0StinEcha, n1Sample, "");}

void TEcnaHistos::Plot1DHisto(const TString& User_X_Quantity, const TString& User_Y_Quantity,
			      const Int_t&  n1StexStin,      const Int_t&  i0StinEcha, const Int_t& n1Sample,
			      const TString& UserPlotOption)
{
  TString CallingMethod = "1D";

  TString StandardPlotOption = fCnaParHistos->BuildStandardPlotOption(CallingMethod, UserPlotOption); 

  TString Standard_X_Quantity = fCnaParHistos->BuildStandard1DHistoCodeX(CallingMethod, User_X_Quantity);
  TString Standard_Y_Quantity = fCnaParHistos->BuildStandard1DHistoCodeY(CallingMethod, User_Y_Quantity);
  
  Int_t i0Sample = n1Sample-1;
  
  if( Standard_X_Quantity != "?" && Standard_Y_Quantity != "?" )
    {
      TString TechHistoCode = fCnaParHistos->GetTechHistoCode(Standard_X_Quantity, Standard_Y_Quantity);
      ViewHisto(fReadHistoDummy, fTobeRead, n1StexStin, i0StinEcha, i0Sample, TechHistoCode, StandardPlotOption);
    }
  else
    {fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
      cout << "!TEcnaHistos::Plot1DHisto(...)> Histo cannot be reached." << fTTBELL << endl;}
}

//---------------------------------------------------------------------------------------
//
//                              (ReadAnd)PlotHistory
//
//---------------------------------------------------------------------------------------
void TEcnaHistos::PlotHistory(const TString& User_X_Quantity, const TString& User_Y_Quantity,
			      const TString& list_of_run_file_name,
			      const Int_t&  StexStin_A, const Int_t& i0StinEcha)
{PlotHistory(User_X_Quantity, User_Y_Quantity, list_of_run_file_name, StexStin_A, i0StinEcha, "");}

void TEcnaHistos::PlotHistory(const TString& User_X_Quantity, const TString& User_Y_Quantity,
			      const TString& list_of_run_file_name,
			      const Int_t&  StexStin_A, const Int_t& i0StinEcha,
			      const TString& UserPlotOption)
{
  TString CallingMethod = "Time";

  TString StandardPlotOption = fCnaParHistos->BuildStandardPlotOption(CallingMethod, UserPlotOption); 

  TString Standard_X_Quantity = fCnaParHistos->BuildStandard1DHistoCodeX(CallingMethod, User_X_Quantity);
  TString Standard_Y_Quantity = fCnaParHistos->BuildStandard1DHistoCodeY(CallingMethod, User_Y_Quantity);

  if( Standard_X_Quantity != "?" && Standard_Y_Quantity != "?" )
    { 
      TString TechHistoCode = fCnaParHistos->GetTechHistoCode(Standard_X_Quantity, Standard_Y_Quantity);
      ViewHistime(list_of_run_file_name, StexStin_A, i0StinEcha, TechHistoCode, StandardPlotOption);
    }
  else
    {fFlagUserHistoMin = "OFF"; fFlagUserHistoMax = "OFF";
      cout << "!TEcnaHistos::PlotHistory(...)> Histo cannot be reached." << fTTBELL << endl;} 
}

//=============================================================================================
//
//                            " V I E W "    M E T H O D S
//
//=============================================================================================

//=============================================================================================================
//
//                                       ViewMatrix(***)
//     
//     arg_read_matrix:   2D array
//     arg_AlreadyRead:   =1 <=> use arg_read_matrix 
//                        =0 <=> read the 2D array in this method with TEcnaRead
//     StexStin_A:        tower  number in SM (if EB) OR SC  "ECNA" number in Dee (if EE)
//     StexStin_B:        tower' number in SM (if EB) OR SC' "ECNA" number in Dee (if EE)
//     MatrixBinIndex:    channel number in tower (if EB) OR in SC (if EE)
//     CorOrCov:          flag CORRELATION/COVARIANCE
//     BetweenWhat:       flag BETWEEN SAMPLES / BETWEEN CHANNELS / BETWEEN TOWERS / BETWEEN SCs / LF, HF, ...
//     PlotOption:        ROOT 2D histos draw options (COLZ, LEGO, ...) + additional (ASCII)
//
//     MatrixBinIndex:  = i0StinEcha if cov(s,s'), cor(s,s')
//                      = 0          if cov(c,c'), cor(c,c'), cov(Stin,Stin'), cor(Stin,Stin')
//
//     ViewMatrix(StexStin_A, StexStin_B, MatrixBinIndex, CorOrCov, BetweenWhat, PlotOption)
//     ViewMatrix(StexStin_A,          0,     i0StinEcha, CorOrCov,       "Mss", PlotOption)      
//     Output:
//     Plot of cov(s,s') or cor(s,s') matrix for i0StinEcha of StexStin_A              
//
//     ViewMatrix(StexStin_A, StexStin_B, MatrixBinIndex, CorOrCov, BetweenWhat, PlotOption)
//     ViewMatrix(StexStin_A, StexStin_B,              0, CorOrCov,       "Mcc", PlotOption)
//     Output:
//     Plot LF-HF Corcc matrix for Stins: (StexStin_A, StexStin_B)
//
//     ViewMatrix(StexStin_A, StexStin_B, MatrixBinIndex, CorOrCov, BetweenWhat, PlotOption)
//     ViewMatrix(         0,          0,              0, CorOrCov,       "Mcc", PlotOption)
//     Output:
//     Plot of LF-HF Corcc matrix for Stex (big matrix)
//
//     ViewMatrix(StexStin_A, StexStin_B, MatrixBinIndex, CorOrCov, BetweenWhat, PlotOption)
//     ViewMatrix(         0,          0,              0, CorOrCov,       "Mtt", PlotOption)
//     Output:
//     Plot of LF-HF Cortt matrix
//
//=============================================================================================================
void TEcnaHistos::ViewMatrix(const TMatrixD& arg_read_matrix, const Int_t&  arg_AlreadyRead,
			     const Int_t&    StexStin_A,      const Int_t&  StexStin_B,
			     const Int_t&    MatrixBinIndex,  const TString& CorOrCov,
			     const TString&   BetweenWhat,     const TString& PlotOption)
{
  //Plot correlation or covariance matrix between samples or channels or Stins

  if( (fFapStexNumber > 0) &&  (fFapStexNumber <= fEcal->MaxStexInStas()) )
    {
      Bool_t OKArray = kFALSE;
      Bool_t OKData  = kFALSE;
      TVectorD vStin(fEcal->MaxStinEcnaInStex());

      if( arg_AlreadyRead == fTobeRead )
	{
	  fMyRootFile->PrintNoComment();
	  fMyRootFile->FileParameters(fFapAnaType,          fFapNbOfSamples,
				      fFapRunNumber,        fFapFirstReqEvtNumber,
				      fFapLastReqEvtNumber, fFapReqNbOfEvts,
				      fFapStexNumber,       fCfgResultsRootFilePath.Data());
	  OKArray = fMyRootFile->LookAtRootFile();
	  if( OKArray == kTRUE )
	    {
	      fFapNbOfEvts = fMyRootFile->GetNumberOfEvents(fFapReqNbOfEvts, fFapStexNumber);
	      TString fp_name_short = fMyRootFile->GetRootFileNameShort();
	      // cout << "*TEcnaHistos::ViewMatrix(...)> Data are analyzed from file ----> "
	      //      << fp_name_short << endl;
	      //...................................................................... (ViewMatrix) 
	      for(Int_t i=0; i<fEcal->MaxStinEcnaInStex(); i++){vStin(i)=(Double_t)0.;}
	      vStin = fMyRootFile->ReadStinNumbers(fEcal->MaxStinEcnaInStex());

	      fStartDate = fMyRootFile->GetStartDate();
	      fStopDate  = fMyRootFile->GetStopDate();
	      fRunType   = fMyRootFile->GetRunType();

	      if( fMyRootFile->DataExist() == kTRUE ){OKData = kTRUE;}
	    }

	}
      if( arg_AlreadyRead >= 1 )
	{
	  OKArray = kTRUE;
	  OKData  = kTRUE;
	  if( fFlagSubDet == "EB") 
	    {
	      for(Int_t i=0; i<fEcal->MaxStinEcnaInStex(); i++){vStin(i)=i;}
	    }
	  if( fFlagSubDet == "EE") 
	    {
	      for(Int_t i=0; i<fEcal->MaxStinEcnaInStex(); i++)
		{vStin(i)= fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fFapStexNumber, (Int_t)vStin(i));}
	    }
	}

      if ( OKArray == kTRUE )                         //  (ViewMatrix) 
	{
	  fStatusFileFound = kTRUE;
      
	  if( OKData == kTRUE )
	    {
	      fStatusDataExist = kTRUE;

	      Int_t Stin_X_ok = 0;
	      Int_t Stin_Y_ok = 0;
	  
	      if( (BetweenWhat == fLFBetweenStins) || (BetweenWhat == fHFBetweenStins) )
		{Stin_X_ok = 1; Stin_Y_ok = 1;}
	      if( BetweenWhat == fBetweenSamples )
		{Stin_Y_ok = 1;}
	  
	      for (Int_t index_Stin = 0; index_Stin < fEcal->MaxStinEcnaInStex(); index_Stin++)
		{
		  if ( vStin(index_Stin) == StexStin_A ){Stin_X_ok = 1;}
		  if ( vStin(index_Stin) == StexStin_B ){Stin_Y_ok = 1;}
		}
	      //................................................................. (ViewMatrix)
	      if( Stin_X_ok == 1 && Stin_Y_ok == 1 )
		{
		  Int_t MatSize      = -1; 
		  Int_t ReadMatSize  = -1; 
		  Int_t i0StinEcha   = -1;
	      
		  //-------------------------- Set values of ReadMatSize, MatSize, i0StinEcha
		  if( BetweenWhat == fBetweenSamples )
		    {ReadMatSize = fFapNbOfSamples; MatSize = fEcal->MaxSampADC(); i0StinEcha=(Int_t)MatrixBinIndex;}

		  if( BetweenWhat == fLFBetweenChannels || BetweenWhat == fHFBetweenChannels )
		    {ReadMatSize = fEcal->MaxCrysInStin(); MatSize = fEcal->MaxCrysInStin(); /*i0Sample=MatrixBinIndex;*/}

		  if( (BetweenWhat == fLFBetweenStins) || (BetweenWhat == fHFBetweenStins) )
		    {ReadMatSize = fEcal->MaxStinEcnaInStex(); MatSize = fEcal->MaxStinInStex();}
 
		  //------------------------------------------------------------------------------------- (ViewMatrix)
		  if( ( BetweenWhat == fLFBetweenStins    || BetweenWhat == fHFBetweenStins    ) ||
		      ( BetweenWhat == fLFBetweenChannels || BetweenWhat == fHFBetweenChannels 
			/* && (i0Sample  >= 0) && (i0Sample  < fFapNbOfSamples ) */ ) ||
		      ( (BetweenWhat == fBetweenSamples) && (i0StinEcha >= 0) && (i0StinEcha < fEcal->MaxCrysInStin()) ) )
		    {
		      TMatrixD read_matrix(ReadMatSize, ReadMatSize);
		      for(Int_t i=0; i-ReadMatSize<0; i++)
			{for(Int_t j=0; j-ReadMatSize<0; j++){read_matrix(i,j)=(Double_t)0.;}}
		 
		      Bool_t OKData = kFALSE;
		      if( arg_AlreadyRead == fTobeRead )
			{
			  if( BetweenWhat == fBetweenSamples && CorOrCov == fCovarianceMatrix )
			    {read_matrix =
				fMyRootFile->ReadCovariancesBetweenSamples(StexStin_A, i0StinEcha, ReadMatSize);}

			  if( BetweenWhat == fBetweenSamples && CorOrCov == fCorrelationMatrix )
			    {read_matrix =
				fMyRootFile->ReadCorrelationsBetweenSamples(StexStin_A, i0StinEcha, ReadMatSize);}
			  
			  if( BetweenWhat == fLFBetweenChannels && CorOrCov == fCovarianceMatrix )
			    {read_matrix =
				fMyRootFile->ReadLowFrequencyCovariancesBetweenChannels(StexStin_A, StexStin_B, ReadMatSize);}

			  if( BetweenWhat == fLFBetweenChannels && CorOrCov == fCorrelationMatrix )
			    {read_matrix =
				fMyRootFile->ReadLowFrequencyCorrelationsBetweenChannels(StexStin_A, StexStin_B, ReadMatSize);}
			  
			  if( BetweenWhat == fHFBetweenChannels && CorOrCov == fCovarianceMatrix )
			    {read_matrix =
				fMyRootFile->ReadHighFrequencyCovariancesBetweenChannels(StexStin_A, StexStin_B, ReadMatSize);}

			  if( BetweenWhat == fHFBetweenChannels && CorOrCov == fCorrelationMatrix )
			    {read_matrix =
				fMyRootFile->ReadHighFrequencyCorrelationsBetweenChannels(StexStin_A, StexStin_B, ReadMatSize);}
			  
			  if( BetweenWhat == fLFBetweenStins && CorOrCov == fCorrelationMatrix )
			    {read_matrix =
				fMyRootFile->ReadLowFrequencyMeanCorrelationsBetweenStins(ReadMatSize);}

			  if( BetweenWhat == fHFBetweenStins && CorOrCov == fCorrelationMatrix )
			    {read_matrix =
				fMyRootFile->ReadHighFrequencyMeanCorrelationsBetweenStins(ReadMatSize);
			    }

			  OKData = fMyRootFile->DataExist();
			}
		      else
			{
			  read_matrix = arg_read_matrix;
			  OKData = kTRUE;
			}
		      //.......................................................... (ViewMatrix)
		      if( OKData == kTRUE )
			{
			  fStatusDataExist = kTRUE;

			  if( PlotOption == "ASCII" )
			    {
			      WriteMatrixAscii(BetweenWhat, CorOrCov, 
					       StexStin_A, MatrixBinIndex, ReadMatSize, read_matrix);
			    }
			  else
			    {
			      //......................... matrix title  (ViewMatrix)
			      char* f_in_mat_tit = new char[fgMaxCar];               fCnew++;
			  
			      if( BetweenWhat == fBetweenSamples && CorOrCov == fCovarianceMatrix )
				{sprintf(f_in_mat_tit, "Covariance(Sample, Sample')");}
			      if( BetweenWhat == fBetweenSamples && CorOrCov == fCorrelationMatrix )
				{sprintf(f_in_mat_tit, "Correlation(Sample, Sample')");}

			      if(fFlagSubDet == "EB" )
				{
				  if( BetweenWhat == fLFBetweenStins && CorOrCov == fCorrelationMatrix )
				    {sprintf(f_in_mat_tit,
					     "Mean LF |Cor(Xtal,Xtal')| for each (Tower,Tower')");}
				  if( BetweenWhat == fHFBetweenStins && CorOrCov == fCorrelationMatrix )
				    {sprintf(f_in_mat_tit,
					     "Mean HF |Cor(Xtal,Xtal')| for each (Tower,Tower')");}
				}
			      if(fFlagSubDet == "EE" )
				{
				  if( BetweenWhat == fLFBetweenStins && CorOrCov == fCorrelationMatrix )
				    {sprintf(f_in_mat_tit,
					     "Mean LF |Cor(Xtal,Xtal')| for each (SC,SC')");}
				  if( BetweenWhat == fHFBetweenStins && CorOrCov == fCorrelationMatrix )
				    {sprintf(f_in_mat_tit,
					     "Mean HF |Cor(Xtal,Xtal')| for each (SC,SC')");}
				}

			      if( BetweenWhat == fLFBetweenChannels && CorOrCov == fCorrelationMatrix )
				{
				  if( fFlagSubDet == "EB" )
				    {sprintf(f_in_mat_tit, "LF Cor(Xtal,Xtal') matrix elts for (Tow,Tow')");}
				  if( fFlagSubDet == "EE" )
				    {sprintf(f_in_mat_tit, "LF Cor(Xtal,Xtal') matrix elts for (SC,SC')");}
				}
			      if( BetweenWhat == fHFBetweenChannels && CorOrCov == fCorrelationMatrix )
				{
				  if( fFlagSubDet == "EB" )
				    {sprintf(f_in_mat_tit, "HF Cor(Xtal,Xtal') matrix elts for (Tow,Tow')");}
				  if( fFlagSubDet == "EE" )
				    {sprintf(f_in_mat_tit, "LF Cor(Xtal,Xtal') matrix elts for (SC,SC')");}
				}
			  
			      //................................. Axis parameters (ViewMatrix)
			      TString axis_x_var_name;
			      TString axis_y_var_name;
			  
			      char* f_in_axis_x = new char[fgMaxCar];               fCnew++;
			      char* f_in_axis_y = new char[fgMaxCar];               fCnew++;
			  
			      if( BetweenWhat == fLFBetweenStins || BetweenWhat == fHFBetweenStins )
				{
				  if( fFlagSubDet == "EB" )
				    {sprintf(f_in_axis_x, " %s number  ", fFapStinName.Data());}
				  if( fFlagSubDet == "EE" )
				    {sprintf(f_in_axis_x, " %s number for construction ", fFapStinName.Data());}

				  axis_x_var_name = f_in_axis_x; axis_y_var_name = f_in_axis_x;
				}      
			      if( BetweenWhat == fBetweenSamples)
				{
				  axis_x_var_name = " Sample      "; axis_y_var_name = "    Sample ";
				}
			      if( BetweenWhat == fLFBetweenChannels || BetweenWhat == fHFBetweenChannels ){
				sprintf(f_in_axis_x, " Crystal %s %d   ", fFapStinName.Data(), StexStin_A);
				sprintf(f_in_axis_y, " Crystal %s %d   ", fFapStinName.Data(),StexStin_B);
				axis_x_var_name = f_in_axis_x; axis_y_var_name = f_in_axis_y;}
			  
			      Int_t  nb_binx  = MatSize;
			      Int_t  nb_biny  = MatSize;
			      Axis_t xinf_bid = (Axis_t)0.;
			      Axis_t xsup_bid = (Axis_t)MatSize;
			      Axis_t yinf_bid = (Axis_t)0.;
			      Axis_t ysup_bid = (Axis_t)MatSize;   
			  
			      if( (fFlagSubDet == "EE") &&
				  (BetweenWhat == fLFBetweenStins || BetweenWhat == fHFBetweenStins) )
				{
				  if( fFapStexNumber == 1 || fFapStexNumber == 3 )
				    {
				      xinf_bid += fEcal->MaxStinInStex();
				      xsup_bid += fEcal->MaxStinInStex();
				      yinf_bid += fEcal->MaxStinInStex();
				      ysup_bid += fEcal->MaxStinInStex();
				    }
				}
			      //...................................................  histogram booking (ViewMatrix)
			      TH2D* h_fbid0 = new TH2D("bidim", f_in_mat_tit,
						       nb_binx, xinf_bid, xsup_bid,
						       nb_biny, yinf_bid, ysup_bid);     fCnewRoot++;
			      h_fbid0->Reset();
			  
			      h_fbid0->GetXaxis()->SetTitle(axis_x_var_name);
			      h_fbid0->GetYaxis()->SetTitle(axis_y_var_name);
			  
			      //------------------------------------------------  F I L L    H I S T O  (ViewMatrix)
			      if( (fFlagSubDet == "EE") &&
				  (BetweenWhat == fLFBetweenStins || BetweenWhat == fHFBetweenStins) )
				{
				  for(Int_t i = 0 ; i < ReadMatSize ; i++)
				    {
				      for(Int_t j = 0 ; j < ReadMatSize ; j++)
					{
					  Int_t ip = i+1;
					  Double_t xi_bid =
					    (Double_t)fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fFapStexNumber, ip);
					  Int_t jp = j+1;
					  Double_t xj_bid =
					    (Double_t)fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fFapStexNumber, jp);
					  if( xi_bid > 0 && xj_bid > 0 )
					    {
					      Int_t xi_bid_m = xi_bid-1;
					      Int_t xj_bid_m = xj_bid-1;
					      h_fbid0->Fill(xi_bid_m, xj_bid_m, read_matrix(i,j));
					    }
					}
				    }
				}
			      else
				{
				  for(Int_t i = 0 ; i - ReadMatSize < 0 ; i++)
				    {
				      Double_t xi = (Double_t)i;
				      for(Int_t j = 0 ; j < ReadMatSize ; j++)
					{
					  Double_t xj      = (Double_t)j;
					  Double_t mat_val = (Double_t)read_matrix(i,j);
					  h_fbid0->Fill(xi, xj, (Double_t)mat_val);
					}
				    }
				}
			      //--------------- H I S T O   M I N / M A X   M A N A G E M E N T   (ViewMatrix)
			  
			      //................................ Put histo min max values
			      TString quantity_code = "D_MCs_ChNb";
			      if ( CorOrCov == fCorrelationMatrix )
				{
				  if( BetweenWhat == fBetweenSamples ){quantity_code = "D_MCs_ChNb";}

				  if( BetweenWhat == fLFBetweenChannels  ){quantity_code = "H2LFccMosMatrix";}
				  if( BetweenWhat == fHFBetweenChannels ){quantity_code = "H2HFccMosMatrix";}

				  if( BetweenWhat == fLFBetweenStins  ){quantity_code = "H2LFccMosMatrix";}
				  if( BetweenWhat == fHFBetweenStins ){quantity_code = "H2HFccMosMatrix";}
				}
			      if( CorOrCov == fCovarianceMatrix ){quantity_code = "H2HFccMosMatrix";}
			      //.......... default if flag not set to "ON"
			      SetYminMemoFromValue(quantity_code, fCnaParHistos->GetYminDefaultValue(quantity_code));
			      SetYmaxMemoFromValue(quantity_code, fCnaParHistos->GetYmaxDefaultValue(quantity_code));

			      if( fUserHistoMin == fUserHistoMax ){fFlagUserHistoMin = "AUTO"; fFlagUserHistoMax = "AUTO";}
			      //................................. User's min and/or max  (ViewMatrix)
			      if( fFlagUserHistoMin == "ON" )
				{SetYminMemoFromValue(quantity_code, fUserHistoMin); fFlagUserHistoMin = "OFF";}
			      if( fFlagUserHistoMax == "ON" )
				{SetYmaxMemoFromValue(quantity_code, fUserHistoMax); fFlagUserHistoMax = "OFF";}
			      //................................. automatic min and/or max
			      if( fFlagUserHistoMin == "AUTO" )
				{SetYminMemoFromValue(quantity_code, h_fbid0->GetMinimum()); fFlagUserHistoMin = "OFF";}
			      if( fFlagUserHistoMax == "AUTO" )
				{SetYmaxMemoFromValue(quantity_code, h_fbid0->GetMaximum()); fFlagUserHistoMax = "OFF";}
			      //...................................... histo set ymin and ymax  (ViewMatrix)	
			      if( CorOrCov == fCorrelationMatrix )
				{
				  if(BetweenWhat == fBetweenSamples)
				    {SetHistoFrameYminYmaxFromMemo((TH1D*)h_fbid0, "D_MCs_ChNb");}
				  if( BetweenWhat == fLFBetweenStins  || BetweenWhat == fLFBetweenChannels  )
				    {SetHistoFrameYminYmaxFromMemo((TH1D*)h_fbid0, "H2LFccMosMatrix");}
				  if( BetweenWhat == fHFBetweenStins || BetweenWhat == fHFBetweenChannels )
				    {SetHistoFrameYminYmaxFromMemo((TH1D*)h_fbid0, "H2HFccMosMatrix");}
				  //************************** A GARDER EN RESERVE ******************************
				  //............. special contour level for correlations (square root wise scale)
				  // Int_t nb_niv  = 9;
				  // Double_t* cont_niv = new Double_t[nb_niv];                  fCnew++;
				  // SqrtContourLevels(nb_niv, &cont_niv[0]);
				  // h_fbid0->SetContour(nb_niv, &cont_niv[0]);		  
				  // delete [] cont_niv;                                  fCdelete++;
				  //******************************** (FIN RESERVE) ****************************** 
				}
			      if( CorOrCov == fCovarianceMatrix )
				{
				  if (BetweenWhat == fBetweenSamples)
				    {SetYminMemoFromPreviousMemo("D_TNo_ChNb");   // covariance => same level as sigmas
				    SetYmaxMemoFromPreviousMemo("D_TNo_ChNb");
				    SetHistoFrameYminYmaxFromMemo((TH1D*)h_fbid0, "D_TNo_ChNb");}
				  if ( BetweenWhat == fLFBetweenStins || BetweenWhat == fHFBetweenStins ||
				       BetweenWhat == fLFBetweenChannels || BetweenWhat == fHFBetweenChannels )
				    {SetHistoFrameYminYmaxFromMemo((TH1D*)h_fbid0, "H2HFccMosMatrix");}
				}
			  
			      // ----------------------------------------------- P L O T S  (ViewMatrix)
			      char* f_in = new char[fgMaxCar];          fCnew++;
			      //...................... Taille/format canvas
			      UInt_t canv_w = fCnaParHistos->CanvasFormatW("petit");
			      UInt_t canv_h = fCnaParHistos->CanvasFormatH("petit");
			  
			      //............................. options generales 
			      TString HistoType;
			      Int_t MaxCar = fgMaxCar;
			      HistoType.Resize(MaxCar);
			      HistoType = "(no quantity type info)";
			  
			      if (PlotOption == "COLZ"  ){HistoType = "colz";}
			      if (PlotOption == "BOX"   ){HistoType = "colz";}
			      if (PlotOption == "TEXT"  ){HistoType = "colz";}
			      if (PlotOption == "CONTZ" ){HistoType = "colz";}
			      if (PlotOption == "LEGO2Z"){HistoType = "lego";}
			      if (PlotOption == "SURF1Z"){HistoType = "surf";}
			      if (PlotOption == "SURF2Z"){HistoType = "surf";}
			      if (PlotOption == "SURF3Z"){HistoType = "surf";}
			      if (PlotOption == "SURF4" ){HistoType = "surf";}

			      if( fFlagSubDet == "EB" )
				{
				  fFapStexBarrel = fEcalNumbering->GetSMHalfBarrel(fFapStexNumber);
				  SetAllPavesViewMatrix(BetweenWhat.Data(), StexStin_A, StexStin_B, i0StinEcha);
				}
			      if( fFlagSubDet == "EE" )
				{
				  fFapStexType = fEcalNumbering->GetEEDeeType(fFapStexNumber);
				  fFapStinQuadType = fEcalNumbering->GetSCQuadFrom1DeeSCEcna(StexStin_A);
				  SetAllPavesViewMatrix(BetweenWhat.Data(), StexStin_A, StexStin_B, i0StinEcha);
				}

			      //---------------------------------------- Canvas name (ViewMatrix)
			      TString  name_cov_cor;
			      MaxCar = fgMaxCar;
			      name_cov_cor.Resize(MaxCar);
			      name_cov_cor = "?";
			      if( CorOrCov == fCovarianceMatrix){name_cov_cor = "Covariance";}
			      if( CorOrCov == fCorrelationMatrix){name_cov_cor = "Correlation";}
			  
			      TString name_chan_samp;
			      MaxCar = fgMaxCar;
			      name_chan_samp.Resize(MaxCar);
			      name_chan_samp = "?";
			  
			      if( BetweenWhat == fLFBetweenStins ){name_chan_samp = "LFccMos";}
			      if( BetweenWhat == fHFBetweenStins ){name_chan_samp = "HFccMos"; }

			      if( BetweenWhat == fLFBetweenChannels ){name_chan_samp = "LF_cc";}
			      if( BetweenWhat == fHFBetweenChannels ){name_chan_samp = "HF_cc";}
		
			      if(BetweenWhat == fBetweenSamples)
				{
				  name_chan_samp = "Between_Samples";  // MatrixBinIndex = i0StinEcha
				}

			      TString name_visu;
			      MaxCar = fgMaxCar;
			      name_visu.Resize(MaxCar);
			      name_visu = "?";
			  
			      name_visu = PlotOption;
			  
			      if( (BetweenWhat == fLFBetweenStins) || (BetweenWhat == fHFBetweenStins) ){
				sprintf(f_in, "%s_%s_%s_S1_%d_R%d_%d_%d_%s%d_%s",
					name_cov_cor.Data(), name_chan_samp.Data(),
					fFapAnaType.Data(), fFapNbOfSamples, fFapRunNumber,
					fFapFirstReqEvtNumber, fFapLastReqEvtNumber,
					fFapStexName.Data(), fFapStexNumber,
					name_visu.Data());}

			      if( BetweenWhat == fLFBetweenChannels || BetweenWhat == fHFBetweenChannels ){
				sprintf(f_in, "%s_%s_%s_S1_%d_R%d_%d_%d_%s%d_%sX%d_%sY%d_%s",
					name_cov_cor.Data(), name_chan_samp.Data(),
					fFapAnaType.Data(), fFapNbOfSamples, fFapRunNumber,
					fFapFirstReqEvtNumber, fFapLastReqEvtNumber,
					fFapStexName.Data(), fFapStexNumber,
					fFapStexName.Data(), StexStin_A, fFapStexName.Data(), StexStin_B,
					name_visu.Data());}
			  			  
			      if( BetweenWhat == fBetweenSamples ){
				sprintf(f_in, "%s_%s_%s_S1_%d_R%d_%d_%d_%s%d_%sX%d_%sY%d_ElecChannel_%d_%s",
					name_cov_cor.Data(), name_chan_samp.Data(),
					fFapAnaType.Data(), fFapNbOfSamples, fFapRunNumber,
					fFapFirstReqEvtNumber, fFapLastReqEvtNumber,
					fFapStexName.Data(), fFapStexNumber,
					fFapStexName.Data(), StexStin_A,  fFapStexName.Data(), StexStin_B,
					MatrixBinIndex,
					name_visu.Data());}
			  
			      //----------------------------------------------------------	(ViewMatrix)

			      SetHistoPresentation((TH1D*)h_fbid0, HistoType);
			      TCanvas *MainCanvas = new TCanvas(f_in, f_in, canv_w , canv_h);   fCnewRoot++;
			      fCurrentCanvas = MainCanvas; fCurrentCanvasName = f_in;
			  
			      // cout << "*TEcnaHistos::ViewMatrix(...)> Plot is displayed on canvas ----> "
			      //      << fCurrentCanvasName << endl;
			      // cout << "*TEcnaHistos::ViewMatrix(...)> fCurrentCanvas = " << fCurrentCanvas << endl;
			  
			      delete [] f_in; f_in = 0;                         fCdelete++;

			      if( fPavComGeneralTitle != 0 ){fPavComGeneralTitle->Draw();}
			      fPavComStex->Draw();

			      if(BetweenWhat == fLFBetweenChannels || BetweenWhat == fHFBetweenChannels)
				{fPavComStin->Draw();}
			      if(BetweenWhat == fBetweenSamples)
				{fPavComStin->Draw(); fPavComXtal->Draw();}

			      fPavComAnaRun->Draw();
			      fPavComNbOfEvts->Draw();

			      Double_t x_margin = fCnaParHistos->BoxLeftX("bottom_left_box") - 0.005;
			      Double_t y_margin = fCnaParHistos->BoxTopY("bottom_right_box") + 0.005;
			      MainCanvas->Divide(1, 1, x_margin, y_margin);
			      gPad->cd(1);

			      //----------------------------------------------------------	(ViewMatrix)      
			      Int_t logy = 0;  
			      gPad->SetLogy(logy);
			      if( (BetweenWhat == fLFBetweenStins) ||
				  (BetweenWhat == fHFBetweenStins) ){gPad->SetGrid(1,1);}
			      h_fbid0->DrawCopy(PlotOption);
			      h_fbid0->SetStats((Bool_t)1);    
			      gPad->Update();
			      h_fbid0->Delete();  h_fbid0 = 0;              fCdeleteRoot++;
			  
			      //MainCanvas->Delete();                 fCdeleteRoot++;
			      delete [] f_in_axis_x;  f_in_axis_x  = 0;       fCdelete++;
			      delete [] f_in_axis_y;  f_in_axis_y  = 0;       fCdelete++;
			      delete [] f_in_mat_tit; f_in_mat_tit = 0;       fCdelete++;
			    }
			} // end of if ( OKData == kTRUE )
		      else
			{
			  fStatusDataExist = kFALSE;
			}
		    } // end of if ((BetweenWhat == fLFBetweenStins) || (BetweenWhat == fHFBetweenStins)  ) ||
		      //( (BetweenWhat == fBetweenSamples) && (i0StinEcha>= 0) && (i0StinEcha<fEcal->MaxCrysInStin())) ||
		      //( (BetweenWhat == fLFBetweenChannels || BetweenWhat == fHFBetweenChannels)
	              // /* && (i0Sample  >= 0) && (i0Sample  < fFapNbOfSamples ) */ ) )
		  else
		    {
		      if(BetweenWhat == fBetweenSamples)
			{
			  cout << "*TEcnaHistos::ViewMatrix(...)> *ERROR* ==> Wrong channel number in "
			       << fFapStinName.Data() << ". Value = "
			       << i0StinEcha << " (required range: [0, "
			       << fEcal->MaxCrysInStin()-1 << "] )"
			       << fTTBELL << endl;
			}

		     // if( BetweenWhat == fLFBetweenChannels || BetweenWhat == fHFBetweenChannels )
		     //	{
			 // cout << "*TEcnaHistos::ViewMatrix(...)> *ERROR* ==> Wrong sample index. Value = "
			 //      << i0Sample << " (required range: [0, "
			 //      << fFapNbOfSamples-1 << "] )"
			 //      << fTTBELL << endl;
			//}
		    }
		}
	      else    // else of the if ( Stin_X_ok ==1 && Stin_Y_ok ==1 )
		{
		  //----------------------------------------------------------	(ViewMatrix)
		  if ( Stin_X_ok != 1 )
		    {
		      if( fFlagSubDet == "EB") 
			{
			  cout << "*TEcnaHistos::ViewMatrix(...)> *ERROR* =====> "
			       << fFapStinName.Data() << " "
			       << StexStin_A << ", "
			       << fFapStinName.Data() << " not found. Available numbers = ";
			  for(Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++)
			    {
			      if( vStin(i) > 0 )
				{
				  cout << vStin(i) << ", ";
				}
			    }
			}

		      if( fFlagSubDet == "EE") 
			{
			  cout << "*TEcnaHistos::ViewMatrix(...)> *ERROR* =====> "
			       << fFapStinName.Data() << " "
			       << fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fFapStexNumber, StexStin_A) << ", "
			       << fFapStinName.Data() << " not found. Available numbers = ";
			  for(Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++)
			    {
			      if( vStin(i) > 0 )
				{
				  cout << fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fFapStexNumber, (Int_t)vStin(i)) << ", ";
				}
			    }
			}
		      cout << fTTBELL << endl;
		    }
		  if ( Stin_Y_ok != 1 )
		    {

		      if( fFlagSubDet == "EB") 
			{
			  cout << "*TEcnaHistos::ViewMatrix(...)> *ERROR* =====> "
			       << fFapStinName.Data() << " "
			       << StexStin_B << ", "
			       << fFapStinName.Data() << " not found. Available numbers = ";
			  for(Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++)
			    {
			      if( vStin(i) > 0 )
				{
				  cout << vStin(i) << ", ";
				}
			    }
			}

		      if( fFlagSubDet == "EE") 
			{
			  cout << "*TEcnaHistos::ViewMatrix(...)> *ERROR* =====> "
			       << fFapStinName.Data() << " "
			       << fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fFapStexNumber, StexStin_B) << ", "
			       << fFapStinName.Data() << " not found. Available numbers = ";
			  for(Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++)
			    {
			      if( vStin(i) > 0 )
				{
				  cout << fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fFapStexNumber, (Int_t)vStin(i)) << ", ";
				}
			    }
			}     
		      cout << fTTBELL << endl;
		    }
		}
	    } // end of if ( fMyRootFile->DataExist() == kTRUE )
	  else
	    {
	      fStatusDataExist = kFALSE;
	      cout  << "!TEcnaHistos::ViewMatrix(...)> *ERROR* =====> "
		    << " Histo not available." << fTTBELL << endl;
	      fFlagUserHistoMin = "OFF";
	      fFlagUserHistoMax = "OFF";
	    }
	} // end of if ( fMyRootFile->LookAtRootFile() == kTRUE )
      else
	{
	  fStatusFileFound = kFALSE;
	  cout  << "!TEcnaHistos::ViewMatrix(...)> *ERROR* =====> "
		<< " ROOT file not found" << fTTBELL << endl;
	}
    } // ---- end of if( (fFapStexNumber > 0) &&  (fFapStexNumber <= fEcal->MaxStexInStas()) ) -----
  else
    {
      cout << "!TEcnaHistos::ViewMatrix(...)> " << fFapStexName.Data()
	   << " = " << fFapStexNumber << ". Out of range (range = [1,"
	   << fEcal->MaxStexInStas() << "]) " << fTTBELL << endl;
    }
}  // end of ViewMatrix(...)

//==========================================================================
//
//                         ViewStin   ( => option COLZ )
//   
//==========================================================================

void TEcnaHistos::CorrelationsBetweenSamples(const Int_t& StinNumber)
{
  TString   CorOrCov = fCorrelationMatrix;
  ViewStin(StinNumber, CorOrCov);
}

void TEcnaHistos::CovariancesBetweenSamples(const Int_t& StinNumber)
{
  TString   CorOrCov = fCovarianceMatrix;
  ViewStin(StinNumber, CorOrCov);
}

//==========================================================================
//
//                         ViewStin   ( => option COLZ )   
//
//  StexStin ==>
//  (sample,sample) cor or cov matrices for all the crystal of StexStin              
//
//
//==========================================================================
void TEcnaHistos::ViewStin(const Int_t& cStexStin, const TString& CorOrCov)
{
  //cor(s,s') or cov(s,s') matrices for all the crystals of one given Stin. Option COLZ mandatory.

  // cStexStin = number for cons (in case of EE)
  // StexStin   = ECNA number

  if( (fFapStexNumber > 0) &&  fFapStexNumber <= fEcal->MaxStexInStas() )
    {
      Int_t StexStin =  cStexStin; 
      if(fFlagSubDet == "EE" )
	{StexStin = fEcalNumbering->Get1DeeSCEcnaFromDeeSCCons(fFapStexNumber, cStexStin);}

      fMyRootFile->PrintNoComment();
      fMyRootFile->FileParameters(fFapAnaType,          fFapNbOfSamples,
				  fFapRunNumber,        fFapFirstReqEvtNumber,
				  fFapLastReqEvtNumber, fFapReqNbOfEvts,
				  fFapStexNumber,       fCfgResultsRootFilePath.Data());
      
      if ( fMyRootFile->LookAtRootFile() == kTRUE )              //  (ViewStin)
	{
	  fStatusFileFound = kTRUE;

	  fFapNbOfEvts = fMyRootFile->GetNumberOfEvents(fFapReqNbOfEvts, fFapStexNumber);
	  TString fp_name_short = fMyRootFile->GetRootFileNameShort(); 
	  // cout << "*TEcnaHistos::ViewStin(...)> Data are analyzed from file ----> "
	  //      << fp_name_short << endl;

	  TVectorD vStin(fEcal->MaxStinEcnaInStex());
	  for(Int_t i=0; i<fEcal->MaxStinEcnaInStex(); i++){vStin(i)=(Double_t)0.;}
	  vStin = fMyRootFile->ReadStinNumbers(fEcal->MaxStinEcnaInStex());

	  if ( fMyRootFile->DataExist() == kTRUE )
	    {
	      fStatusDataExist = kTRUE;

	      Int_t Stin_ok = 0;
	      for (Int_t index_Stin = 0; index_Stin < fEcal->MaxStinEcnaInStex(); index_Stin++)
		{
		  if ( vStin(index_Stin) == StexStin ){Stin_ok++;}
		}

	      if( Stin_ok == 1)
		{
		  fStartDate = fMyRootFile->GetStartDate();
		  fStopDate  = fMyRootFile->GetStopDate();
		  fRunType   = fMyRootFile->GetRunType();
	      
		  //......................... matrix title                              (ViewStin)
		  char* f_in_mat_tit = new char[fgMaxCar];               fCnew++;
	      
		  if ( CorOrCov == fCovarianceMatrix )
		    {sprintf(f_in_mat_tit, "Xtal's Cov(s,s') matrices in %s.",
			     fFapStinName.Data());}
		  if ( CorOrCov == fCorrelationMatrix )
		    {sprintf(f_in_mat_tit, "Xtal's Cor(s,s') matrices in %s.",
			     fFapStinName.Data());}

		  //................................. Bidim parameters
		  Int_t  GeoBidSize = fEcal->MaxSampADC()*fEcal->MaxCrysHocoInStin(); 
		  Int_t  nb_binx  = GeoBidSize;
		  Int_t  nb_biny  = GeoBidSize;
		  Axis_t xinf_bid = (Axis_t)0.;
		  Axis_t xsup_bid = (Axis_t)GeoBidSize;
		  Axis_t yinf_bid = (Axis_t)0.;
		  Axis_t ysup_bid = (Axis_t)GeoBidSize;   
      
		  //--------------------------------------------------------- (ViewStin)
		  //............. matrices reading and histogram filling
      
		  TH2D* h_geo_bid = new TH2D("geobidim_ViewStin", f_in_mat_tit,
					     nb_binx, xinf_bid, xsup_bid,
					     nb_biny, yinf_bid, ysup_bid);     fCnewRoot++;
	  
		  h_geo_bid->Reset();

		  //======================================================== (ViewStin)
	  
		  //----------------------------------------------- Geographical bidim filling
		  Int_t ReadMatSize = fFapNbOfSamples;
		  Int_t MatSize     = fEcal->MaxSampADC();
		  TMatrixD read_matrix(ReadMatSize, ReadMatSize);
		  for(Int_t i=0; i-ReadMatSize < 0; i++){for(Int_t j=0; j-ReadMatSize < 0; j++)
		    {read_matrix(i,j)=(Double_t)0.;}}

		  Int_t i_data_exist = 0;

		  for(Int_t n_crys = 0; n_crys < fEcal->MaxCrysInStin(); n_crys++)
		    {
		      if( CorOrCov == fCovarianceMatrix )
			{read_matrix = fMyRootFile->ReadCovariancesBetweenSamples(StexStin, n_crys, ReadMatSize);}
		      if ( CorOrCov == fCorrelationMatrix )
			{read_matrix = fMyRootFile->ReadCorrelationsBetweenSamples(StexStin, n_crys, ReadMatSize);}

		      if( fMyRootFile->DataExist() == kFALSE )
			{
			  fStatusDataExist = kFALSE;
			  break;   // <= if no data: exiting loop over the channels
			}
		      else
			{
			  fStatusDataExist = kTRUE;
			  i_data_exist++;

			  for(Int_t i_samp = 0 ; i_samp < ReadMatSize ; i_samp++)
			    {
			      Int_t i_xgeo = GetXSampInStin(fFapStexNumber, StexStin, n_crys, i_samp);
			      for(Int_t j_samp = 0; j_samp < ReadMatSize ; j_samp++)
				{
				  Int_t j_ygeo = GetYSampInStin(fFapStexNumber,
								StexStin,     n_crys, j_samp);
				  h_geo_bid->Fill((Double_t)i_xgeo, (Double_t)j_ygeo,
						  (Double_t)read_matrix(i_samp, j_samp));
				}
			    }
			}
		    }

		  //===========  H I S T O   M I N / M A X   M A N A G E M E N T  ========  (ViewStin)
		  //................................ Put histo min max values
		  TString quantity_code = "D_MCs_ChNb";
		  if( CorOrCov == fCorrelationMatrix ){quantity_code = "D_MCs_ChNb";}
		  if( CorOrCov == fCovarianceMatrix ){quantity_code = "H2HFccMosMatrix";}
	      
		  //.......... default if flag not set to "ON"
		  SetYminMemoFromValue(quantity_code, fCnaParHistos->GetYminDefaultValue(quantity_code));
		  SetYmaxMemoFromValue(quantity_code, fCnaParHistos->GetYmaxDefaultValue(quantity_code));
	      
		  if( fUserHistoMin == fUserHistoMax ){fFlagUserHistoMin = "AUTO"; fFlagUserHistoMax = "AUTO";}
		  //.......... user's min and/or max values
		  if( fFlagUserHistoMin == "ON" )
		    {SetYminMemoFromValue(quantity_code, fUserHistoMin); fFlagUserHistoMin = "OFF";}
		  if( fFlagUserHistoMax == "ON" )
		    {SetYmaxMemoFromValue(quantity_code, fUserHistoMax); fFlagUserHistoMax = "OFF";}
		  //................................. automatic min and/or max
		  if( fFlagUserHistoMin == "AUTO" )
		    {SetYminMemoFromValue(quantity_code, h_geo_bid->GetMinimum()); fFlagUserHistoMin = "OFF";}
		  if( fFlagUserHistoMax == "AUTO" )
		    {SetYmaxMemoFromValue(quantity_code, h_geo_bid->GetMaximum()); fFlagUserHistoMax = "OFF";}
		  //...................................... histo set ymin and ymax   (ViewStin)
		  if ( CorOrCov == fCorrelationMatrix )
		    {SetHistoFrameYminYmaxFromMemo((TH1D*)h_geo_bid, "D_MCs_ChNb");
		
		    // ************************** A GARDER EN RESERVE *******************************
		    //............. special  contour level for correlations (square root wise scale)
		    //Int_t nb_niv  = 9;
		    //Double_t* cont_niv = new Double_t[nb_niv];                  fCnew++;
		    //SqrtContourLevels(nb_niv, &cont_niv[0]);
		    //h_geo_bid->SetContour(nb_niv, &cont_niv[0]);	      
		    //delete [] cont_niv;                                  fCdelete++;
		    // ******************************** (FIN RESERVE) *******************************
		    }
		  if ( CorOrCov == fCovarianceMatrix )
		    {SetHistoFrameYminYmaxFromMemo((TH1D*)h_geo_bid, "D_TNo_ChNb");}

		  // =================================== P L O T S ========================  (ViewStin)
		  if( i_data_exist > 0 )
		    {		  
		      char* f_in = new char[fgMaxCar];                           fCnew++;

		      //...................... Taille/format canvas  
		      UInt_t canv_w = fCnaParHistos->CanvasFormatW("petit");
		      UInt_t canv_h = fCnaParHistos->CanvasFormatH("petit");
		  
		      //.................................................. paves commentaires (ViewStin)	  
		      SetAllPavesViewStin(StexStin);
		  
		      //------------------------------------ Canvas name ----------------- (ViewStin)  
		      TString name_cov_cor;
		      Int_t MaxCar = fgMaxCar;
		      name_cov_cor.Resize(MaxCar);
		      name_cov_cor = "?";
		      if( CorOrCov == fCovarianceMatrix ){name_cov_cor = "CovSS_Matrices_in_";}
		      if( CorOrCov == fCorrelationMatrix){name_cov_cor = "CorSS_Matrices_in_";}
		  
		      TString name_visu;
		      MaxCar = fgMaxCar;
		      name_visu.Resize(MaxCar);
		      name_visu = "colz";
		  
		      sprintf(f_in, "%s_%s_%s_S1_%d_R%d_%d_%d_%s%d_%s%d_%s",
			      name_cov_cor.Data(), fFapStinName.Data(),
			      fFapAnaType.Data(), fFapNbOfSamples, fFapRunNumber,
			      fFapFirstReqEvtNumber, fFapLastReqEvtNumber,
			      fFapStexName.Data(), fFapStexNumber,
			      fFapStinName.Data(), StexStin, name_visu.Data()); 
		  
		      SetHistoPresentation((TH1D*)h_geo_bid, "Stin");
		  
		      TCanvas *MainCanvas = new TCanvas(f_in, f_in, canv_w, canv_h);   fCnewRoot++;
		      fCurrentCanvas = MainCanvas; fCurrentCanvasName = f_in;
		  
		      // cout << "*TEcnaHistos::ViewStin(...)> Plot is displayed on canvas ----> " << f_in << endl;
		  
		      delete [] f_in; f_in = 0;                                 fCdelete++;
		  
		      //------------------------ Canvas draw and update ------------ (ViewStin)  
		      if( fPavComGeneralTitle != 0 ){fPavComGeneralTitle->Draw();}
		      fPavComStex->Draw();
		      fPavComStin->Draw();
		      fPavComAnaRun->Draw();
		      fPavComNbOfEvts->Draw();

		      Double_t x_margin = fCnaParHistos->BoxLeftX("bottom_left_box") - 0.005;
		      Double_t y_margin = fCnaParHistos->BoxTopY("bottom_right_box") + 0.005;		  
		      MainCanvas->Divide(1, 1, x_margin, y_margin);
		      gPad->cd(1);

		      Int_t logy = 0;  
		      gPad->SetLogy(logy);
		  
		      h_geo_bid->DrawCopy("COLZ");
		  
		      //--------------------------------------------------------------------------- (ViewStin)       
		      Int_t size_Hoco    = fEcal->MaxCrysHocoInStin();
		      Int_t size_Veco    = fEcal->MaxCrysVecoInStin();
		  
		      ViewStinGrid(fFapStexNumber, StexStin, MatSize, size_Hoco, size_Veco, " ");
		  
		      gPad->Update();
		  
		      h_geo_bid->SetStats((Bool_t)1);    

		      //      delete MainCanvas;              fCdeleteRoot++;
		    }
		  delete [] f_in_mat_tit;   f_in_mat_tit = 0;        fCdelete++;
		  
		  h_geo_bid->Delete();   h_geo_bid = 0;             fCdeleteRoot++;
		}
	      else
		{
		  cout << "!TEcnaHistos::ViewStin(...)> *ERROR* =====> "
		       << fFapStinName.Data() << " "
		       << cStexStin << " not found."
		       << " Available numbers = ";
		  for(Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++)
		    {
		      if( vStin(i) > 0 )
			{
			  if( fFlagSubDet == "EB" ){cout << (Int_t)vStin(i) << ", ";}
			  if( fFlagSubDet == "EE" )
			    {cout << fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fFapStexNumber, (Int_t)vStin(i)) << ", ";}
			}
		    }
		  cout << fTTBELL << endl;  
		}
	    }  // end of if ( myRootFile->DataExist() == kTRUE )
	  else
	    {
	      fStatusDataExist = kFALSE;
	    }
	} // end of if ( fMyRootFile->LookAtRootFile() == kTRUE )
      else
	{
	  fStatusFileFound = kFALSE;

	  cout << "!TEcnaHistos::ViewStin(...)> *ERROR* =====> "
	       << " ROOT file not found" << fTTBELL << endl;
	}
    }
  else
    {
      cout << "!TEcnaHistos::ViewStin(...)> " << fFapStexName.Data()
	   << " = " << fFapStexNumber << ". Out of range (range = [1,"
	   << fEcal->MaxStexInStas() << "]) " << fTTBELL << endl;
    }
}  // end of ViewStin(...)

//====================================================================================
//
//                         StinCrystalNumbering
//              independent of the ROOT file => StexNumber as argument
//
//====================================================================================  
void TEcnaHistos::StinCrystalNumbering(const Int_t& StexNumber, const Int_t& cStexStin)
{
//display the crystal numbering of one Stin
// cStexStin = Tower number in case of EB or SC number for construction in case of EE

  if( fFlagSubDet == "EB" ){TowerCrystalNumbering(StexNumber, cStexStin);}
  if( fFlagSubDet == "EE" ){SCCrystalNumbering(StexNumber, cStexStin);}
}
//---------------->  end of StinCrystalNumbering()

//====================================================================================
//
//                         TowerCrystalNumbering
//              independent of the ROOT file => SMNumber as argument
//
//====================================================================================  
void TEcnaHistos::TowerCrystalNumbering(const Int_t& SMNumber, const Int_t& n1SMTow)
{
  //display the crystal numbering of one tower

  if( (SMNumber > 0) && (SMNumber <= fEcal->MaxSMInEB()) )
    {
      fFapStexBarrel = fEcalNumbering->GetSMHalfBarrel(SMNumber);

      Int_t MatSize   = fEcal->MaxSampADC();
      Int_t size_eta  = fEcal->MaxCrysEtaInTow();
      Int_t size_phi  = fEcal->MaxCrysPhiInTow();

      //---------------------------------- bidim

      Int_t nb_bins  = fEcal->MaxSampADC();
      Int_t nx_gbins = nb_bins*size_eta;
      Int_t ny_gbins = nb_bins*size_phi;

      Axis_t xinf_gbid = (Axis_t)0.;
      Axis_t xsup_gbid = (Axis_t)fEcal->MaxSampADC()*size_eta;
      Axis_t yinf_gbid = (Axis_t)0.;
      Axis_t ysup_gbid = (Axis_t)fEcal->MaxSampADC()*size_phi;

      TString fg_name = "M0' crystals";
      TString fg_tit  = "Xtal numbering (chan. in tow, chan. in SM, Xtal in SM, hashed)"; 
 
      //----------------------- empty 2D histo for pave coordinates registration
      TH2D *h_gbid;
      h_gbid = new TH2D(fg_name.Data(),  fg_tit.Data(),
			nx_gbins, xinf_gbid, xsup_gbid,
			ny_gbins, yinf_gbid, ysup_gbid);    fCnewRoot++;
      h_gbid->Reset();

      //-----------------  T R A C E  D E S   P L O T S ------ (TowerCrystalNumbering)

      char* f_in = new char[fgMaxCar];                           fCnew++;
	  
      //...................... Taille/format canvas
  
      UInt_t canv_w = fCnaParHistos->CanvasFormatW("petit");
      UInt_t canv_h = fCnaParHistos->CanvasFormatH("petit");

      //........................................ couleurs
      Color_t couleur_noir       = fCnaParHistos->SetColorsForNumbers("crystal");
      Color_t couleur_rouge      = fCnaParHistos->SetColorsForNumbers("lvrb_top");
      Color_t couleur_bleu_fonce = fCnaParHistos->SetColorsForNumbers("lvrb_bottom");

      gStyle->SetPalette(1,0);          // Rainbow spectrum

      //.................................... options generales
      fCnaParHistos->SetViewHistoStyle("Stin");
          
      //.................................... paves commentaires (TowerCrystalNumbering)
  
      SetAllPavesViewStinCrysNb(SMNumber, n1SMTow);
 
      //---------------------------------------------- (TowerCrystalNumbering)

      //..................... Canvas name
      sprintf(f_in, "Crystal_Numbering_for_%s_X_%d_%s%d",
	      fFapStinName.Data(), n1SMTow, fFapStexName.Data(), SMNumber);
  
      SetHistoPresentation((TH1D*)h_gbid, "Stin");

      TCanvas *MainCanvas = new TCanvas(f_in, f_in, canv_w , canv_h);    fCnewRoot++;
      fCurrentCanvas = MainCanvas; fCurrentCanvasName = f_in;

      // cout << "*TEcnaHistosEB::TowerCrystalNumbering(...)> Plot is displayed on canvas ----> "
      //      << f_in << endl;

      Double_t x_margin = fCnaParHistos->BoxLeftX("bottom_left_box") - 0.005;
      Double_t y_margin = fCnaParHistos->BoxTopY("bottom_right_box") + 0.005;  
      MainCanvas->Divide(1, 1, x_margin, y_margin);

      fPavComStex->Draw();
      fPavComStin->Draw();
      fPavComLVRB->Draw();
  
      Bool_t b_true = 1; 
      Bool_t b_false = 0;
      gPad->cd(1);

      gStyle->SetMarkerColor(couleur_rouge);
  
      Int_t logy = 0;
      gPad->SetLogy(logy);
  
      //............................... bidim .......... (TowerCrystalNumbering)   

      h_gbid->SetStats(b_false);
      h_gbid->DrawCopy("COLZ");
    
      //..... Ecriture des numeros de channels dans la grille..... (TowerCrystalNumbering)
      //      et des numeros SM des cristaux

      //............... prepa arguments fixes appels [TText]->DrawText()
      char* f_in_elec = new char[fgMaxCar];                                         fCnew++;
      TString TowerLvrbType = fEcalNumbering->GetTowerLvrbType(n1SMTow) ;
      TText *text_elec_num = new TText();                                           fCnewRoot++;
      if ( TowerLvrbType == "top"    ){text_elec_num->SetTextColor(couleur_rouge);}
      if ( TowerLvrbType == "bottom" ){text_elec_num->SetTextColor(couleur_bleu_fonce);}
      text_elec_num->SetTextSize(0.04);

      char* f_in_sme = new char[fgMaxCar];                                         fCnew++;
      TText *text_sme_num = new TText();                                           fCnewRoot++;
      if ( TowerLvrbType == "top"    ){text_sme_num->SetTextColor(couleur_rouge);}
      if ( TowerLvrbType == "bottom" ){text_sme_num->SetTextColor(couleur_bleu_fonce);}
      text_sme_num->SetTextSize(0.03);

      char* f_in_sm = new char[fgMaxCar];                                             fCnew++;
      TText *text_sm_num = new TText();                                               fCnewRoot++;
      text_sm_num->SetTextColor(couleur_noir);
      text_sm_num->SetTextSize(0.03);

      char* f_in_hsd = new char[fgMaxCar];                                             fCnew++;
      TText *text_hsd_num = new TText();                                               fCnewRoot++;
      text_hsd_num->SetTextColor(couleur_noir);
      text_hsd_num->SetTextSize(0.03);

      //............... prepa arguments fixes appels GetXGeo(...) et GetYGeo(...)
      Int_t    i_samp  = 0;
      //Double_t off_set = (Double_t)(fEcal->MaxSampADC()/4);
      Double_t off_set = (Double_t)1.;

      //------------------ LOOP ON THE CRYSTAL ELECTRONIC CHANNEL NUMBER  (TowerCrystalNumbering)

      for (Int_t i_chan = 0; i_chan < fEcal->MaxCrysInTow(); i_chan++)
	{
	  Int_t i_xgeo = GetXSampInStin(SMNumber, n1SMTow, i_chan, i_samp);
	  Int_t i_ygeo = GetYSampInStin(SMNumber, n1SMTow, i_chan, i_samp);

	  Double_t xgi     =  i_xgeo + 3.*off_set;
	  Double_t ygj     =  i_ygeo + 7.*off_set;

	  Double_t xgi_sme =  i_xgeo + 3.*off_set;
	  Double_t ygj_sme =  i_ygeo + 5.*off_set;

	  Double_t xgi_sm  =  i_xgeo + 3.*off_set;
	  Double_t ygj_sm  =  i_ygeo + 3.*off_set;

	  Double_t xgi_hsd =  i_xgeo + 3.*off_set;
	  Double_t ygj_hsd =  i_ygeo + 1.*off_set;

	  Int_t i_crys_sme = fEcalNumbering->Get0SMEchaFrom1SMTowAnd0TowEcha(n1SMTow, i_chan);
	  Int_t i_crys_sm  = fEcalNumbering->Get1SMCrysFrom1SMTowAnd0TowEcha(n1SMTow, i_chan);

	  Double_t Eta = fEcalNumbering->GetEta(SMNumber, n1SMTow, i_chan);
	  Double_t Phi = fEcalNumbering->GetPhi(SMNumber, n1SMTow, i_chan);

	  Int_t i_crys_hsd = fEcalNumbering->GetHashedNumberFromIEtaAndIPhi((Int_t)Eta, (Int_t)Phi);

	  //------------------------------------------------------- TowerCrystalNumbering

	  sprintf(f_in_elec, "%d", i_chan);
	  text_elec_num->DrawText(xgi, ygj, f_in_elec);

	  sprintf(f_in_sme, "%d", i_crys_sme);
	  text_sme_num->DrawText(xgi_sme, ygj_sme, f_in_sme);

	  sprintf(f_in_sm, "%d", i_crys_sm);
	  text_sm_num->DrawText(xgi_sm, ygj_sm, f_in_sm);

	  sprintf(f_in_hsd, "%d", i_crys_hsd);
	  text_sm_num->DrawText(xgi_hsd, ygj_hsd, f_in_hsd);
	}
      text_sm_num->Delete();   text_sm_num   = 0;        fCdeleteRoot++;
      text_sme_num->Delete();  text_sme_num  = 0;        fCdeleteRoot++;
      text_elec_num->Delete(); text_elec_num = 0;        fCdeleteRoot++;
      text_hsd_num->Delete();  text_hsd_num  = 0;        fCdeleteRoot++;

      ViewStinGrid(SMNumber, n1SMTow, MatSize, size_eta, size_phi, "CrystalNumbering");

      gPad->Update();
      h_gbid->SetStats(b_true);

      h_gbid->Delete();     h_gbid = 0;             fCdeleteRoot++;

      delete [] f_in;       f_in      = 0;          fCdelete++; 
      delete [] f_in_sm;    f_in_sm   = 0;          fCdelete++;
      delete [] f_in_sme;   f_in_sme  = 0;          fCdelete++;
      delete [] f_in_elec;  f_in_elec = 0;          fCdelete++;
    }
  else
    {
      cout << "!TEcnaHistos::TowerCrystalNumbering(...)> SM = " << SMNumber
	   << ". Out of range ( range = [1," << fEcal->MaxSMInEB() << "] )" << fTTBELL << endl;
    }
}
//---------------->  end of TowerCrystalNumbering()

//====================================================================================
//
//                         SCCrystalNumbering
//              independent of the ROOT file => DeeNumber and n1DeeSCEcna as argument
//
//====================================================================================  
void TEcnaHistos::SCCrystalNumbering(const Int_t& DeeNumber, const Int_t& n1DeeSCCons)
{
  //display the crystal numbering of one SC

  if( (DeeNumber > 0) && (DeeNumber <= fEcal->MaxDeeInEE()) )
    {
      Int_t n1DeeSCEcna = fEcalNumbering->Get1DeeSCEcnaFromDeeSCCons(DeeNumber, n1DeeSCCons);
      fFapStexType      = fEcalNumbering->GetEEDeeType(DeeNumber);
      fFapStinQuadType  = fEcalNumbering->GetSCQuadFrom1DeeSCEcna(n1DeeSCEcna);

      //Int_t MatSize  = fEcal->MaxSampADC();
      Int_t size_IX  = fEcal->MaxCrysIXInSC();
      Int_t size_IY  = fEcal->MaxCrysIYInSC();

      //---------------------------------- bidim

      Int_t nb_bins  = fEcal->MaxSampADC();
      Int_t nx_gbins = nb_bins*size_IX;
      Int_t ny_gbins = nb_bins*size_IY;

      Axis_t xinf_gbid = (Axis_t)0.;
      Axis_t xsup_gbid = (Axis_t)fEcal->MaxSampADC()*size_IX;
      Axis_t yinf_gbid = (Axis_t)0.;
      Axis_t ysup_gbid = (Axis_t)fEcal->MaxSampADC()*size_IY;

      TString fg_name = "crystalnbring";
      TString fg_tit  = "Xtal numbering for construction"; 
  
      TH2D *h_gbid;
      h_gbid = new TH2D(fg_name.Data(),  fg_tit.Data(),
			nx_gbins, xinf_gbid, xsup_gbid,
			ny_gbins, yinf_gbid, ysup_gbid);    fCnewRoot++;
      h_gbid->Reset();

      //-----------------  T R A C E  D E S   P L O T S ------ (SCCrystalNumbering)

      char* f_in = new char[fgMaxCar];                           fCnew++;
	  
      //...................... Taille/format canvas
  
      UInt_t canv_w = fCnaParHistos->CanvasFormatW("petit");
      UInt_t canv_h = fCnaParHistos->CanvasFormatH("petit");
      //........................................ couleurs
      // Color_t couleur_noir       = fCnaParHistos->ColorDefinition("noir");
      Color_t couleur_rouge      = fCnaParHistos->ColorDefinition("rouge");
      // Color_t couleur_bleu_fonce = fCnaParHistos->ColorDefinition("bleu_fonce");

      gStyle->SetPalette(1,0);          // Rainbow spectrum
      //.................................... options generales
      fCnaParHistos->SetViewHistoStyle("Stin");
          
      //.................................... paves commentaires (SCCrystalNumbering)
      SetAllPavesViewStinCrysNb(DeeNumber, n1DeeSCEcna);

      //---------------------------------------------- (SCCrystalNumbering)
      //..................... Canvas name
      sprintf(f_in, "Crystal_Numbering_for_%s_X_%d_%s%d",
	      fFapStinName.Data(), n1DeeSCEcna,  fFapStexName.Data(), DeeNumber);

      SetHistoPresentation((TH1D*)h_gbid, "Stin");

      TCanvas *MainCanvas = new TCanvas(f_in, f_in, canv_w , canv_h);    fCnewRoot++;
      fCurrentCanvas = MainCanvas; fCurrentCanvasName = f_in;

      // cout << "*TEcnaHistosEE::SCCrystalNumbering(...)> Plot is displayed on canvas ----> "
      //      << f_in << endl;

      Double_t x_margin = fCnaParHistos->BoxLeftX("bottom_left_box") - 0.005;
      Double_t y_margin = fCnaParHistos->BoxTopY("bottom_right_box") + 0.005;

      MainCanvas->Divide(1, 1, x_margin, y_margin);

      fPavComStex->Draw();
      fPavComStin->Draw();
      fPavComCxyz->Draw();

      Bool_t b_true  = 1; 
      Bool_t b_false = 0;
      gPad->cd(1);

      gStyle->SetMarkerColor(couleur_rouge);
  
      Int_t logy = 0;
      gPad->SetLogy(logy);
  
      //............................... bidim .......... (SCCrystalNumbering)   
      h_gbid->SetStats(b_false); 
      fCnaParHistos->SetViewHistoOffsets((TH1D*)h_gbid, "Stin", " ");
      h_gbid->DrawCopy("COLZ");

      //..... Ecriture des numeros de channels dans la grille..... (SCCrystalNumbering)
      //      et des numeros Dee des cristaux
      TString SCQuadType = fEcalNumbering->GetSCQuadFrom1DeeSCEcna(n1DeeSCEcna);
      TString DeeDir     = fEcalNumbering->GetDeeDirViewedFromIP(DeeNumber);
      TString DeeEndcap  = fEcalNumbering->GetEEDeeEndcap(DeeNumber);
      Color_t couleur_SC = GetSCColor(DeeEndcap, DeeDir, SCQuadType);
      //............... prepa arguments fixes appels [TText]->DrawText()
      char* f_in_elec = new char[fgMaxCar];                                           fCnew++;
      TText *text_elec_num   = new TText();                                           fCnewRoot++;
      text_elec_num->SetTextColor(couleur_SC);
      text_elec_num->SetTextSize(0.06);

      //............... prepa arguments fixes appels GetXGeo(...) et GetYGeo(...)
      Int_t    i_samp  = 0;
      Double_t off_set = (Double_t)(fEcal->MaxSampADC()/3);

      //------------------ LOOP ON THE CRYSTAL ELECTRONIC CHANNEL NUMBER  (SCCrystalNumbering)

      for (Int_t i_chan = 0; i_chan < fEcal->MaxCrysInSC(); i_chan++)
	{
	  Int_t i_xgeo = GetXSampInStin(DeeNumber, n1DeeSCEcna, i_chan, i_samp);
	  Int_t i_ygeo = GetYSampInStin(DeeNumber, n1DeeSCEcna, i_chan, i_samp);

	  Double_t xgi = i_xgeo + off_set;
	  Double_t ygj = i_ygeo + 2*off_set;

	  //------------------------------------------------------- SCCrystalNumbering
	  Int_t i_chan_p = i_chan+1;
	  sprintf(f_in_elec, "%d", i_chan_p);   // offset = +1 (Xtal for construction numbering, CMS NOTE 2006/027)
	  text_elec_num->DrawText(xgi, ygj, f_in_elec);
	}
      text_elec_num->Delete();   text_elec_num = 0;           fCdeleteRoot++;

      ViewStinGrid(DeeNumber, n1DeeSCEcna, fEcal->MaxSampADC(), size_IX, size_IY, "CrystalNumbering");

      gPad->Update();
      h_gbid->SetStats(b_true);

      h_gbid->Delete();     h_gbid = 0;                         fCdeleteRoot++;

      delete [] f_in;       f_in      = 0;          fCdelete++; 
      delete [] f_in_elec;  f_in_elec = 0;          fCdelete++;
    }
  else
    {
      cout << "!TEcnaHistos::SCCrystalNumbering(...)> Dee = " << DeeNumber
	   << ". Out of range ( range = [1," << fEcal->MaxDeeInEE() << "] )" << fTTBELL << endl;
    }
}
//---------------->  end of SCCrystalNumbering()
              
//==================================================================================
//
//                       GetXSampInStin, GetYSampInStin
//
//==================================================================================
Int_t TEcnaHistos::GetXSampInStin(const Int_t& StexNumber,  const Int_t& StexStin,
				  const Int_t& i0StinEcha,  const Int_t& i_samp) 
{
//Gives the X coordinate in the geographic view of one Stin

  Int_t ix_geo = -1;

  if( fFlagSubDet == "EB" )
    {TString ctype = fEcalNumbering->GetStinLvrbType(StexStin);
    TString btype = fEcalNumbering->GetStexHalfStas(StexNumber);
    if( (btype == "EB+" && ctype == "bottom")  || (btype == "EB-" && ctype == "top") )
      {ix_geo = ( (fEcal->MaxCrysHocoInStin()-1)-(i0StinEcha/fEcal->MaxCrysHocoInStin()) )
	 *fEcal->MaxSampADC() + i_samp;}
    if( (btype == "EB+" &&  ctype  == "top")   || (btype == "EB-" && ctype == "bottom") )
      {ix_geo = ( i0StinEcha/fEcal->MaxCrysHocoInStin() )*fEcal->MaxSampADC() + i_samp;}}
  
  if( fFlagSubDet == "EE" )
    {  TString DeeDir = fEcalNumbering->GetDeeDirViewedFromIP(StexNumber);
    if( DeeDir == "right" )
      {ix_geo = (fEcalNumbering->GetIXCrysInSC(StexNumber, StexStin, i0StinEcha)-1)*fEcal->MaxSampADC() + i_samp;}
    if( DeeDir == "left"  )
      {ix_geo = (fEcal->MaxCrysHocoInStin() - fEcalNumbering->GetIXCrysInSC(StexNumber, StexStin, i0StinEcha))*
	 fEcal->MaxSampADC() + i_samp;}}

  return ix_geo;
}
//--------------------------------------------------------------------------------------------
Int_t TEcnaHistos::GetYSampInStin(const Int_t& StexNumber, const Int_t& StexStin,
				  const Int_t& i0StinEcha, const Int_t& j_samp)
{
//Gives the Y coordinate in the geographic view of one Stin

  Int_t jy_geo = -1;

  if( fFlagSubDet == "EB" )
    {
      TString ctype = fEcalNumbering->GetStinLvrbType(StexStin);
      TString btype = fEcalNumbering->GetStexHalfStas(StexNumber);
      
      //.......................... jy_geo for the EB+ (and beginning for the EB-)
      
      if( (btype == "EB+" && ctype == "top")    ||  (btype == "EB-" && ctype == "bottom") )
	{
	  if( i0StinEcha >=  0 && i0StinEcha <=  4 ) {jy_geo =  (i0StinEcha -  0)*fEcal->MaxSampADC() + j_samp;}
	  if( i0StinEcha >=  5 && i0StinEcha <=  9 ) {jy_geo = -(i0StinEcha -  9)*fEcal->MaxSampADC() + j_samp;}
	  if( i0StinEcha >= 10 && i0StinEcha <= 14 ) {jy_geo =  (i0StinEcha - 10)*fEcal->MaxSampADC() + j_samp;}
	  if( i0StinEcha >= 15 && i0StinEcha <= 19 ) {jy_geo = -(i0StinEcha - 19)*fEcal->MaxSampADC() + j_samp;}
	  if( i0StinEcha >= 20 && i0StinEcha <= 24 ) {jy_geo =  (i0StinEcha - 20)*fEcal->MaxSampADC() + j_samp;}
	}
      
      if( (btype == "EB+" && ctype == "bottom") ||  (btype == "EB-" && ctype == "top") )
	{
	  if( i0StinEcha >=  0 && i0StinEcha <=  4 )
	    {jy_geo = ( (fEcal->MaxCrysVecoInStin()-1) - (i0StinEcha- 0))*fEcal->MaxSampADC() + j_samp;}  
	  if( i0StinEcha >=  5 && i0StinEcha <=  9 )
	    {jy_geo = ( (fEcal->MaxCrysVecoInStin()-1) + (i0StinEcha- 9))*fEcal->MaxSampADC() + j_samp;}
	  if( i0StinEcha >= 10 && i0StinEcha <= 14 )
	    {jy_geo = ( (fEcal->MaxCrysVecoInStin()-1) - (i0StinEcha-10))*fEcal->MaxSampADC() + j_samp;}
	  if( i0StinEcha >= 15 && i0StinEcha <= 19 )
	    {jy_geo = ( (fEcal->MaxCrysVecoInStin()-1) + (i0StinEcha-19))*fEcal->MaxSampADC() + j_samp;}
	  if( i0StinEcha >= 20 && i0StinEcha <= 24 )
	    {jy_geo = ( (fEcal->MaxCrysVecoInStin()-1) - (i0StinEcha-20))*fEcal->MaxSampADC() + j_samp;}
	}
    }
  
  if( fFlagSubDet == "EE" )
    {jy_geo =
    (fEcalNumbering->GetJYCrysInSC(StexNumber, StexStin, i0StinEcha) - 1)*fEcal->MaxSampADC() + j_samp;}

  return jy_geo;
}

//===============================================================================
//
//                           ViewStinGrid
//              independent of the ROOT file => StexNumber as argument
//
//===============================================================================
void TEcnaHistos::ViewStinGrid(const Int_t& StexNumber, 
			       const Int_t&  StexStin,   const Int_t& MatSize,
			       const Int_t&  size_Hoco,  const Int_t& size_Veco,
			       const TString& chopt)
{
  //Grid of one Stin with axis Hoco and Veco

  if( fFlagSubDet == "EB"){ViewTowerGrid(StexNumber, StexStin, MatSize,
					 size_Hoco,   size_Veco,  chopt);}
  if( fFlagSubDet == "EE"){ViewSCGrid(StexNumber, StexStin, MatSize,
				      size_Hoco,   size_Veco,  chopt);}

} // end of ViewStinGrid

//===============================================================================
//
//                           ViewTowerGrid
//              independent of the ROOT file => SMNumber as argument
//
//===============================================================================
void TEcnaHistos::ViewTowerGrid(const Int_t&  SMNumber, 
				const Int_t&  n1SMTow,  const Int_t& MatSize,
				const Int_t&  size_eta, const Int_t& size_phi,
				const TString& chopt)
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

  Double_t eta_min = fEcalNumbering->GetIEtaMin(SMNumber, n1SMTow);
  Double_t eta_max = fEcalNumbering->GetIEtaMax(SMNumber, n1SMTow);

  TString  x_var_name  = GetEtaPhiAxisTitle("ietaTow");
  TString  x_direction = fEcalNumbering->GetXDirectionEB(SMNumber);

  Float_t tit_siz_x = fCnaParHistos->AxisTitleSize();
  Float_t lab_siz_x = fCnaParHistos->AxisLabelSize();
  Float_t tic_siz_x = fCnaParHistos->AxisTickSize();
  Float_t tit_off_x = fCnaParHistos->AxisTitleOffset("Towx");
  Float_t lab_off_x = fCnaParHistos->AxisLabelOffset("Towx");
  
  new TF1("f1", x_direction.Data(), eta_min, eta_max);                fCnewRoot++;

  TGaxis* sup_axis_x = 0;

  if ( x_direction == "-x" )   // NEVER  IN THIS CASE: xmin->xmax <=> right->left ("-x") direction
    {sup_axis_x = new TGaxis( -(Float_t)MatSize, (Float_t)0, (Float_t)(size_eta*MatSize), (Float_t)0.,
			      "f1", size_eta, "BCS" , 0.);                                fCnewRoot++;
    cout << "TEcnaHistosEB::ViewTowerGrid()> non foreseen case. eta with -x direction." << fTTBELL << endl;}

  if ( x_direction == "x" )    // ALWAYS IN THIS CASE: xmin->xmax <=> left->right ("x") direction
    {sup_axis_x = new TGaxis( (Float_t)0.      , (Float_t)0., (Float_t)(size_eta*MatSize), (Float_t)0.,
			      "f1", size_eta, "CS" , 0.);                                fCnewRoot++;}
  
  sup_axis_x->SetTitle(x_var_name);
  sup_axis_x->SetTitleSize(tit_siz_x);
  sup_axis_x->SetTitleOffset(tit_off_x);
  sup_axis_x->SetLabelSize(lab_siz_x);
  sup_axis_x->SetLabelOffset(lab_off_x);
  sup_axis_x->SetTickSize(tic_siz_x);
  sup_axis_x->Draw("SAME");

  //...................................................... Axe phi (y right)  (ViewTowerGrid)
  Float_t tit_siz_y = fCnaParHistos->AxisTitleSize();
  Float_t lab_siz_y = fCnaParHistos->AxisLabelSize();
  Float_t tic_siz_y = fCnaParHistos->AxisTickSize();
  Float_t tit_off_y = fCnaParHistos->AxisTitleOffset("Towy");
  Float_t lab_off_y = fCnaParHistos->AxisLabelOffset("Towy");

  if( chopt == "CrystalNumbering" )
    {
      Double_t phi_min     = fEcalNumbering->GetPhiMin(SMNumber, n1SMTow);
      Double_t phi_max     = fEcalNumbering->GetPhiMax(SMNumber, n1SMTow);
      
      TString  y_var_name  = GetEtaPhiAxisTitle("phi");
      TString  y_direction = fEcalNumbering->GetYDirectionEB(SMNumber);
  
      new TF1("f2", y_direction.Data(), phi_min, phi_max);               fCnewRoot++;
      TGaxis* sup_axis_y = 0;
      
      if ( y_direction == "-x" )  // ALWAYS IN THIS CASE: ymin->ymax <=> top->bottom ("-x") direction
	{sup_axis_y = new TGaxis( (Float_t)(size_eta*MatSize), (Float_t)0.,
				  (Float_t)(size_eta*MatSize), (Float_t)(size_phi*MatSize),
				  "f2", size_phi, "+CS", 0.);                fCnewRoot++;}
      
      if ( y_direction == "x" )   // NEVER  IN THIS CASE: ymin->ymax <=> bottom->top ("x") direction
	{sup_axis_y = new TGaxis( (Float_t)0.,  (Float_t)0., (Float_t) 0., (Float_t)(size_phi*MatSize),
				  "f2", size_phi, "BCS", 0.);                fCnewRoot++;}
      
      sup_axis_y->SetTitle(y_var_name);
      sup_axis_y->SetTitleSize(tit_siz_y);
      sup_axis_y->SetTitleOffset(tit_off_y);
      sup_axis_y->SetLabelSize(lab_siz_y);
      sup_axis_y->SetLabelOffset(lab_off_y);
      sup_axis_y->SetTickSize(tic_siz_y);
      sup_axis_y->Draw("SAME");
    }
  //...................................................... Axe j(phi) (y left)  (ViewTowerGrid)

  Double_t j_phi_min = fEcalNumbering->GetJPhiMin(SMNumber, n1SMTow);
  Double_t j_phi_max = fEcalNumbering->GetJPhiMax(SMNumber, n1SMTow);

  TString  jy_var_name  = GetEtaPhiAxisTitle("jphiTow");
  TString  jy_direction = fEcalNumbering->GetJYDirectionEB(SMNumber);

  new TF1("f3", jy_direction.Data(), j_phi_min, j_phi_max);               fCnewRoot++;
  TGaxis* sup_axis_jy = 0;

  sup_axis_jy = new TGaxis( (Float_t)0., (Float_t)0.,
			    (Float_t)0., (Float_t)(size_phi*MatSize),
			    "f3", size_phi, "SC", 0.);                fCnewRoot++;
  
  sup_axis_jy->SetTitle(jy_var_name);
  sup_axis_jy->SetTitleSize(tit_siz_y);
  sup_axis_jy->SetTitleOffset(tit_off_y);
  sup_axis_jy->SetLabelSize(lab_siz_y);
  sup_axis_jy->SetLabelOffset(lab_off_y);
  sup_axis_jy->SetTickSize(tic_siz_y);
  sup_axis_jy->Draw("SAME");
} // end of ViewTowerGrid

//===============================================================================
//
//                           ViewSCGrid
//              independent of the ROOT file => DeeNumber as argument
//
//===============================================================================
void TEcnaHistos::ViewSCGrid(const Int_t& DeeNumber, const Int_t&  n1DeeSCEcna,
			     const Int_t& MatSize,   const Int_t&  size_IX,
			     const Int_t& size_IY,   const TString& chopt)
{
  //Grid of one SC with axis IX and IY
  //.......................... lignes verticales
  Double_t xline = 0.;
  
  Double_t yline_bot = 0.;
  Double_t yline_top = (Double_t)(MatSize*size_IX);
  
  for( Int_t i = 0 ; i < size_IX ; i++)
    {  
      xline = xline + (Double_t)MatSize;
      TLine *lin;
      lin = new TLine(xline, yline_bot, xline, yline_top); fCnewRoot++;
      lin->Draw();
      // delete lin;             fCdeleteRoot++;
    }
  //............................. lignes horizontales
  Double_t xline_left  = 0;
  Double_t xline_right = (Double_t)(MatSize*size_IX);
  
  Double_t yline = -(Double_t)MatSize;
  
  for( Int_t j = 0 ; j < size_IX+1 ; j++)
    {
      yline = yline + (Double_t)MatSize;
      TLine *lin;
      lin = new TLine(xline_left, yline, xline_right, yline); fCnewRoot++;
      lin->Draw();
      // delete lin;             fCdeleteRoot++;
    }
 
  //------------------ trace axes en IX et IY --------------- ViewSCGrid

  //...................................................... Axe i(IX) (x bottom)  (ViewSCGrid)

  Double_t IX_min = fEcalNumbering->GetIIXMin(n1DeeSCEcna) - 0.5;
  Double_t IX_max = fEcalNumbering->GetIIXMax(n1DeeSCEcna) + 0.5;

  Float_t axis_x_inf  = 0;
  Float_t axis_x_sup  = 0;
  Float_t axis_y_inf  = 0;
  Float_t axis_y_sup  = 0;
  Int_t   axis_nb_div = 1;
  Double_t IX_values_min = 0;
  Double_t IX_values_max = 0;
  Option_t* axis_chopt = "CS";

  Float_t tit_siz_x = fCnaParHistos->AxisTitleSize();
  Float_t lab_siz_x = fCnaParHistos->AxisLabelSize();
  Float_t tic_siz_x = fCnaParHistos->AxisTickSize();
  Float_t tit_off_x = fCnaParHistos->AxisTitleOffset("SCx");
  Float_t lab_off_x = fCnaParHistos->AxisLabelOffset("SCx");

  TString StrDee = "iIXSC";
  if( DeeNumber == 1 ){StrDee = "iIXDee1";}
  if( DeeNumber == 2 ){StrDee = "iIXDee2";}
  if( DeeNumber == 3 ){StrDee = "iIXDee3";}
  if( DeeNumber == 4 ){StrDee = "iIXDee4";}

  TString  x_var_name  = GetIXIYAxisTitle(StrDee.Data());
  TString  x_direction = fEcalNumbering->GetXDirectionEE(DeeNumber);

  TGaxis* sup_axis_x = 0;

  if( DeeNumber == 1 ) //  -xmin -> -xmax <=> left->right
    {
      axis_x_inf    = 0; axis_y_inf  = 0;  axis_x_sup    = size_IX*MatSize;  axis_y_sup    = 0;
      axis_nb_div   = size_IX;
      IX_values_min = -IX_min ;   IX_values_max = -IX_max; axis_chopt = "CS";
    }
  if( DeeNumber == 2 ) //  xmin -> xmax <=> right->left
    {
      axis_x_inf    = 0; axis_y_inf  = 0;  axis_x_sup    = size_IX*MatSize;  axis_y_sup    = 0;
      axis_nb_div   = size_IX;
      IX_values_min = IX_min ;    IX_values_max = IX_max;   axis_chopt = "CS";
    }
  if( DeeNumber == 3 )  //  xmin -> xmax <=>  left->right
    {
      axis_x_inf    = 0; axis_y_inf  = 0;  axis_x_sup    = size_IX*MatSize;  axis_y_sup    = 0;
      axis_nb_div   = size_IX;
      IX_values_min = IX_min ;    IX_values_max = IX_max;   axis_chopt = "CS";
    }
  if( DeeNumber == 4 )  //  -xmin -> -xmax <=> right->left
    {
      axis_x_inf    = 0; axis_y_inf  = 0;  axis_x_sup    = size_IX*MatSize;  axis_y_sup    = 0;
      axis_nb_div   = size_IX;
      IX_values_min = -IX_min ;   IX_values_max = -IX_max; axis_chopt = "CS";
    }

  new TF1("f1", x_direction.Data(), IX_values_min, IX_values_max);    fCnewRoot++;
  sup_axis_x = new TGaxis( axis_x_inf, axis_y_inf, axis_x_sup, axis_y_sup,
			   "f1", axis_nb_div, axis_chopt , 0.);   fCnewRoot++;

  sup_axis_x->SetTitle(x_var_name);
  sup_axis_x->SetTitleSize(tit_siz_x);
  sup_axis_x->SetTitleOffset(tit_off_x);
  sup_axis_x->SetLabelSize(lab_siz_x);
  sup_axis_x->SetLabelOffset(lab_off_x);
  sup_axis_x->SetTickSize(tic_siz_x);     // <===== NE MARCHE QU'AVEC L'OPTION "S"
  sup_axis_x->Draw("SAME");

  //...................................................... Axe j(IY) (ViewSCGrid)

  Float_t tit_siz_y = fCnaParHistos->AxisTitleSize();
  Float_t lab_siz_y = fCnaParHistos->AxisLabelSize();
  Float_t tic_siz_y = fCnaParHistos->AxisTickSize();
  Float_t tit_off_y = fCnaParHistos->AxisTitleOffset("SCy");
  Float_t lab_off_y = fCnaParHistos->AxisLabelOffset("SCy");

  Double_t j_IY_min = fEcalNumbering->GetJIYMin(DeeNumber, n1DeeSCEcna) - 0.5;
  Double_t j_IY_max = fEcalNumbering->GetJIYMax(DeeNumber, n1DeeSCEcna) + 0.5;

  TString  jy_var_name  = GetIXIYAxisTitle("jIYSC");
  TString  jy_direction = fEcalNumbering->GetJYDirectionEE(DeeNumber);

  new TF1("f2", jy_direction.Data(), j_IY_min, j_IY_max);             fCnewRoot++;

  TGaxis* sup_axis_jy = new TGaxis( (Float_t)0., (Float_t)0.,
			    (Float_t)0., (Float_t)(size_IY*MatSize),
			    "f2", size_IY, "CS", 0.);                     fCnewRoot++;

  sup_axis_jy->SetTitle(jy_var_name);
  sup_axis_jy->SetTitleSize(tit_siz_y);
  sup_axis_jy->SetTitleOffset(tit_off_y);
  sup_axis_jy->SetLabelSize(lab_siz_y);
  sup_axis_jy->SetLabelOffset(lab_off_y);
  sup_axis_jy->SetTickSize(tic_siz_y);     // <===== NE MARCHE QU'AVEC L'OPTION "S"
  sup_axis_jy->Draw();

} // end of ViewSCGrid

//=======================================================================================
//
//                              ViewStex(***)     
//
//           (Hoco,Veco) matrices for all the Stins of a Stex             
//
//     arg_read_histo:    1D array containing the quantity for each channel in the Stex
//                        (dim = MaxCrysInStex())
//     arg_AlreadyRead:   =1 <=> arg_read_histo 
//                        =0 <=> read the 1D array in this method with TEcnaRead
//     
//      HistoCode:        code for the plotted quantity
//
//=======================================================================================
void TEcnaHistos::ViewStex(const TVectorD& arg_read_histo, const Int_t& arg_AlreadyRead,
			   const TString&   HistoCode)
{
// (Hoco, Veco) matrices for all the Stins of a Stex

  Bool_t OKFileExists = kFALSE;
  Bool_t OKData  = kFALSE;

  Int_t n1StexStin = -1;

  if( arg_AlreadyRead == fTobeRead )
    {
      fMyRootFile->PrintNoComment();
      fMyRootFile->FileParameters(fFapAnaType,          fFapNbOfSamples,
				  fFapRunNumber,        fFapFirstReqEvtNumber,
				  fFapLastReqEvtNumber, fFapReqNbOfEvts,
				  fFapStexNumber,       fCfgResultsRootFilePath.Data());
      
      if( fMyRootFile->LookAtRootFile() == kTRUE ){OKFileExists = kTRUE;}

      if( OKFileExists == kTRUE )
	{
	  fFapNbOfEvts = fMyRootFile->GetNumberOfEvents(fFapReqNbOfEvts, fFapStexNumber);
	  TString fp_name_short = fMyRootFile->GetRootFileNameShort();
	  // cout << "*TEcnaHistos::ViewStex(...)> Data are analyzed from file ----> "
	  //      << fp_name_short << endl;
	  
	  fStartDate = fMyRootFile->GetStartDate();
	  fStopDate  = fMyRootFile->GetStopDate();
	  fRunType   = fMyRootFile->GetRunType();
	}
    }
  if( arg_AlreadyRead >= 1 )
    {
      OKFileExists = kTRUE;
    }

  if( OKFileExists == kTRUE ) 
    {
      fStatusFileFound = kTRUE;

      //......................... matrix title    (ViewStex)
      char* f_in_mat_tit = new char[fgMaxCar];               fCnew++;
      sprintf(f_in_mat_tit, "?");

      if (HistoCode == "D_NOE_ChNb") {sprintf(f_in_mat_tit, "Number of events");}
      if (HistoCode == "D_Ped_ChNb") {sprintf(f_in_mat_tit, "Pedestals");}
      if (HistoCode == "D_TNo_ChNb") {sprintf(f_in_mat_tit, "Total noise");}
      if (HistoCode == "D_MCs_ChNb") {sprintf(f_in_mat_tit, "Mean cor(s,s')");}
      if (HistoCode == "D_LFN_ChNb") {sprintf(f_in_mat_tit, "Low frequency noise");}
      if (HistoCode == "D_HFN_ChNb") {sprintf(f_in_mat_tit, "High frequency noise");}
      if (HistoCode == "D_SCs_ChNb") {sprintf(f_in_mat_tit, "Sigma of cor(s,s')");}
      
      //................................. Axis parameters
      Int_t  GeoBidSizeHoco = fEcal->MaxStinHocoInStex()*fEcal->MaxCrysHocoInStin();
      Int_t  GeoBidSizeVeco = fEcal->MaxStinVecoInStex()*fEcal->MaxCrysVecoInStin();

      Int_t  nb_binx  = GeoBidSizeHoco;
      Int_t  nb_biny  = GeoBidSizeVeco;
      Axis_t xinf_bid = (Axis_t)0.;
      Axis_t xsup_bid = (Axis_t)GeoBidSizeHoco;
      Axis_t yinf_bid = (Axis_t)0.;
      Axis_t ysup_bid = (Axis_t)GeoBidSizeVeco;   
      
      TString axis_x_var_name = "  #Hoco  ";
      TString axis_y_var_name = "  #Veco  ";
      
      //............. matrices reading and histogram filling   (ViewStex)
      
      TH2D* h_geo_bid = new TH2D("geobidim_ViewStex", f_in_mat_tit,
				 nb_binx, xinf_bid,  xsup_bid,
				 nb_biny, yinf_bid,  ysup_bid);     fCnewRoot++;

      h_geo_bid->Reset();

      //............................................... 1D histo reading  (ViewStex)
      TVectorD partial_histp(fEcal->MaxCrysEcnaInStex());
      for(Int_t i=0; i<fEcal->MaxCrysEcnaInStex(); i++){partial_histp(i)=(Double_t)0.;}

      if( arg_AlreadyRead == fTobeRead )
	{
	  if (HistoCode == "D_NOE_ChNb" ){partial_histp = fMyRootFile->ReadNumberOfEvents(fEcal->MaxCrysEcnaInStex());}
	  if (HistoCode == "D_Ped_ChNb" ){
	    partial_histp = fMyRootFile->ReadPedestals(fEcal->MaxCrysEcnaInStex());}
	  if (HistoCode == "D_TNo_ChNb" ){
	    partial_histp = fMyRootFile->ReadTotalNoise(fEcal->MaxCrysEcnaInStex());}
	  if (HistoCode == "D_MCs_ChNb" ){
	    partial_histp = fMyRootFile->ReadMeanCorrelationsBetweenSamples(fEcal->MaxCrysEcnaInStex());}
	  if (HistoCode == "D_LFN_ChNb" ){
	    partial_histp = fMyRootFile->ReadLowFrequencyNoise(fEcal->MaxCrysEcnaInStex());}
	  if (HistoCode == "D_HFN_ChNb" ){
	    partial_histp = fMyRootFile->ReadHighFrequencyNoise(fEcal->MaxCrysEcnaInStex());}
	  if (HistoCode == "D_SCs_ChNb" ){
	    partial_histp = fMyRootFile->ReadSigmaOfCorrelationsBetweenSamples(fEcal->MaxCrysEcnaInStex());}

	  OKData = fMyRootFile->DataExist();
	}

      if( arg_AlreadyRead >= 1 )
	{
	  partial_histp = arg_read_histo;
	  OKData = kTRUE;
	}

      //------------------------------- Build 2D matrix to be ploted from 1D read histo  (ViewStex)
      TMatrixD read_matrix(nb_binx, nb_biny);
      for(Int_t i=0; i<nb_binx; i++)
	{for(Int_t j=0; j<nb_biny; j++){read_matrix(i,j)=(Double_t)0.;}}

      if ( OKData == kTRUE )
	{
	  fStatusDataExist = kTRUE;

	  for(Int_t i0StexStinEcna=0; i0StexStinEcna<fEcal->MaxStinEcnaInStex(); i0StexStinEcna++)
	    {
	      if( arg_AlreadyRead == fTobeRead )	      
		{n1StexStin = fMyRootFile->GetStexStinFromIndex(i0StexStinEcna);}
	      if( arg_AlreadyRead >= 1 )
		{n1StexStin = i0StexStinEcna+1;}

	      if (n1StexStin != -1)
		{
		  //------------------ Geographical bidim filling   (ViewStex)
		  for(Int_t i0StinEcha=0; i0StinEcha<fEcal->MaxCrysInStin(); i0StinEcha++)
		    {
		      Int_t iStexEcha = (n1StexStin-1)*fEcal->MaxCrysInStin() + i0StinEcha;
		      Int_t i_xgeo = GetXCrysInStex(fFapStexNumber, n1StexStin, i0StinEcha);
		      Int_t i_ygeo = GetYCrysInStex(fFapStexNumber, n1StexStin, i0StinEcha);
		      
		      if(i_xgeo >=0 && i_xgeo < nb_binx && i_ygeo >=0 && i_ygeo < nb_biny)
			{
			  read_matrix(i_xgeo, i_ygeo) = partial_histp(iStexEcha);
			  h_geo_bid->Fill((Double_t)i_xgeo, (Double_t)i_ygeo,
					  (Double_t)read_matrix(i_xgeo, i_ygeo));
			}	   
		    }
		}
	    }
	  
	  //===============  H I S T O   M I N / M A X   M A N A G E M E N T ============  (ViewStex)
	  
	  //................................ Put histo min max values
	  //.......... default if flag not set to "ON"
	  SetYminMemoFromValue(HistoCode.Data(), fCnaParHistos->GetYminDefaultValue(HistoCode.Data()));
	  SetYmaxMemoFromValue(HistoCode.Data(), fCnaParHistos->GetYmaxDefaultValue(HistoCode.Data()));
	  
	  if( fUserHistoMin == fUserHistoMax ){fFlagUserHistoMin = "AUTO"; fFlagUserHistoMax = "AUTO";}
	  //.......... user's value if flag set to "ON"
	  if( fFlagUserHistoMin == "ON" )
	    {SetYminMemoFromValue(HistoCode.Data(), fUserHistoMin); fFlagUserHistoMin = "OFF";}
	  if( fFlagUserHistoMax == "ON" )
	    {SetYmaxMemoFromValue(HistoCode.Data(), fUserHistoMax); fFlagUserHistoMax = "OFF";}
	  //................................. automatic min and/or max
	  if( fFlagUserHistoMin == "AUTO" )
	    {SetYminMemoFromValue(HistoCode.Data(), h_geo_bid->GetMinimum()); fFlagUserHistoMin = "OFF";}
	  if( fFlagUserHistoMax == "AUTO" )
	    {SetYmaxMemoFromValue(HistoCode.Data(), h_geo_bid->GetMaximum()); fFlagUserHistoMax = "OFF";}
	  //...................................... histo set ymin and ymax
	  SetHistoFrameYminYmaxFromMemo((TH1D*)h_geo_bid, HistoCode);
	  
	  // ************************** A GARDER EN RESERVE *******************************
	  //............. special contour level for correlations (square root wise scale)
	  //if ( HistoCode == "D_MCs_ChNb" )
	  //{
	  //  Int_t nb_niv  = 9;
	  //  Double_t* cont_niv = new Double_t[nb_niv];           fCnew++;
	  //  SqrtContourLevels(nb_niv, &cont_niv[0]);      
	  //  h_geo_bid->SetContour(nb_niv, &cont_niv[0]);	      
	  //  delete [] cont_niv;                                  fCdelete++;
	  //}
	  // ******************************** (FIN RESERVE) *******************************
	  
	  // =================================== P L O T S ========================   (ViewStex) 
	  
	  char* f_in = new char[fgMaxCar];                           fCnew++;

	  //...................... Taille/format canvas
	  UInt_t canv_h = fCnaParHistos->CanvasFormatH("petit");
	  UInt_t canv_w = fCnaParHistos->CanvasFormatW("petit");

	  if( fFlagSubDet == "EB")
	  {canv_h = fCnaParHistos->CanvasFormatH("etaphiSM");
	  canv_w = fCnaParHistos->CanvasFormatW("etaphiSM");}
	  if( fFlagSubDet == "EE")	  
	  {canv_h = fCnaParHistos->CanvasFormatH("IXIYDee");
	  canv_w = fCnaParHistos->CanvasFormatW("IXIYDee");}

	  //............................................... paves commentaires (ViewStex)
	  SetAllPavesViewStex(fFapStexNumber);
	  
	  //------------------------------------ Canvas name ----------------- (ViewStex)  
	  TString name_cov_cor;
	  Int_t MaxCar = fgMaxCar;
	  name_cov_cor.Resize(MaxCar);
	  name_cov_cor = "?";

	  if( HistoCode == "D_NOE_ChNb"){name_cov_cor = "Nb_Of_D_Adc_EvDs";}
	  if( HistoCode == "D_Ped_ChNb"){name_cov_cor = "Pedestals";}
	  if( HistoCode == "D_TNo_ChNb"){name_cov_cor = "Total_noise";}
	  if( HistoCode == "D_MCs_ChNb"){name_cov_cor = "Mean_Corss";}
	  if( HistoCode == "D_LFN_ChNb"){name_cov_cor = "Low_Fq_Noise";}
	  if( HistoCode == "D_HFN_ChNb"){name_cov_cor = "High_Fq_Noise";}
	  if( HistoCode == "D_SCs_ChNb"){name_cov_cor = "Sigma_Corss";}
	  
	  TString name_visu;
	  MaxCar = fgMaxCar;
	  name_visu.Resize(MaxCar);
	  name_visu = "colz";
	  
	  TString flag_already_read;
	  MaxCar = fgMaxCar;
	  flag_already_read.Resize(MaxCar);
	  flag_already_read = "?";
	  sprintf(f_in,"M%d", arg_AlreadyRead); flag_already_read = f_in;

	  sprintf(f_in, "%s_%s_S1_%d_R%d_%d_%d_%s%d_%s_HocoVeco_R%s",
		  name_cov_cor.Data(), fFapAnaType.Data(), fFapNbOfSamples, fFapRunNumber,
		  fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapStexName.Data(), fFapStexNumber,
		  name_visu.Data(), flag_already_read.Data());
	  
	  if( fFlagSubDet == "EB" ){SetHistoPresentation((TH1D*)h_geo_bid, "Stex2DEB");}
	  if( fFlagSubDet == "EE" ){SetHistoPresentation((TH1D*)h_geo_bid, "Stex2DEE");}
	  
	  TCanvas *MainCanvas = new TCanvas(f_in, f_in, canv_w, canv_h);   fCnewRoot++;
	  fCurrentCanvas = MainCanvas; fCurrentCanvasName = f_in;

	  // cout << "*TEcnaHistos::ViewStex(...)> Plot is displayed on canvas ----> " << f_in << endl;
	  
	  delete [] f_in; f_in = 0;                                 fCdelete++;
	  
	  //------------------------ Canvas draw and update ------------ (ViewStex)  
	  if( fPavComGeneralTitle != 0 ){fPavComGeneralTitle->Draw();}
	  fPavComStex->Draw();
	  fPavComAnaRun->Draw();
	  fPavComNbOfEvts->Draw();

	  //----------------------------------------------------------- pad margins
	  Double_t x_low = fCnaParHistos->BoxLeftX("bottom_left_box")    - 0.005;
	  Double_t y_low = fCnaParHistos->BoxTopY("bottom_left_box")     + 0.005;
	  Double_t x_margin = x_low;
	  Double_t y_margin = y_low;	  
	  MainCanvas->Divide( 1,  1, x_margin, y_margin);
	  //           Divide(nx, ny, x_margin, y_margin,    color);	  
	  gPad->cd(1);
	  //........................... specific EE
	  if( fFlagSubDet == "EE" )
	    {Double_t x_up  = fCnaParHistos->BoxRightX("bottom_right_box")  + 0.005;
	    Double_t y_up  = fCnaParHistos->BoxBottomY("top_left_box_Dee") - 0.005;
	    TVirtualPad* main_subpad = gPad;
	    main_subpad->SetPad(x_low, y_low, x_up, y_up);}
	  
	  //------------------------------------------------------------
	  h_geo_bid->GetXaxis()->SetTitle(axis_x_var_name);
	  h_geo_bid->GetYaxis()->SetTitle(axis_y_var_name);
	  
	  h_geo_bid->DrawCopy("COLZ");
	  
	  // trace de la grille: un rectangle = une tour ou un SC ---------------- (ViewStex) 
	  ViewStexGrid(fFapStexNumber, " ");
	  gPad->Draw();
	  gPad->Update();

	  //..................... retour aux options standard
	  Bool_t b_true = 1;
	  h_geo_bid->SetStats(b_true);    
	  h_geo_bid->Delete();  h_geo_bid = 0;              fCdeleteRoot++;

	  //      delete MainCanvas;              fCdeleteRoot++;
	}  // end of if OKData == kTRUE )
      delete [] f_in_mat_tit;    f_in_mat_tit = 0;                        fCdelete++;
    } // end of if OKFileExists == kTRUE )
  else
    {
      fStatusFileFound = kFALSE;

      cout << "!TEcnaHistos::ViewStex(...)> *ERROR* =====> "
	   << " ROOT file not found" << fTTBELL << endl;
    }
}  // end of ViewStex(...)

//===========================================================================
//
//                       StexHocoVecoLHFCorcc(***)
//
//     Geographical view of the cor(c,c) matrices (mean over samples) of
//     all (Stin_A,Stin_A) [case A=B only] of a given Stex (BIG MATRIX)
//
//===========================================================================  
void TEcnaHistos::StexHocoVecoLHFCorcc(const TString& Freq)
{
// (Hoco, Veco) matrices for all the Stins of a Stex

  fMyRootFile->PrintNoComment();
  fMyRootFile->FileParameters(fFapAnaType,          fFapNbOfSamples,
			      fFapRunNumber,        fFapFirstReqEvtNumber,
			      fFapLastReqEvtNumber, fFapReqNbOfEvts,
			      fFapStexNumber,       fCfgResultsRootFilePath.Data());
  
  if ( fMyRootFile->LookAtRootFile() == kTRUE )                 // (StexHocoVecoLHFCorcc)
    {
      fStatusFileFound = kTRUE;

      fFapNbOfEvts = fMyRootFile->GetNumberOfEvents(fFapReqNbOfEvts, fFapStexNumber);
      TString fp_name_short = fMyRootFile->GetRootFileNameShort(); 
      //cout << "*TEcnaHistos::StexHocoVecoLHFCorcc(...)> Data are analyzed from file ----> "
      //     << fp_name_short << endl;

      fStartDate = fMyRootFile->GetStartDate();
      fStopDate  = fMyRootFile->GetStopDate();
      fRunType   = fMyRootFile->GetRunType();
      
      //......................... matrix title  
      char* f_in_mat_tit = new char[fgMaxCar];               fCnew++;
      
      if( fFlagSubDet == "EB" && Freq == "LF" )
	{sprintf(f_in_mat_tit, "LF Cor(Xtal,Xtal') for each tower in SM");}
      if( fFlagSubDet == "EB" && Freq == "HF" )
	{sprintf(f_in_mat_tit, "HF Cor(Xtal,Xtal') for each tower in SM");}
      if( fFlagSubDet == "EE" && Freq == "LF"  )
	{sprintf(f_in_mat_tit, "LF Cor(Xtal,Xtal') for each SC in Dee");}
      if( fFlagSubDet == "EE" && Freq == "HF"  )
	{sprintf(f_in_mat_tit, "HF Cor(Xtal,Xtal') for each SC in Dee");}

      //................................. Axis parameters
      Int_t  GeoBidSizeHoco = fEcal->MaxStinHocoInStex()*fEcal->MaxCrysInStin();
      Int_t  GeoBidSizeVeco = fEcal->MaxStinVecoInStex()*fEcal->MaxCrysInStin();

      Int_t  nb_binx  = GeoBidSizeHoco;
      Int_t  nb_biny  = GeoBidSizeVeco;
      Axis_t xinf_bid = (Axis_t)0.;
      Axis_t xsup_bid = (Axis_t)GeoBidSizeHoco;
      Axis_t yinf_bid = (Axis_t)0.;
      Axis_t ysup_bid = (Axis_t)GeoBidSizeVeco;   
      
      TString axis_x_var_name = "  #Hoco  ";
      TString axis_y_var_name = "  #varVeco  ";

      //======================================================== (StexHocoVecoLHFCorcc)
      TVectorD Stin_numbers(fEcal->MaxStinEcnaInStex());
      for(Int_t i=0; i<fEcal->MaxStinEcnaInStex(); i++){Stin_numbers(i)=(Double_t)0.;}
      Stin_numbers = fMyRootFile->ReadStinNumbers(fEcal->MaxStinEcnaInStex());

      if ( fMyRootFile->DataExist() == kTRUE )
	{
	  fStatusDataExist = kTRUE;

	  //............. matrices reading and histogram filling
	  TMatrixD partial_matrix(fEcal->MaxCrysEcnaInStex(), fEcal->MaxCrysEcnaInStex());
	  for(Int_t i=0; i<fEcal->MaxCrysEcnaInStex(); i++)
	    {for(Int_t j=0; j<fEcal->MaxCrysEcnaInStex(); j++){partial_matrix(i,j)=(Double_t)0.;}}

	  if( Freq == "LF")
	    {
	      partial_matrix = fMyRootFile->ReadLowFrequencyCorrelationsBetweenChannels(fEcal->MaxCrysEcnaInStex());
	    }
	  if( Freq == "HF")
	    {
	      partial_matrix = fMyRootFile->ReadHighFrequencyCorrelationsBetweenChannels(fEcal->MaxCrysEcnaInStex());
	    }

	  if ( fMyRootFile->DataExist() == kTRUE )
	    {
	      fStatusDataExist = kTRUE;
	      
	      //............................... 2D histo booking
	      TH2D* h_geo_bid = new TH2D("geobidim_HocoVecoLHFCorcc", f_in_mat_tit,
					 nb_binx, xinf_bid,  xsup_bid,
					 nb_biny, yinf_bid,  ysup_bid);     fCnewRoot++;
	      h_geo_bid->Reset();

	      fFapStexBarrel = fEcalNumbering->GetStexHalfStas(fFapStexNumber);
	      
	      for(Int_t i0StexStinEcna=0; i0StexStinEcna<fEcal->MaxStinEcnaInStex(); i0StexStinEcna++)
		{
		  Int_t n1StexStin = (Int_t)Stin_numbers(i0StexStinEcna);
		  Int_t offset_x = ((n1StexStin-1)/fEcal->MaxStinVecoInStex())*fEcal->MaxCrysInStin();
		  Int_t offset_y = ((n1StexStin-1)%fEcal->MaxStinVecoInStex())*fEcal->MaxCrysInStin();
		  
		  if (n1StexStin != -1)
		    {
		      //================================================= (StexHocoVecoLHFCorcc)
		      //------------------ Geographical bidim filling
		      for(Int_t i0StinEcha=0; i0StinEcha<fEcal->MaxCrysInStin(); i0StinEcha++)
			{
			  for(Int_t j0StinEcha=0; j0StinEcha<fEcal->MaxCrysInStin(); j0StinEcha++)
			    {
			      Int_t i_xgeo = offset_x + i0StinEcha;
			      Int_t i_ygeo = offset_y + j0StinEcha;
			      
			      if(i_xgeo >=0 && i_xgeo < nb_binx && i_ygeo >=0 && i_ygeo < nb_biny)
				{
				  Int_t iEcha = (n1StexStin-1)*fEcal->MaxCrysInStin() + i0StinEcha;
				  Int_t jEcha = (n1StexStin-1)*fEcal->MaxCrysInStin() + j0StinEcha;
				  
				  h_geo_bid->Fill((Double_t)i_xgeo, (Double_t)i_ygeo,
						  (Double_t)partial_matrix(iEcha, jEcha));
				}
			    }	   
			}
		    }
		}
	      
	      //===============  H I S T O   M I N / M A X   M A N A G E M E N T ============  (StexHocoVecoLHFCorcc)
	      
	      TString HistoCode = "H2CorccInStins";
	      
	      //................................ Put histo min max values
	      //.......... default if flag not set to "ON"
	      SetYminMemoFromValue(HistoCode.Data(), fCnaParHistos->GetYminDefaultValue(HistoCode.Data()));
	      SetYmaxMemoFromValue(HistoCode.Data(), fCnaParHistos->GetYmaxDefaultValue(HistoCode.Data()));
	      
	      if( fUserHistoMin == fUserHistoMax ){fFlagUserHistoMin = "AUTO"; fFlagUserHistoMax = "AUTO";}
	      //.......... user's value if flag set to "ON"
	      if( fFlagUserHistoMin == "ON" )
		{SetYminMemoFromValue(HistoCode.Data(), fUserHistoMin); fFlagUserHistoMin = "OFF";}
	      if( fFlagUserHistoMax == "ON" )
		{SetYmaxMemoFromValue(HistoCode.Data(), fUserHistoMax); fFlagUserHistoMax = "OFF";}
	      //................................. automatic min and/or max
	      if( fFlagUserHistoMin == "AUTO" )
		{SetYminMemoFromValue(HistoCode.Data(), h_geo_bid->GetMinimum()); fFlagUserHistoMin = "OFF";}
	      if( fFlagUserHistoMax == "AUTO" )
		{SetYmaxMemoFromValue(HistoCode.Data(), h_geo_bid->GetMaximum()); fFlagUserHistoMax = "OFF";}
	      //...................................... histo set ymin and ymax
	      SetHistoFrameYminYmaxFromMemo((TH1D*)h_geo_bid, HistoCode);
	      
	      // ----------------------------------- P L O T S   (StexHocoVecoLHFCorcc)
	      
	      char* f_in = new char[fgMaxCar];                           fCnew++;
	      
	      //...................... Taille/format canvas
	      
	      UInt_t canv_h = fCnaParHistos->CanvasFormatH("petit");
	      UInt_t canv_w = fCnaParHistos->CanvasFormatW("petit");
	      
	      if( fFlagSubDet == "EB")
		{canv_h = fCnaParHistos->CanvasFormatH("etaphiSM");
		canv_w = fCnaParHistos->CanvasFormatW("etaphiSM");}
	      if( fFlagSubDet == "EE")	  
		{canv_h = fCnaParHistos->CanvasFormatH("IXIYDee");
		canv_w = fCnaParHistos->CanvasFormatW("IXIYDee");}
	      
	      //..................................... paves commentaires (StexHocoVecoLHFCorcc)
	      SetAllPavesViewStex(fFapStexNumber);	  
	      
	      //----------------- Canvas name ------- (StexHocoVecoLHFCorcc)
	      TString name_cov_cor;
	      Int_t MaxCar = fgMaxCar;
	      name_cov_cor.Resize(MaxCar);
	      if( Freq == "LF" ){name_cov_cor = "StexLFCorcc";}
	      if( Freq == "HF" ){name_cov_cor = "StexHFCorcc";}
	      
	      TString name_visu;
	      MaxCar = fgMaxCar;
	      name_visu.Resize(MaxCar);
	      name_visu = "colz";
	      
	      sprintf(f_in, "%s_%s_S1_%d_R%d_%d_%d_Stex%s%d_%s_HocoVeco",
		      name_cov_cor.Data(), fFapAnaType.Data(), fFapNbOfSamples, fFapRunNumber,
		      fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapStexName.Data(), fFapStexNumber,
		      name_visu.Data());
	      
	      if( fFlagSubDet == "EB" ){SetHistoPresentation((TH1D*)h_geo_bid, "Stex2DEB");}
	      if( fFlagSubDet == "EE" ){SetHistoPresentation((TH1D*)h_geo_bid, "Stex2DEE");}
	      
	      TCanvas *MainCanvas = new TCanvas(f_in, f_in, canv_w, canv_h);   fCnewRoot++;
	      fCurrentCanvas = MainCanvas; fCurrentCanvasName = f_in;

	      // cout << "*TEcnaHistos::StexHocoVecoLHFCorcc(...)> Plot is displayed on canvas ----> "
	      //      << f_in << endl;
	      
	      delete [] f_in; f_in = 0;                                 fCdelete++;
	     
	      //------------ Canvas draw and update ------ (StexHocoVecoLHFCorcc)  
	      if( fPavComGeneralTitle != 0 ){fPavComGeneralTitle->Draw();}
	      fPavComStex->Draw();
	      fPavComAnaRun->Draw();
	      fPavComNbOfEvts->Draw();

	      //----------------------------------------------------------- pad margins
	      Double_t x_low = fCnaParHistos->BoxLeftX("bottom_left_box")    - 0.005;
	      Double_t y_low = fCnaParHistos->BoxTopY("bottom_left_box")     + 0.005;
	      Double_t x_margin = x_low;
	      Double_t y_margin = y_low;	  
	      MainCanvas->Divide( 1,  1, x_margin, y_margin);
	      //           Divide(nx, ny, x_margin, y_margin,    color);	  
	      gPad->cd(1);
	      //........................... specific EE
	      if( fFlagSubDet == "EE" )
		{
		  Double_t x_up  = fCnaParHistos->BoxRightX("bottom_right_box")  + 0.005;
		  Double_t y_up  = fCnaParHistos->BoxBottomY("top_left_box_Dee") - 0.005;
		  TVirtualPad* main_subpad = gPad;
		  main_subpad->SetPad(x_low, y_low, x_up, y_up);
		}
	      
	      h_geo_bid->GetXaxis()->SetTitle(axis_x_var_name);
	      h_geo_bid->GetYaxis()->SetTitle(axis_y_var_name);
	      
	      h_geo_bid->DrawCopy("COLZ");
	      
	      // trace de la grille: un rectangle = une tour (StexHocoVecoLHFCorcc) 
	      ViewStexGrid(fFapStexNumber, "corcc");
	      gPad->Draw();
	      gPad->Update();

	      //..................... retour aux options standard
	      Bool_t b_true = 1;
	      h_geo_bid->SetStats(b_true);    
	      h_geo_bid->Delete();   h_geo_bid = 0;             fCdeleteRoot++;
	      	      
	      //      delete MainCanvas;              fCdeleteRoot++;
	    }
	}
      delete [] f_in_mat_tit;   f_in_mat_tit = 0;               fCdelete++;
    } // end of if ( fMyRootFile->LookAtRootFile() == kTRUE )
  else
    {
      fStatusFileFound = kFALSE;

      cout << "!TEcnaHistos::StexHocoVecoLHFCorcc(...)> *ERROR* =====> "
	   << " ROOT file not found" << fTTBELL << endl;
    }
} // end of StexHocoVecoLHFCorcc

//==================================================================================
//
//                          GetXCrysInStex, GetYCrysInStex
//
//==================================================================================
Int_t TEcnaHistos::GetXCrysInStex(const Int_t&  StexNumber,  const Int_t& n1StexStin,
				  const Int_t&  i0StinEcha) 
{
//Gives the X crystal coordinate in the geographic view of one Stex
// (X = 0 to MaxStinHocoInStex*NbCrysHocoInStin - 1)

  Int_t ix_geo = 0;

  if( fFlagSubDet == "EB")
    {TString ctype = fEcalNumbering->GetStexHalfStas(StexNumber);
    Int_t n1StexCrys = fEcalNumbering->Get1StexCrysFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha, StexNumber);  
    ix_geo = (n1StexCrys-1)/fEcal->MaxCrysVecoInStex();  // ix_geo for EB+
    if( ctype == "EB-"){ix_geo = fEcal->MaxCrysHocoInStex() - ix_geo - 1;}}

  if( fFlagSubDet == "EE")
    {TString DeeDir = fEcalNumbering->GetDeeDirViewedFromIP(StexNumber);  
    ix_geo = 0;
    if( DeeDir == "right" )
      {ix_geo = fEcalNumbering->GetIXCrysInDee(StexNumber, n1StexStin, i0StinEcha) - 1;}
    if( DeeDir == "left"  )
      {ix_geo = fEcal->MaxCrysIXInDee() - fEcalNumbering->GetIXCrysInDee(StexNumber, n1StexStin, i0StinEcha);}}

  return ix_geo;
}

Int_t TEcnaHistos::GetYCrysInStex(const Int_t& StexNumber, const Int_t& n1StexStin,
				  const Int_t& j0StinEcha) 
{
//Gives the Y crystal coordinate in the geographic view of one Stex
// (Y = 0 to MaxStinVecoInStex*NbCrysVecoInStin - 1)

  Int_t iy_geo = 0;

  if( fFlagSubDet == "EB")
    {TString ctype    = fEcalNumbering->GetStexHalfStas(StexNumber);
    Int_t n1StexCrys = fEcalNumbering->Get1StexCrysFrom1StexStinAnd0StinEcha(n1StexStin, j0StinEcha, StexNumber);
    Int_t ix_geo = (n1StexCrys-1)/fEcal->MaxCrysVecoInStex();     // ix_geo for EB+
    iy_geo = n1StexCrys - 1 - ix_geo*fEcal->MaxCrysVecoInStex();  // iy_geo for EB+
    if( ctype == "EB-"){iy_geo = fEcal->MaxCrysVecoInStex() - iy_geo - 1;}}
  
  if( fFlagSubDet == "EE")
    {iy_geo = fEcalNumbering->GetJYCrysInDee(StexNumber, n1StexStin, j0StinEcha) - 1;}
  
  return iy_geo;
}

//===========================================================================
//
//     StexStinNumbering, ViewStexStinNumberingPad
//
//              independent of the ROOT file => StexNumber as argument
//
//===========================================================================  
void TEcnaHistos::StexStinNumbering(const Int_t& StexNumber)
{
//display the Stin numbering of the Stex

  if( fFlagSubDet == "EB" ){SMTowerNumbering(StexNumber);}
  if( fFlagSubDet == "EE" ){DeeSCNumbering(StexNumber);}
}
// end of StexStinNumbering

//=============================================================================
//
//                   ViewStexStinNumberingPad
//            independent of the ROOT file => StexNumber as argument
//
//=============================================================================
void TEcnaHistos::ViewStexStinNumberingPad(const Int_t& StexNumber)
{
//display the Stin numbering of the Stex in a Pad

  if( fFlagSubDet ==  "EB"){ViewSMTowerNumberingPad(StexNumber);}
  if( fFlagSubDet ==  "EE"){ViewDeeSCNumberingPad(StexNumber);}
}
//---------------->  end of ViewStexStinNumberingPad()

//==========================================================================
//
//                       ViewStexGrid
//              independent of the ROOT file => StexNumber as argument
//
//==========================================================================
void TEcnaHistos::ViewStexGrid(const Int_t& StexNumber, const TString&  c_option)
{
 //Grid of one Stex with axis Hoco and Veco

  if( fFlagSubDet ==  "EB"){ViewSMGrid(StexNumber, c_option);}
  if( fFlagSubDet ==  "EE"){ViewDeeGrid(StexNumber, c_option);}

} // end of ViewStexGrid

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  S P E C I F I C  &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

//=======================================      BARREL       ===============================
void TEcnaHistos::SMTowerNumbering(const Int_t& SMNumber)
{
  //display the tower numbering of the super-module

  if( (SMNumber > 0) && (SMNumber <= fEcal->MaxSMInEB()) )
    {
      Int_t  GeoBidSizeEta = fEcal->MaxTowEtaInSM()*fEcal->MaxCrysEtaInTow();
      Int_t  GeoBidSizePhi = fEcal->MaxTowPhiInSM()*fEcal->MaxCrysPhiInTow();

      Int_t  nb_binx  = GeoBidSizeEta;
      Int_t  nb_biny  = GeoBidSizePhi;
      Axis_t xinf_bid = (Axis_t)0.;
      Axis_t xsup_bid = (Axis_t)GeoBidSizeEta;
      Axis_t yinf_bid = (Axis_t)0.;
      Axis_t ysup_bid = (Axis_t)GeoBidSizePhi;   
  
      TString axis_x_var_name = "  #eta  ";
      TString axis_y_var_name = "  #varphi  ";

      //------------------------------------------------------------------- SMTowerNumbering
  
      //............. matrices reading and histogram filling
      char* f_in_mat_tit = new char[fgMaxCar];                           fCnew++;

      sprintf(f_in_mat_tit, "SM tower numbering");

      // il faut tracer un bidim vide pour pouvoir tracer la grille et les axes

      TH2D* h_empty_bid = new TH2D("grid_bidim_eta_phi", f_in_mat_tit,
				   nb_binx, xinf_bid, xsup_bid,
				   nb_biny, yinf_bid, ysup_bid);     fCnewRoot++; 
      h_empty_bid->Reset();
  
      h_empty_bid->GetXaxis()->SetTitle(axis_x_var_name);
      h_empty_bid->GetYaxis()->SetTitle(axis_y_var_name);

      // ------------------------------------------------ P L O T S   (SMTowerNumbering)
  
      char* f_in = new char[fgMaxCar];                           fCnew++;
  
      //...................... Taille/format canvas
  
      UInt_t canv_h = fCnaParHistos->CanvasFormatH("etaphiSM");
      UInt_t canv_w = fCnaParHistos->CanvasFormatW("etaphiSM");
  
      //............................................... options generales

      fFapStexBarrel = fEcalNumbering->GetSMHalfBarrel(SMNumber);

      //............................................... paves commentaires (SMTowerNumbering)
      SetAllPavesViewStex("Numbering", SMNumber);	  

      //------------------------------------ Canvas name ----------------- (SMTowerNumbering)  

      sprintf(f_in, "tower_numbering_for_SuperModule_SM%d", SMNumber);
  
      SetHistoPresentation((TH1D*)h_empty_bid,"Stex2DEB");

      TCanvas *MainCanvas = new TCanvas(f_in, f_in, canv_w, canv_h);   fCnewRoot++;
      fCurrentCanvas = MainCanvas; fCurrentCanvasName = f_in;

      // cout << "*TEcnaHistosEB::ViewSM(...)> Plot is displayed on canvas ----> " << f_in << endl;
  
      delete [] f_in; f_in = 0;                                 fCdelete++;

      //------------------------ Canvas draw and update ------------ (SMTowerNumbering)  
      fPavComStex->Draw();
  
      Double_t x_margin = fCnaParHistos->BoxLeftX("bottom_left_box") - 0.005;
      Double_t y_margin = fCnaParHistos->BoxTopY("bottom_right_box") + 0.005; 
      MainCanvas->Divide(1, 1, x_margin, y_margin);
      gPad->cd(1);

      h_empty_bid->DrawCopy("COL");   // il faut tracer un bidim vide pour pouvoir tracer la grille et les axes

      ViewSMTowerNumberingPad(SMNumber);
      gPad->Update();
  
      //..................... retour aux options standard
      Bool_t b_true = 1;
      h_empty_bid->SetStats(b_true);    
  
      h_empty_bid->Delete();  h_empty_bid = 0;            fCdeleteRoot++;      
  
      //      delete MainCanvas;              fCdeleteRoot++;
  
      delete [] f_in_mat_tit;  f_in_mat_tit = 0;         fCdelete++;
    }
  else
    {
      cout << "!TEcnaHistos::SMTowerNumbering(...)> SM = " << SMNumber
	   << ". Out of range ( range = [1," << fEcal->MaxSMInEB() << "] )" << fTTBELL << endl;
    }
}
// end of SMTowerNumbering

void TEcnaHistos::ViewSMTowerNumberingPad(const Int_t& SMNumber)
{
  //display the tower numbering of the super-module in a Pad

  gStyle->SetTitleW(0.2);        // taille titre histos
  gStyle->SetTitleH(0.07);

  ViewSMGrid(SMNumber, " ");

  Color_t couleur_rouge      = fCnaParHistos->SetColorsForNumbers("lvrb_top");
  Color_t couleur_bleu_fonce = fCnaParHistos->SetColorsForNumbers("lvrb_bottom");

  //..... Ecriture des numeros de tours dans la grille..... (ViewSMTowerNumberingPad)

  char* f_in = new char[fgMaxCar];                           fCnew++;
  gStyle->SetTextSize(0.075);

  // x_channel, y_channel: coordinates of the text "Txx"
  Int_t y_channel = 12;
  Int_t x_channel = 12;

  Int_t max_tow_phi = fEcal->MaxTowPhiInSM()*fEcal->MaxCrysPhiInTow();

  //------------------ LOOP ON THE SM_TOWER NUMBER   (ViewSMTowerNumberingPad)

  TText *text_SMtow_num = new TText();        fCnewRoot++;

  for (Int_t i_SMtow = 1; i_SMtow <= fEcal->MaxTowInSM(); i_SMtow++)
    {
      if(fEcalNumbering->GetTowerLvrbType(i_SMtow) == "top")
	{text_SMtow_num->SetTextColor(couleur_rouge);}
      if(fEcalNumbering->GetTowerLvrbType(i_SMtow) == "bottom")
	{text_SMtow_num->SetTextColor(couleur_bleu_fonce);}

      //................................ x from eta
      Double_t x_from_eta = fEcalNumbering->GetEta(SMNumber, i_SMtow, x_channel) - (Double_t)1;
      if(fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB-")
      {x_from_eta = fEcal->MaxTowEtaInSM()*fEcal->MaxCrysEtaInTow() + x_from_eta + (Double_t)1;}

      //................................ y from phi
      Double_t y_from_phi = max_tow_phi - 1
	- (fEcalNumbering->GetPhi(SMNumber, i_SMtow, y_channel) - fEcalNumbering->GetPhiMin(SMNumber));
      if( fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB-")
	{y_from_phi = - y_from_phi + fEcal->MaxTowPhiInSM()*fEcal->MaxCrysPhiInTow() - (Double_t)1;}

      sprintf(f_in, "%d", i_SMtow);
      text_SMtow_num->DrawText(x_from_eta, y_from_phi, f_in);  // <=== prend du temps si on mets "T%d" dans le sprintf
    }

  text_SMtow_num->Delete();    text_SMtow_num = 0;         fCdeleteRoot++;

  //.................................................... legende (ViewSMTowerNumberingPad)
  Double_t offset_tow_tex_eta = (Double_t)8.;
  Double_t offset_tow_tex_phi = (Double_t)15.;

  Color_t couleur_noir = fCnaParHistos->ColorDefinition("noir");
  Double_t x_legend    = (Double_t)0.;
  Double_t y_legend    = (Double_t)0.;

  Int_t ref_tower = fEcal->MaxTowInSM();

  //.................................................  LVRB TOP (ViewSMTowerNumberingPad)
  gStyle->SetTextSize(0.075);  
  gStyle->SetTextColor(couleur_rouge);
  x_legend = fEcalNumbering->GetEta(SMNumber, ref_tower, x_channel);
  y_legend = fEcalNumbering->GetPhi(SMNumber, ref_tower, y_channel) - fEcalNumbering->GetPhiMin(SMNumber);

  if( fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB+" )
    {
      x_legend = x_legend + offset_tow_tex_eta;
      y_legend = y_legend + offset_tow_tex_phi;
    }
  if( fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB-" )
    {
      x_legend = -x_legend + offset_tow_tex_eta;
      y_legend =  y_legend + offset_tow_tex_phi;
    }

  sprintf( f_in, "xx");
  TText *text_legend_rouge = new TText(x_legend, y_legend, f_in);        fCnewRoot++;
  text_legend_rouge->Draw();
  gStyle->SetTextSize(0.05);
  x_legend = x_legend - (Double_t)3.5;
  y_legend = y_legend - (Double_t)2.;
  sprintf(f_in, "       LVRB     ");
  TText *text_legend_rouge_expl = new TText(x_legend, y_legend, f_in);        fCnewRoot++;
  text_legend_rouge_expl->Draw();
  y_legend = y_legend - (Double_t)1.75;
  if( fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB+" ){sprintf(f_in, "        <---  ");}
  if( fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB-" ){sprintf(f_in, "        --->  ");}
  TText *text_legend_rouge_expm = new TText(x_legend, y_legend, f_in);        fCnewRoot++;
  text_legend_rouge_expm->Draw();
  //text_legend_rouge_expl->Delete();   text_legend_rouge_expl = 0;          fCdeleteRoot++;
  
  //.................................................  LVRB BOTTOM (ViewSMTowerNumberingPad)
  gStyle->SetTextSize(0.075);  
  gStyle->SetTextColor(couleur_bleu_fonce);
  x_legend = fEcalNumbering->GetEta(SMNumber, ref_tower, x_channel);
  y_legend = fEcalNumbering->GetPhi(SMNumber, ref_tower, y_channel) - fEcalNumbering->GetPhiMin(SMNumber); 

  if( fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB+" )
    {
      x_legend = x_legend + offset_tow_tex_eta;
      y_legend = y_legend + offset_tow_tex_phi/3;
    }
  if( fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB-" )
    {
      x_legend = -x_legend + offset_tow_tex_eta;
      y_legend =  y_legend + offset_tow_tex_phi/3;
    }

  sprintf(f_in, "xx");
  TText *text_legend_bleu = new TText(x_legend, y_legend, f_in);        fCnewRoot++;
  text_legend_bleu->Draw();
  //text_legend_bleu->Delete();   text_legend_bleu = 0;          fCdeleteRoot++;
  gStyle->SetTextSize(0.05);
  x_legend = x_legend - (Double_t)3.5;
  y_legend = y_legend - (Double_t)2.;
  sprintf( f_in, "       LVRB     ");
  TText *text_legend_bleu_expl = new TText(x_legend, y_legend, f_in);        fCnewRoot++;
  text_legend_bleu_expl->Draw();
  y_legend = y_legend - (Double_t)1.75;
  if( fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB+" ){sprintf( f_in, "        --->  ");}
  if( fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB-" ){sprintf( f_in, "        <---  ");}
  TText *text_legend_bleu_expm = new TText(x_legend, y_legend, f_in);        fCnewRoot++;
  text_legend_bleu_expm->Draw();
  //text_legend_bleu_expl->Delete();   text_legend_bleu_expl = 0;          fCdeleteRoot++;
  
  delete [] f_in;   f_in = 0;                                    fCdelete++;

  gStyle->SetTextColor(couleur_noir);
}
//---------------->  end of ViewSMTowerNumberingPad()

void TEcnaHistos::ViewSMGrid(const Int_t& SMNumber, const TString& c_option)
{
 //Grid of one supermodule with axis eta and phi

  Int_t  GeoBidSizeEta = fEcal->MaxTowEtaInSM()*fEcal->MaxCrysEtaInTow();
  Int_t  GeoBidSizePhi = fEcal->MaxTowPhiInSM()*fEcal->MaxCrysPhiInTow();

  if ( c_option == "corcc")
    {
      GeoBidSizeEta = fEcal->MaxTowEtaInSM()*fEcal->MaxCrysInTow();
      GeoBidSizePhi = fEcal->MaxTowPhiInSM()*fEcal->MaxCrysInTow();
    }

  Int_t  nb_binx  = GeoBidSizeEta;
  Int_t  nb_biny  = GeoBidSizePhi;
  Axis_t xinf_bid = (Axis_t)0.;
  Axis_t xsup_bid = (Axis_t)GeoBidSizeEta;
  Axis_t yinf_bid = (Axis_t)0.;
  Axis_t ysup_bid = (Axis_t)GeoBidSizePhi;   
  
  //---------------- trace de la grille: un rectangle = une tour

  Int_t size_eta = fEcal->MaxCrysEtaInTow();
  Int_t size_phi = fEcal->MaxCrysPhiInTow();
  if ( c_option == "corcc")
    {
      size_eta = fEcal->MaxCrysInTow();
      size_phi = fEcal->MaxCrysInTow();
    }
  Int_t max_x = nb_binx/size_eta;
  Int_t max_y = nb_biny/size_phi;

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
  
  Color_t coul_surligne = fCnaParHistos->ColorDefinition("noir");
  Color_t coul_textmodu = fCnaParHistos->ColorDefinition("vert36");

  //............................ Mj text
  gStyle->SetTextColor(coul_textmodu);
  gStyle->SetTextSize(0.075);

  char* f_in = new char[fgMaxCar];                           fCnew++;

  for( Int_t i = 0 ; i < max_x ; i++)
    {  
      xline = xline + (Double_t)size_eta;
      TLine *lin;
      lin = new TLine(xline, yline_bot, xline, yline_top); fCnewRoot++;
      
      //............. Surlignage separateur des modules
      if( (fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB-") && (i == 4 || i == 8 || i == 12) )
	{lin->SetLineWidth(2); lin->SetLineColor(coul_surligne);}      
      if( (fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB+") && (i == 5 || i == 9 || i == 13) )
	{lin->SetLineWidth(2); lin->SetLineColor(coul_surligne);}
       
      lin->Draw();
      // delete lin;             fCdeleteRoot++;

      //............. Numeros des modules
      if( (fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB-") && (i == 2 || i == 6 || i == 10 || i == 14)  )
	{
	  if( i ==  2 ){sprintf( f_in, "M4");}
	  if( i ==  6 ){sprintf( f_in, "M3");}
	  if( i == 10 ){sprintf( f_in, "M2");}
	  if( i == 14 ){sprintf( f_in, "M1");}

	  TText *text_num_module = new TText(xline + 1, yline_top + 1, f_in);        fCnewRoot++;
	  text_num_module->Draw();
	  //text_num_module->Delete(); text_num_module = 0;      fCdeleteRoot++;      
	}      
      if( (fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB+") && (i == 3 || i == 7 || i == 11 || i == 15)  )
	{
	  if( i ==  3 ){sprintf( f_in, "M1");}
	  if( i ==  7 ){sprintf( f_in, "M2");}
	  if( i == 11 ){sprintf( f_in, "M3");}
	  if( i == 15 ){sprintf( f_in, "M4");}
	 
	  TText *text_num_module = new TText(xline, yline_top + 1, f_in);        fCnewRoot++;
	  text_num_module->Draw();
	  //text_num_module->Delete();  text_num_module = 0;    fCdeleteRoot++;     
	}
    }
  delete [] f_in;   f_in = 0;    fCdelete++;

  //------------------ trace axes en eta et phi --------------- ViewSMGrid

  Int_t MatSize      = fEcal->MaxCrysEtaInTow();
  if ( c_option == "corcc"){MatSize = fEcal->MaxCrysInTow();}

  Int_t size_eta_sm  = fEcal->MaxTowEtaInSM();
  Int_t size_phi_sm  = fEcal->MaxTowPhiInSM();

  //...................................................... Axe i(eta) (bottom x) ViewSMGrid
  Double_t eta_min = fEcalNumbering->GetIEtaMin(SMNumber);
  Double_t eta_max = fEcalNumbering->GetIEtaMax(SMNumber);

  TString  x_var_name  = GetHocoVecoAxisTitle("ietaSM");;
  TString  x_direction = fEcalNumbering->GetXDirectionEB(SMNumber);

  Float_t tit_siz_x = fCnaParHistos->AxisTitleSize();
  Float_t lab_siz_x = fCnaParHistos->AxisLabelSize("SMx");
  Float_t tic_siz_x = fCnaParHistos->AxisTickSize("SMx");
  Float_t tit_off_x = fCnaParHistos->AxisTitleOffset("SMx");
  Float_t lab_off_x = fCnaParHistos->AxisLabelOffset("SMx");

  new TF1("f1", x_direction.Data(), eta_min, eta_max);          fCnewRoot++;
    TGaxis* sup_axis_x = 0;

  if( x_direction == "-x" ) // NEVER  IN THIS CASE: xmin->xmax <=> right->left ("-x") direction
    {sup_axis_x = new TGaxis( (Float_t)0., (Float_t)0., (Float_t)(size_eta_sm*MatSize), (Float_t)0.,
			      "f1", size_eta_sm, "SC" , 0.);   fCnewRoot++;}

  if( x_direction == "x" )  // ALWAYS IN THIS CASE: xmin->xmax <=> left->right ("x") direction    
    {sup_axis_x = new TGaxis( (Float_t)0., (Float_t)0., (Float_t)(size_eta_sm*MatSize), (Float_t)0.,
			      "f1", size_eta_sm, "SC" , 0.);   fCnewRoot++;}
  
  sup_axis_x->SetTitle(x_var_name);
  sup_axis_x->SetTitleSize(tit_siz_x);
  sup_axis_x->SetTitleOffset(tit_off_x);
  sup_axis_x->SetLabelSize(lab_siz_x);
  sup_axis_x->SetLabelOffset(lab_off_x);
  sup_axis_x->SetTickSize(tic_siz_x);
  sup_axis_x->Draw("SAME");

  //...................................................... Axe phi (y) ViewSMGrid
  Double_t phi_min = fEcalNumbering->GetPhiMin(SMNumber);
  Double_t phi_max = fEcalNumbering->GetPhiMax(SMNumber);

  TString  y_var_name  = GetHocoVecoAxisTitle("phi");
  TString  y_direction = fEcalNumbering->GetYDirectionEB(SMNumber);

  Float_t tit_siz_y = fCnaParHistos->AxisTitleSize();
  Float_t lab_siz_y = fCnaParHistos->AxisLabelSize("SMy");
  Float_t tic_siz_y = fCnaParHistos->AxisTickSize("SMy");
  Float_t tit_off_y = fCnaParHistos->AxisTitleOffset("SMy");
  Float_t lab_off_y = fCnaParHistos->AxisLabelOffset("SMy");

  new TF1("f2", y_direction.Data(), phi_min, phi_max);           fCnewRoot++;
  TGaxis* sup_axis_y = 0;
  
  if ( y_direction == "-x" ) // ALWAYS IN THIS CASE: ymin->ymax <=> top->bottom ("-x") direction
    {sup_axis_y = new TGaxis(-(Float_t)1.5*(Float_t)size_eta, (Float_t)0.,
			     -(Float_t)1.5*(Float_t)size_eta, (Float_t)(size_phi_sm*MatSize),
			     "f2", (Int_t)size_phi_sm, "SC", 0.);   fCnewRoot++;}
  
  if ( y_direction == "x" )  // NEVER  IN THIS CASE: ymin->ymax <=> bottom->top ("x") direction
    {sup_axis_y = new TGaxis(-(Float_t)1.5*(Float_t)size_eta, (Float_t)0.,
			     -(Float_t)1.5*(Float_t)size_eta, (Float_t)(size_phi_sm*MatSize),
			     "f2", (Int_t)size_phi_sm, "SC", 0.);   fCnewRoot++;}
  
  sup_axis_y->SetTitle(y_var_name);
  sup_axis_y->SetTitleSize(tit_siz_y);
  sup_axis_y->SetTitleOffset(tit_off_y);
  sup_axis_y->SetLabelSize(lab_siz_y);
  sup_axis_y->SetLabelOffset(lab_off_y);
  sup_axis_y->SetTickSize(tic_siz_y);
  sup_axis_y->Draw("SAME");

  //...................................................... Axe jphi (jy) ViewSMGrid
  Double_t jphi_min     = fEcalNumbering->GetJPhiMin(SMNumber);
  Double_t jphi_max     = fEcalNumbering->GetJPhiMax(SMNumber);

  TString  jy_var_name  = " ";
  TString  jy_direction = fEcalNumbering->GetJYDirectionEB(SMNumber);

  new TF1("f3", jy_direction.Data(), jphi_min, jphi_max);           fCnewRoot++;
  TGaxis* sup_axis_jy = 0;
  
  //............; essai
  sup_axis_jy = new TGaxis((Float_t)0., (Float_t)0.,
			   (Float_t)0., (Float_t)(size_phi_sm*MatSize),
			   "f3", (Int_t)size_phi_sm, "SC", 0.);   fCnewRoot++;
  
  if ( jy_direction == "-x" ) // IN THIS CASE FOR EB+: ymin->ymax <=> top->bottom ("-x") direction
    {jy_var_name  = GetEtaPhiAxisTitle("jphiSMB+");}
  
  if ( jy_direction == "x" )  // IN THIS CASE FOR EB-: ymin->ymax <=> bottom->top ("x") direction
    {jy_var_name  = GetEtaPhiAxisTitle("jphiSMB-");}
  
  lab_off_y = fCnaParHistos->AxisLabelOffset("SMyInEB");

  sup_axis_jy->SetTitle(jy_var_name);
  sup_axis_jy->SetTitleSize(tit_siz_y);
  sup_axis_jy->SetTitleOffset(tit_off_y);
  sup_axis_jy->SetLabelSize(lab_siz_y);
  sup_axis_jy->SetLabelOffset(lab_off_y);
  sup_axis_jy->SetTickSize(tic_siz_y);
  sup_axis_jy->Draw("SAME");

  //--------------------------- ViewSMGrid
	  
  gStyle->SetTextColor(fCnaParHistos->ColorDefinition("noir"));

} // end of ViewSMGrid

//=======================================      ENDCAP       ===============================
void TEcnaHistos::DeeSCNumbering(const Int_t& DeeNumber)
{
  //display the SC numbering of the Dee

  if( (DeeNumber > 0) && (DeeNumber <= fEcal->MaxDeeInEE()) )
    {
      Int_t  GeoBidSizeIX = fEcal->MaxSCIXInDee()*fEcal->MaxCrysIXInSC();
      Int_t  GeoBidSizeIY = fEcal->MaxSCIYInDee()*fEcal->MaxCrysIYInSC();

      Int_t  nb_binx  = GeoBidSizeIX;
      Int_t  nb_biny  = GeoBidSizeIY;
      Axis_t xinf_bid = (Axis_t)0.;
      Axis_t xsup_bid = (Axis_t)GeoBidSizeIX;
      Axis_t yinf_bid = (Axis_t)0.;
      Axis_t ysup_bid = (Axis_t)GeoBidSizeIY;   
  
      TString axis_x_var_name = "  IX  ";
      TString axis_y_var_name = "  IY  ";

      //------------------------------------------------------------------- DeeSCNumbering
      //........................................... empty histogram filling
      char* f_in_mat_tit = new char[fgMaxCar];                           fCnew++;

      sprintf(f_in_mat_tit, " Dee SC numbering ");

      // il faut tracer un bidim vide pour pouvoir tracer la grille et les axes

      TH2D* h_empty_bid = new TH2D("grid_bidim_IX_IY", f_in_mat_tit,
				   nb_binx, xinf_bid,  xsup_bid,
				   nb_biny, yinf_bid,  ysup_bid);     fCnewRoot++;
      h_empty_bid->Reset();
  
      h_empty_bid->GetXaxis()->SetTitle(axis_x_var_name);
      h_empty_bid->GetYaxis()->SetTitle(axis_y_var_name);

      // ------------------------------------------------ P L O T S   (DeeSCNumbering)
  
      char* f_in = new char[fgMaxCar];                           fCnew++;
  
      //...................... Taille/format canvas
  
      UInt_t canv_h = fCnaParHistos->CanvasFormatH("IXIYDee");
      UInt_t canv_w = fCnaParHistos->CanvasFormatW("IXIYDee");
  
      //............................................... options generales
      fFapStexType = fEcalNumbering->GetEEDeeType(DeeNumber);

      //............................................... paves commentaires (DeeSCNumbering)
      SetAllPavesViewStex("Numbering", DeeNumber);

      //------------------------------------ Canvas name ----------------- (DeeSCNumbering)  

      sprintf(f_in, "SC_numbering_for_Dee_Dee%d", DeeNumber);
      SetHistoPresentation((TH1D*)h_empty_bid,"Stex2DEENb");
      TCanvas *MainCanvas = new TCanvas(f_in, f_in, canv_w, canv_h);   fCnewRoot++;
      fCurrentCanvas = MainCanvas; fCurrentCanvasName = f_in;

      // cout << "*TEcnaHistosEE::ViewDee(...)> Plot is displayed on canvas ----> " << f_in << endl;
  
      delete [] f_in; f_in = 0;                                 fCdelete++;

      //------------------------ Canvas draw and update ------------ (DeeSCNumbering)  
      fPavComStex->Draw();
      fPavComCxyz->Draw();
  
      Double_t x_low = fCnaParHistos->BoxLeftX("bottom_left_box")    - 0.005;
      Double_t x_up  = fCnaParHistos->BoxRightX("bottom_right_box")  + 0.005;
      Double_t y_low = fCnaParHistos->BoxTopY("bottom_left_box")     + 0.005;
      Double_t y_up  = fCnaParHistos->BoxBottomY("top_left_box_Dee") - 0.005;
  
      Double_t x_margin = x_low;
      Double_t y_margin = y_low;
  
      MainCanvas->Divide( 1,  1, x_margin, y_margin);
      //           Divide(nx, ny, x_margin, y_margin,    color);
  
      gPad->cd(1);
      TVirtualPad* main_subpad = gPad;
      main_subpad->SetPad(x_low, y_low, x_up, y_up);
  
      h_empty_bid->DrawCopy("COL");   // il faut tracer un bidim vide pour pouvoir tracer la grille et les axes
      ViewDeeSCNumberingPad(DeeNumber);
      gPad->Update();   // prend beaucoup de temps...
  
      //..................... retour aux options standard
      Bool_t b_true = 1;
      h_empty_bid->SetStats(b_true);    
  
      h_empty_bid->Delete(); h_empty_bid = 0;             fCdeleteRoot++;      

      //      delete MainCanvas;              fCdeleteRoot++;
  
      delete [] f_in_mat_tit;  f_in_mat_tit = 0;         fCdelete++;
    }
  else
    {
      cout << "!TEcnaHistos::DeeSCNumbering(...)> Dee = " << DeeNumber
	   << ". Out of range ( range = [1," << fEcal->MaxDeeInEE() << "] )" << fTTBELL << endl;
    }
}
// end of DeeSCNumbering

void TEcnaHistos::ViewDeeSCNumberingPad(const Int_t&   DeeNumber)
{
//display the SC numbering of the Dee in a Pad

  gStyle->SetTitleW(0.4);        // taille titre histos
  gStyle->SetTitleH(0.08);

  ViewDeeGrid(DeeNumber, " ");

  //..... SC numbers writing in the grid .... (ViewDeeSCNumberingPad)

  char* f_in = new char[fgMaxCar];                           fCnew++;
  gStyle->SetTextSize(0.0325);

  //------------------ LOOP ON THE Dee_SC NUMBER   (ViewDeeSCNumberingPad)
  Int_t x_channel =  0;    // => defined here after according to DeeDir and SCQuadType
  TText *text_DSSC_num      = new TText();        fCnewRoot++;
  TText *text_DeeSCCons_num = new TText();        fCnewRoot++;

  for (Int_t n1DeeSCEcna = 1; n1DeeSCEcna <= fEcal->MaxSCEcnaInDee(); n1DeeSCEcna++)
    {
      TString DeeDir     = fEcalNumbering->GetDeeDirViewedFromIP(DeeNumber);
      TString SCQuadType = fEcalNumbering->GetSCQuadFrom1DeeSCEcna(n1DeeSCEcna);
      if( SCQuadType == "top"    &&  DeeDir == "right"){x_channel = 13;}
      if( SCQuadType == "top"    &&  DeeDir == "left" ){x_channel =  7;}
      if( SCQuadType == "bottom" &&  DeeDir == "left" ){x_channel = 11;}  
      if( SCQuadType == "bottom" &&  DeeDir == "right"){x_channel = 17;}
      Int_t i_SCEcha = (Int_t)x_channel;

      Double_t x_from_IX = (Double_t)GetXCrysInStex(DeeNumber, n1DeeSCEcna, i_SCEcha);
      Double_t y_from_IY = (Double_t)GetYCrysInStex(DeeNumber, n1DeeSCEcna, i_SCEcha);
      Double_t y_from_IYp = y_from_IY + (Double_t)1.;
      Double_t y_from_IYm = y_from_IY - (Double_t)1.;

      TString DeeEndcap  = fEcalNumbering->GetEEDeeEndcap(DeeNumber);
      Color_t couleur_SC = GetSCColor(DeeEndcap, DeeDir, SCQuadType);
      text_DSSC_num->SetTextColor(couleur_SC);
      text_DeeSCCons_num->SetTextColor((Color_t)1);

      Int_t i_DSSC      = fEcalNumbering->GetDSSCFrom1DeeSCEcna(DeeNumber, n1DeeSCEcna);
      Int_t i_DeeSCCons = fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(DeeNumber, n1DeeSCEcna);
      if( i_DSSC > 0 )
	{
	  if(
	     //.................................................... (D2,S9) , (D4,S1)
	      !(i_DeeSCCons ==  33 && n1DeeSCEcna ==  60) &&
	      !(i_DeeSCCons ==  33 && n1DeeSCEcna == 119) &&
	      //................................................... (D2,S8) , (D4,S2)
	      !(i_DeeSCCons ==  29 && n1DeeSCEcna ==  32) &&   // !(29c and 58c)
	      !(i_DeeSCCons ==  29 && n1DeeSCEcna == 138) &&
	      !(i_DeeSCCons ==  29 && n1DeeSCEcna == 157) &&
	      !(i_DeeSCCons ==  58 && n1DeeSCEcna == 176) &&
	      !(i_DeeSCCons ==  58 && n1DeeSCEcna == 193) &&
	      //.................................................... (D2,S7) , (D4,S3)
	      !(i_DeeSCCons == 149 && n1DeeSCEcna == 188) &&
	      //.................................................... (D2,S6) , (D4,S4)
	      !(i_DeeSCCons == 112 && n1DeeSCEcna ==  29) &&
	      !(i_DeeSCCons == 112 && n1DeeSCEcna == 144) &&
	      !(i_DeeSCCons == 112 && n1DeeSCEcna == 165) &&
	      !(i_DeeSCCons == 119 && n1DeeSCEcna == 102) &&
	      !(i_DeeSCCons == 119 && n1DeeSCEcna == 123) &&
	      //.................................................... (D2,S5) , (D4,S5)
	      !(i_DeeSCCons == 132 && n1DeeSCEcna ==  41) &&
	      //----------------------------------------------------------------------
	      //.................................................... (D1,S1) , (D3,S9)
	      !(i_DeeSCCons == 182 && n1DeeSCEcna ==  60) &&
	      !(i_DeeSCCons == 182 && n1DeeSCEcna == 119) &&
	      //.................................................... (D1,S2) , (D3,S8)
	      !(i_DeeSCCons == 178 && n1DeeSCEcna ==  32) &&   // !(178c and 207c)
	      !(i_DeeSCCons == 178 && n1DeeSCEcna == 138) &&
	      !(i_DeeSCCons == 178 && n1DeeSCEcna == 157) &&
	      !(i_DeeSCCons == 207 && n1DeeSCEcna == 176) &&
	      !(i_DeeSCCons == 207 && n1DeeSCEcna == 193) &&
	      //.................................................... (D1,S3) , (D3,S7)
	      !(i_DeeSCCons == 298 && n1DeeSCEcna == 188) &&
	      //.................................................... (D1,S4) , (D3,S6)
	      !(i_DeeSCCons == 261 && n1DeeSCEcna ==  29) &&   // !(261a and 268a)
	      !(i_DeeSCCons == 261 && n1DeeSCEcna == 144) &&
	      !(i_DeeSCCons == 261 && n1DeeSCEcna == 165) &&
	      !(i_DeeSCCons == 268 && n1DeeSCEcna == 102) &&
	      !(i_DeeSCCons == 268 && n1DeeSCEcna == 123) &&
	      //.................................................... (D1,S5) , (D3,S5)
	      !(i_DeeSCCons == 281 && n1DeeSCEcna ==  41) 
	      )
	    {
	      sprintf(f_in, "%d", i_DSSC);
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);      // <=== DrawText: prend du temps
	      sprintf(f_in, "%d", i_DeeSCCons);
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in); // <=== DrawText: prend du temps
	    }

	  //.................................................... (D2,S9) , (D4,S1)

	  if( i_DeeSCCons ==  33 && n1DeeSCEcna ==  60 )
	    {
	      sprintf(f_in, "30a");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "33a");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  if( i_DeeSCCons ==  33 && n1DeeSCEcna == 119 )
	    {
	      sprintf(f_in, "30b");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "33b");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  //.................................................... (D2,S8) , (D4,S2)
	  if( i_DeeSCCons ==  29 && n1DeeSCEcna == 32 )
	    {
	      sprintf(f_in, " 3c-25c");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "29c-58c");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  if( i_DeeSCCons ==  29 && n1DeeSCEcna == 138 )
	    {
	      sprintf(f_in, "3a");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "29a");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  if( i_DeeSCCons ==  29 && n1DeeSCEcna == 157 )
	    {
	      sprintf(f_in, "3b");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "29b");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }

	  if( i_DeeSCCons ==  58 && n1DeeSCEcna == 176 )
	    {
	      sprintf(f_in, "25a");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "58a");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  if( i_DeeSCCons ==  58 && n1DeeSCEcna == 193 )
	    {
	      sprintf(f_in, "25b");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "58b");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  //.................................................... (D2,S7) , (D4,S3)
	  if( i_DeeSCCons == 149 && n1DeeSCEcna == 188 )
	    {
	      sprintf(f_in, "34a");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "149a");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  //.................................................... (D2,S6) , (D4,S4)
	  if( i_DeeSCCons == 112 && n1DeeSCEcna == 29 )
	    {
	      sprintf(f_in, " 14a-21a");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "112a-119a");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  if( i_DeeSCCons == 112 && n1DeeSCEcna == 144 )
	    {
	      sprintf(f_in, "14c");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "112c");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  if( i_DeeSCCons == 112 && n1DeeSCEcna == 165 )
	    {
	      sprintf(f_in, "14b");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "112b");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }

	  if( i_DeeSCCons == 119 && n1DeeSCEcna == 102 )
	    {
	      sprintf(f_in, "21c");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "119c");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  if( i_DeeSCCons == 119 && n1DeeSCEcna == 123 )
	    {
	      sprintf(f_in, "21b");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "119b");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  //.................................................... (D2,S5) , (D4,S5)
	  if( i_DeeSCCons == 132 && n1DeeSCEcna ==  41 )
	    {
	      sprintf(f_in, "3a");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "132a");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }

	  //.................................................... (D1,S1) , (D3,S9)
	  if( i_DeeSCCons == 182 && n1DeeSCEcna ==  60 )
	    {
	      sprintf(f_in, "30a");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "182a");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  if( i_DeeSCCons == 182 && n1DeeSCEcna == 119 )
	    {
	      sprintf(f_in, "30b");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "182b");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  //.................................................... (D1,S2) , (D3,S8)
	  if( i_DeeSCCons == 178 && n1DeeSCEcna == 32 )
	    {
	      sprintf(f_in, "  3c-25c");
	      text_DSSC_num->DrawText(x_from_IX-6, y_from_IYp, f_in);
	      sprintf(f_in, "178c-207c");
	      text_DeeSCCons_num->DrawText(x_from_IX-6, y_from_IYm, f_in);
	    }
	  if( i_DeeSCCons == 178 && n1DeeSCEcna == 138 )
	    {
	      sprintf(f_in, "3a");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "178a");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  if( i_DeeSCCons == 178 && n1DeeSCEcna == 157 )
	    {
	      sprintf(f_in, "3b");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "178b");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }

	  if( i_DeeSCCons == 207 && n1DeeSCEcna == 176 )
	    {
	      sprintf(f_in, "25a");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "207a");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  if( i_DeeSCCons == 207 && n1DeeSCEcna == 193 )
	    {
	      sprintf(f_in, "25b");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "207b");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  //.................................................... (D1,S3) , (D3,S7)
	  if( i_DeeSCCons == 298 && n1DeeSCEcna == 188 )
	    {
	      sprintf(f_in, "34a");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "298a");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  //.................................................... (D1,S4) , (D3,S6)
	  if( i_DeeSCCons == 261 && n1DeeSCEcna == 29 )
	    {
	      sprintf(f_in, " 14a-21a");
	      text_DSSC_num->DrawText(x_from_IX-6, y_from_IYp, f_in);
	      sprintf(f_in, "261a-268a");
	      text_DeeSCCons_num->DrawText(x_from_IX-6, y_from_IYm, f_in);
	    }
	  if( i_DeeSCCons == 261 && n1DeeSCEcna == 144 )
	    {
	      sprintf(f_in, "14c");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "261c");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  if( i_DeeSCCons == 261 && n1DeeSCEcna == 165 )
	    {
	      sprintf(f_in, "14b");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "261b");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }

	  if( i_DeeSCCons == 268 && n1DeeSCEcna == 102 )
	    {
	      sprintf(f_in, "21c");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "268c");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  if( i_DeeSCCons == 268 && n1DeeSCEcna == 123 )
	    {
	      sprintf(f_in, "21b");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "268b");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	  //.................................................... (D1,S5) , (D3,S5)
	  if( i_DeeSCCons == 281 && n1DeeSCEcna ==  41 )
	    {
	      sprintf(f_in, "20a");
	      text_DSSC_num->DrawText(x_from_IX, y_from_IYp, f_in);
	      sprintf(f_in, "281a");
	      text_DeeSCCons_num->DrawText(x_from_IX, y_from_IYm, f_in);
	    }
	}
    }

  // delete text_DSSC_num;             fCdeleteRoot++;
 
  //......................... mention "color, black"  
  Color_t coul_textcolors = fCnaParHistos->ColorDefinition("noir");
  sprintf( f_in, "color: nb in Data Sector, black: nb for construction");
  Int_t x_colors = 3;
  Int_t y_colors = -14;

  TText *text_colors = new TText(x_colors, y_colors, f_in);        fCnewRoot++;
  text_colors->SetTextSize(0.03);
  text_colors->SetTextColor(coul_textcolors);
  text_colors->Draw();

  delete [] f_in;  f_in = 0;                                     fCdelete++;

  Color_t couleur_noir = fCnaParHistos->ColorDefinition("noir");
  gStyle->SetTextColor(couleur_noir);
}
//---------------->  end of ViewDeeSCNumberingPad()

void TEcnaHistos::ViewDeeGrid(const Int_t& DeeNumber, const TString& c_option)
{
 //Grid of one Dee with axis IX and IY

  Int_t  GeoBidSizeIX = fEcal->MaxSCIXInDee()*fEcal->MaxCrysIXInSC();
  Int_t  GeoBidSizeIY = fEcal->MaxSCIYInDee()*fEcal->MaxCrysIYInSC();

  if ( c_option == "corcc")
    {
      GeoBidSizeIX = fEcal->MaxSCIXInDee()*fEcal->MaxCrysInSC();
      GeoBidSizeIY = fEcal->MaxSCIYInDee()*fEcal->MaxCrysInSC();
    }

  Int_t  nb_binx  = GeoBidSizeIX;
  Int_t  nb_biny  = GeoBidSizeIY;
  Axis_t xinf_bid = (Axis_t)0.;
  Axis_t xsup_bid = (Axis_t)GeoBidSizeIX;
 
  Axis_t   yinf_bid = (Axis_t)0.;
  Axis_t   ysup_bid = (Axis_t)GeoBidSizeIY;
  Double_t ymid_bid = (Double_t)(ysup_bid-yinf_bid)/2.;
 
  //---------------- trace de la grille: un rectangle = un super-cristal

  Int_t size_IX = fEcal->MaxCrysIXInSC();
  Int_t size_IY = fEcal->MaxCrysIYInSC();

  if ( c_option == "corcc"){size_IX = fEcal->MaxCrysInSC(); size_IY = fEcal->MaxCrysInSC();}

  Int_t max_x  = nb_binx/size_IX;
  Int_t max_y  = nb_biny/size_IY;
  Int_t max_yd = max_y/2;

  //= SURLIGNAGES (unite de coordonnees: le cristal ou 5 fois le cristal si option corcc)
  //........................... multplicative coefficient for corcc option
  Int_t coefcc_x = 1;
  Int_t coefcc_y = 1;
  if ( c_option == "corcc"){coefcc_x = fEcal->MaxCrysIXInSC(); coefcc_y = fEcal->MaxCrysIYInSC();}

  //............................. lignes horizontales
  Double_t yline = (Double_t)yinf_bid - (Double_t)size_IY;

  Double_t xline_beg = (Double_t)xinf_bid;
  Double_t xline_end = (Double_t)xsup_bid;

  //           k  =   0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10
  Int_t x_min[11] = {11,11, 7, 0, 0, 0, 0, 0, 0, 0, 0};
  Int_t x_max[11] = {50,50,47,45,45,42,37,35,30,15,50}; 
  for(Int_t i=0;i<11;i++){x_min[i] = coefcc_x*x_min[i]; x_max[i] = coefcc_x*x_max[i];}
  
  for( Int_t j = 0 ; j < max_y ; j++)
    {
      if( j < max_yd )  // j = 0,1,2,3,4,5,6,7,8,9
	{
	  if( DeeNumber == 1 || DeeNumber == 3 ) 
	    {
	      xline_beg = xinf_bid + (Double_t)x_min[10-j];
	      xline_end = xinf_bid + (Double_t)x_max[10-j];
	    }
	  if( DeeNumber == 2 || DeeNumber == 4 ) 
	    {
	      xline_beg = xsup_bid - (Double_t)x_max[10-j];
	      xline_end = xsup_bid - (Double_t)x_min[10-j];
	    }
	}

      if( j == max_yd ) // j = 10
	{
	  if( DeeNumber == 1 || DeeNumber == 3 ) 
	    {
	      xline_beg = xinf_bid + (Double_t)x_min[0];
	      xline_end = xinf_bid + (Double_t)x_max[0];
	    }
	  if( DeeNumber == 2 || DeeNumber == 4 ) 
	    {
	      xline_beg = xsup_bid - (Double_t)x_max[0];
	      xline_end = xsup_bid - (Double_t)x_min[0];
	    }
	}

      if( j > max_yd ) // j = 11,12,13,14,15,16,17,18,19,20
	{
	  if( DeeNumber == 1 || DeeNumber == 3 ) 
	    {
	      xline_beg = xinf_bid + (Double_t)x_min[j-10];
	      xline_end = xinf_bid + (Double_t)x_max[j-10];
	    }
	  if( DeeNumber == 2 || DeeNumber == 4 ) 
	    {
	      xline_beg = xsup_bid - (Double_t)x_max[j-10];
	      xline_end = xsup_bid - (Double_t)x_min[j-10];
	    }
	}

      yline = yline + (Double_t)size_IY;
      TLine *lin;
      lin = new TLine(xline_beg, yline, xline_end, yline); fCnewRoot++;
      lin->Draw();
      //lin->Delete();   // => si on delete, pas de trace de la ligne
      // delete lin;             fCdeleteRoot++;
    }

  //.......................... lignes verticales
  Double_t xline = (Double_t)xinf_bid - (Double_t)size_IX;
  
  Double_t yline_haut_bot = (Double_t)ymid_bid;
  Double_t yline_haut_top = (Double_t)ysup_bid;

  Double_t yline_bas_bot = (Double_t)yinf_bid;
  Double_t yline_bas_top = (Double_t)ymid_bid;

  // coordonnees demi-lignes 
  //           l  =   0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10
  Int_t y_min[11] = { 0,11, 7, 0, 0, 0, 0, 0, 0, 0, 0};
  Int_t y_max[11] = {50,50,47,45,45,42,38,35,30,15,10};
  for(Int_t i=0;i<11;i++){y_min[i] = coefcc_y*y_min[i]; y_max[i] = coefcc_y*y_max[i];}

  gStyle->SetTextSize(0.075);   //  ===> pourquoi pas avant?

  for( Int_t i = 0 ; i <= max_x ; i++)
    {
      if( DeeNumber == 1 || DeeNumber == 3 ) 
	{
	  yline_haut_bot = ymid_bid + (Double_t)y_min[i];
	  yline_haut_top = ymid_bid + (Double_t)y_max[i];
	}
      if( DeeNumber == 2 || DeeNumber == 4 ) 
	{
	  yline_haut_bot = ymid_bid + (Double_t)y_min[10-i];
	  yline_haut_top = ymid_bid + (Double_t)y_max[10-i];
	}
      yline_bas_bot  = ysup_bid - yline_haut_top;
      yline_bas_top  = ysup_bid - yline_haut_bot;
      
      xline = xline + (Double_t)size_IX;
      TLine *lin_haut;
      lin_haut = new TLine(xline, yline_haut_bot, xline, yline_haut_top); fCnewRoot++;
      lin_haut->Draw();
      // delete lin_haut;             fCdeleteRoot++;
      TLine *lin_bas;
      lin_bas = new TLine(xline, yline_bas_bot, xline, yline_bas_top); fCnewRoot++;
      lin_bas->Draw();
      // delete lin_bas;             fCdeleteRoot++;
    }

  EEDataSectors(coefcc_x, coefcc_y, DeeNumber, "Dee");
  EEGridAxis(coefcc_x, coefcc_y, DeeNumber, "Dee", c_option);

} // end of ViewDeeGrid

//=================================================================================
//
//             SqrtContourLevels(const Int_t& nb_niv, Double_t* cont_niv)
//
//=================================================================================
void TEcnaHistos::SqrtContourLevels(const Int_t& nb_niv, Double_t* cont_niv)
{
//Calculation of levels in z coordinate for 3D plots. Square root scale
  
  Int_t nb_niv2 = (nb_niv+1)/2;
  
  for (Int_t num_niv = 0; num_niv < nb_niv2; num_niv++)
    {
      Int_t ind_niv = num_niv + nb_niv2 - 1;
      if ( ind_niv < 0 || ind_niv > nb_niv )
	{
	  cout << "!TEcnaHistos::ContourLevels(...)> *** ERROR *** "
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
	  cout << "!TEcnaHistos::ContourLevels(...)> *** ERROR *** "
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
//                    GetHocoVecoAxisTitle
//
//==========================================================================
TString TEcnaHistos::GetHocoVecoAxisTitle(const TString& chcode)
{
  TString xname = " ";

  if ( fFlagSubDet == "EB" ){xname = GetEtaPhiAxisTitle(chcode);}
  if ( fFlagSubDet == "EE" ){xname = GetIXIYAxisTitle(chcode);}

  return xname;
}

TString TEcnaHistos::GetEtaPhiAxisTitle(const TString& chcode)
{
  TString xname = " ";

  if ( chcode == "ietaEB" ){xname = "i#eta Xtal ";}
  if ( chcode == "ietaSM" ){xname = "i#eta Xtal ";}
  if ( chcode == "ietaTow"){xname = "i#eta Xtal        ";}

  if ( chcode == "iphiEB"   ){xname = "         i#varphi Xtal";}
  if ( chcode == "jphiEB+"  ){xname = "         i#varphi Xtal";}
  if ( chcode == "jphiEB-"  ){xname = "         i#varphi Xtal";}
  if ( chcode == "jphiSMB+" ){xname = "         i#varphi Xtal";}
  if ( chcode == "jphiSMB-" ){xname = "i#varphi Xtal    ";}
  if ( chcode == "jphiTow"  ){xname = "i#varphi Xtal in SM        ";}
  if ( chcode == "phi"      ){xname = "i#varphi Xtal in EB  ";}

  return xname;
}

TString TEcnaHistos::GetIXIYAxisTitle(const TString& chcode)
{
  TString xname = " ";

  if ( chcode == "iIXDee"  ){xname = "IX(SC)";}

  if ( chcode == "iIXDee1" ){xname = "                                                    -IX Xtal";}
  if ( chcode == "iIXDee2" ){xname = " IX Xtal                                                     ";}
  if ( chcode == "iIXDee3" ){xname = " IX Xtal";}
  if ( chcode == "iIXDee4" ){xname = "-IX Xtal                                                             ";}

  if ( chcode == "iIXEE"   ){xname = "                      IX Xtal";}

  if ( chcode == "iIXSC"  ){xname = "IX Xtal";}

  if ( chcode == "jIYDee" ){xname = "IY Xtal";}
  if ( chcode == "jIYSC"  ){xname = "IY Xtal";}
  if ( chcode == "IY"     ){xname = "IY";}

  return xname;
}

//=======================================================================================
//
//                              ViewStas(***)     
//
//           (Hoco,Veco) matrices for all the Stex's of a Stas             
//                        Stas = EB or EE
//
//=======================================================================================
void TEcnaHistos::ViewStas(const TVectorD& arg_read_histo, const Int_t& arg_AlreadyRead,
			   const TString&   HistoCode)
{
// (Hoco, Veco) matrices for all the Stex's of a Stas

  //......................... matrix title  
  char* f_in_mat_tit = new char[fgMaxCar];               fCnew++;
	  
  if (HistoCode == "D_NOE_ChNb"){sprintf(f_in_mat_tit, "Number of Events");}
  if (HistoCode == "D_Ped_ChNb"){sprintf(f_in_mat_tit, "Pedestals");}
  if (HistoCode == "D_TNo_ChNb"){sprintf(f_in_mat_tit, "Total noise");}
  if (HistoCode == "D_MCs_ChNb"){sprintf(f_in_mat_tit, "Mean cor(s,s')");}
  if (HistoCode == "D_LFN_ChNb"){sprintf(f_in_mat_tit, "Low frequency noise");}
  if (HistoCode == "D_HFN_ChNb"){sprintf(f_in_mat_tit, "High frequency noise");}
  if (HistoCode == "D_SCs_ChNb"){sprintf(f_in_mat_tit, "Sigma of cor(s,s')");}
	  
  //.... Axis parameters: *** WARNING *** EB ===>  x (Bid Hoco) = phi (StinVeco),  y (Bid Veco) = eta (StinHoco)
  Int_t GeoBidSizeHoco = fEcal->MaxStinVecoInStas();
  Int_t GeoBidSizeVeco = fEcal->MaxStinHocoInStas();

  Int_t vertic_empty_strips  = 3;
  Int_t vertic_empty_strip_1 = 1;

  if ( fFlagSubDet == "EE" )
    {
      // for empty vertical strips: before EE-, between EE- and EE+, after EE+ on plot
      GeoBidSizeHoco = fEcal->MaxStinHocoInStas() + vertic_empty_strips; 
      GeoBidSizeVeco = fEcal->MaxStinVecoInStas();
    }

  Int_t  nb_binx  = GeoBidSizeHoco;
  Int_t  nb_biny  = GeoBidSizeVeco;
  Axis_t xinf_bid = (Axis_t)0.;
  Axis_t xsup_bid = (Axis_t)GeoBidSizeHoco;
  Axis_t yinf_bid = (Axis_t)0.;
  Axis_t ysup_bid = (Axis_t)GeoBidSizeVeco;   
      
  TString axis_x_var_name = "  #Hoco  ";
  TString axis_y_var_name = "  #varVeco  ";

  //............. matrices reading and histogram filling   (ViewStas)

  TH2D* h_geo_bid = new TH2D("geobidim_ViewStas", f_in_mat_tit,
			     nb_binx, xinf_bid,   xsup_bid,
			     nb_biny, yinf_bid,   ysup_bid);     fCnewRoot++;
  h_geo_bid->Reset();

  Int_t CounterExistingFile = 0;
  Int_t CounterDataExist = 0;

  Int_t* xFapNbOfEvts = new Int_t[fEcal->MaxStexInStas()];      fCnew++;
  for(Int_t i=0; i<fEcal->MaxStexInStas(); i++){xFapNbOfEvts[i]=0;}

  //Int_t* NOFE_int = new Int_t[fEcal->MaxCrysEcnaInStex()];      fCnew++;

  //......................................................................... (ViewStas)
  for(Int_t iStasStex=0; iStasStex<fEcal->MaxStexInStas(); iStasStex++)
    {
      TVectorD partial_histp(fEcal->MaxStinEcnaInStex());
      for(Int_t i=0; i<fEcal->MaxStinEcnaInStex(); i++){partial_histp(i)=(Double_t)0.;}

      Bool_t OKFileExists = kFALSE;
      Bool_t OKDataExist  = kFALSE;

      if( arg_AlreadyRead == fTobeRead )
	{
	  fMyRootFile->PrintNoComment();
	  Int_t n1StasStex = iStasStex+1;
	  fMyRootFile->FileParameters(fFapAnaType,          fFapNbOfSamples,
				      fFapRunNumber,        fFapFirstReqEvtNumber,
				      fFapLastReqEvtNumber, fFapReqNbOfEvts,
				      n1StasStex,           fCfgResultsRootFilePath.Data());
	  
	  if ( fMyRootFile->LookAtRootFile() == kTRUE ){OKFileExists = kTRUE;}       //   (ViewStas)

	  if( OKFileExists == kTRUE )
	    {
	      xFapNbOfEvts[iStasStex] = fMyRootFile->GetNumberOfEvents(fFapReqNbOfEvts, n1StasStex);
	      TString fp_name_short = fMyRootFile->GetRootFileNameShort();
	      // cout << "*TEcnaHistos::ViewStas(...)> Data are analyzed from file ----> "
	      //      << fp_name_short << endl;
	      
	      //....................... search for first and last dates
	      if( iStasStex == 0 )
		{
		  fStartTime = fMyRootFile->GetStartTime();
		  fStopTime  = fMyRootFile->GetStopTime();
		  fStartDate = fMyRootFile->GetStartDate();
		  fStopDate  = fMyRootFile->GetStopDate();
		}
	      
	      time_t  xStartTime = fMyRootFile->GetStartTime();
	      time_t  xStopTime  = fMyRootFile->GetStopTime();
	      TString xStartDate = fMyRootFile->GetStartDate();
	      TString xStopDate  = fMyRootFile->GetStopDate();

	      if( xStartTime < fStartTime ){fStartTime = xStartTime; fStartDate = xStartDate;}
	      if( xStopTime  > fStopTime  ){fStopTime  = xStopTime;  fStopDate  = xStopDate;}
	      
	      fRunType = fMyRootFile->GetRunType();

	      //----------------------------------------------------------------------------- file reading   (ViewStas)
	      if( HistoCode == "D_NOE_ChNb" ){
		partial_histp = fMyRootFile->ReadAverageNumberOfEvents(fEcal->MaxStinEcnaInStex());}
	      if( HistoCode == "D_Ped_ChNb" ){
		 partial_histp = fMyRootFile->ReadAveragePedestals(fEcal->MaxStinEcnaInStex());}
	      if (HistoCode == "D_TNo_ChNb" ){
		partial_histp = fMyRootFile->ReadAverageTotalNoise(fEcal->MaxStinEcnaInStex());}
	      if( HistoCode == "D_MCs_ChNb" ){
		partial_histp = fMyRootFile->ReadAverageMeanCorrelationsBetweenSamples(fEcal->MaxStinEcnaInStex());}
	      if( HistoCode == "D_LFN_ChNb" ){
		partial_histp = fMyRootFile->ReadAverageLowFrequencyNoise(fEcal->MaxStinEcnaInStex());}
	      if( HistoCode == "D_HFN_ChNb" ){
		partial_histp = fMyRootFile->ReadAverageHighFrequencyNoise(fEcal->MaxStinEcnaInStex());}
	      if( HistoCode == "D_SCs_ChNb" ){
		partial_histp = fMyRootFile->ReadAverageSigmaOfCorrelationsBetweenSamples(fEcal->MaxStinEcnaInStex());}
	  
	      if ( fMyRootFile->DataExist() == kTRUE ){OKDataExist = kTRUE;}
	    }
	  else
	    {
	      fStatusFileFound = kFALSE;
	      cout << "!TEcnaHistos::ViewStas(...)> *ERROR* =====> "
		   << " ROOT file not found" << fTTBELL << endl;
	    }
	}

      if( arg_AlreadyRead == 1 )
	{
	  OKDataExist = kTRUE;
	  for(Int_t i0Stin=0; i0Stin<fEcal->MaxStinEcnaInStex(); i0Stin++)
	    {
	      partial_histp(i0Stin) = arg_read_histo(fEcal->MaxStinEcnaInStex()*iStasStex+i0Stin);
	    }
	}
      
      if( OKDataExist == kTRUE)
	{
	  fStatusFileFound = kTRUE;
	  CounterExistingFile++;

	  //.................................................................	  (ViewStas)
	  TMatrixD read_matrix(nb_binx, nb_biny);
	  for(Int_t i=0; i<nb_binx; i++)
	    {for(Int_t j=0; j<nb_biny; j++){read_matrix(i,j)=(Double_t)0.;}}

	  if ( OKDataExist == kTRUE )
	    {
	      fStatusDataExist = kTRUE;
	      CounterDataExist++;

	      for(Int_t i0StexStinEcna=0; i0StexStinEcna<fEcal->MaxStinEcnaInStex(); i0StexStinEcna++)
		{
		  //-------------------------------------- Geographical bidim filling   (ViewStas)
		  Int_t i_xgeo = GetXStinInStas(iStasStex, i0StexStinEcna, vertic_empty_strip_1);
		  Int_t i_ygeo = GetYStinInStas(iStasStex, i0StexStinEcna);

		  if(i_xgeo >=0 && i_xgeo < nb_binx && i_ygeo >=0 && i_ygeo < nb_biny)
		    {
		      Int_t n1StexStinEcna = i0StexStinEcna+1;

		      if( fFlagSubDet == "EB" )
			{	     
			  read_matrix(i_xgeo, i_ygeo) = partial_histp(i0StexStinEcna);
			  h_geo_bid->Fill((Double_t)i_xgeo, (Double_t)i_ygeo,
					  (Double_t)read_matrix(i_xgeo, i_ygeo));
			}

		      if( fFlagSubDet == "EE" )
			{
			  //---------------------> do not draw bin for SCEcna = 10 or 11	  (ViewStas)
			  if( !( (n1StexStinEcna == 10 || n1StexStinEcna == 11 ||
				  n1StexStinEcna == 29 || n1StexStinEcna == 32) ) )
			    {
			      read_matrix(i_xgeo, i_ygeo) = partial_histp(i0StexStinEcna);
			      h_geo_bid->Fill((Double_t)i_xgeo, (Double_t)i_ygeo,
					      (Double_t)read_matrix(i_xgeo, i_ygeo));
			    }
			  if( n1StexStinEcna == 29 )
			    {
			      //----------------------------------------------------------------- (ViewStas)
			      //   Average on SCEcna 29 (x1+x2+x3+x6+x7) and SCEcna 10: (x11)
			      //   (x = Xtal# in SC; see CMS NOTE 2006/027, p.10)
			      // 
			      //   (x1+x2+x3+x6+x7)/5 = partial_histp(29-1)   ;   x11 = partial_histp(10-1) 
			      //
			      //    => (x1+x2+x3+x6+x7+x11)/6 = partial_histp(29-1)*5/6 + partial_histp(10-1)/6
			      //
			      //   //  except for "D_NOE_ChNb" because average done in ReadAverageNumberOfEvents
			      //   //  (no averaged NbOfEvts in root file)
			      //---------------------------------------------------------------------------------
			      read_matrix(i_xgeo, i_ygeo) =
				partial_histp(i0StexStinEcna)*(Double_t)(5./6.) + partial_histp(9)/(Double_t)6.;
			      h_geo_bid->Fill((Double_t)i_xgeo, (Double_t)i_ygeo,
					      (Double_t)read_matrix(i_xgeo, i_ygeo));
			    }
			  //if( n1StexStinEcna == 32 && HistoCode != "D_NOE_ChNb" )	  (ViewStas)
			  if( n1StexStinEcna == 32 )
			    {
			      //---- same as previous case: replace SCEcna 29 by 32 AND SCEcna 10 by 11
			      //---->  (x1+x2+x3+x6+x7+x11)/6 = partial_histp(32-1)*5/6 + partial_histp(11-1)/6
			      read_matrix(i_xgeo, i_ygeo) =
				partial_histp(i0StexStinEcna)*(Double_t)(5./6.) + partial_histp(10)/(Double_t)6.;
			      h_geo_bid->Fill((Double_t)i_xgeo, (Double_t)i_ygeo,
					      (Double_t)read_matrix(i_xgeo, i_ygeo));
			    }
			} // end of if( fFlagSubDet == "EE" )
		    } // end of if(i_xgeo >=0 && i_xgeo < nb_binx && i_ygeo >=0 && i_ygeo < nb_biny)
		}  // end of for(Int_t i0StexStinEcna=0; i0StexStinEcna<fEcal->MaxStinEcnaInStex(); i0StexStinEcna++)
	    } // end of if ( fMyRootFile->DataExist() == kTRUE )
	  else
	    {
	      fStatusDataExist = kFALSE;

	      cout << "!TEcnaHistos::ViewStas(...)>  "
		   << " Data not available for " << fFapStexName << " " << iStasStex+1
		   << " (Quantity not present in the ROOT file)" << fTTBELL << endl;
	    }
	} // end of if( fMyRootFile->LookAtRootFile() == kTRUE )	  (ViewStas)
      else
	{
	  fStatusFileFound = kFALSE;

	  cout << "!TEcnaHistos::ViewStas(...)>  "
	       << " Data not available for " << fFapStexName << " " << iStasStex+1
	       << " (ROOT file not found)"  << fTTBELL << endl;
	}

      if( fFapNbOfEvts <= xFapNbOfEvts[iStasStex] ){fFapNbOfEvts = xFapNbOfEvts[iStasStex];}

    } // end of for(Int_t iStasStex=0; iStasStex<fEcal->MaxStexInStas(); iStasStex++)

  //delete [] NOFE_int;     NOFE_int = 0;       fCdelete++;
  delete [] xFapNbOfEvts; xFapNbOfEvts = 0;   fCdelete++;

  if( CounterExistingFile > 0 && CounterDataExist > 0 )
    {
      //===============  H I S T O   M I N / M A X   M A N A G E M E N T ============  (ViewStas)
      //................................ Put histo min max values
      //.......... default if flag not set to "ON"
      SetYminMemoFromValue(HistoCode.Data(), fCnaParHistos->GetYminDefaultValue(HistoCode.Data()));
      SetYmaxMemoFromValue(HistoCode.Data(), fCnaParHistos->GetYmaxDefaultValue(HistoCode.Data()));
	  
      if( fUserHistoMin == fUserHistoMax ){fFlagUserHistoMin = "AUTO"; fFlagUserHistoMax = "AUTO";}
      //.......... user's value if flag set to "ON"
      if( fFlagUserHistoMin == "ON" )
	{SetYminMemoFromValue(HistoCode.Data(), fUserHistoMin); fFlagUserHistoMin = "OFF";}
      if( fFlagUserHistoMax == "ON" )
	{SetYmaxMemoFromValue(HistoCode.Data(), fUserHistoMax); fFlagUserHistoMax = "OFF";}
      //................................. automatic min and/or max
      if( fFlagUserHistoMin == "AUTO" )
	{SetYminMemoFromValue(HistoCode.Data(), h_geo_bid->GetMinimum()); fFlagUserHistoMin = "OFF";}
      if( fFlagUserHistoMax == "AUTO" )
	{SetYmaxMemoFromValue(HistoCode.Data(), h_geo_bid->GetMaximum()); fFlagUserHistoMax = "OFF";}
      //...................................... histo set ymin and ymax
      SetHistoFrameYminYmaxFromMemo((TH1D*)h_geo_bid, HistoCode);
	  
      // ************************** A GARDER EN RESERVE *******************************
      //............. special contour level for correlations (square root wise scale)
      //if ( HistoCode == "D_MCs_ChNb" )
      //{
      //  Int_t nb_niv  = 9;
      //  Double_t* cont_niv = new Double_t[nb_niv];           fCnew++;
      //  SqrtContourLevels(nb_niv, &cont_niv[0]);      
      //  h_geo_bid->SetContour(nb_niv, &cont_niv[0]);	      
      //  delete [] cont_niv;                                  fCdelete++;
      //}
      // ******************************** (FIN RESERVE) *******************************

      // =================================== P L O T S ========================   (ViewStas) 
	  
      char* f_in = new char[fgMaxCar];                           fCnew++;

      //...................... Taille/format canvas
      UInt_t canv_h = fCnaParHistos->CanvasFormatH("petit");
      UInt_t canv_w = fCnaParHistos->CanvasFormatW("petit");

      if( fFlagSubDet == "EB")
	{canv_w = fCnaParHistos->CanvasFormatW("phietaEB");
	canv_h = fCnaParHistos->CanvasFormatH("phietaEB");}
      if( fFlagSubDet == "EE")	  
	{canv_w = fCnaParHistos->CanvasFormatW("IYIXEE");
	canv_h = fCnaParHistos->CanvasFormatH("IYIXEE");}

      //............................................... paves commentaires (ViewStas)
      SetAllPavesViewStas();

      //------------------------------------ Canvas name ----------------- (ViewStas)  
      TString name_cov_cor;
      Int_t MaxCar = fgMaxCar;
      name_cov_cor.Resize(MaxCar);
      name_cov_cor = "?";
	  
      if( HistoCode == "D_NOE_ChNb"){name_cov_cor = "Number_of_Events";}
      if( HistoCode == "D_Ped_ChNb"){name_cov_cor = "Pedestals";}
      if( HistoCode == "D_TNo_ChNb"){name_cov_cor = "Total_noise";}
      if( HistoCode == "D_MCs_ChNb"){name_cov_cor = "Mean_Corss";}
      if( HistoCode == "D_LFN_ChNb"){name_cov_cor = "Low_Fq_Noise";}
      if( HistoCode == "D_HFN_ChNb"){name_cov_cor = "High_Fq_Noise";}
      if( HistoCode == "D_SCs_ChNb"){name_cov_cor = "Sigma_Corss";}
	  
      TString name_visu;
      MaxCar = fgMaxCar;
      name_visu.Resize(MaxCar);
      name_visu = "colz";
	  
      sprintf(f_in, "%s_%s_S1_%d_R%d_%d_%d_%s_%s_HocoVeco_R%d",
	      name_cov_cor.Data(),   fFapAnaType.Data(),   fFapNbOfSamples, fFapRunNumber,
	      fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFlagSubDet.Data(),
	      name_visu.Data(),      arg_AlreadyRead);
	  
      if( fFlagSubDet == "EB" ){SetHistoPresentation((TH1D*)h_geo_bid, "Stas2DEB");}
      if( fFlagSubDet == "EE" ){SetHistoPresentation((TH1D*)h_geo_bid, "Stas2DEE");}

      TCanvas *MainCanvas = new TCanvas(f_in, f_in, canv_w, canv_h);   fCnewRoot++;
      fCurrentCanvas = MainCanvas; fCurrentCanvasName = f_in;

      // cout << "*TEcnaHistos::ViewStas(...)> Plot is displayed on canvas ----> " << f_in << endl;
	  
      delete [] f_in; f_in = 0;                                 fCdelete++;

      //------------------------ Canvas draw and update ------------ (ViewStas)  
      if( fPavComGeneralTitle != 0 ){fPavComGeneralTitle->Draw();}

      fPavComStas->Draw();
      fPavComAnaRun->Draw();
      fPavComNbOfEvts->Draw();

      //----------------------------------------------------------- pad margins
      Double_t x_low = fCnaParHistos->BoxLeftX("bottom_left_box") - 0.005;
      Double_t y_low = fCnaParHistos->BoxTopY("bottom_left_box")  + 0.005;
      Double_t x_margin = x_low;
      Double_t y_margin = y_low;	  
      MainCanvas->Divide( 1,  1, x_margin, y_margin);
      //           Divide(nx, ny, x_margin, y_margin,    color);	  
      gPad->cd(1);
      //........................... specific EE
      if( fFlagSubDet == "EE" ){
	Double_t x_up  = fCnaParHistos->BoxRightX("bottom_right_box")  + 0.005;
	Double_t y_up  = fCnaParHistos->BoxBottomY("top_left_box_EE") - 0.005;
	TVirtualPad* main_subpad = gPad;
	main_subpad->SetPad(x_low, y_low, x_up, y_up);}
	  
      //------------------------------------------------------------
      h_geo_bid->GetXaxis()->SetTitle(axis_x_var_name);
      h_geo_bid->GetYaxis()->SetTitle(axis_y_var_name);
	  
      h_geo_bid->DrawCopy("COLZ");
	  
      // trace de la grille  ---------------- (ViewStas) 
      ViewStasGrid(vertic_empty_strips);
      gPad->Draw();
      gPad->Update();

      //      delete MainCanvas;              fCdeleteRoot++;
    }
  //..................... retour aux options standard

  Bool_t b_true = 1;
  h_geo_bid->SetStats(b_true);    
  h_geo_bid->Delete();  h_geo_bid = 0;              fCdeleteRoot++;
  
  delete [] f_in_mat_tit;    f_in_mat_tit = 0;             fCdelete++;

}  // end of ViewStas(...)

//==================================================================================
//
//                         GetXStinInStas , GetYStinInStas
//
//==================================================================================
Int_t TEcnaHistos::GetXStinInStas(const Int_t& iStasStex, const Int_t& StexStinEcna,
				  const Int_t& vertic_empty_strip_1) 
{
//Gives the X Stin coordinate in the geographic view of the Stas
// (X = 0 to MaxStexHocoInStas*MaxStinHocoInStex - 1 + vertic_empty_strips(EE only))

  Int_t ix_geo = 0;
  Int_t n1StasStex = iStasStex+1;
  TString ctype = fEcalNumbering->GetStexHalfStas(n1StasStex);

  if( fFlagSubDet == "EB")
    {
      if( ctype == "EB-")
	{
	  ix_geo = (iStasStex - fEcal->MaxStexInStasMinus())*fEcal->MaxStinVecoInStex()
	    + StexStinEcna%fEcal->MaxStinVecoInStex();
	}
      if( ctype == "EB+")
	{
	  ix_geo = iStasStex*fEcal->MaxStinVecoInStex()
	    + fEcal->MaxStinVecoInStex()- 1 - StexStinEcna%fEcal->MaxStinVecoInStex();
	}
    }

  if( fFlagSubDet == "EE")
    {
      TString LeftRightFromIP = fEcalNumbering->GetDeeDirViewedFromIP(n1StasStex);

      if( ctype == "EE-" && LeftRightFromIP == "left"  )
	{
      	  ix_geo = fEcal->MaxStinHocoInStex() - StexStinEcna/fEcal->MaxStinVecoInStex() - 1 + vertic_empty_strip_1;
	}
      if( ctype == "EE-" && LeftRightFromIP == "right" )
	{
      	  ix_geo = fEcal->MaxStinHocoInStex() + StexStinEcna/fEcal->MaxStinVecoInStex() + vertic_empty_strip_1;
	}
      if( ctype == "EE+" && LeftRightFromIP == "left"  )
	{
	  ix_geo = (Int_t)fCnaParHistos->DeeOffsetX(fFlagSubDet, n1StasStex)
	    + fEcal->MaxStinHocoInStex() - StexStinEcna/fEcal->MaxStinVecoInStex() - 1;
	}
      if( ctype == "EE+" && LeftRightFromIP == "right" )
	{
	  ix_geo = (Int_t)fCnaParHistos->DeeOffsetX(fFlagSubDet, n1StasStex)
	    + StexStinEcna/fEcal->MaxStinVecoInStex();
	}
    }
  return ix_geo;
}

Int_t TEcnaHistos::GetYStinInStas(const Int_t& iStasStex, const Int_t& StexStinEcna) 
{
//Gives the Y crystal coordinate in the geographic view of one Stex
// (Y = 0 to MaxStexVecoInStas*MaxStinVecoInStex - 1)

  Int_t iy_geo = 0;
  
  if( fFlagSubDet == "EB")
    {
      Int_t n1StasStex = iStasStex+1;
      TString ctype = fEcalNumbering->GetStexHalfStas(n1StasStex);
      if( ctype == "EB+")
	{iy_geo = StexStinEcna/fEcal->MaxStinVecoInStex() + fEcal->MaxStinHocoInStex(); }
      if( ctype == "EB-")
	{iy_geo = fEcal->MaxStinHocoInStex() - 1 - StexStinEcna/fEcal->MaxStinVecoInStex();}
    }
  
  if( fFlagSubDet == "EE")
    {iy_geo = StexStinEcna%fEcal->MaxStinVecoInStex();} 
  return iy_geo;
}

//==========================================================================
//
//                       ViewStasGrid
//              independent of the ROOT file
//
//==========================================================================
void TEcnaHistos::ViewStasGrid(const Int_t & vertic_empty_strips)
{
 //Grid of Stas with axis Hoco and Veco

  if( fFlagSubDet ==  "EB"){ViewEBGrid();}
  if( fFlagSubDet ==  "EE"){ViewEEGrid(vertic_empty_strips);}

} // end of ViewStasGrid

void TEcnaHistos::ViewEBGrid()
{
 //Grid of EB with axis Hoco and Veco

  Int_t  GeoBidSizeEta = fEcal->MaxSMEtaInEB()*fEcal->MaxTowEtaInSM();
  Int_t  GeoBidSizePhi = fEcal->MaxSMPhiInEB()*fEcal->MaxTowPhiInSM();

  Int_t size_y = fEcal->MaxTowEtaInSM();
  Int_t size_x = fEcal->MaxTowPhiInSM();

  Int_t  nb_binx  = GeoBidSizePhi;
  Int_t  nb_biny  = GeoBidSizeEta;
  Axis_t xinf_bid = (Axis_t)0.;
  Axis_t xsup_bid = (Axis_t)nb_binx;
  Axis_t yinf_bid = (Axis_t)0.;
  Axis_t ysup_bid = (Axis_t)nb_biny;   
  
  //---------------- trace de la grille: un rectangle = un SM

  Int_t max_x = nb_binx/size_x;  // = fEcal->MaxSMPhiInEB()
  Int_t max_y = nb_biny/size_y;  // = fEcal->MaxSMEtaInEB()

  //............................. lignes horizontales  (ViewEBGrid)
  Double_t yline = (Double_t)yinf_bid;

  Double_t xline_left  = (Double_t)xinf_bid;
  Double_t xline_right = (Double_t)xsup_bid;
  
  for( Int_t j = 0 ; j < max_y ; j++)
    {
      yline = yline + (Double_t)size_y;
      TLine *lin;
      lin = new TLine(xline_left, yline, xline_right, yline); fCnewRoot++;
      lin->Draw();
      // delete lin;             fCdeleteRoot++;
    }

  //-------------------------------- lignes verticales
  Double_t xline = (Double_t)xinf_bid - (Double_t)size_x;
  
  Double_t yline_bot = (Double_t)yinf_bid;
  Double_t yline_top = (Double_t)ysup_bid;

  for( Int_t i = 0 ; i < max_x ; i++)
    {  
      xline = xline + (Double_t)size_x;
      TLine *lin;
      lin = new TLine(xline, yline_bot, xline, yline_top); fCnewRoot++;
      lin->Draw();
    }

  //-------------------------------- Numeros des SM
  Double_t yTextBot = yline_bot - (yline_top - yline_bot)/25.;
  Double_t yTextTop = yline_top + (yline_top - yline_bot)/120.;
  xline = (Double_t)xinf_bid - (Double_t)size_x;

  char* f_in = new char[fgMaxCar];                  fCnew++;
  TText *text_SM = new TText();              fCnewRoot++;
  for( Int_t i = 0 ; i < max_x ; i++)
    {  
      xline = xline + (Double_t)size_x;
      text_SM->SetTextColor(fCnaParHistos->ColorDefinition("bleu_fonce"));
      text_SM->SetTextSize((Double_t)0.03);
      sprintf( f_in, "  +%d", i+1 );
      text_SM->DrawText(xline, yTextTop, f_in);
      sprintf( f_in, "  %d", -i-1 );
      text_SM->DrawText(xline, yTextBot, f_in);
    }
  delete [] f_in;                                   fCdelete++;

  //------------------ trace axes en eta et phi --------------- ViewEBGrid

  Int_t SMNumber = 1;

  //...................................................... Axe i(phi) (bottom x) ViewEBGrid
  Int_t MatSize    = fEcal->MaxTowPhiInSM();
  Int_t size_x_eb  = fEcal->MaxSMPhiInEB();
  Double_t phi_min =   0;
  Double_t phi_max = 360;

  TString  x_var_name  = GetHocoVecoAxisTitle("iphiEB");;
  TString  x_direction = fEcalNumbering->GetXDirectionEB(SMNumber);

  new TF1("f1", x_direction.Data(), phi_min, phi_max);          fCnewRoot++;
    TGaxis* sup_axis_x = 0;

  if( x_direction == "-x" ) // NEVER  IN THIS CASE: xmin->xmax <=> right->left ("-x") direction
    {sup_axis_x = new TGaxis( (Float_t)0., (Float_t)0., (Float_t)(size_x_eb*MatSize), (Float_t)0.,
			      "f1", size_x_eb, "SC" , 0.);   fCnewRoot++;}

  if( x_direction == "x" )  // ALWAYS IN THIS CASE: xmin->xmax <=> left->right ("x") direction    
    {sup_axis_x = new TGaxis( (Float_t)0., (Float_t)0., (Float_t)(size_x_eb*MatSize), (Float_t)0.,
			      "f1", size_x_eb, "SC" , 0.);   fCnewRoot++;} 

  Float_t tit_siz_x = fCnaParHistos->AxisTitleSize();
  Float_t lab_siz_x = fCnaParHistos->AxisLabelSize("EBx");
  Float_t tic_siz_x = fCnaParHistos->AxisTickSize("SMx");
  Float_t tit_off_x = fCnaParHistos->AxisTitleOffset("EBx");
  Float_t lab_off_x = fCnaParHistos->AxisLabelOffset("EBx");
 
  sup_axis_x->SetTitle(x_var_name);
  sup_axis_x->SetTitleSize(tit_siz_x);
  sup_axis_x->SetTitleOffset(tit_off_x);
  sup_axis_x->SetLabelSize(lab_siz_x);
  sup_axis_x->SetLabelOffset(lab_off_x);
  sup_axis_x->SetTickSize(tic_siz_x);
  sup_axis_x->Draw("SAME");

  //...................................................... Axe eta (y) ViewEBGrid
  MatSize = fEcal->MaxTowEtaInSM();
  Int_t size_y_eb = fEcal->MaxSMEtaInEB(); 

  Double_t eta_min = (Double_t)(-85.); 
  Double_t eta_max = (Double_t)85.; 

  TString y_var_name = GetHocoVecoAxisTitle("ietaEB");

  TGaxis* sup_axis_y = 0;
  sup_axis_y = new TGaxis((Float_t)0., (Float_t)0.,
			  (Float_t)0., (Float_t)(size_y_eb*MatSize),
			  eta_min, eta_max, MatSize/2, "SC", 0.);         fCnewRoot++;

  Float_t tit_siz_y = fCnaParHistos->AxisTitleSize();
  Float_t lab_siz_y = fCnaParHistos->AxisLabelSize("EBy");
  Float_t tic_siz_y = fCnaParHistos->AxisTickSize("SMy");
  Float_t tit_off_y = fCnaParHistos->AxisTitleOffset("EBy");
  Float_t lab_off_y = fCnaParHistos->AxisLabelOffset("EBy");
  
  sup_axis_y->SetTitle(y_var_name);
  sup_axis_y->SetTitleSize(tit_siz_y);
  sup_axis_y->SetTitleOffset(tit_off_y);
  sup_axis_y->SetLabelColor(1);
  sup_axis_y->SetLabelSize(lab_siz_y);
  sup_axis_y->SetLabelOffset(lab_off_y);
  sup_axis_y->SetTickSize(tic_siz_y);
  sup_axis_y->Draw("SAME");

  //f2 = 0;

  gStyle->SetTextColor(fCnaParHistos->ColorDefinition("noir"));

} // end of ViewEBGrid
//---------------------------------------------------------------------
void TEcnaHistos::ViewEEGrid(const Int_t& vertic_empty_strips)
{
 //Grid of EE with axis Hoco and Veco

  Float_t coefcc_x = (Float_t)1./(Float_t)5.;
  Float_t coefcc_y = (Float_t)1./(Float_t)5.;
  
  for( Int_t DeeNumber = 1; DeeNumber <= 4; DeeNumber++)
    {
      EEDataSectors(coefcc_x, coefcc_y, DeeNumber, "EE");
      EEGridAxis(coefcc_x, coefcc_y, DeeNumber, "EE", " "); 
    }

  // vertical line between the two endcaps
  Double_t xline = coefcc_x*( 2*fEcal->MaxCrysIXInDee()
			      + ((Double_t)vertic_empty_strips)/2.*fEcal->MaxCrysIXInSC() );
  
  Double_t yline_bot = coefcc_y*(Double_t)0.;
  Double_t yline_top = coefcc_y*(Double_t)fEcal->MaxCrysIYInDee();

  TLine *lin;
  lin = new TLine(xline, yline_bot, xline, yline_top); fCnewRoot++;
  lin->Draw();

  // vertical line in the midles of the two endcaps
  //  xline = xline + coefcc_x*( fEcal->MaxCrysIXInDee()+ 0.5*fEcal->MaxCrysIXInSC() );
  xline = coefcc_x*(3*fEcal->MaxCrysIXInDee()
		    + ((Double_t)vertic_empty_strips-1.)*fEcal->MaxCrysIXInSC() );
  TLine *lin12;
  lin12 = new TLine(xline, yline_bot, xline, yline_top); fCnewRoot++;
  lin12->SetLineStyle(2);
  lin12->Draw();

  xline = coefcc_x*(fEcal->MaxCrysIXInDee()
		     + ((Double_t)vertic_empty_strips)/3.*fEcal->MaxCrysIXInSC() );
  TLine *lin34;
  lin34 = new TLine(xline, yline_bot, xline, yline_top); fCnewRoot++;
  lin34->SetLineStyle(2);
  lin34->Draw();

  // horizontal line at IY = 50
  Double_t xline_end = coefcc_x*( 4*fEcal->MaxCrysIXInDee() + vertic_empty_strips*fEcal->MaxCrysIXInSC());
  Double_t yline_mid = coefcc_x*fEcal->MaxCrysIYInDee()/2;

  TLine *linh;
  linh = new TLine( 0., yline_mid, xline_end, yline_mid); fCnewRoot++;
  linh->SetLineStyle(2);
  linh->Draw();

} // end of ViewEEGrid

//==================================================================================================
void TEcnaHistos::EEDataSectors(const Float_t& coefcc_x,  const Float_t& coefcc_y,
				const Int_t&   DeeNumber, const TString& opt_plot)
{
  //Surlignage des bords du Dee et des Data Sectors. Numeros des secteurs.

  // Epaisseur du trait selon option
  Int_t LineWidth = 2;   // DEFAULT => option "EE"
  if( opt_plot == "Dee" ){LineWidth = 3;}

  Int_t ngmax = 0;
  // surlignage du bord interne du Dee (unite de coordonnees: le cristal)
  ngmax = 13;
  Float_t xg_dee_int_bot[13] = { 0, 5, 5, 7, 7, 8, 8, 9, 9,10,10,11,11};
  Float_t yg_dee_int_bot[13] = {39,39,40,40,41,41,42,42,43,43,45,45,50};
  for(Int_t i=0;i<ngmax;i++){
    xg_dee_int_bot[i] = coefcc_x*xg_dee_int_bot[i];
    yg_dee_int_bot[i] = coefcc_y*yg_dee_int_bot[i];}
  
  Float_t XgDeeIntBotRight[13]; Float_t YgDeeIntBotRight[13];
  Float_t XgDeeIntTopRight[13]; Float_t YgDeeIntTopRight[13];
  
  for( Int_t i=0; i<ngmax; i++)
    {
      XgDeeIntBotRight[i] = xg_dee_int_bot[i];
      YgDeeIntBotRight[i] = yg_dee_int_bot[i];
      XgDeeIntTopRight[i] = XgDeeIntBotRight[i];
      YgDeeIntTopRight[i] = coefcc_y*fEcal->MaxCrysIYInDee() - YgDeeIntBotRight[i];
      if ( DeeNumber == 2 || DeeNumber == 4 )
	{
	  XgDeeIntBotRight[i] = -XgDeeIntBotRight[i] + coefcc_x*fEcal->MaxCrysIXInDee();
	  XgDeeIntTopRight[i] = -XgDeeIntTopRight[i] + coefcc_x*fEcal->MaxCrysIXInDee();
	}
      XgDeeIntBotRight[i] += fCnaParHistos->DeeOffsetX(opt_plot, DeeNumber);
      XgDeeIntTopRight[i] += fCnaParHistos->DeeOffsetX(opt_plot, DeeNumber);
    }

  TGraph *BDeeIntBotRight = new TGraph(ngmax, XgDeeIntBotRight, YgDeeIntBotRight);
  BDeeIntBotRight->SetLineWidth(LineWidth);
  BDeeIntBotRight->Draw();
  
  TGraph *BDeeIntTopRight = new TGraph(ngmax, XgDeeIntTopRight, YgDeeIntTopRight);
  BDeeIntTopRight->SetLineWidth(LineWidth);
  BDeeIntTopRight->Draw();

  // surlignage du bord externe du Dee  (unite de coordonnees: le cristal)
  ngmax = 21;
  Float_t xg_dee_ext_bot[21] = {0,10,10,15,15,25,25,30,30,35,35,37,37,42,42,45,45,47,47,50,50};
  Float_t yg_dee_ext_bot[21] = {0, 0, 3, 3, 5, 5, 8, 8,13,13,15,15,20,20,25,25,35,35,40,40,50};
  for(Int_t i=0;i<ngmax;i++){
    xg_dee_ext_bot[i] = coefcc_x*xg_dee_ext_bot[i];
    yg_dee_ext_bot[i] = coefcc_y*yg_dee_ext_bot[i];}
  
  Float_t XgDeeExtBotRight[21]; Float_t YgDeeExtBotRight[21];
  Float_t XgDeeExtTopRight[21]; Float_t YgDeeExtTopRight[21];
  
  for( Int_t i=0; i<ngmax; i++)
    {
      XgDeeExtBotRight[i] = xg_dee_ext_bot[i];
      YgDeeExtBotRight[i] = yg_dee_ext_bot[i];
      XgDeeExtTopRight[i] = XgDeeExtBotRight[i];
      YgDeeExtTopRight[i] = coefcc_y*fEcal->MaxCrysIYInDee() - YgDeeExtBotRight[i];
      if ( DeeNumber == 2 || DeeNumber == 4 )
	{
	  XgDeeExtBotRight[i] = -XgDeeExtBotRight[i] + coefcc_x*fEcal->MaxCrysIXInDee();
	  XgDeeExtTopRight[i] = -XgDeeExtTopRight[i] + coefcc_x*fEcal->MaxCrysIXInDee();
	}
      XgDeeExtBotRight[i] += fCnaParHistos->DeeOffsetX(opt_plot, DeeNumber);
      XgDeeExtTopRight[i] += fCnaParHistos->DeeOffsetX(opt_plot, DeeNumber);
    }
  
  TGraph *BDeeExtBotRight = new TGraph(ngmax, XgDeeExtBotRight, YgDeeExtBotRight);
  BDeeExtBotRight->SetLineWidth(LineWidth);
  BDeeExtBotRight->Draw();
  
  TGraph *BDeeExtTopRight = new TGraph(ngmax, XgDeeExtTopRight, YgDeeExtTopRight);
  BDeeExtTopRight->SetLineWidth(LineWidth);
  BDeeExtTopRight->Draw();

  char* f_in = new char[fgMaxCar];                           fCnew++;
  
  //............. Surlignage separateurs des secteurs en phi (Data sectors)

  //================== S9 -> S1 (EE-) option "EE" seulement
  if( opt_plot == "EE" )
    {
      ngmax = 2;
      Float_t xg_dee_data_sec9[2] = { 0,  0};
      Float_t yg_dee_data_sec9[2] = {61,100};
      for(Int_t i=0;i<ngmax;i++){
	xg_dee_data_sec9[i] = coefcc_x*xg_dee_data_sec9[i];
	yg_dee_data_sec9[i] = coefcc_y*yg_dee_data_sec9[i];}
      
      Float_t XgDeeDataSec9[11]; Float_t YgDeeDataSec9[11];
      for( Int_t i=0; i<ngmax; i++)
	{
	  XgDeeDataSec9[i] = xg_dee_data_sec9[i]; YgDeeDataSec9[i] = yg_dee_data_sec9[i];
	  if ( DeeNumber == 2 || DeeNumber == 4 )
	    {
	      XgDeeDataSec9[i] = -XgDeeDataSec9[i] + coefcc_x*fEcal->MaxCrysIXInDee();
	    }
	  XgDeeDataSec9[i] += fCnaParHistos->DeeOffsetX(opt_plot, DeeNumber);
	}      
      TGraph *BDeeDataSec9 = new TGraph(ngmax, XgDeeDataSec9, YgDeeDataSec9);
      BDeeDataSec9->SetLineWidth(LineWidth);
      BDeeDataSec9->Draw();      
    }

  //================= S1->S2(EE-)
  ngmax = 11;
  Float_t xg_dee_data_sec1[11] = { 7,10,10,15,15,20,20,25,25,30,30};
  Float_t yg_dee_data_sec1[11] = {60,60,65,65,70,70,75,75,85,85,87};
  for(Int_t i=0;i<ngmax;i++){
    xg_dee_data_sec1[i] = coefcc_x*xg_dee_data_sec1[i];
    yg_dee_data_sec1[i] = coefcc_y*yg_dee_data_sec1[i];}
  
  Float_t XgDeeDataSec1[11]; Float_t YgDeeDataSec1[11];
  for( Int_t i=0; i<ngmax; i++)
    {
      XgDeeDataSec1[i] = xg_dee_data_sec1[i]; YgDeeDataSec1[i] = yg_dee_data_sec1[i];
      if ( DeeNumber == 2 || DeeNumber == 4 )
	{
	  XgDeeDataSec1[i] = -XgDeeDataSec1[i] + coefcc_x*fEcal->MaxCrysIXInDee();
	}
      XgDeeDataSec1[i] += fCnaParHistos->DeeOffsetX(opt_plot, DeeNumber);
    }

  TGraph *BDeeDataSec1 = new TGraph(ngmax, XgDeeDataSec1, YgDeeDataSec1);
  BDeeDataSec1->SetLineWidth(LineWidth);
  BDeeDataSec1->Draw();

  //================= S2->S3(EE-)
  ngmax = 6;
  Float_t xg_dee_data_sec2[6] = {11,15,15,40,40,47};
  Float_t yg_dee_data_sec2[6] = {50,50,55,55,60,60};
  for(Int_t i=0;i<ngmax;i++){
    xg_dee_data_sec2[i] = coefcc_x*xg_dee_data_sec2[i];
    yg_dee_data_sec2[i] = coefcc_y*yg_dee_data_sec2[i];}
  
  Float_t XgDeeDataSec2[6]; Float_t YgDeeDataSec2[6];
  for( Int_t i=0; i<ngmax; i++)
    {
      XgDeeDataSec2[i] = xg_dee_data_sec2[i]; YgDeeDataSec2[i] = yg_dee_data_sec2[i];
      if ( DeeNumber == 2 || DeeNumber == 4 )
	{
	  XgDeeDataSec2[i] = -XgDeeDataSec2[i] + coefcc_x*fEcal->MaxCrysIXInDee();
	}
      XgDeeDataSec2[i] += fCnaParHistos->DeeOffsetX(opt_plot, DeeNumber);
    }
  TGraph *BDeeDataSec2 = new TGraph(ngmax, XgDeeDataSec2, YgDeeDataSec2);
  BDeeDataSec2->SetLineWidth(LineWidth);
  BDeeDataSec2->Draw();
 
  //================= S3->S4(EE-)
  ngmax = 10;
  Float_t xg_dee_data_sec3[10] = {10,15,15,20,20,30,30,40,40,42};
  Float_t yg_dee_data_sec3[10] = {45,45,40,40,35,35,30,30,25,25};
  for(Int_t i=0;i<ngmax;i++){
    xg_dee_data_sec3[i] = coefcc_x*xg_dee_data_sec3[i];
    yg_dee_data_sec3[i] = coefcc_y*yg_dee_data_sec3[i];}
  
  Float_t XgDeeDataSec3[10]; Float_t YgDeeDataSec3[10];
  for( Int_t i=0; i<ngmax; i++)
    {
      XgDeeDataSec3[i] = xg_dee_data_sec3[i]; YgDeeDataSec3[i] = yg_dee_data_sec3[i];
      if ( DeeNumber == 2 || DeeNumber == 4 )
	{
	  XgDeeDataSec3[i] = -XgDeeDataSec3[i] + coefcc_x*fEcal->MaxCrysIXInDee();
	}
      XgDeeDataSec3[i] += fCnaParHistos->DeeOffsetX(opt_plot, DeeNumber);
    }
  TGraph *BDeeDataSec3 = new TGraph(ngmax, XgDeeDataSec3, YgDeeDataSec3);
  BDeeDataSec3->SetLineWidth(LineWidth);
  BDeeDataSec3->Draw();
 
  //================= S4->S5(EE-)
  ngmax = 6;
  Float_t xg_dee_data_sec4[6] = { 5, 5,10,10,15,15};
  Float_t yg_dee_data_sec4[6] = {40,30,30,15,15, 5};
  for(Int_t i=0;i<ngmax;i++){
    xg_dee_data_sec4[i] = coefcc_x*xg_dee_data_sec4[i];
    yg_dee_data_sec4[i] = coefcc_y*yg_dee_data_sec4[i];}
  
  Float_t XgDeeDataSec4[6]; Float_t YgDeeDataSec4[6];
  for( Int_t i=0; i<ngmax; i++)
    {
      XgDeeDataSec4[i] = xg_dee_data_sec4[i]; YgDeeDataSec4[i] = yg_dee_data_sec4[i];
      if ( DeeNumber == 2 || DeeNumber == 4 )
	{
	  XgDeeDataSec4[i] = -XgDeeDataSec4[i] + coefcc_x*fEcal->MaxCrysIXInDee();
	}
      XgDeeDataSec4[i] += fCnaParHistos->DeeOffsetX(opt_plot, DeeNumber);
    }
  TGraph *BDeeDataSec4 = new TGraph(ngmax, XgDeeDataSec4, YgDeeDataSec4);
  BDeeDataSec4->SetLineWidth(LineWidth);
  BDeeDataSec4->Draw();
  

  //..................................... Numeros des secteurs S_i (option "Dee" seulement)
  if( opt_plot == "Dee" || opt_plot == "EE" )
    {
      //............. Coordonnees des numeros de secteurs
      ngmax = 5;
      Float_t  xg_coord_sector[5] = { 16, 41, 45, 33, -2};
      Float_t  yg_coord_sector[5] = { 96, 83, 30,  5, -8};

      //....... Reprise secteurs 3 et 7
      if(opt_plot == "Dee" && (DeeNumber == 1) ){xg_coord_sector[2] += 0.5;}
      if(opt_plot == "Dee" && (DeeNumber == 2) ){xg_coord_sector[2] -= 1. ;}
      if(opt_plot == "Dee" && (DeeNumber == 3) ){xg_coord_sector[2] += 0.7;}
      if(opt_plot == "Dee" && (DeeNumber == 4) ){xg_coord_sector[2] -= 1.2;}

      if(opt_plot == "EE"  && (DeeNumber == 2 || DeeNumber == 3) ){xg_coord_sector[2] += 0.55;}
      if(opt_plot == "EE"  && (DeeNumber == 4 ) ){xg_coord_sector[2] -= 0.2;}

      for(Int_t i=0;i<ngmax;i++){
	xg_coord_sector[i] = coefcc_x*xg_coord_sector[i];
	yg_coord_sector[i] = coefcc_y*yg_coord_sector[i];}

      Float_t  xg_sector[9];
      Float_t  yg_sector[9];
      Int_t ns1 = 1;
      Int_t ns2 = 5;
      Float_t xinv_d2d4 = coefcc_x*44;
      
      if( DeeNumber == 1 )
	{
	  ns1 = 1; ns2 = 5;
	  xg_sector[1-ns1] = xg_coord_sector[1-ns1];  yg_sector[1-ns1] = yg_coord_sector[1-ns1];
	  xg_sector[2-ns1] = xg_coord_sector[2-ns1];  yg_sector[2-ns1] = yg_coord_sector[2-ns1];
	  xg_sector[3-ns1] = xg_coord_sector[3-ns1];  yg_sector[3-ns1] = yg_coord_sector[3-ns1];
	  xg_sector[4-ns1] = xg_coord_sector[4-ns1];  yg_sector[4-ns1] = yg_coord_sector[4-ns1];
	  xg_sector[5-ns1] = xg_coord_sector[5-ns1];  yg_sector[5-ns1] = yg_coord_sector[5-ns1];
	}
      
      if( DeeNumber == 2 )
	{
	  ns1 = 5; ns2 = 9;
	  xg_sector[ns2-1] = xinv_d2d4-xg_coord_sector[1-1];  yg_sector[ns2-1] = yg_coord_sector[1-1];
	  xg_sector[ns2-2] = xinv_d2d4-xg_coord_sector[2-1];  yg_sector[ns2-2] = yg_coord_sector[2-1];
	  xg_sector[ns2-3] = xinv_d2d4-xg_coord_sector[3-1];  yg_sector[ns2-3] = yg_coord_sector[3-1];
	  xg_sector[ns2-4] = xinv_d2d4-xg_coord_sector[4-1];  yg_sector[ns2-4] = yg_coord_sector[4-1];
	  xg_sector[ns2-5] = xinv_d2d4-xg_coord_sector[5-1];  yg_sector[ns2-5] = yg_coord_sector[5-1];
	}
      if( DeeNumber == 3 )
	{
	  ns1 = 5; ns2 = 9;
	  xg_sector[ns2-1]= xg_coord_sector[1-1];  yg_sector[ns2-1] = yg_coord_sector[1-1];
	  xg_sector[ns2-2]= xg_coord_sector[2-1];  yg_sector[ns2-2] = yg_coord_sector[2-1];
	  xg_sector[ns2-3]= xg_coord_sector[3-1];  yg_sector[ns2-3] = yg_coord_sector[3-1];
	  xg_sector[ns2-4]= xg_coord_sector[4-1];  yg_sector[ns2-4] = yg_coord_sector[4-1];
	  xg_sector[ns2-5]= xg_coord_sector[5-1];  yg_sector[ns2-5] = yg_coord_sector[5-1];
	}
      if( DeeNumber == 4 )
	{
	  ns1 = 1; ns2 = 5;
	  xg_sector[1-ns1]= xinv_d2d4-xg_coord_sector[1-ns1];  yg_sector[1-ns1] = yg_coord_sector[1-ns1];
	  xg_sector[2-ns1]= xinv_d2d4-xg_coord_sector[2-ns1];  yg_sector[2-ns1] = yg_coord_sector[2-ns1];
	  xg_sector[3-ns1]= xinv_d2d4-xg_coord_sector[3-ns1];  yg_sector[3-ns1] = yg_coord_sector[3-ns1];
	  xg_sector[4-ns1]= xinv_d2d4-xg_coord_sector[4-ns1];  yg_sector[4-ns1] = yg_coord_sector[4-ns1];
	  xg_sector[5-ns1]= xinv_d2d4-xg_coord_sector[5-ns1];  yg_sector[5-ns1] = yg_coord_sector[5-ns1];
	}
      
      Color_t coul_textsector = fCnaParHistos->ColorDefinition("vert37");
      for(Int_t ns=ns1; ns<= ns2; ns++)
	{
	  xg_sector[ns-1] += fCnaParHistos->DeeOffsetX(opt_plot, DeeNumber);
	  if( DeeNumber == 1 || DeeNumber == 2 ){sprintf( f_in, "+%d", ns);}
	  if( DeeNumber == 3 || DeeNumber == 4 ){sprintf( f_in, "-%d", ns);}
	  TText *text_num_module = new TText(xg_sector[ns-1], yg_sector[ns-1], f_in);        fCnewRoot++;
	  if(opt_plot == "Dee"){text_num_module->SetTextSize(0.065);}
	  if(opt_plot == "EE" ){text_num_module->SetTextSize(0.045);}
	  text_num_module->SetTextColor(coul_textsector);
	  if( opt_plot == "Dee" ||
	      ( opt_plot == "EE" && !( (DeeNumber == 3 && ns == 5) || (DeeNumber == 1 && ns == 5) ) ) )
	    {text_num_module->Draw();}

	  // text_num_module->Delete(); text_num_module = 0;     fCdeleteRoot++;     
	}
      
      //............................ numeros des dee's
      ngmax = 4;
      Float_t  xg_coord_dee[4] = { 0,  0,  0,  0};
      Float_t  yg_coord_dee[4] = {48, 48, 48, 48};
      
      xg_coord_dee[DeeNumber-1] = coefcc_x*xg_coord_dee[DeeNumber-1];
      yg_coord_dee[DeeNumber-1] = coefcc_y*yg_coord_dee[DeeNumber-1];

      Float_t  xg_dee = xg_coord_dee[DeeNumber-1];
      Float_t  yg_dee = yg_coord_dee[DeeNumber-1];

      Color_t coul_textdee = fCnaParHistos->ColorDefinition("noir");
   
      xg_dee += fCnaParHistos->DeeOffsetX(opt_plot, DeeNumber)
	+ fCnaParHistos->DeeNumberOffsetX(opt_plot, DeeNumber);

      if( DeeNumber == 1 ){sprintf( f_in, "D1");}
      if( DeeNumber == 2 ){sprintf( f_in, "D2");}
      if( DeeNumber == 3 ){sprintf( f_in, "D3");}
      if( DeeNumber == 4 ){sprintf( f_in, "D4");}

      TText *text_num_module = new TText(xg_dee, yg_dee, f_in);        fCnewRoot++;
      if( opt_plot == "EE" ){text_num_module->SetTextSize(0.045);}
      if( opt_plot == "Dee"){text_num_module->SetTextSize(0.085);}
      text_num_module->SetTextColor(coul_textdee);
      text_num_module->Draw();
    }

  //..................................... Numeros des Dee et indication EE+- (option "EE" seulement)
  if( opt_plot == "EE" )
    {
      //............................ indication EE+-
      ngmax = 4;
      Float_t  xg_coord_eepm[4] = { 0,  0,  0,  0};
      Float_t  yg_coord_eepm[4] = {95, 95, 95, 95};
     
      xg_coord_eepm[DeeNumber-1] = coefcc_x*xg_coord_eepm[DeeNumber-1];
      yg_coord_eepm[DeeNumber-1] = coefcc_y*yg_coord_eepm[DeeNumber-1];

      Float_t  xg_eepm = xg_coord_eepm[DeeNumber-1];
      Float_t  yg_eepm = yg_coord_eepm[DeeNumber-1];

      Color_t coul_texteepm = fCnaParHistos->ColorDefinition("noir");
   
      xg_eepm += fCnaParHistos->DeeOffsetX(opt_plot, DeeNumber)
	+ fCnaParHistos->DeeNameOffsetX(DeeNumber);

      if( DeeNumber == 1 ){sprintf( f_in, "EE+F");}
      if( DeeNumber == 2 ){sprintf( f_in, "EE+N");}
      if( DeeNumber == 3 ){sprintf( f_in, "EE-N");}
      if( DeeNumber == 4 ){sprintf( f_in, "EE-F");}

      TText *text_num_eepm = new TText(xg_eepm, yg_eepm, f_in);        fCnewRoot++;
      text_num_eepm->SetTextSize(0.04);
      text_num_eepm->SetTextColor(coul_texteepm);
      text_num_eepm->Draw();
    }

  //......................... mention "viewed from IP"
  Color_t coul_textfromIP = fCnaParHistos->ColorDefinition("rouge49");
  sprintf( f_in, "viewed from IP");
  Float_t x_from_ip = 15.;
  Float_t y_from_ip = -10.;
  if( opt_plot == "EE" ){y_from_ip = -16.;}
  x_from_ip = coefcc_x*x_from_ip;
  y_from_ip = coefcc_x*y_from_ip;
  if( opt_plot == "EE" && DeeNumber == 3 ){x_from_ip += 1.4*fCnaParHistos->DeeOffsetX(opt_plot, DeeNumber);}
  TText *text_from_ip = new TText(x_from_ip, y_from_ip, f_in);        fCnewRoot++;
  text_from_ip->SetTextSize(0.045);
  if( opt_plot == "EE" ){text_from_ip->SetTextSize(0.035);}
  text_from_ip->SetTextColor(coul_textfromIP);
  if( opt_plot == "Dee" || (opt_plot == "EE" && DeeNumber == 3) ){text_from_ip->Draw();}

  delete [] f_in;      f_in = 0;                                 fCdelete++;

}  // ------ end of EEDataSectors() ------

//==========================================================================================

void TEcnaHistos::EEGridAxis(const Float_t& coefcc_x,  const Float_t& coefcc_y,
			     const Int_t&   DeeNumber, const TString& opt_plot,  const TString& c_option)
{
  //------------------ trace axes en IX et IY --------------- EEGridAxis
  //=============================================================================== Axe IX
  Int_t size_IX_dee  = fEcal->MaxSCIXInDee();

  Double_t IX_min = fEcalNumbering->GetIIXMin(1) - 0.5;                        // IX_min = 0.5  pour les 4 dee's
  Double_t IX_max = fEcalNumbering->GetIIXMax()*fEcal->MaxCrysIXInSC() + 0.5;  // IX_max = 50.5 pour les 4 dee's

  Int_t MatSize = 1;
  if( opt_plot == "Dee" && c_option == "corcc" )
    {
      MatSize = fEcal->MaxCrysInSC();
      IX_min = fEcalNumbering->GetIIXMin() - 0.5;
      IX_max = fEcalNumbering->GetIIXMax() + 0.5;
    }
  if( opt_plot == "EE"  && c_option == "corcc" ){return;}     // => a voir...

  if( opt_plot == "Dee" && c_option != "corcc" ){MatSize = fEcal->MaxCrysIXInSC();}
  if( opt_plot == "EE"  && c_option != "corcc" ){MatSize = 1;}

  TString  x_var_name  = " ";

  Float_t axis_x_inf  = 0;
  Float_t axis_x_sup  = 0;
  Float_t axis_y_inf  = 0;
  Float_t axis_y_sup  = 0;
  Int_t   axis_nb_div = 205;   // DEFAULT: option "EE"
  Double_t IX_values_min = 0;
  Double_t IX_values_max = 0;
  Option_t* chopt = "C";

  //........................................................................EEGridAxis
  if( DeeNumber == 1 ) //  xmin -> xmax <=> right->left
    {
      //.....axis min->max/left->right: first draw axis with -ticksize and no label
      axis_x_inf    = size_IX_dee*MatSize;
      axis_x_sup    = 0;
      axis_y_inf    = 0; 
      axis_y_sup    = 0;
      IX_values_min = -IX_max;   // -50.5 right
      IX_values_max = -IX_min;   // - 0.5 left
      if( opt_plot == "Dee" ){x_var_name = GetIXIYAxisTitle("iIXDee1");}
      if( opt_plot == "EE"  ){x_var_name = GetIXIYAxisTitle("iIXEE");}
      if( opt_plot == "Dee" ){axis_nb_div = size_IX_dee;} 
      chopt         = "-CSU";
    }
  if( DeeNumber == 2 ) //  xmin -> xmax <=> right->left
    {
      //.....axis min->max/left->right: first draw axis with -ticksize and no label
      axis_x_inf    = size_IX_dee*MatSize;
      axis_x_sup    = 0;
      axis_y_inf    = 0;
      axis_y_sup    = 0;
      IX_values_min = IX_min;   // + 0.5 right
      IX_values_max = IX_max;   // +50.5 left
      if( opt_plot == "Dee" ){x_var_name = GetIXIYAxisTitle("iIXDee2");}
      if( opt_plot == "EE"  ){x_var_name = " ";}
      if( opt_plot == "Dee" ){axis_nb_div = size_IX_dee;}  
      chopt         = "-CSU";
    }
  if( DeeNumber == 3 )  //  xmin -> xmax <=> left->right
    {
      axis_x_inf    = 0;
      axis_x_sup    = size_IX_dee*MatSize;
      axis_y_inf    = 0; 
      axis_y_sup    = 0;
      IX_values_min = IX_min;   // + 0.5 left
      IX_values_max = IX_max;   // +50.5 right
      if( opt_plot == "Dee" ){x_var_name = GetIXIYAxisTitle("iIXDee3");}
      if( opt_plot == "EE"  ){x_var_name = " ";}
      if( opt_plot == "Dee" ){axis_nb_div = size_IX_dee;} 
      chopt         = "CS";
    }
  if( DeeNumber == 4 )  //  xmin -> xmax <=> left->right
    {
      axis_x_inf    = 0;
      axis_x_sup    = size_IX_dee*MatSize;
      axis_y_inf    = 0; 
      axis_y_sup    = 0;
      IX_values_min = -IX_max;   // -50.5 left
      IX_values_max = -IX_min;   // - 0.5 right
      if( opt_plot == "Dee" ){x_var_name = GetIXIYAxisTitle("iIXDee4");}
      if( opt_plot == "EE"  ){x_var_name = " ";}
      if( opt_plot == "Dee" ){axis_nb_div = size_IX_dee;} 
      chopt         = "CS"; 
    }

  //.................................................................... EEGridAxis
  axis_x_inf += fCnaParHistos->DeeOffsetX(opt_plot, DeeNumber);
  axis_x_sup += fCnaParHistos->DeeOffsetX(opt_plot, DeeNumber);

  TGaxis* sup_axis_x = 0;
  sup_axis_x = new TGaxis(axis_x_inf, axis_y_inf, axis_x_sup, axis_y_sup,
			  IX_values_min, IX_values_max, axis_nb_div, chopt, 0.);   fCnewRoot++;

  Float_t tit_siz_x = fCnaParHistos->AxisTitleSize();
  Float_t lab_siz_x = fCnaParHistos->AxisLabelSize();

  Float_t tic_siz_x = fCnaParHistos->AxisTickSize("Deex");
  if(opt_plot == "EE"){tic_siz_x = fCnaParHistos->AxisTickSize("EEx");}

  Float_t tit_off_x = fCnaParHistos->AxisTitleOffset("Deex");
  if(opt_plot == "EE"){tit_off_x = fCnaParHistos->AxisTitleOffset("EEx");}

  Float_t lab_off_x = fCnaParHistos->AxisLabelOffset("Deex");
  if(opt_plot == "EE"){lab_off_x = fCnaParHistos->AxisLabelOffset("EEx");}

  sup_axis_x->SetTitle(x_var_name);
  sup_axis_x->SetTitleSize(tit_siz_x);
  sup_axis_x->SetTitleOffset(tit_off_x);
  sup_axis_x->SetLabelSize(lab_siz_x);
  sup_axis_x->SetLabelOffset(lab_off_x);
  sup_axis_x->SetTickSize(tic_siz_x);
  sup_axis_x->Draw("SAME");

  //.....axis min->max/right->left: redraw axis with ticksize = 0 and with -labelOffset
  if( DeeNumber == 1 || DeeNumber == 2 )
    {
      chopt = "CS";
      TGaxis* sup_axis_x_bis = 0;
      sup_axis_x_bis = new TGaxis(axis_x_inf, axis_y_inf, axis_x_sup, axis_y_sup,
				  IX_values_min, IX_values_max, axis_nb_div, chopt, 0.);   fCnewRoot++;
      sup_axis_x_bis->SetTickSize(0.);
      lab_siz_x = sup_axis_x->GetLabelSize();
      sup_axis_x_bis->SetLabelSize(lab_siz_x);
      lab_off_x = sup_axis_x->GetLabelOffset();
      sup_axis_x_bis->SetLabelOffset(-lab_off_x);
      sup_axis_x_bis->Draw("SAME");
    }

  //================================================================== Axe IY  EEGridAxis

  if( opt_plot == "Dee" || (opt_plot == "EE" && DeeNumber == 4) )
    {
      Int_t size_IY_dee  = fEcal->MaxSCIYInDee();
      Int_t size_IY_axis = size_IY_dee;

      if( opt_plot == "Dee" ){axis_nb_div = size_IY_axis;}
      if( opt_plot == "EE"  ){axis_nb_div = 210;}

      Double_t jIY_min = fEcalNumbering->GetJIYMin(DeeNumber, 1) - 0.5;
      Double_t jIY_max = fEcalNumbering->GetJIYMax(DeeNumber)*fEcal->MaxCrysIYInSC() + 0.5;

      TString  jy_var_name  = " ";
      TString  jy_direction = "x";

      Float_t tit_siz_y = fCnaParHistos->AxisTitleSize();
      Float_t lab_siz_y = fCnaParHistos->AxisLabelSize();

      Float_t tic_siz_y = fCnaParHistos->AxisTickSize("Deey");
      if(opt_plot == "EE"){tic_siz_y = fCnaParHistos->AxisTickSize("EEy");}

      Float_t tit_off_y = fCnaParHistos->AxisTitleOffset("Deey");
      if(opt_plot == "EE"){tit_off_y = fCnaParHistos->AxisTitleOffset("EEy");}

      Float_t lab_off_y = fCnaParHistos->AxisLabelOffset("Deey");
      if(opt_plot == "EE"){lab_off_y = fCnaParHistos->AxisLabelOffset("EEy");}

      TGaxis* axis_jy_plus = 0;
      axis_jy_plus = new TGaxis((Float_t)0., (Float_t)0.,
				(Float_t)0., (Float_t)(size_IY_axis*MatSize),
				jIY_min, jIY_max, axis_nb_div, "SC", 0.);   fCnewRoot++;

      jy_var_name  = GetIXIYAxisTitle("jIYDee");  
      axis_jy_plus->SetTitle(jy_var_name);
      axis_jy_plus->SetTitleSize(tit_siz_y);
      axis_jy_plus->SetTitleOffset(tit_off_y);
      axis_jy_plus->SetLabelSize(lab_siz_y);
      axis_jy_plus->SetLabelOffset(lab_off_y);
      axis_jy_plus->SetTickSize(tic_siz_y);
      axis_jy_plus->Draw("SAME");
    }

//---------------------------------- 2 axes (0,50) et (0,-50)
#define IYAX
#ifndef IYAX
  if( opt_plot == "Dee" || (opt_plot == "EE" && DeeNumber == 4) )
    {
      Int_t size_IY_dee  = fEcal->MaxSCIYInDee();
      Int_t size_IY_axis = size_IY_dee/2;

      if( opt_plot == "Dee" ){axis_nb_div = (Int_t)size_IY_axis;}
      if( opt_plot == "EE"  ){axis_nb_div = 5;}

      Double_t jIY_min = fEcalNumbering->GetJIYMin(DeeNumber, 1) - 0.5;
      Double_t jIY_max = (fEcalNumbering->GetJIYMax(DeeNumber)/2)*fEcal->MaxCrysIYInSC() + 0.5;

      TString  jy_var_name  = " ";
      TString  jy_direction = "x";

      Float_t tit_siz_y = fCnaParHistos->AxisTitleSize();
      Float_t lab_siz_y = fCnaParHistos->AxisLabelSize();

      Float_t tic_siz_y = fCnaParHistos->AxisTickSize("Deey");
      if(opt_plot == "EE"){tic_siz_y = fCnaParHistos->AxisTickSize("EEy");}

      Float_t tit_off_y = fCnaParHistos->AxisTitleOffset("Deey");
      if(opt_plot == "EE"){tit_off_y = fCnaParHistos->AxisTitleOffset("EEy");}

      Float_t lab_off_y = fCnaParHistos->AxisLabelOffset("Deey");
      if(opt_plot == "EE"){lab_off_y = fCnaParHistos->AxisLabelOffset("EEy");}

      TGaxis* axis_jy_plus = 0;
      axis_jy_plus = new TGaxis((Float_t)0., (Float_t)(size_IY_dee*MatSize/2),
				(Float_t)0., (Float_t)(2*size_IY_dee*MatSize/2),
				jIY_min, jIY_max, axis_nb_div, "SC", 0.);   fCnewRoot++;

      jy_var_name  = GetIXIYAxisTitle("jIYDee");  
      axis_jy_plus->SetTitle(jy_var_name);
      axis_jy_plus->SetTitleSize(tit_siz_y);
      axis_jy_plus->SetTitleOffset(tit_off_y);
      axis_jy_plus->SetLabelSize(lab_siz_y);
      axis_jy_plus->SetLabelOffset(lab_off_y);
      axis_jy_plus->SetTickSize(tic_siz_y);
      axis_jy_plus->Draw("SAME");

      TGaxis* axis_jy_minus = 0;
      axis_jy_minus = new TGaxis((Float_t)0., (Float_t)(size_IY_dee*MatSize/2),
				 (Float_t)0., (Float_t)0.,
				 -jIY_min, -jIY_max, axis_nb_div, "-SC", 0.);   fCnewRoot++;

      jy_var_name  = GetIXIYAxisTitle("jIYDee");  
      axis_jy_minus->SetTitle(jy_var_name);
      axis_jy_minus->SetTitleSize(tit_siz_y);
      axis_jy_minus->SetTitleOffset(tit_off_y);
      axis_jy_minus->SetLabelSize(lab_siz_y);
      axis_jy_minus->SetLabelOffset(lab_off_y);
      axis_jy_minus->SetTickSize(tic_siz_y);
      axis_jy_minus->Draw("SAME");
    }
#endif // IYAX

} // ------------- end of EEGridAxis(...) --------------

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//                               ViewHisto(***)
// 
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//======================== D_MSp_SpNb
void TEcnaHistos::XtalSamplesEv(const TVectorD& arg_read_histo, const Int_t& arg_AlreadyRead,
				const Int_t&    n1StexStin,     const Int_t& i0StinEcha)
{XtalSamplesEv(arg_read_histo, arg_AlreadyRead, n1StexStin, i0StinEcha, "ONLYONE");}
void TEcnaHistos::XtalSamplesEv(const TVectorD& arg_read_histo, const Int_t& arg_AlreadyRead,
				const Int_t&    n1StexStin,     const Int_t& i0StinEcha,
				const TString&   PlotOption)
{
  if( fFapStexNumber > 0 )
    {
      if( PlotOption == fAllXtalsInStinPlot )
	{
	  Int_t StexStin_A = n1StexStin;
	  if( fFlagSubDet == "EE" )
	    {StexStin_A = fEcalNumbering->Get1DeeSCEcnaFromDeeSCCons(fFapStexNumber, n1StexStin);}

	  Bool_t aOKData = kFALSE;
	  TVectorD read_histo(fEcal->MaxCrysInStin()*fEcal->MaxSampADC());

	  if( arg_AlreadyRead == fTobeRead )
	    {
	      fMyRootFile->PrintNoComment();
	      fMyRootFile->FileParameters(fFapAnaType,          fFapNbOfSamples,
					  fFapRunNumber,        fFapFirstReqEvtNumber,
					  fFapLastReqEvtNumber, fFapReqNbOfEvts,
					  fFapStexNumber,       fCfgResultsRootFilePath.Data());
	      
	      if ( fMyRootFile->LookAtRootFile() == kTRUE )
		{
		  fStatusFileFound = kTRUE;
		  read_histo = fMyRootFile->ReadSampleMeans(StexStin_A, fEcal->MaxCrysInStin()*fEcal->MaxSampADC());
		  if( fMyRootFile->DataExist() == kTRUE ){fStatusDataExist = kTRUE;}
		}
	      else
		{
		  fStatusFileFound = kFALSE;
		  cout << "!TEcnaHistos::XtalSamplesEv(...)> Data not available (ROOT file not found)." << endl;
		}
	      if( fStatusFileFound == kTRUE && fStatusDataExist == kTRUE ){aOKData = kTRUE;}
	    }
	  if( arg_AlreadyRead >= 1 )
	    {
	      for(Int_t i=0; i<fEcal->MaxCrysInStin()*fEcal->MaxSampADC(); i++){read_histo[i] = arg_read_histo[i];}
	      fStatusDataExist = kTRUE;
	      aOKData = kTRUE;
	    }

	  if( aOKData == kTRUE )
	    {
	      TVectorD read_histo_samps(fFapNbOfSamples);
	      
	      Int_t xAlreadyRead = 1;
	      for( Int_t i0_stin_echa=0; i0_stin_echa<fEcal->MaxCrysInStin(); i0_stin_echa++)
		{
		  if( fFapStexName == "SM" )
		    {cout << "*TEcnaHistos::XtalSamplesEv(...)> channel " << setw(2) << i0_stin_echa << ": ";}
		  if( fFapStexName == "Dee" )
		    {cout << "*TEcnaHistos::XtalSamplesEv(...)> Xtal " << setw(2) << i0_stin_echa+1 << ": ";}
		  
		  for( Int_t i0_samp=0; i0_samp<fFapNbOfSamples; i0_samp++ )
		    {
		      read_histo_samps(i0_samp) = read_histo(i0_stin_echa*fFapNbOfSamples+i0_samp);
		      cout << setprecision(4) << setw(8) << read_histo_samps(i0_samp) << ", " ;
		    }
		  cout << endl;
		  ViewHisto(read_histo_samps, xAlreadyRead,
			    StexStin_A, i0_stin_echa, fZerv, "D_MSp_SpNb", fAllXtalsInStinPlot);
		  xAlreadyRead++;
		}
	      xAlreadyRead = 0;
	    }
	  else
	    {
	      cout << "!TEcnaHistos::XtalSamplesEv(...)> Data not available." << endl;
	    }
	}
      
      if( !(PlotOption == fAllXtalsInStinPlot) )      
	{
	  Int_t StexStin_A = n1StexStin;
	  if( fFlagSubDet == "EE" )
	    {StexStin_A = fEcalNumbering->Get1DeeSCEcnaFromDeeSCCons(fFapStexNumber, n1StexStin);}
	  ViewHisto(fReadHistoDummy, fTobeRead, StexStin_A, i0StinEcha, fZerv, "D_MSp_SpNb", PlotOption);
	}
    }
  else
    {
      cout << "!TEcnaHistos::XtalSamplesEv(...)> " << fFapStexName.Data() << " number = " << fFapStexNumber
	   << " out of range (range = [1," << fEcal->MaxStexInStas() << "])" << fTTBELL << endl;
    }
}

//======================== D_MSp_SpDs
void TEcnaHistos::EvSamplesXtals(const TVectorD& arg_read_histo, const Int_t& arg_AlreadyRead,
				 const Int_t&    n1StexStin,     const Int_t& i0StinEcha)
{EvSamplesXtals(arg_read_histo, arg_AlreadyRead, n1StexStin, i0StinEcha, "ONLYONE");}
void TEcnaHistos::EvSamplesXtals(const TVectorD& arg_read_histo, const Int_t& arg_AlreadyRead,
				 const Int_t&    n1StexStin,     const Int_t& i0StinEcha,
				 const TString&   PlotOption)
{
  if( fFapStexNumber > 0 )
    {
      if( PlotOption == fAllXtalsInStinPlot )
	{
	  Int_t StexStin_A = n1StexStin;
	  if( fFlagSubDet == "EE" )
	    {StexStin_A = fEcalNumbering->Get1DeeSCEcnaFromDeeSCCons(fFapStexNumber, n1StexStin);}

	  Bool_t aOKData = kFALSE;
	  TVectorD read_histo(fEcal->MaxCrysInStin()*fEcal->MaxSampADC());

	  if( arg_AlreadyRead == fTobeRead )
	    {
	      fMyRootFile->PrintNoComment();
	      fMyRootFile->FileParameters(fFapAnaType,          fFapNbOfSamples,
					  fFapRunNumber,        fFapFirstReqEvtNumber,
					  fFapLastReqEvtNumber, fFapReqNbOfEvts,
					  fFapStexNumber,       fCfgResultsRootFilePath.Data());

	      if ( fMyRootFile->LookAtRootFile() == kTRUE )
		{
		  fStatusFileFound = kTRUE;
		  read_histo = fMyRootFile->ReadSampleMeans(StexStin_A, fEcal->MaxCrysInStin()*fEcal->MaxSampADC());
		  if( fMyRootFile->DataExist() == kTRUE ){fStatusDataExist = kTRUE;}
		}
	      else
		{
		  fStatusFileFound = kFALSE;
		  cout << "!TEcnaHistos::EvSamplesXtals(...)> Data not available (ROOT file not found)." << endl;
		}
	      if( fStatusFileFound == kTRUE && fStatusDataExist == kTRUE ){aOKData = kTRUE;}
	    }
	  if( arg_AlreadyRead >= 1 )
	    {
	      for(Int_t i=0; i<fEcal->MaxCrysInStin()*fEcal->MaxSampADC(); i++){read_histo[i] = arg_read_histo[i];}
	      fStatusDataExist = kTRUE;
	      aOKData = kTRUE;
	    }
	  if( aOKData == kTRUE )
	    {
	      TVectorD read_histo_samps(fFapNbOfSamples);
	      
	      Int_t xAlreadyRead = 1;
	      for( Int_t i0_stin_echa=0; i0_stin_echa<fEcal->MaxCrysInStin(); i0_stin_echa++)
		{
		  if( fFapStexName == "SM" )
		    {cout << "*TEcnaHistos::EvSamplesXtals(...)> channel " << setw(2) << i0_stin_echa << ": ";}
		  if( fFapStexName == "Dee" )
		    {cout << "*TEcnaHistos::EvSamplesXtals(...)> Xtal " << setw(2) << i0_stin_echa+1 << ": ";}
		  
		  for( Int_t i0_samp=0; i0_samp<fFapNbOfSamples; i0_samp++ )
		    {
		      read_histo_samps(i0_samp) = read_histo(i0_stin_echa*fFapNbOfSamples+i0_samp);
		      cout << setprecision(4) << setw(8) << read_histo_samps(i0_samp) << ", " ;
		    }
		  cout << endl;
		  ViewHisto(read_histo_samps, xAlreadyRead,
			    StexStin_A, i0_stin_echa, fZerv, "D_MSp_SpDs", fAllXtalsInStinPlot);
		  xAlreadyRead++;
		}
	      xAlreadyRead = 0;
	    }
	  else
	    {
	      cout << "!TEcnaHistos::EvSamplesXtals(...)> Data not available." << endl;
	    }
	}
      
      if( !(PlotOption == fAllXtalsInStinPlot) )      
	{
	  Int_t StexStin_A = n1StexStin;
	  if( fFlagSubDet == "EE" )
	    {StexStin_A = fEcalNumbering->Get1DeeSCEcnaFromDeeSCCons(fFapStexNumber, n1StexStin);}
	  ViewHisto(fReadHistoDummy, fTobeRead, StexStin_A, i0StinEcha, fZerv, "D_MSp_SpDs", PlotOption);
	}
    }
  else
    {
      cout << "!TEcnaHistos::EvSamplesXtals(...)> " << fFapStexName.Data() << " number = " << fFapStexNumber
	   << " out of range (range = [1," << fEcal->MaxStexInStas() << "])" << fTTBELL << endl;
    }
} // end of EvSamplesXtals(...)

//======================== D_SSp_SpNb
void TEcnaHistos::XtalSamplesSigma(const TVectorD& arg_read_histo, const Int_t& arg_AlreadyRead,
				   const Int_t&    n1StexStin,     const Int_t& i0StinEcha)
{XtalSamplesSigma(arg_read_histo, arg_AlreadyRead, n1StexStin, i0StinEcha, "ONLYONE");}
void TEcnaHistos::XtalSamplesSigma(const TVectorD& arg_read_histo, const Int_t& arg_AlreadyRead,
				   const Int_t&    n1StexStin,     const Int_t& i0StinEcha,
				   const TString&   PlotOption)
{
  if( fFapStexNumber > 0 )
    {  
      if( PlotOption == fAllXtalsInStinPlot )
	{
	  Int_t StexStin_A = n1StexStin;
	  if( fFlagSubDet == "EE" )
	    {StexStin_A = fEcalNumbering->Get1DeeSCEcnaFromDeeSCCons(fFapStexNumber, n1StexStin);}
	  
	  Bool_t aOKData = kFALSE;
	  TVectorD read_histo(fEcal->MaxCrysInStin()*fEcal->MaxSampADC());
	  
	  if( arg_AlreadyRead == fTobeRead )
	    {
	      fMyRootFile->PrintNoComment();
	      fMyRootFile->FileParameters(fFapAnaType,          fFapNbOfSamples,
					  fFapRunNumber,        fFapFirstReqEvtNumber,
					  fFapLastReqEvtNumber, fFapReqNbOfEvts,
					  fFapStexNumber,       fCfgResultsRootFilePath.Data());
	      
	      if ( fMyRootFile->LookAtRootFile() == kTRUE )
		{
		  fStatusFileFound = kTRUE;
		  read_histo = fMyRootFile->ReadSampleSigmas(StexStin_A, fEcal->MaxCrysInStin()*fEcal->MaxSampADC());
		  if( fMyRootFile->DataExist() == kTRUE ){fStatusDataExist = kTRUE;}
		}
	      else
		{
		  fStatusFileFound = kFALSE;
		  cout << "!TEcnaHistos::XtalSamplesSigma(...)> Data not available (ROOT file not found)." << endl;
		}
	      if( fStatusFileFound == kTRUE && fStatusDataExist == kTRUE ){aOKData = kTRUE;}
	    }
	  if( arg_AlreadyRead >= 1 )
	    {
	      for(Int_t i=0; i<fEcal->MaxCrysInStin()*fEcal->MaxSampADC(); i++){read_histo[i] = arg_read_histo[i];}
	      fStatusDataExist = kTRUE;
	      aOKData = kTRUE;
	    }
	  if( aOKData == kTRUE )
	    {
	      TVectorD read_histo_samps(fFapNbOfSamples);
	      
	      Int_t xAlreadyRead = 1;
	      for( Int_t i0_stin_echa=0; i0_stin_echa<fEcal->MaxCrysInStin(); i0_stin_echa++)
		{
		  if( fFapStexName == "SM" )
		    {cout << "*TEcnaHistos::XtalSamplesSigma(...)> channel " << setw(2) << i0_stin_echa << ": ";}
		  if( fFapStexName == "Dee" )
		    {cout << "*TEcnaHistos::XtalSamplesSigma(...)> Xtal " << setw(2) << i0_stin_echa+1 << ": ";}
		  
		  for( Int_t i0_samp=0; i0_samp<fFapNbOfSamples; i0_samp++ )
		    {
		      read_histo_samps(i0_samp) = read_histo(i0_stin_echa*fFapNbOfSamples+i0_samp);
		      cout << setprecision(3) << setw(6) << read_histo_samps(i0_samp) << ", " ;
		    }
		  cout << endl;
		  ViewHisto(read_histo_samps, xAlreadyRead,
			    StexStin_A, i0StinEcha, fZerv, "D_SSp_SpNb", fAllXtalsInStinPlot);
		  xAlreadyRead++;    
		}
	      xAlreadyRead = 0;
	    }
	  else
	    {
	      cout << "!TEcnaHistos::XtalSamplesSigma(...)> Data not available." << endl;
	    }
	}

      if( !(PlotOption == fAllXtalsInStinPlot) )      
	{
	  Int_t StexStin_A = n1StexStin;
	  if( fFlagSubDet == "EE" )
	    {StexStin_A = fEcalNumbering->Get1DeeSCEcnaFromDeeSCCons(fFapStexNumber, n1StexStin);}
	  ViewHisto(fReadHistoDummy, fTobeRead, StexStin_A, i0StinEcha, fZerv, "D_SSp_SpNb", PlotOption);
	} 
    }
  else
    {
      cout << "!TEcnaHistos::XtalSamplesSigma(...)> " << fFapStexName.Data() << " number = " << fFapStexNumber
	   << " out of range (range = [1," << fEcal->MaxStexInStas() << "])" << fTTBELL << endl;
    }
}


//======================== D_SSp_SpDs
void TEcnaHistos::SigmaSamplesXtals(const TVectorD& arg_read_histo, const Int_t& arg_AlreadyRead,
				    const Int_t&    n1StexStin,     const Int_t& i0StinEcha)
{SigmaSamplesXtals(arg_read_histo, arg_AlreadyRead, n1StexStin, i0StinEcha, "ONLYONE");}
void TEcnaHistos::SigmaSamplesXtals(const TVectorD& arg_read_histo, const Int_t& arg_AlreadyRead,
				    const Int_t&    n1StexStin,     const Int_t& i0StinEcha,
				    const TString&   PlotOption)
{
  if( fFapStexNumber > 0 )
    {  
      if( PlotOption == fAllXtalsInStinPlot )
	{
	  Int_t StexStin_A = n1StexStin;
	  if( fFlagSubDet == "EE" )
	    {StexStin_A = fEcalNumbering->Get1DeeSCEcnaFromDeeSCCons(fFapStexNumber, n1StexStin);}
	  	  
	  Bool_t aOKData = kFALSE;
	  TVectorD read_histo(fEcal->MaxCrysInStin()*fEcal->MaxSampADC());
	  
	  if( arg_AlreadyRead == fTobeRead )
	    {
	      fMyRootFile->PrintNoComment();
	      fMyRootFile->FileParameters(fFapAnaType,          fFapNbOfSamples,
					  fFapRunNumber,        fFapFirstReqEvtNumber,
					  fFapLastReqEvtNumber, fFapReqNbOfEvts,
					  fFapStexNumber,       fCfgResultsRootFilePath.Data());
	      if ( fMyRootFile->LookAtRootFile() == kTRUE )
		{
		  fStatusFileFound = kTRUE;
		  read_histo = fMyRootFile->ReadSampleSigmas(StexStin_A, fEcal->MaxCrysInStin()*fEcal->MaxSampADC());
		  if( fMyRootFile->DataExist() == kTRUE ){fStatusDataExist = kTRUE;}
		}
	      else
		{
		  fStatusFileFound = kFALSE;
		  cout << "!TEcnaHistos::SigmaSamplesXtals(...)> Data not available (ROOT file not found)." << endl;
		}
	      if( fStatusFileFound == kTRUE && fStatusDataExist == kTRUE ){aOKData = kTRUE;}
	    }
	  
	  if( arg_AlreadyRead >= 1 )
	    {
	      for(Int_t i=0; i<fEcal->MaxCrysInStin()*fEcal->MaxSampADC(); i++){read_histo[i] = arg_read_histo[i];}
	      fStatusDataExist = kTRUE;
	      aOKData = kTRUE;
	    }
	  if( aOKData == kTRUE )
	    {
	      TVectorD read_histo_samps(fFapNbOfSamples);
	      
	      Int_t xAlreadyRead = 1;
	      for( Int_t i0_stin_echa=0; i0_stin_echa<fEcal->MaxCrysInStin(); i0_stin_echa++)
		{
		  if( fFapStexName == "SM" )
		    {cout << "*TEcnaHistos::SigmaSamplesXtals(...)> channel " << setw(2) << i0_stin_echa << ": ";}
		  if( fFapStexName == "Dee" )
		    {cout << "*TEcnaHistos::SigmaSamplesXtals(...)> Xtal " << setw(2) << i0_stin_echa+1 << ": ";}
		  
		  for( Int_t i0_samp=0; i0_samp<fFapNbOfSamples; i0_samp++ )
		    {
		      read_histo_samps(i0_samp) = read_histo(i0_stin_echa*fFapNbOfSamples+i0_samp);
		      cout << setprecision(3) << setw(6) << read_histo_samps(i0_samp) << ", " ;
		    }
		  cout << endl;
		  ViewHisto(read_histo_samps, xAlreadyRead,
			    StexStin_A, i0StinEcha, fZerv, "D_SSp_SpDs", fAllXtalsInStinPlot);
		  xAlreadyRead++;    
		}
	      xAlreadyRead = 0;
	    }
	  else
	    {
	      cout << "!TEcnaHistos::SigmaSamplesXtals(...)> Data not available." << endl;
	    }
	}
      
      if( !(PlotOption == fAllXtalsInStinPlot) )      
	{
	  Int_t StexStin_A = n1StexStin;
	  if( fFlagSubDet == "EE" )
	    {StexStin_A = fEcalNumbering->Get1DeeSCEcnaFromDeeSCCons(fFapStexNumber, n1StexStin);}
	  ViewHisto(fReadHistoDummy, fTobeRead, StexStin_A, i0StinEcha, fZerv, "D_SSp_SpDs", PlotOption);
	} 
    }
  else
    {
      cout << "!TEcnaHistos::SigmaSamplesXtals(...)> " << fFapStexName.Data() << " number = " << fFapStexNumber
	   << " out of range (range = [1," << fEcal->MaxStexInStas() << "])" << fTTBELL << endl;
    }
} // end of SigmaSamplesXtals(...)

//==========================================================================================
//
//                         ViewHisto
//
//    arg_read_histo  = array containing the values
//    arg_AlreadyRead = histo flag: =1 => arg_read_histo exists,
//                                  =0 => values will be read by internal
//                                        call to TEcnaRead inside ViewHisto
//    StexStin_A      = [1,68] or [1,150]  ==> tower# if EB,  SC# if EE
//    i0StinEcha      = [0,24] = Electronic channel# in tower (if EB) or SC (if EE) 
//    i0Sample        = [0,9]  = sample#
//    HistoCode       = String for histo type (pedestal, total noise, mean cor(s,s), ...)  
//    opt_plot_arg    = String for plot option (SAME or not SAME)
//
//===========================================================================================
void TEcnaHistos::ViewHisto(const TVectorD& arg_read_histo, const Int_t&  arg_AlreadyRead,
			    const Int_t&    StexStin_A,     const Int_t&  i0StinEcha,
			    const Int_t&    i0Sample,       const TString& HistoCode,
			    const TString&   opt_plot_arg)
{
  //Histogram of the quantities (one run)

  TString opt_plot  = opt_plot_arg;
  fPlotAllXtalsInStin = 0;

  if( opt_plot_arg == fAllXtalsInStinPlot ){opt_plot = fOnlyOnePlot; fPlotAllXtalsInStin = 1;}

  TString HistoType = fCnaParHistos->GetHistoType(HistoCode.Data());

  Int_t OKHisto = 0;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Canvas already closed in option SAME %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Int_t xCanvasExists = 1; // a priori ==> Canvas exists                                   // (ViewHisto)
  if( opt_plot != fOnlyOnePlot && GetMemoFlag(HistoCode, opt_plot) == "Busy" )
    {
      TVirtualPad* main_subpad = 0; 
      //---------------- Call to ActivePad
      main_subpad = ActivePad(HistoCode.Data(), opt_plot.Data());  // => return 0 if canvas has been closed
      if( main_subpad == 0 )
	{
	  cout << "*TEcnaHistos::ViewHisto(...)> WARNING ===> Canvas has been closed in option SAME or SAME n."
	       << endl
	       << "                              Please, restart with a new canvas."
	       << fTTBELL << endl;
	  
	  ReInitCanvas(HistoCode, opt_plot);
	  xCanvasExists = 0;
	}
    }
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  //%%%%%%%%%%%%%%%%%%%%%%%% Change of X variable in option SAME n with no proj %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Int_t SameXVarMemo = 1;   //  a priori ==> SAME n option: X variable OK                     (ViewHisto)
  if( !(HistoType == "Proj" || HistoType == "SampProj" || HistoType == "EvolProj") && 
      !(arg_AlreadyRead >= 1) )
    {
      TString XVarHisto = fCnaParHistos->GetXVarHisto(HistoCode.Data(), fFlagSubDet.Data(), fFapStexNumber);
      TString YVarHisto = fCnaParHistos->GetYVarHisto(HistoCode.Data(), fFlagSubDet.Data(), fFapStexNumber);
      if( (opt_plot == fSameOnePlot ) && GetMemoFlag(HistoCode, opt_plot) == "Free" )
	{
	  SetXVarMemo(HistoCode, opt_plot, XVarHisto);  SetYVarMemo(HistoCode, opt_plot, YVarHisto); SameXVarMemo = 1;
	}
      if( (opt_plot == fSameOnePlot ) && GetMemoFlag(HistoCode, opt_plot) == "Busy" )
	{
	  TString XVariableMemo = GetXVarFromMemo(HistoCode, opt_plot);
	  TString YVariableMemo = GetYVarFromMemo(HistoCode, opt_plot);
	  
	  if( XVarHisto != XVariableMemo )
	    {
	      cout << "!TEcnaHistos::ViewHisto(...)> *** ERROR *** ===> X coordinate changed in option SAME n." << endl
		   << "                              Present  X = " << XVarHisto << endl
		   << "                              Present  Y = " << YVarHisto << endl
		   << "                              Previous X = " << XVariableMemo << endl
		   << "                              Previous Y = " << YVariableMemo 
		   << fTTBELL << endl;
	      SameXVarMemo = 0;
	    }
	  else
	    {SetYVarMemo(HistoCode, opt_plot, YVarHisto);}
	}
    }
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Change of Y variable in option SAME n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  Int_t SameYVarMemo = 1;   //  a priori ==> SAME n option: Y variable OK                     (ViewHisto)
  if( (HistoType == "Proj" || HistoType == "SampProj" || HistoType == "EvolProj") && 
      !(arg_AlreadyRead >= 1) )
    {
      TString XVarHisto = fCnaParHistos->GetXVarHisto(HistoCode.Data(), fFlagSubDet.Data(), fFapStexNumber);
      TString YVarHisto = fCnaParHistos->GetYVarHisto(HistoCode.Data(), fFlagSubDet.Data(), fFapStexNumber);
      if( (opt_plot == fSameOnePlot ) && GetMemoFlag(HistoCode, opt_plot) == "Free" )
	{
	  SetYVarMemo(HistoCode, opt_plot, YVarHisto);  SetYVarMemo(HistoCode, opt_plot, YVarHisto); SameYVarMemo = 1;
	}
      if( (opt_plot == fSameOnePlot ) && GetMemoFlag(HistoCode, opt_plot) == "Busy" )
	{
	  TString XVariableMemo = GetXVarFromMemo(HistoCode, opt_plot);
	  TString YVariableMemo = GetYVarFromMemo(HistoCode, opt_plot);
	  
	  if( YVarHisto != YVariableMemo )
	    {
	      cout << "!TEcnaHistos::ViewHisto(...)> *** ERROR *** ===> Y coordinate changed in option SAME n." << endl
		   << "                              Present  X = " << XVarHisto << endl
		   << "                              Present  Y = " << YVarHisto << endl
		   << "                              Previous X = " << XVariableMemo << endl
		   << "                              Previous Y = " << YVariableMemo 
		   << fTTBELL << endl;
	      SameYVarMemo = 0;
	    }
	  else
	    {SetYVarMemo(HistoCode, opt_plot, YVarHisto);}
	}
    }
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  //%%%%%%%%%%%%%%%%%%%%%%%%%%% Number of bins change in option SAME or SAME n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Int_t OkBinsMemoSameOne = 1;   //  a priori ==> SAME n option: Nb bins OK                     (ViewHisto)

  Int_t SizeForPlot = GetHistoSize(HistoCode.Data(), "plot");
  Int_t xNbBins = GetHistoNumberOfBins(HistoCode.Data(), SizeForPlot);  

  if( (opt_plot == fSameOnePlot || opt_plot == fSeveralPlot) && GetMemoFlag(HistoCode, opt_plot) == "Free" )
    {
      SetNbBinsMemo(HistoCode, opt_plot, xNbBins); OkBinsMemoSameOne = 1;
    }

  if( (opt_plot == fSameOnePlot || opt_plot == fSeveralPlot) && GetMemoFlag(HistoCode, opt_plot) == "Busy" )
    {
      Int_t NbBinsMemo = GetNbBinsFromMemo(HistoCode, opt_plot);
      if( xNbBins != NbBinsMemo )
	{
	  cout << "!TEcnaHistos::ViewHisto(...)> *** ERROR *** ===> Number of bins changed in option SAME or SAME n."
	       << " Present number = " << xNbBins << ", requested number = " << NbBinsMemo << fTTBELL << endl;
	  OkBinsMemoSameOne = 0;
	}
    }

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if( xCanvasExists == 1 && SameXVarMemo == 1 && SameYVarMemo == 1 && OkBinsMemoSameOne == 1 ){OKHisto = 1;}

  //======================== Histo accepted                                                       (ViewHisto)
  if( OKHisto == 1 )
    {
      Int_t opt_scale_x = fOptScaleLinx;
      if (fFlagScaleX == "LIN" ){opt_scale_x = fOptScaleLinx;}
      if (fFlagScaleX == "LOG" ){opt_scale_x = fOptScaleLogx;}

      Int_t opt_scale_y = fOptScaleLiny;
      if (fFlagScaleY == "LIN" ){opt_scale_y = fOptScaleLiny;}
      if (fFlagScaleY == "LOG" ){opt_scale_y = fOptScaleLogy;}

      fCnaParHistos->SetColorPalette(fFlagColPal);
      TString fp_name_short = " ";
  
      //-------------------- read_histo size
      Int_t SizeForRead = GetHistoSize(HistoCode.Data(), "read");

      //............................................... allocation/init_histo
      TVectorD histo_for_plot(SizeForPlot);
      for(Int_t i=0; i<SizeForPlot; i++){histo_for_plot[i]=(Double_t)0;}

      TVectorD histo_for_plot_memo(SizeForPlot);
      for(Int_t i=0; i<SizeForPlot; i++){histo_for_plot_memo[i]=(Double_t)0;}

      Int_t i_data_exist = 0;
      Int_t OKPlot = 0;

      //------------------------------------- histos Global, (Global)Proj, SampGlobal and SampProj
      if( HistoType == "Global"   || HistoType == "Proj" || HistoType == "SampGlobal" ||
	  HistoType == "SampProj" )
	{     
	  if( fFapStexNumber == 0 )
	    {
	      Bool_t ok_view_histo  = kFALSE;

	      //--------------------------------------------------------------------- Stas Histo      (ViewHisto)
	      Int_t CounterExistingFile = 0;
	      Int_t CounterDataExist = 0;

	      Int_t* xFapNbOfEvts = new Int_t[fEcal->MaxStexInStas()];     fCnew++;
	      for(Int_t i=0; i<fEcal->MaxStexInStas(); i++){xFapNbOfEvts[i]=0;}

	      //Int_t* NOFE_int = new Int_t[fEcal->MaxCrysEcnaInStex()];     fCnew++;

	      for(Int_t iStasStex=0; iStasStex<fEcal->MaxStexInStas(); iStasStex++)
		{
		  Bool_t OKFileExists   = kFALSE;
		  Bool_t ok_data_exists = kFALSE;

		  TVectorD read_histo(fEcal->MaxStinEcnaInStex());
		  for(Int_t i=0; i<fEcal->MaxStinEcnaInStex(); i++){read_histo(i)=(Double_t)0.;}

		  if( arg_AlreadyRead == 0 )
		    {
		      //----------------------------------------------------------------------------- file reading
		      fMyRootFile->PrintNoComment();
		      Int_t n1StasStex = iStasStex+1;
		      fMyRootFile->FileParameters(fFapAnaType,          fFapNbOfSamples,
						  fFapRunNumber,        fFapFirstReqEvtNumber,
						  fFapLastReqEvtNumber, fFapReqNbOfEvts,
						  n1StasStex,           fCfgResultsRootFilePath.Data());
		      
		      if( fMyRootFile->LookAtRootFile() == kTRUE ){OKFileExists = kTRUE;}            //   (ViewHisto, Stas)
		      if( OKFileExists == kTRUE )
			{
			  xFapNbOfEvts[iStasStex] = fMyRootFile->GetNumberOfEvents(fFapReqNbOfEvts, n1StasStex);
			  fp_name_short = fMyRootFile->GetRootFileNameShort();
			  // cout << "*TEcnaHistos::ViewHisto(...)> Data are analyzed from file ----> "
			  //      << fp_name_short << endl;
			  //....................... search for first and last dates
			  if( iStasStex == 0 )
			    {
			      fStartTime = fMyRootFile->GetStartTime();
			      fStopTime  = fMyRootFile->GetStopTime();
			      fStartDate = fMyRootFile->GetStartDate();
			      fStopDate  = fMyRootFile->GetStopDate();
			    }

			  time_t xStartTime = fMyRootFile->GetStartTime();
			  time_t xStopTime  = fMyRootFile->GetStopTime();
			  TString xStartDate = fMyRootFile->GetStartDate();
			  TString xStopDate  = fMyRootFile->GetStopDate();

			  if( xStartTime < fStartTime ){fStartTime = xStartTime; fStartDate = xStartDate;}
			  if( xStopTime  > fStopTime  ){fStopTime  = xStopTime;  fStopDate  = xStopDate;}

			  fRunType = fMyRootFile->GetRunType();
			  ok_view_histo =
			    GetOkViewHisto(fMyRootFile, StexStin_A, i0StinEcha, i0Sample, HistoCode.Data());

			  if( ok_view_histo == kTRUE )
			    {
			      //............................................... histo reading   (ViewHisto, Stas)
			      if( HistoCode == "D_NOE_ChNb" || HistoCode == "D_NOE_ChDs" ){
				read_histo = fMyRootFile->ReadAverageNumberOfEvents(fEcal->MaxStinEcnaInStex());}
			      if( HistoCode == "D_Ped_ChNb" || HistoCode == "D_Ped_ChDs" ){
				read_histo = fMyRootFile->ReadAveragePedestals(fEcal->MaxStinEcnaInStex());}
			      if( HistoCode == "D_TNo_ChNb" || HistoCode == "D_TNo_ChDs" ){
				read_histo = fMyRootFile->ReadAverageTotalNoise(fEcal->MaxStinEcnaInStex());}
			      if( HistoCode == "D_MCs_ChNb" || HistoCode == "D_MCs_ChDs" ){
				read_histo = fMyRootFile->ReadAverageMeanCorrelationsBetweenSamples(fEcal->MaxStinEcnaInStex());}
			      if( HistoCode == "D_LFN_ChNb" || HistoCode == "D_LFN_ChDs" ){
				read_histo = fMyRootFile->ReadAverageLowFrequencyNoise(fEcal->MaxStinEcnaInStex());}
			      if( HistoCode == "D_HFN_ChNb" || HistoCode == "D_HFN_ChDs" ){
				read_histo = fMyRootFile->ReadAverageHighFrequencyNoise(fEcal->MaxStinEcnaInStex());}
			      if( HistoCode == "D_SCs_ChNb" || HistoCode == "D_SCs_ChDs" ){
				read_histo = fMyRootFile->ReadAverageSigmaOfCorrelationsBetweenSamples(fEcal->MaxStinEcnaInStex());}
			      if( fMyRootFile->DataExist() == kTRUE ){ok_data_exists = kTRUE;}
			    }
			}
		    }
		  
		  if( arg_AlreadyRead >= 1 )
		    {
		      ok_data_exists = kTRUE;
		      for(Int_t i0Stin=0; i0Stin<fEcal->MaxStinEcnaInStex(); i0Stin++ )
			{read_histo(i0Stin) = arg_read_histo(fEcal->MaxStinEcnaInStex()*iStasStex+i0Stin);}
		    }

		  if( ok_data_exists == kTRUE )
		    {
		      fStatusFileFound = kTRUE;
		      CounterExistingFile++;


		      //...........................................................
		      if( ok_data_exists == kTRUE )
			{
			  fStatusDataExist = kTRUE;
			  CounterDataExist++;

			  for(Int_t i0StexStinEcna=0; i0StexStinEcna<fEcal->MaxStinEcnaInStex(); i0StexStinEcna++)
			    {
			      //Int_t n1StexStinEcna = i0StexStinEcna+1;
			      //-------------------------------------- Stas histo filling   (ViewHisto, Stas)
			      Int_t i_xgeo = -1;
			      //...................................... EB
			      if( fFlagSubDet == "EB" )
				{
				  i_xgeo = iStasStex*fEcal->MaxStinEcnaInStex() + i0StexStinEcna;
				  if( i_xgeo >= 0 && i_xgeo < SizeForPlot )
				    {
				      histo_for_plot[i_xgeo] = read_histo[i0StexStinEcna];
				    }
				  else
				    {
				      cout << "!TEcnaHistos::ViewHisto(...)> <EB> i_xgeo = " << i_xgeo
					   << ". OUT OF RANGE ( range = [0,"<< SizeForPlot << "] " << endl;
				    }  
				}
			      //...................................... EE    (ViewHisto)
			      //-------> Dee order: D4, D3, D2, D1		      
			      if( fFlagSubDet == "EE" )
				{
				  Int_t DeeOffset = 0;
				  Int_t DSOffset  = 0;
			      
				  Int_t DeeNumber = iStasStex+1;
				  Int_t n1DeeSCEcna = i0StexStinEcna+1;
			      
				  //................................................ Dee offset
				  if( DeeNumber == 3 ){DeeOffset +=   fEcal->MaxSCForConsInDee();}     // 149
				  if( DeeNumber == 2 ){DeeOffset += 3*fEcal->MaxSCForConsInDee()-1;}   // 446
				  if( DeeNumber == 1 ){DeeOffset += 4*fEcal->MaxSCForConsInDee()-1;}   // 595
			      
				  //................................................ Data Sector offset   (ViewHisto, Stas)
				  Int_t StexDataSector = fEcalNumbering->GetDSFrom1DeeSCEcna(DeeNumber, n1DeeSCEcna);
				  //.... returns 0 if n1DeeSCEcna corresponds to an empty "ECNA-SC"
				  
				  //................................................ SC final coordinate   (ViewHisto, Stas)
				  Int_t StexDSStin = fEcalNumbering->GetDSSCFrom1DeeSCEcna(DeeNumber, n1DeeSCEcna);
				  //--> return StexDSStin = 25 (not  3) for n1DeeSCEcna = 32
				  //--> return StexDSStin = 14 (not 21) for n1DeeSCEcna = 29
				  //--> return StexDSStin = -1 for n1DeeSCEcna = 10 and n1DeeSCEcna = 11 

				  if( StexDataSector >= 1 && StexDataSector <= 9 )
				    {
				      if( DeeNumber == 4 ) // Sectors 1,2,3,4,5a
					{
					  for(Int_t is=2; is<=5; is++)
					    { if( StexDataSector >= is )
						{Int_t ism = is-1; DSOffset += fEcalNumbering->GetMaxSCInDS(ism);}}
					}
				  
				      if( DeeNumber == 3 ) // Sectors 5b,6,7,8,9
					{
					  if( StexDataSector >= 6 )
					    {DSOffset += fEcalNumbering->GetMaxSCInDS(5)/2;}
					  for(Int_t is=7; is<=9; is++)
					    { if( StexDataSector >= is )
					      {Int_t ism = is-1; DSOffset += fEcalNumbering->GetMaxSCInDS(ism);}}
					}
				  
				      if( DeeNumber == 2 ) // Sectors 9,8,7,6,5a
					{
					  if( StexDataSector >= 6 )
					    {DSOffset -= fEcalNumbering->GetMaxSCInDS(5)/2;}
					  for(Int_t is=7; is<=9; is++)
					    {if( StexDataSector >= is )
					      {Int_t ism = is-1; DSOffset -= fEcalNumbering->GetMaxSCInDS(ism);}}
					}
				  
				      if( DeeNumber == 1 ) // Sectors 5b,4,3,2,1
					{
					  for(Int_t is=2; is<=5; is++)
					    { if( StexDataSector >= is )
					      {Int_t ism = is-1; DSOffset -= fEcalNumbering->GetMaxSCInDS(ism);}}
					}
				  
				      if( StexDSStin >=1 && StexDSStin <= fEcalNumbering->GetMaxSCInDS(StexDataSector) )
					{
					  if( DeeNumber == 4 ) // Sectors 1,2,3,4,5a
					    {
					      if(StexDataSector != 5)
						{i_xgeo = DeeOffset + DSOffset + (StexDSStin - 1);}
					      if( StexDataSector == 5)
						{i_xgeo = DeeOffset + DSOffset + (StexDSStin - 1);}
					    }
					  if( DeeNumber == 3 ) // Sectors 5b,6,7,8,9
					    {
					      if(StexDataSector != 5)
						{i_xgeo = DeeOffset + DSOffset + (StexDSStin - 1);}
					      if( StexDataSector == 5)
						{i_xgeo = DeeOffset + DSOffset + (StexDSStin-17) - 1;}
					    }
					  if( DeeNumber == 2 ) // Sectors 5a,6,7,8,9
					    {
					      if(StexDataSector != 5)
						{i_xgeo = DeeOffset + DSOffset
						   - fEcalNumbering->GetMaxSCInDS(StexDataSector) + StexDSStin;}
					      if( StexDataSector == 5)
						{i_xgeo = DeeOffset + DSOffset
						   - fEcalNumbering->GetMaxSCInDS(StexDataSector)/2 + StexDSStin;}
					    }
					  if( DeeNumber == 1 ) // Sectors 1,2,3,4,5b
					    {
					      if(StexDataSector != 5)
						{i_xgeo = DeeOffset + DSOffset
						   - fEcalNumbering->GetMaxSCInDS(StexDataSector) + StexDSStin;}
					      if( StexDataSector == 5)
						{i_xgeo = DeeOffset + DSOffset
						   - fEcalNumbering->GetMaxSCInDS(StexDataSector)/2 +(StexDSStin-17);}
					    }
					  
					}// end of if(StexDSStin >=1 && StexDSStin <= fEcalNumbering->GetMaxSCInDS(StexDataSector))
				      else
					{
					  cout << "!TEcnaHistos::ViewHisto(...)> <EE>  StexDSStin = " << StexDSStin
					       << ". OUT OF RANGE ( range = [1,"
					       << fEcalNumbering->GetMaxSCInDS(StexDataSector)
					       << "]. DeeNumber =  " << DeeNumber
					       << ", n1DeeSCEcna = " << n1DeeSCEcna
					       << ", StexDataSector = "  << StexDataSector 
					       << ", i_xgeo = "  << i_xgeo << endl;
					}
				    }// end of if( StexDataSector >= 1 && StexDataSector <= 9 )
				  else
				    {
				      //cout << "!TEcnaHistos::ViewHisto(...)> <EE>  StexDataSector = " << StexDataSector
				      //     << ". OUT OF RANGE ( range = [1,9]. DeeNumber = " << DeeNumber
				      //     << ", n1DeeSCEcna = " << n1DeeSCEcna
				      //     << ", i_xgeo = "  << i_xgeo << endl;
				    }
				  //......................................... transfert read_histo -> histo_for_plot
				  if( i_xgeo >= -1 && i_xgeo < SizeForPlot )
				    {
				      // special treatement for not connected & mixed SC's
				      if( n1DeeSCEcna ==  29 || n1DeeSCEcna ==  32 ||   //  261a, 207c, 268a, 178c 
					                                                // [ 14a,  25c,  21a,   3c]
					  n1DeeSCEcna == 144 || n1DeeSCEcna == 165 ||   //  261c, 261b [14c, 14b]
					  n1DeeSCEcna == 176 || n1DeeSCEcna == 193 ||   //  207a, 207b [25a, 25b]
					  n1DeeSCEcna ==  60 || n1DeeSCEcna == 119 ||   //  182a, 182b [30a, 30b]
					  n1DeeSCEcna == 102 || n1DeeSCEcna == 123 ||   //  268c, 268b [21c, 21b]
					  n1DeeSCEcna == 138 || n1DeeSCEcna == 157 )    //  178a, 178b [ 3a,  3b] 
					{
					  //--------------- DSSC 14
					  if( n1DeeSCEcna ==  29 && i_xgeo >= 0 )
					    {histo_for_plot[i_xgeo] += read_histo[i0StexStinEcna]/(Double_t)5.;}
					  if( (n1DeeSCEcna ==  144 || n1DeeSCEcna == 165) && i_xgeo >= 0 )
					    {histo_for_plot[i_xgeo] +=
					       read_histo[i0StexStinEcna]*(Double_t)10./(Double_t)25.;}
					  
					  //--------------- DSSC 25
					  if( n1DeeSCEcna ==  32 && i_xgeo >= 0 )
					    {histo_for_plot[i_xgeo] += read_histo[i0StexStinEcna]/(Double_t)5.;}
					  if( (n1DeeSCEcna ==  176 || n1DeeSCEcna == 193) && i_xgeo >= 0 )
					    {histo_for_plot[i_xgeo] +=
					       read_histo[i0StexStinEcna]*(Double_t)10./(Double_t)25.;}
					  
					  //--------------- DSSC 30 
					  if( (n1DeeSCEcna == 60 || n1DeeSCEcna == 119) && i_xgeo >= 0 )
					    {histo_for_plot[i_xgeo] += read_histo[i0StexStinEcna]/(Double_t)2.;}
					  
					  //--------------- DSSC 21 (Add SC translated at 10-1 only once, i_xgeo = -1 accepted)
					  if( n1DeeSCEcna == 102 )
					    {histo_for_plot[i_xgeo] += read_histo[9]/(Double_t)21.
					       + read_histo[i0StexStinEcna]*(Double_t)10./(Double_t)21.;}
					  if( n1DeeSCEcna == 123 && i_xgeo >= 0 )
					    {histo_for_plot[i_xgeo] +=
					       read_histo[i0StexStinEcna]*(Double_t)10./(Double_t)21.;}
					  
					  //--------------- DSSC 3 (Add SC translated at 11-1 only once, i_xgeo = -1 accepted)
					  if( n1DeeSCEcna == 138 )
					    {histo_for_plot[i_xgeo] += read_histo[10]/(Double_t)21.
					       + read_histo[i0StexStinEcna]*(Double_t)10./(Double_t)21.;}
					  if( n1DeeSCEcna == 157 && i_xgeo >= 0 )
					    {histo_for_plot[i_xgeo] +=
					       read_histo[i0StexStinEcna]*(Double_t)10./(Double_t)21.;}
					}
				      else
					{
					  if( i_xgeo >= 0 )
					    {histo_for_plot[i_xgeo] += read_histo[i0StexStinEcna];} // standard treatment
					}
				    } // end of if( i_xgeo >= -1 && i_xgeo < SizeForPlot )
				  else
				    {
				      //cout << "!TEcnaHistos::ViewHisto(...)> <EE>  i_xgeo = " << i_xgeo
				      //     << ". OUT OF RANGE ( range = [0,"<< SizeForPlot << "] " << endl;
				    }
				}// end of if( fFlagSubDet == "EE" )
			    }// end of for(Int_t i0StexStinEcna=0; i0StexStinEcna<fEcal->MaxStinEcnaInStex(); i0StexStinEcna++)
			}
		      else
			{
			  cout << "!TEcnaHistos::ViewHisto(...)>  "
			       << " Data not available for " << fFapStexName << " " << iStasStex+1
			       << " (Quantity not present in the ROOT file)" << endl;
			}
		    } // end of if ( fMyRootFile->LookAtRootFile() == kTRUE )   (ViewHisto/Stas)
		  else
		    {
		      fStatusFileFound = kFALSE;

		      cout << "!TEcnaHistos::ViewHisto(...)>  "
			   << " Data not available for " << fFapStexName << " " << iStasStex+1
			   << " (ROOT file not found)" << endl;
		    }

		  if( fFapNbOfEvts <= xFapNbOfEvts[iStasStex] ){fFapNbOfEvts = xFapNbOfEvts[iStasStex];}

		} // end of for(Int_t iStasStex=0; iStasStex<fEcal->MaxStexInStas(); iStasStex++)

	      //delete [] NOFE_int; NOFE_int = 0;               fCdelete++;
	      delete [] xFapNbOfEvts; xFapNbOfEvts = 0;       fCdelete++;
	    
	      if( CounterExistingFile > 0 && CounterDataExist > 0 ){OKPlot = 1;} 
	  
	    } // end of if( fFapStexNumber == 0 )
	
	  //---------------------------------------------------------------------------- (ViewHisto [Stex])

	  if( fFapStexNumber > 0 )
	    {
	      Bool_t OKFileExists  = kFALSE ;
	      Bool_t ok_view_histo = kFALSE;

	      if( arg_AlreadyRead == 0 )
		{
		  fMyRootFile->PrintNoComment();
		  fMyRootFile->FileParameters(fFapAnaType,          fFapNbOfSamples,
					      fFapRunNumber,        fFapFirstReqEvtNumber,
					      fFapLastReqEvtNumber, fFapReqNbOfEvts,
					      fFapStexNumber,       fCfgResultsRootFilePath.Data());
		  
		  if ( fMyRootFile->LookAtRootFile() == kTRUE ){OKFileExists = kTRUE;}       //   (ViewHisto, Stex)	       
		  
		  if( OKFileExists == kTRUE )
		    {
		      fFapNbOfEvts = fMyRootFile->GetNumberOfEvents(fFapReqNbOfEvts, fFapStexNumber);
		      fp_name_short = fMyRootFile->GetRootFileNameShort();
		      // cout << "*TEcnaHistos::ViewHisto(...)> Data are analyzed from file ----> "
		      //      << fp_name_short << endl;
		      
		      fStartDate = fMyRootFile->GetStartDate();
		      fStopDate  = fMyRootFile->GetStopDate();
		      fRunType   = fMyRootFile->GetRunType();

		      ok_view_histo =
			GetOkViewHisto(fMyRootFile, StexStin_A, i0StinEcha, i0Sample, HistoCode.Data());
		    }
		}
	      
	      if( arg_AlreadyRead >= 1 )
		{
		  OKFileExists = kTRUE; ok_view_histo = kTRUE;
		}
	      
	      if( OKFileExists == kTRUE ) 
		{
		  fStatusFileFound = kTRUE;
		  //---------------------------------------------------------------------------- (ViewHisto [Stex])
	      
		  if( ok_view_histo == kTRUE )
		    {
		      //------------ EB or EE with SampGlobal or SampProj (histo_for_plot = read_histo)
		      if( fFlagSubDet == "EB" || 
			  ( fFlagSubDet == "EE" && ( HistoType == "SampGlobal" || HistoType == "SampProj" )  )  )
			{
			  histo_for_plot = GetHistoValues(arg_read_histo, arg_AlreadyRead, fMyRootFile, HistoCode.Data(),
							  SizeForPlot, SizeForRead,
							  StexStin_A,  i0StinEcha, i0Sample, i_data_exist);
			  if( i_data_exist > 0 ){OKPlot = 1;}
			  if( OKPlot == 1 && opt_plot == "ASCII" && ( HistoType == "Global" || HistoType == "Proj" ) )
			    {WriteHistoAscii(HistoCode.Data(), SizeForPlot, histo_for_plot);}
			}
		  
		      //------------ EE  except for SampGlobal and SampProj) (histo_for_plot # read_histo)
		      if( fFlagSubDet == "EE" && !( HistoType == "SampGlobal" || HistoType == "SampProj" ) )
			{
			  TVectorD read_histo(SizeForRead);
			  for(Int_t i=0; i<SizeForRead; i++){read_histo(i)=(Double_t)0.;}

			  read_histo = GetHistoValues(arg_read_histo, arg_AlreadyRead, fMyRootFile, HistoCode.Data(),
						      SizeForRead, SizeForRead,
						      StexStin_A, i0StinEcha, i0Sample, i_data_exist);
			  if( i_data_exist > 0 ){OKPlot = 1;}
			  if( OKPlot == 1 && opt_plot == "ASCII" )
			    {
			      WriteHistoAscii(HistoCode.Data(), fEcal->MaxCrysEcnaInDee(), read_histo);
			    }
			  if( OKPlot == 1 && opt_plot != "ASCII" )
			    {
			      //..................... Build histo_for_plot from read_histo (ViewHisto [Stex])
			      Int_t DeeNumber = fFapStexNumber;
			      TString DeeDir  = fEcalNumbering->GetDeeDirViewedFromIP(DeeNumber);
			  
			      //%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOOP ON Echa (Ecna) %%%%%%%%%%%%%%%%%%%%%%%%%% (ViewHisto [Stex])
			      for(Int_t i0DeeEcha=0; i0DeeEcha<fEcal->MaxCrysEcnaInDee(); i0DeeEcha++)
				{
				  Int_t n1SCEcha    = fEcalNumbering->Get1SCEchaFrom0DeeEcha(i0DeeEcha);
				  Int_t n1DeeSCEcna = i0DeeEcha/fEcal->MaxCrysInSC()+1;
			      
				  Int_t DataSector = fEcalNumbering->GetDSFrom1DeeSCEcna(DeeNumber, n1DeeSCEcna);
				  Int_t SC_in_DS   = fEcalNumbering->GetDSSCFrom1DeeSCEcna(DeeNumber, n1DeeSCEcna, n1SCEcha);
			      
				  Int_t i_xgeo = -1;		      
			      
				  if( n1SCEcha >= 1 && n1SCEcha <= fEcal->MaxCrysInSC() )
				    {
				      if( n1DeeSCEcna >= 1 && n1DeeSCEcna <= fEcal->MaxSCEcnaInDee() )
					{
					  if( DataSector >= 1 && DataSector <= 9 )
					    {
					      if( SC_in_DS >= 1 && SC_in_DS <= fEcalNumbering->GetMaxSCInDS(DataSector) )
						{
						  if( read_histo[i0DeeEcha] != 0 )
						    {
						      //................................... Data Sector offset
						      Int_t DSOffset = GetDSOffset(DeeNumber, DataSector);
					      
						      //........................ Super-Crystal (SC) offset (ViewHisto [Stex])
						      Int_t SCOffset = GetSCOffset(DeeNumber, DataSector, SC_in_DS);
					      
						      //........................ Xtal final bin
						      Int_t nSCCons = fEcalNumbering->
							GetDeeSCConsFrom1DeeSCEcna(DeeNumber, n1DeeSCEcna, n1SCEcha);

						      Int_t n1FinalSCEcha = n1SCEcha;
						      
						      if( fEcalNumbering->GetSCType(nSCCons) == "NotConnected" || 
							  fEcalNumbering->GetSCType(nSCCons) == "NotComplete"  )
							{ //----- not complete and not connected SC's
							  // no i_xgeo value if SC = 14 or 25 and channel 11
							  if( !( (SC_in_DS == 14 || SC_in_DS == 25 ) && n1SCEcha == 11 )  )
							    {
							      n1FinalSCEcha =
								ModifiedSCEchaForNotConnectedSCs(DeeNumber, nSCCons, SC_in_DS,
												 n1DeeSCEcna, n1SCEcha);
							      i_xgeo = DSOffset + SCOffset + (n1FinalSCEcha-1);
							    }
							  // change SC 14 -> 21 and channel 11 -> 21
							  if( SC_in_DS ==  14 && n1SCEcha == 11 )
							    {
							      SCOffset = GetSCOffset(DeeNumber, DataSector, 21);
							      n1FinalSCEcha = 21;
							      i_xgeo = DSOffset + SCOffset + (n1FinalSCEcha-1);
							    }
							  // change SC 25 -> 3 for channel 11 -> 21
							  if( SC_in_DS ==  25 && n1SCEcha == 11 )
							    {
							      SCOffset = GetSCOffset(DeeNumber, DataSector, 3);
							      n1FinalSCEcha = 21;
							      i_xgeo = DSOffset + SCOffset + (n1FinalSCEcha-1);
							    }
							}
						      else
							{ //----------- Complete SCs
							  i_xgeo = DSOffset + SCOffset + (n1FinalSCEcha-1);
							}

						      histo_for_plot_memo[i_xgeo]++;
						      if( histo_for_plot_memo[i_xgeo] >= 2 )
							{
							  cout << "! histo_memo[" << i_xgeo
							       << "] = " << histo_for_plot_memo[i_xgeo]
							       << ", nSCCons = " <<  nSCCons
							       << ", SC_in_DS = " << SC_in_DS
							       << ", DSOffset = " << DSOffset
							       << ", SCOffset = " << SCOffset
							       << ", n1DeeSCEcna = " << n1DeeSCEcna
							       << ", n1SCEcha = " << n1SCEcha
							       << ", n1FinalSCEcha = " << n1FinalSCEcha << endl;
							}
						      //.............................. transfert read_histo -> histo_for_plot
						      if( i_xgeo >= 0 && i_xgeo < SizeForPlot )
							{
							  if( n1FinalSCEcha > 0 )
							    {histo_for_plot[i_xgeo] += read_histo[i0DeeEcha];}
							}
						      else
							{
							  cout << "!TEcnaHistos::ViewHisto(...)> <EE>  i_xgeo = " << i_xgeo
							       << ". OUT OF RANGE ( range = [0,"<< SizeForPlot << "] " << endl;
							}
						    } // end of  if( read_histo[i0DeeEcha] > 0 )
						} // end of if( SC_in_DS >= 1 && SC_in_DS <= fEcalNumbering->GetMaxSCInDS(DataSector) )
					      else
						{
						  cout << "!TEcnaHistos::ViewHisto(...)> <EE>  SC_in_DS = " << SC_in_DS
						       << ". OUT OF RANGE ( range = [1,"
						       << fEcalNumbering->GetMaxSCInDS(DataSector) << "] "
						       << ", DataSector = " << DataSector
						       << ", n1DeeSCEcna = " << n1DeeSCEcna
						       << ", n1SCEcha = " << n1SCEcha
						       << ", i0DeeEcha = " << i0DeeEcha
						       << endl;
						}
					    } // end of if( DataSector >= 1 && DataSector <= 9 )
					  else
					    {
					      if( DataSector != 0 )
						{
						  cout << "!TEcnaHistos::ViewHisto(...)> <EE>  DataSector = " << DataSector
						       << ". OUT OF RANGE ( range = [1,9] "
						       << ", n1DeeSCEcna = " << n1DeeSCEcna
						       << ", n1SCEcha = " << n1SCEcha
						       << ", i0DeeEcha = " << i0DeeEcha
						       << endl;
						}
					    }
					} // end of if( n1DeeSCEcna >= 1 && n1DeeSCEcna <= fEcal->MaxSCEcnaInDee() )
				      else
					{
					  cout << "!TEcnaHistos::ViewHisto(...)> <EE>  n1DeeSCEcna = " << n1DeeSCEcna
					       << ". OUT OF RANGE ( range = [1,"<< fEcal->MaxSCEcnaInDee() << "] "
					       << ", n1SCEcha = " << n1SCEcha
					       << ", i0DeeEcha = " << i0DeeEcha
					       << endl;
					}
				    } // end of if(n1SCEcha >= 1 && n1SCEcha <= fEcal->MaxCrysInSC() )
				  else
				    {
				      cout << "!TEcnaHistos::ViewHisto(...)> <EE>  n1SCEcha = " << n1SCEcha
					   << ". OUT OF RANGE ( range = [1,"<< fEcal->MaxCrysInSC() << "] "
					   << ", i0DeeEcha = " << i0DeeEcha
					   << endl;
				    }
				}
			    } // end of if( OKPlot == 1 && opt_plot != "ASCII" )
			} // end of if(fFlagSubDet == "EE")
		    } // end of if(ok_view_histo == kTRUE)
		  else
		    {
		      cout << "!TEcnaHistos::ViewHisto(...)> *ERROR* =====> "
			   << " ok_view_histo != kTRUE " << fTTBELL << endl;
		    }
		} // end of if(fMyRootFile->LookAtRootFile() == kTRUE)
	      else
		{
		  fStatusFileFound = kFALSE;

		  cout << "!TEcnaHistos::ViewHisto(...)> *ERROR* =====> "
		       << " ROOT file not found" << fTTBELL << endl;
		}
	    } // end of if(fFapStexNumber > 0)
	} // end of if( HistoType == "Global" || HistoType == "Proj" || HistoType == "SampGlobal" || HistoType == "SampProj" )
      else	  
	{
	  //--------------------------------------------------------------------- not Global-Proj Histo
	  if( (fFapStexNumber > 0) && (fFapStexNumber <= fEcal->MaxStexInStas()) )
	    {
	      Bool_t OKFileExists = kFALSE;

	      if( !(arg_AlreadyRead > 1) )
		{
		  fMyRootFile->PrintNoComment();
		  fMyRootFile->FileParameters(fFapAnaType,          fFapNbOfSamples,
					      fFapRunNumber,        fFapFirstReqEvtNumber,
					      fFapLastReqEvtNumber, fFapReqNbOfEvts,
					      fFapStexNumber,       fCfgResultsRootFilePath.Data());
		  OKFileExists = fMyRootFile->LookAtRootFile();
		  if( OKFileExists == kTRUE ){fFapNbOfEvts = fMyRootFile->GetNumberOfEvents(fFapReqNbOfEvts, fFapStexNumber);}
		}
	      else
		{
		  OKFileExists = kTRUE;
		}
	      
	      if( OKFileExists == kTRUE )    //   (ViewHisto, not Global-Proj)
		{
		  fStatusFileFound = kTRUE;

		  for(Int_t i=0; i<SizeForPlot; i++){histo_for_plot[i]=(Double_t)0;}

		  histo_for_plot = GetHistoValues(arg_read_histo, arg_AlreadyRead, fMyRootFile, HistoCode.Data(),
						  SizeForPlot, SizeForRead,
						  StexStin_A, i0StinEcha, i0Sample, i_data_exist);

		  fFapNbOfEvts = fMyRootFile->GetNumberOfEvents(fFapReqNbOfEvts, fFapStexNumber);
		  fStartDate = fMyRootFile->GetStartDate();
		  fStopDate  = fMyRootFile->GetStopDate();
		  fRunType   = fMyRootFile->GetRunType();
		  
		  if( i_data_exist > 0 ){OKPlot = 1;}
		}
	      else
		{
		  cout << "!TEcnaHistos::ViewHisto(...)> *ERROR* =====> "
		       << " ROOT file not found" << fTTBELL << endl;
		}
	    }
	  else
	    {
	      cout << "!TEcnaHistos::ViewHisto(...)> " << fFapStexName.Data()
		   << " = " << fFapStexNumber << ". Out of range (range = [1,"
		   << fEcal->MaxStexInStas() << "]) " << fTTBELL << endl;
	    }
	}

      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLOT accepted 

      if( (    HistoType == "Global"     || HistoType == "Proj" ||
	       HistoType == "SampGlobal" || HistoType == "SampProj" || 
	       HistoType == "H1Basic"    || HistoType == "H1BasicProj" ) ||
	  ( !( HistoType == "Global"     || HistoType == "Proj" ||
	       HistoType == "SampGlobal" || HistoType == "SampProj" || 
	       HistoType == "H1Basic"    || HistoType == "H1BasicProj" ) &&
	    ( (fFapStexNumber > 0) && (fFapStexNumber <= fEcal->MaxStexInStas()) ) ) )
	{
	  if( opt_plot != "ASCII" )
	    {
	      if( OKPlot > 0 )
		{
		  //... Ymin and ymax default values will be taken if fFlagUserHistoMin/Max = "OFF"
		  //    and if "Free" for "SAME" and "SAME n" options
		  if( (opt_plot == fOnlyOnePlot && ( arg_AlreadyRead == 0 || arg_AlreadyRead == 1 ) ) ||
		      (opt_plot == fSeveralPlot && GetMemoFlag(HistoCode, opt_plot) == "Free") ||
		      (opt_plot == fSameOnePlot && GetMemoFlag(HistoCode, opt_plot) == "Free") )
		    {
		      SetYminMemoFromValue(HistoCode.Data(), fCnaParHistos->GetYminDefaultValue(HistoCode.Data()));
		      SetYmaxMemoFromValue(HistoCode.Data(), fCnaParHistos->GetYmaxDefaultValue(HistoCode.Data()));
		    }
		  
		  //====  H I S T O  P R O J   X I N F / X S U P   M A N A G E M E N T  ========  (ViewHisto)
		  //
		  //  must be done before booking because of the x <-> y permutation in case of "Proj"
		  //
		  //-----------------------------------------------------------------------------------------
		  //
		  //        CASE:    HistoType == "Proj"   OR   HistoType == "SampProj"
		  //
		  //                 Xinf and Xsup must be calculated from ymin and ymax
		  //                 of the direct ("Global") histo
		  //
		  //-----------------------------------------------------------------------------------------
		  if( HistoType == "Proj" || HistoType == "SampProj" || HistoType == "H1BasicProj" )
		    {
		      TString HistoCodi = HistoCode;     // HistoCodi = direct histo

		      if( HistoCode == "D_NOE_ChDs" ){HistoCodi = "D_NOE_ChNb";}
		      if( HistoCode == "D_Ped_ChDs" ){HistoCodi = "D_Ped_ChNb";}
		      if( HistoCode == "D_TNo_ChDs" ){HistoCodi = "D_TNo_ChNb";}
		      if( HistoCode == "D_MCs_ChDs" ){HistoCodi = "D_MCs_ChNb";}
		      if( HistoCode == "D_LFN_ChDs" ){HistoCodi = "D_LFN_ChNb";}
		      if( HistoCode == "D_HFN_ChDs" ){HistoCodi = "D_HFN_ChNb";}
		      if( HistoCode == "D_SCs_ChDs" ){HistoCodi = "D_SCs_ChNb";}
		      if( HistoCode == "D_MSp_SpDs" ){HistoCodi = "D_MSp_SpNb";}
		      if( HistoCode == "D_SSp_SpDs" ){HistoCodi = "D_SSp_SpNb";}
		      if( HistoCode == "D_Adc_EvDs" ){HistoCodi = "D_Adc_EvNb";}		      

		      TString TitleHisto = ";";
		      if( opt_plot != fSameOnePlot )
			{TitleHisto = fCnaParHistos->GetQuantityName(HistoCodi);}		      

		      if( fUserHistoMin >= fUserHistoMax ){fFlagUserHistoMin = "AUTO"; fFlagUserHistoMax = "AUTO";}

		      //--------------------------------------------------------------------------- (ViewHisto)
		      //
		      //    fOnlyOnePlot => compute Xinf and Xsup at each time
		      //    fSeveralPlot => compute Xinf and Xsup once when flag = "Free" for each HistoCode
		      //    fSameOnePlot => compute Xinf and Xsup once
		      //
		      //--------------------------------------------------------------------------------------
		      if( (opt_plot == fOnlyOnePlot && ( arg_AlreadyRead == 0 || arg_AlreadyRead == 1 ) ) ||
			  (opt_plot == fSeveralPlot && GetMemoFlag(HistoCode, fSeveralPlot) == "Free" ) ||
			  (opt_plot == fSameOnePlot && GetMemoFlag(HistoCode, fSameOnePlot) == "Free" ) )
			{
			  Double_t XinfProj =(Double_t)0;
			  Double_t XsupProj =(Double_t)0;

			  //...................................................................... (ViewHisto)
			  if( fFlagUserHistoMin == "AUTO" || fFlagUserHistoMax == "AUTO" )
			    {
			      Int_t HisSiza = GetHistoSize(HistoCodi.Data(), "plot");
			      Int_t ReadHisSiza = HisSiza;
			      //..............................  prepa direct histogram booking (ViewHisto)
			      Axis_t xinf_hisa = GetHistoXinf(HistoCodi.Data(), HisSiza, opt_plot);
			      Axis_t xsup_hisa = GetHistoXsup(HistoCodi.Data(), HisSiza, opt_plot);
			      Int_t nb_binxa   = GetHistoNumberOfBins(HistoCodi.Data(), HisSiza);
			      //..............................  direct ("Global") histogram booking (ViewHisto)
			      TH1D* h_hisa =
				new TH1D("histoa", TitleHisto.Data(), nb_binxa, xinf_hisa, xsup_hisa); fCnewRoot++;
			      h_hisa->Reset();
			      //.... direct histogram filling to get its ymin (=> xminProj) and ymax (=> xmaxProj)
			      FillHisto(h_hisa, histo_for_plot, HistoCodi.Data(), ReadHisSiza);
			      //... Get direct histo ymin and/or ymax and keep them as xinf and xsup
			      //    in memo for the plotted histo 
			      XinfProj = fUserHistoMin;
			      XsupProj = fUserHistoMax;
			      if( fFlagUserHistoMin == "AUTO" ){XinfProj = h_hisa->GetMinimum();}
			      if( fFlagUserHistoMax == "AUTO" ){XsupProj = h_hisa->GetMaximum();}
			      XsupProj += (XsupProj-XinfProj)*fCnaParHistos->GetMarginAutoMinMax(); // to see the last bin
			      h_hisa->Delete();  h_hisa = 0;     fCdeleteRoot++;
			    } // end of  if( fFlagUserHistoMin == "AUTO" || fFlagUserHistoMax == "AUTO" )
			  else
			    {
			      if( fFlagUserHistoMin == "OFF" )
				{
				  SetYminMemoFromValue(HistoCode.Data(),
						       fCnaParHistos->GetYminDefaultValue(HistoCode.Data()));
				  XinfProj = GetYminValueFromMemo(HistoCode.Data());
				}

			      if( fFlagUserHistoMax == "OFF" )
				{
				  SetYmaxMemoFromValue(HistoCode.Data(),
						       fCnaParHistos->GetYmaxDefaultValue(HistoCode.Data()));
				  XsupProj = GetYmaxValueFromMemo(HistoCode.Data());	  
				}
			      if( fFlagUserHistoMin == "ON" ){XinfProj = fUserHistoMin;}
			      if( fFlagUserHistoMax == "ON" ){XsupProj = fUserHistoMax;}
			    }

			  if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot )
			    {
			      SetXinfMemoFromValue(HistoCode.Data(), XinfProj);
			      SetXsupMemoFromValue(HistoCode.Data(), XsupProj);			  
			    }
			  else
			    {
			      SetXinfMemoFromValue(XinfProj);
			      SetXsupMemoFromValue(XsupProj);
			    }
			} // end of if( (opt_plot == fOnlyOnePlot) || 
		          // (opt_plot == fSeveralPlot  && GetMemoFlag(HistoCode, opt_plot) == "Free") ||
		          // (opt_plot == fSameOnePlot  && GetMemoFlag(HistoCode, opt_plot) == "Free") )
		    } // end of  if( HistoType == "Proj" || HistoType == "SampProj" || HistoType == "H1BasicProj" )

		  //===============  H I S T O   B O O K I N G   A N D   F I L L I N G  ========  (ViewHisto)
		  //..............................  prepa histogram booking (ViewHisto)
		  
		  //.......... Set number of bins: forcing to fNbBinsProj if "HistoType" == "Proj"
		  Int_t xNbBins = GetHistoNumberOfBins(HistoCode.Data(), SizeForPlot);
		  
		  Double_t cXinf = (Double_t)0.;
		  Double_t cXsup = (Double_t)0.;

		  //.......... Set Xinf and Xsup at each time because of simultaneous SAME options
		  if( HistoType == "Proj" || HistoType == "SampProj" || HistoType == "H1BasicProj")
		    {
		      if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot )
			{
			  cXinf = GetXinfValueFromMemo(HistoCode.Data());
			  cXsup = GetXsupValueFromMemo(HistoCode.Data());
			}
		      else
			{
			  cXinf = GetXinfValueFromMemo();
			  cXsup = GetXsupValueFromMemo();
			}
		    }
		  else
		    {
		      cXinf = GetHistoXinf(HistoCode.Data(), SizeForPlot, opt_plot);
		      cXsup = GetHistoXsup(HistoCode.Data(), SizeForPlot, opt_plot);
		    }

		  //..............................  histogram booking (ViewHisto)
		  Axis_t xinf_his = cXinf;  // ancillary variables since no const in arguments of TH1D
		  Axis_t xsup_his = cXsup;
		  Int_t   nb_binx = xNbBins;

		  TString TitleHisto = ";";
		  if( opt_plot != fSameOnePlot )
		    {TitleHisto = fCnaParHistos->GetQuantityName(HistoCode.Data());}
		  TH1D* h_his0 = new TH1D("histo", TitleHisto.Data(), nb_binx, xinf_his, xsup_his); fCnewRoot++;
		  h_his0->Reset();
		  //............................... histogram filling
		  FillHisto(h_his0, histo_for_plot, HistoCode.Data(), SizeForPlot);
		  
		  //===============  H I S T O   Y M I N / Y M A X   M A N A G E M E N T  ===========  (ViewHisto)
		  if( opt_plot == fOnlyOnePlot ||
		      (opt_plot == fSeveralPlot && GetMemoFlag(HistoCode, opt_plot) == "Free") || 
		      (opt_plot == fSameOnePlot && GetMemoFlag(HistoCode, opt_plot) == "Free") )
		    {
		      if( opt_plot == fSameOnePlot ){fHistoCodeFirst = HistoCode;} // registration of first HistoCode
		      //................................. Automatic min and/or max for other options than "Proj" 
		      if( HistoType != "Proj" && HistoType != "SampProj" && HistoType != "H1BasicProj" )
			{
			  if( fUserHistoMin >= fUserHistoMax ){fFlagUserHistoMin = "AUTO"; fFlagUserHistoMax = "AUTO";}
			  //................................. user's min and/or max
			  if( fFlagUserHistoMin == "ON" )
			    {SetYminMemoFromValue(HistoCode.Data(), fUserHistoMin); fFlagUserHistoMin = "OFF";}
			  if( fFlagUserHistoMax == "ON" )
			    {SetYmaxMemoFromValue(HistoCode.Data(), fUserHistoMax); fFlagUserHistoMax = "OFF";}
			  //................................. automatic min and/or max
			  if( fFlagUserHistoMin == "AUTO" )
			    {
			      //.............. no bottom margin if ymin = 0
			      Double_t ymin = GetYminFromHistoFrameAndMarginValue(h_his0, (Double_t)0.);
			      if( ymin != (Double_t)0. )
				{ymin =
				   GetYminFromHistoFrameAndMarginValue(h_his0, fCnaParHistos->GetMarginAutoMinMax());}
			      SetYminMemoFromValue(HistoCode.Data(),ymin);
			      fFlagUserHistoMin = "OFF";
			    }
			  if( fFlagUserHistoMax == "AUTO" )
			    {
			      Double_t ymax =
				GetYmaxFromHistoFrameAndMarginValue(h_his0,fCnaParHistos->GetMarginAutoMinMax());
			      SetYmaxMemoFromValue(HistoCode.Data(),ymax);
			      fFlagUserHistoMax = "OFF";
			    }
			  //................................. Set YMin and YMax of histo (ViewHisto)
			  SetYminMemoFromPreviousMemo(HistoCode);
			  SetYmaxMemoFromPreviousMemo(HistoCode);
			} // end of if( HistoType != "Proj" && HistoType != "SampProj" && HistoType != "H1BasicProj" )

		      //.......... Find maximum in case of Proj (LIN scale only) // PS: EvolProj => in ViewHistime
		      if( ( HistoType == "Proj" || HistoType == "SampProj" ||
			    HistoType == "H1BasicProj" ) && fFlagScaleY == "LIN" )
		      	{
		      	  SetYmaxMemoFromValue
		      	    (HistoCode.Data(),
		      	     GetYmaxFromHistoFrameAndMarginValue(h_his0, fCnaParHistos->GetMarginAutoMinMax()));
		      	}
		    } // end of  if( opt_plot == fOnlyOnePlot ||
		  // (opt_plot == fSeveralPlot && GetMemoFlag(HistoCode, opt_plot) == "Free") || 
		  // (opt_plot == fSameOnePlot && GetMemoFlag(HistoCode, opt_plot) == "Free") )
		  
		  //--- Set ymin and ymax to the first HistoCode values for option SAME n
		  if( opt_plot == fSameOnePlot && GetMemoFlag(HistoCode, opt_plot) == "Busy" )		  
		    {
		      Double_t ymin = GetYminValueFromMemo(fHistoCodeFirst.Data());
		      SetYminMemoFromValue(HistoCode.Data(), ymin);

		      Double_t ymax = GetYmaxValueFromMemo(fHistoCodeFirst.Data());
		      SetYmaxMemoFromValue(HistoCode.Data(), ymax);
		    }

		  //... histogram set ymin and ymax and consequently margin at top of the plot
		  Int_t  xFlagAutoYsupMargin = SetHistoFrameYminYmaxFromMemo(h_his0, HistoCode);

		  //==================================== P L O T ==============================  (ViewHisto)
		  HistoPlot(h_his0,           SizeForPlot,   xinf_his, xsup_his,
			    HistoCode.Data(), HistoType.Data(),
			    StexStin_A,       i0StinEcha,    i0Sample,
			    opt_scale_x,      opt_scale_y,   opt_plot, arg_AlreadyRead,
			    xFlagAutoYsupMargin);
		  h_his0->Delete();   h_his0 = 0;             fCdeleteRoot++;
		  //===========================================================================

		  //--- Recover ymin and ymax from user's values in option SAME n
		  if( (opt_plot == fSameOnePlot && GetMemoFlag(HistoCode, opt_plot) == "Busy") )		  
		    {
		      SetYminMemoFromValue(HistoCode.Data(), fUserHistoMin);
		      SetYmaxMemoFromValue(HistoCode.Data(), fUserHistoMax);
		    }
		} // end of if( OKPlot > 0 )
	      else
		{
		  cout << "!TEcnaHistos::ViewHisto(...)> Histo not available."
		       << fTTBELL << endl;
		}
	    }
	}
    } // end of  if( OKHisto == 1 )

}  // end of ViewHisto(...)

//------------------------------------------------------------------------------------
Int_t TEcnaHistos::GetDSOffset(const Int_t& DeeNumber, const Int_t& DataSector)
{
  // gives the DataSector Offset on 1D histos for option "Global"

  Int_t DSOffset = 0;

  if( DeeNumber == 4 )
    {
      if( DataSector >= 1 ){}
      if( DataSector >= 2 ){DSOffset += fEcalNumbering->GetMaxSCInDS(1)*fEcal->MaxCrysInSC();}
      if( DataSector >= 3 ){DSOffset += fEcalNumbering->GetMaxSCInDS(2)*fEcal->MaxCrysInSC();}
      if( DataSector >= 4 ){DSOffset += fEcalNumbering->GetMaxSCInDS(3)*fEcal->MaxCrysInSC();}
      if( DataSector >= 5 ){DSOffset += fEcalNumbering->GetMaxSCInDS(4)*fEcal->MaxCrysInSC();}
    }
  if( DeeNumber == 3 )
    {
      if( DataSector >= 5 ){}
      if( DataSector >= 6 ){DSOffset += (fEcalNumbering->GetMaxSCInDS(5)/2)*fEcal->MaxCrysInSC();}
      if( DataSector >= 7 ){DSOffset += fEcalNumbering->GetMaxSCInDS(6)*fEcal->MaxCrysInSC();}
      if( DataSector >= 8 ){DSOffset += fEcalNumbering->GetMaxSCInDS(7)*fEcal->MaxCrysInSC();}
      if( DataSector >= 9 ){DSOffset += fEcalNumbering->GetMaxSCInDS(8)*fEcal->MaxCrysInSC();}
    }
  if( DeeNumber == 2 )
    {
      if( DataSector <= 9 ){}
      if( DataSector <= 8 ){DSOffset += fEcalNumbering->GetMaxSCInDS(9)*fEcal->MaxCrysInSC();}
      if( DataSector <= 7 ){DSOffset += fEcalNumbering->GetMaxSCInDS(8)*fEcal->MaxCrysInSC();}
      if( DataSector <= 6 ){DSOffset += fEcalNumbering->GetMaxSCInDS(7)*fEcal->MaxCrysInSC();}
      if( DataSector <= 5 ){DSOffset += fEcalNumbering->GetMaxSCInDS(6)*fEcal->MaxCrysInSC();}
    }
  if( DeeNumber == 1 )
    {
      if( DataSector <= 5 ){}
      if( DataSector <= 4 ){DSOffset += (fEcalNumbering->GetMaxSCInDS(5)/2)*fEcal->MaxCrysInSC();}
      if( DataSector <= 3 ){DSOffset += fEcalNumbering->GetMaxSCInDS(4)*fEcal->MaxCrysInSC();}
      if( DataSector <= 2 ){DSOffset += fEcalNumbering->GetMaxSCInDS(3)*fEcal->MaxCrysInSC();}
      if( DataSector <= 1 ){DSOffset += fEcalNumbering->GetMaxSCInDS(2)*fEcal->MaxCrysInSC();}
    }
  return DSOffset;			      
}
//------------------------------------------------------------------------------------
Int_t TEcnaHistos::GetSCOffset(const Int_t& DeeNumber, const Int_t& DataSector, const Int_t& SC_in_DS)
{
  // gives the SC (Super-Crystal) Offset on 1D histos for option "Global"

  Int_t SCOffset = 0;

  if( DeeNumber == 1 || DeeNumber == 3 )
    {
      if( DataSector == 5 ){SCOffset += ((SC_in_DS-17)-1)*fEcal->MaxCrysInSC();}
      if( DataSector != 5 ){SCOffset += (SC_in_DS-1)*fEcal->MaxCrysInSC();}
    }

  if( DeeNumber == 2 || DeeNumber == 4 ){SCOffset += (SC_in_DS-1)*fEcal->MaxCrysInSC();}

  return SCOffset;	
}
//------------------------------------------------------------------------------------
Int_t TEcnaHistos::ModifiedSCEchaForNotConnectedSCs(const Int_t& n1DeeNumber,
						   const Int_t& nSCCons,     const Int_t& SC_in_DS,
						   const Int_t& n1DeeSCEcna, const Int_t& n1SCEcha)
{
  //------------------------ Modification of n1SCEcha number for not connected SC's

  Int_t ModifiedSCEcha = -1;
  TString SCQuad = fEcalNumbering->GetSCQuadFrom1DeeSCEcna(n1DeeSCEcna);  // SCQuad = top  OR bottom
  TString DeeDir = fEcalNumbering->GetDeeDirViewedFromIP(n1DeeNumber);    // DeeDir = left OR right

  TString TypQuad = "?";
  if( SCQuad == "top"    && DeeDir == "right" ){TypQuad = "TR";}
  if( SCQuad == "top"    && DeeDir == "left"  ){TypQuad = "TL";}
  if( SCQuad == "bottom" && DeeDir == "left"  ){TypQuad = "BL";}
  if( SCQuad == "bottom" && DeeDir == "right" ){TypQuad = "BR";}

  //------------------------------------------------------------------------------------------- top

  //..... (D1,S1), (D3,S9) SC_in_DS = 30, n1DeeSCEcna =  60 -> 182a for construction top/right
  //..... (D1,S2), (D3,S8) SC_in_DS =  3, n1DeeSCEcna = 138 -> 178a for construction top/right
  if( (SC_in_DS == 30 && n1DeeSCEcna ==  60 && TypQuad == "TR") ||
      (SC_in_DS ==  3 && n1DeeSCEcna == 138 && TypQuad == "TR") ){if(n1SCEcha > 15){ModifiedSCEcha = n1SCEcha - 15;}}

  //..... (D4,S1), (D2,S9) SC_in_DS = 30, n1DeeSCEcna =  60 ->  33a for construction top/left
  //..... (D4,S2), (D2,S8) SC_in_DS =  3, n1DeeSCEcna = 138 ->  29a for construction top/left
  if( (SC_in_DS == 30 && n1DeeSCEcna ==  60 && TypQuad == "TL") ||
      (SC_in_DS ==  3 && n1DeeSCEcna == 138 && TypQuad == "TL") )
    {
      if(n1SCEcha ==  4){ModifiedSCEcha =  1;}
      if(n1SCEcha ==  5){ModifiedSCEcha =  2;}
      if(n1SCEcha ==  9){ModifiedSCEcha =  3;}
      if(n1SCEcha == 10){ModifiedSCEcha =  4;}
      if(n1SCEcha == 14){ModifiedSCEcha =  5;}
      if(n1SCEcha == 15){ModifiedSCEcha =  6;}
      if(n1SCEcha == 19){ModifiedSCEcha =  7;}
      if(n1SCEcha == 20){ModifiedSCEcha =  8;}
      if(n1SCEcha == 24){ModifiedSCEcha =  9;}
      if(n1SCEcha == 25){ModifiedSCEcha = 10;}
    }

  //..... (D1,S1), (D3,S9) SC_in_DS = 30, n1DeeSCEcna = 119 -> 182b for construction top/right
  if( SC_in_DS == 30 && n1DeeSCEcna == 119 && TypQuad == "TR" ){if(n1SCEcha > 5){ModifiedSCEcha = n1SCEcha - 5;}}

  //..... (D4,S1), (D2,S9) SC_in_DS = 30, n1DeeSCEcna = 119 ->  33b for construction top/left
  if( SC_in_DS == 30 && n1DeeSCEcna == 119 && TypQuad == "TL" )
    {
      if(n1SCEcha ==  4){ModifiedSCEcha = 11;}
      if(n1SCEcha ==  5){ModifiedSCEcha = 12;}
      if(n1SCEcha ==  9){ModifiedSCEcha = 13;}
      if(n1SCEcha == 10){ModifiedSCEcha = 14;}
      if(n1SCEcha == 14){ModifiedSCEcha = 15;}
      if(n1SCEcha == 15){ModifiedSCEcha = 16;}
      if(n1SCEcha == 19){ModifiedSCEcha = 17;}
      if(n1SCEcha == 20){ModifiedSCEcha = 18;}
      if(n1SCEcha == 24){ModifiedSCEcha = 19;}
      if(n1SCEcha == 25){ModifiedSCEcha = 20;}
    }

  //..... (D1,S1), (D3,S9) SC_in_DS = 12, n1DeeSCEcna =  13 -> 161  for construction top/right
  //..... (D4,S1), (D2,S9) SC_in_DS = 12, n1DeeSCEcna =  13 ->  12  for construction top/left
  if( SC_in_DS == 12 && n1DeeSCEcna ==  13 && TypQuad == "TR" )
    {
      ModifiedSCEcha = n1SCEcha;
    } 
  if( SC_in_DS == 12 && n1DeeSCEcna ==  13 && TypQuad == "TL" )
    {
      if( n1SCEcha >=  1 &&  n1SCEcha <=  4 ){ModifiedSCEcha = n1SCEcha;}
      if( n1SCEcha >=  6 &&  n1SCEcha <=  9 ){ModifiedSCEcha = n1SCEcha-1;}
      if( n1SCEcha >= 11 &&  n1SCEcha <= 14 ){ModifiedSCEcha = n1SCEcha-2;}
      if( n1SCEcha >= 16 &&  n1SCEcha <= 19 ){ModifiedSCEcha = n1SCEcha-3;}
      if( n1SCEcha >= 21 &&  n1SCEcha <= 24 ){ModifiedSCEcha = n1SCEcha-4;}
    }

  //..... (D1,S2), (D3,S8) SC_in_DS = 25, n1DeeSCEcna = 176 -> 207a for construction top/right
  if( SC_in_DS == 25 && n1DeeSCEcna == 176 && TypQuad == "TR" )
    {
      if(n1SCEcha ==  4){ModifiedSCEcha =  1;}
      if(n1SCEcha ==  5){ModifiedSCEcha =  2;}
      if(n1SCEcha ==  9){ModifiedSCEcha =  3;}
      if(n1SCEcha == 10){ModifiedSCEcha =  4;}
      if(n1SCEcha == 14){ModifiedSCEcha =  5;}
      if(n1SCEcha == 15){ModifiedSCEcha =  6;}
      if(n1SCEcha == 19){ModifiedSCEcha =  7;}
      if(n1SCEcha == 20){ModifiedSCEcha =  8;}
      if(n1SCEcha == 24){ModifiedSCEcha =  9;}
      if(n1SCEcha == 25){ModifiedSCEcha = 10;}
    }

  //..... (D4,S2), (D2,S8) SC_in_DS = 25, n1DeeSCEcna = 176 ->  58a for construction top/left
  if( SC_in_DS == 25 && n1DeeSCEcna == 176 && TypQuad == "TL" )
    {
      if(n1SCEcha == 16){ModifiedSCEcha =  1;}
      if(n1SCEcha == 21){ModifiedSCEcha =  2;}
      if(n1SCEcha == 17){ModifiedSCEcha =  3;}
      if(n1SCEcha == 22){ModifiedSCEcha =  4;}
      if(n1SCEcha == 18){ModifiedSCEcha =  5;}
      if(n1SCEcha == 23){ModifiedSCEcha =  6;}
      if(n1SCEcha == 19){ModifiedSCEcha =  7;}
      if(n1SCEcha == 24){ModifiedSCEcha =  8;}
      if(n1SCEcha == 20){ModifiedSCEcha =  9;}
      if(n1SCEcha == 25){ModifiedSCEcha = 10;}
    }

  //..... (D1,S2), (D3,S8) SC_in_DS =  3, n1DeeSCEcna = 157 -> 178b for construction top/right
  //..... (D1,S2), (D3,S8) SC_in_DS = 25, n1DeeSCEcna = 193 -> 207b for construction top/right
  if( (SC_in_DS ==  3 && n1DeeSCEcna == 157 && TypQuad == "TR") ||
      (SC_in_DS == 25 && n1DeeSCEcna == 193 && TypQuad == "TR") )
    {
      if(n1SCEcha ==  4){ModifiedSCEcha = 11;}
      if(n1SCEcha ==  5){ModifiedSCEcha = 12;}
      if(n1SCEcha ==  9){ModifiedSCEcha = 13;}
      if(n1SCEcha == 10){ModifiedSCEcha = 14;}
      if(n1SCEcha == 14){ModifiedSCEcha = 15;}
      if(n1SCEcha == 15){ModifiedSCEcha = 16;}
      if(n1SCEcha == 19){ModifiedSCEcha = 17;}
      if(n1SCEcha == 20){ModifiedSCEcha = 18;}
      if(n1SCEcha == 24){ModifiedSCEcha = 19;}
      if(n1SCEcha == 25){ModifiedSCEcha = 20;}
    }

  //..... (D4,S2), (D2,S8) SC_in_DS =  3, n1DeeSCEcna = 157 ->  29b for construction top/left
  //..... (D4,S2), (D2,S8) SC_in_DS = 25, n1DeeSCEcna = 193 ->  58b for construction top/left
  if( (SC_in_DS ==  3 && n1DeeSCEcna == 157 && TypQuad == "TL") ||
      (SC_in_DS == 25 && n1DeeSCEcna == 193 && TypQuad == "TL") )
    {
      if(n1SCEcha == 16){ModifiedSCEcha = 11;}
      if(n1SCEcha == 21){ModifiedSCEcha = 12;}
      if(n1SCEcha == 17){ModifiedSCEcha = 13;}
      if(n1SCEcha == 22){ModifiedSCEcha = 14;}
      if(n1SCEcha == 18){ModifiedSCEcha = 15;}
      if(n1SCEcha == 23){ModifiedSCEcha = 16;}
      if(n1SCEcha == 19){ModifiedSCEcha = 17;}
      if(n1SCEcha == 24){ModifiedSCEcha = 18;}
      if(n1SCEcha == 20){ModifiedSCEcha = 19;}
      if(n1SCEcha == 25){ModifiedSCEcha = 20;}
    }

  //..... (D1,S2), (D3,S8) SC_in_DS = 32, n1DeeSCEcna =  51 -> 216  for construction top/right
  if( SC_in_DS == 32 && n1DeeSCEcna ==  51 && TypQuad == "TR" )
    {
      if( n1SCEcha >=  1 && n1SCEcha <=  4 ){ModifiedSCEcha = n1SCEcha;}
      if( n1SCEcha >=  6 && n1SCEcha <=  9 ){ModifiedSCEcha = n1SCEcha-1;}
      if( n1SCEcha >= 11 && n1SCEcha <= 14 ){ModifiedSCEcha = n1SCEcha-2;}
      if( n1SCEcha >= 16 && n1SCEcha <= 19 ){ModifiedSCEcha = n1SCEcha-3;}
      if( n1SCEcha >= 21 && n1SCEcha <= 24 ){ModifiedSCEcha = n1SCEcha-4;}
    }

  //..... (D4,S2), (D2,S8) SC_in_DS = 32, n1DeeSCEcna =  51 ->  67  for construction top/left
  if( SC_in_DS == 32 && n1DeeSCEcna ==  51 && TypQuad == "TL" )
    {
      ModifiedSCEcha = n1SCEcha;
    }

  // **************************** Special case: TWO SC's IN THE SAME SC-Ecna place *************************
  //========================================================================================== D1,D3 ======
  //      (D1,S2), (D3,S8) SC_in_DS =  3, n1DeeSCEcna =  32 -> 178c for construction top/right
  //      (D1,S2), (D3,S8) SC_in_DS = 25, n1DeeSCEcna =  32 -> 207c for construction top/right
  //       For  n1DeeSCEcna =  32: ONLY "25" IN ARRAY fT2d_DSSC[][] (see TEcnaNumbering.cc)
  //       fT2d_DSSC[dee-1][32-1] =  25;  // also 3;  // ( (207c, 58c) also (178c, 29c) for construction)
  //       is recovered from number for construction
  //=======================================================================================================
  if( n1DeeSCEcna ==  32 && TypQuad == "TR" )
    {
      if( nSCCons == 207 )
	{
	  if(n1SCEcha ==  1){ModifiedSCEcha = 21;}
	  if(n1SCEcha ==  2){ModifiedSCEcha = 22;}
	  if(n1SCEcha ==  3){ModifiedSCEcha = 23;}
	  if(n1SCEcha ==  6){ModifiedSCEcha = 24;}
	  if(n1SCEcha ==  7){ModifiedSCEcha = 25;}
	}
      if( nSCCons == 178 )
	{
	  if(n1SCEcha == 11){ModifiedSCEcha = 21;}
	}
    }

  //========================================================================================== D2,D4 ======
  //      (D4,S2), (D2,S8) SC_in_DS =  3, n1DeeSCEcna =  32 ->  29c for construction top/left
  //      (D4,S2), (D2,S8) SC_in_DS = 25, n1DeeSCEcna =  32 ->  58c for construction top/left
  //       For  n1DeeSCEcna =  32: ONLY "25" IN ARRAY fT2d_DSSC[][] (see TEcnaNumbering.cc)
  //       fT2d_DSSC[dee-1][32-1] =  25;  // also 3;  // ( (207c, 58c) also (178c, 29c) for construction)
  //       is recovered from number for construction
  //=======================================================================================================
  if( n1DeeSCEcna ==  32 && TypQuad == "TL" )
    {
      if( nSCCons == 58 )
	{
	  if(n1SCEcha ==  1){ModifiedSCEcha = 21;}
	  if(n1SCEcha ==  2){ModifiedSCEcha = 22;}
	  if(n1SCEcha ==  3){ModifiedSCEcha = 23;}
	  if(n1SCEcha ==  6){ModifiedSCEcha = 24;}
	  if(n1SCEcha ==  7){ModifiedSCEcha = 25;}
	}
      if( nSCCons == 29 )
	{
	  if(n1SCEcha == 11){ModifiedSCEcha = 21;}
	}
    }
  //****************************************************************************************************

  //------------------------------------------------------------------------------------------- bottom

  // **************************** Special case: TWO SC's IN THE SAME SC-Ecna place *************************
  //========================================================================================== D1,D3 ======
  //      (D1,S4), (D3,S6) SC_in_DS = 14, n1DeeSCEcna =  29 -> 261a for construction bottom/right
  //      (D1,S4), (D3,S6) SC_in_DS = 21, n1DeeSCEcna =  29 -> 268a for construction bottom/right
  //       For  n1DeeSCEcna =  29: ONLY "14" IN ARRAY fT2d_DSSC[][] (see TEcnaNumbering.cc)
  //       fT2d_DSSC[dee-1][29-1] = 14; // also 21;  //  ( (261a, 112a) also (268a, 119a) for construction)
  //       is recovered from number for construction
  //=======================================================================================================
  if( n1DeeSCEcna ==  29 && TypQuad == "BR" )
    {
      if( nSCCons == 261 )
	{
	  if(n1SCEcha ==  1){ModifiedSCEcha = 21;}
	  if(n1SCEcha ==  2){ModifiedSCEcha = 22;}
	  if(n1SCEcha ==  3){ModifiedSCEcha = 23;}
	  if(n1SCEcha ==  6){ModifiedSCEcha = 24;}
	  if(n1SCEcha ==  7){ModifiedSCEcha = 25;}
	}
      if( nSCCons == 268 )
	{
	  if(n1SCEcha == 11){ModifiedSCEcha = 21;}
	}
    }

  //========================================================================================== D2,D4 ======
  //      (D4,S4), (D2,S6) SC_in_DS = 14, n1DeeSCEcna =  29 -> 112a for construction bottom/left
  //      (D4,S4), (D2,S6) SC_in_DS = 21, n1DeeSCEcna =  29 -> 119a for construction bottom/left
  //       For  n1DeeSCEcna =  29: ONLY "14" IN ARRAY fT2d_DSSC[][] (see TEcnaNumbering.cc)
  //       fT2d_DSSC[dee-1][29-1] = 14; // also 21;  //  ( (261a, 112a) also (268a, 119a) for construction)
  //       is recovered from number for construction
  //======================================================================================================= 
  if( n1DeeSCEcna ==  29 && TypQuad == "BL" )
    {
      if( nSCCons == 119 )
	{
	  if(n1SCEcha == 11){ModifiedSCEcha = 21;}
	}
      if( nSCCons == 112 )
	{
	  if(n1SCEcha ==  1){ModifiedSCEcha = 21;}
	  if(n1SCEcha ==  2){ModifiedSCEcha = 22;}
	  if(n1SCEcha ==  3){ModifiedSCEcha = 23;}
	  if(n1SCEcha ==  6){ModifiedSCEcha = 24;}
	  if(n1SCEcha ==  7){ModifiedSCEcha = 25;}
	}
    }

  // ****************************************************************************************************

  //..... (D1,S3), (D3,S7) SC_in_DS = 34, n1DeeSCEcna = 188 -> 298a for construction bottom/right
  //..... (D1,S4), (D3,S6) SC_in_DS = 14, n1DeeSCEcna = 165 -> 261b for construction bottom/right
  if( (SC_in_DS == 34 && n1DeeSCEcna == 188 && TypQuad == "BR") || 
      (SC_in_DS == 14 && n1DeeSCEcna == 165 && TypQuad == "BR") ){if(n1SCEcha > 15){ModifiedSCEcha = n1SCEcha - 15;}}

  //..... (D4,S3), (D2,S7) SC_in_DS = 34, n1DeeSCEcna = 188 -> 149a for construction bottom/left
  //..... (D4,S4), (D2,S6) SC_in_DS = 14, n1DeeSCEcna = 165 -> 112b for construction bottom/left
  if( (SC_in_DS == 34 && n1DeeSCEcna == 188 && TypQuad == "BL") ||
      (SC_in_DS == 14 && n1DeeSCEcna == 165 && TypQuad == "BL") )
    {
      if(n1SCEcha ==  4){ModifiedSCEcha =  1;}
      if(n1SCEcha ==  5){ModifiedSCEcha =  2;}
      if(n1SCEcha ==  9){ModifiedSCEcha =  3;}
      if(n1SCEcha == 10){ModifiedSCEcha =  4;}
      if(n1SCEcha == 14){ModifiedSCEcha =  5;}
      if(n1SCEcha == 15){ModifiedSCEcha =  6;}
      if(n1SCEcha == 19){ModifiedSCEcha =  7;}
      if(n1SCEcha == 20){ModifiedSCEcha =  8;}
      if(n1SCEcha == 24){ModifiedSCEcha =  9;}
      if(n1SCEcha == 25){ModifiedSCEcha = 10;}
    }

  //..... (D1,S3), (D3,S7) SC_in_DS = 10, n1DeeSCEcna =  50 -> 224  for construction bottom/right
  if( SC_in_DS == 10 && n1DeeSCEcna ==  50 && TypQuad == "BR" )
    {
      ModifiedSCEcha = n1SCEcha;
    }

  //..... (D4,S3), (D2,S7) SC_in_DS = 10, n1DeeSCEcna =  50 ->  75  for construction bottom/left 
  if( SC_in_DS == 10 && n1DeeSCEcna ==  50 && TypQuad == "BL")
    {
      if( n1SCEcha >=  1 &&  n1SCEcha <=  4 ){ModifiedSCEcha = n1SCEcha;}
      if( n1SCEcha >=  6 &&  n1SCEcha <=  9 ){ModifiedSCEcha = n1SCEcha-1;}
      if( n1SCEcha >= 11 &&  n1SCEcha <= 14 ){ModifiedSCEcha = n1SCEcha-2;}
      if( n1SCEcha >= 16 &&  n1SCEcha <= 19 ){ModifiedSCEcha = n1SCEcha-3;}
      if( n1SCEcha >= 21 &&  n1SCEcha <= 24 ){ModifiedSCEcha = n1SCEcha-4;}
    }
  
  //..... (D1,S4), (D3,S6) SC_in_DS = 14, n1DeeSCEcna = 144 -> 261c for construction bottom/right
  if( SC_in_DS == 14 && n1DeeSCEcna == 144 && TypQuad == "BR" ){if(n1SCEcha > 5){ModifiedSCEcha = n1SCEcha - 5;}}

  //..... (D4,S4), (D2,S6) SC_in_DS = 14, n1DeeSCEcna = 144 -> 112c for construction bottom/left
  if( SC_in_DS == 14 && n1DeeSCEcna == 144 && TypQuad == "BL" )
    {
      if(n1SCEcha ==  4){ModifiedSCEcha = 11;}
      if(n1SCEcha ==  5){ModifiedSCEcha = 12;}
      if(n1SCEcha ==  9){ModifiedSCEcha = 13;}
      if(n1SCEcha == 10){ModifiedSCEcha = 14;}
      if(n1SCEcha == 14){ModifiedSCEcha = 15;}
      if(n1SCEcha == 15){ModifiedSCEcha = 16;}
      if(n1SCEcha == 19){ModifiedSCEcha = 17;}
      if(n1SCEcha == 20){ModifiedSCEcha = 18;}
      if(n1SCEcha == 24){ModifiedSCEcha = 19;}
      if(n1SCEcha == 25){ModifiedSCEcha = 20;}
    }

  //..... (D1,S4), (D3,S6) SC_in_DS = 21, n1DeeSCEcna = 123 -> 268b for construction bottom/right
  //..... (D1,S5), (D3,S5) SC_in_DS = 20, n1DeeSCEcna =  21 -> 281a for construction bottom/right
  if( (SC_in_DS == 21 && n1DeeSCEcna == 123 && TypQuad == "BR") ||
      (SC_in_DS == 20 && n1DeeSCEcna ==  41 && TypQuad == "BR") )
    {
      if(n1SCEcha ==  4){ModifiedSCEcha =  1;}
      if(n1SCEcha ==  5){ModifiedSCEcha =  2;}
      if(n1SCEcha ==  9){ModifiedSCEcha =  3;}
      if(n1SCEcha == 10){ModifiedSCEcha =  4;}
      if(n1SCEcha == 14){ModifiedSCEcha =  5;}
      if(n1SCEcha == 15){ModifiedSCEcha =  6;}
      if(n1SCEcha == 19){ModifiedSCEcha =  7;}
      if(n1SCEcha == 20){ModifiedSCEcha =  8;}
      if(n1SCEcha == 24){ModifiedSCEcha =  9;}
      if(n1SCEcha == 25){ModifiedSCEcha = 10;}
    }

  //..... (D4,S4), (D2,S6) SC_in_DS = 21, n1DeeSCEcna = 123 -> 119b for construction bottom/left
  //..... (D4,S5), (D2,S5) SC_in_DS =  3, n1DeeSCEcna =  41 -> 132a for construction bottom/left
  if( (SC_in_DS == 21 && n1DeeSCEcna == 123 && TypQuad == "BL") ||
      (SC_in_DS ==  3 && n1DeeSCEcna ==  41 && TypQuad == "BL") ){if(n1SCEcha > 15){ModifiedSCEcha = n1SCEcha - 15;}}


  //..... (D1,S4), (D3,S6) SC_in_DS = 21, n1DeeSCEcna = 102 -> 268c for construction bottom/right
  if( SC_in_DS == 21 && n1DeeSCEcna == 102 && TypQuad == "BR" )
    {
      if(n1SCEcha ==  4){ModifiedSCEcha = 11;}
      if(n1SCEcha ==  5){ModifiedSCEcha = 12;}
      if(n1SCEcha ==  9){ModifiedSCEcha = 13;}
      if(n1SCEcha == 10){ModifiedSCEcha = 14;}
      if(n1SCEcha == 14){ModifiedSCEcha = 15;}
      if(n1SCEcha == 15){ModifiedSCEcha = 16;}
      if(n1SCEcha == 19){ModifiedSCEcha = 17;}
      if(n1SCEcha == 20){ModifiedSCEcha = 18;}
      if(n1SCEcha == 24){ModifiedSCEcha = 19;}
      if(n1SCEcha == 25){ModifiedSCEcha = 20;}
    }

  //..... (D4,S4), (D2,S6) SC_in_DS = 21, n1DeeSCEcna = 102 -> 119c for construction bottom/left
  if( SC_in_DS == 21 && n1DeeSCEcna == 102 && TypQuad == "BL" )
    {
      if(n1SCEcha == 16){ModifiedSCEcha = 11;}
      if(n1SCEcha == 21){ModifiedSCEcha = 12;}
      if(n1SCEcha == 17){ModifiedSCEcha = 13;}
      if(n1SCEcha == 22){ModifiedSCEcha = 14;}
      if(n1SCEcha == 18){ModifiedSCEcha = 15;}
      if(n1SCEcha == 23){ModifiedSCEcha = 16;}
      if(n1SCEcha == 19){ModifiedSCEcha = 17;}
      if(n1SCEcha == 24){ModifiedSCEcha = 18;}
      if(n1SCEcha == 20){ModifiedSCEcha = 19;}
      if(n1SCEcha == 25){ModifiedSCEcha = 20;}
    }

  //..... (D1,S5), (D3,S5) SC_in_DS = 23, n1DeeSCEcna =   8 -> 286 for construction bottom/right
  if( SC_in_DS == 23 && n1DeeSCEcna ==   8 && TypQuad == "BR" )
    {
      if( n1SCEcha >=  1 &&  n1SCEcha <=  4 ){ModifiedSCEcha = n1SCEcha;}
      if( n1SCEcha >=  6 &&  n1SCEcha <=  9 ){ModifiedSCEcha = n1SCEcha-1;}
      if( n1SCEcha >= 11 &&  n1SCEcha <= 14 ){ModifiedSCEcha = n1SCEcha-2;}
      if( n1SCEcha >= 16 &&  n1SCEcha <= 19 ){ModifiedSCEcha = n1SCEcha-3;}
      if( n1SCEcha >= 21 &&  n1SCEcha <= 24 ){ModifiedSCEcha = n1SCEcha-4;}
    }

  //..... (D4,S5), (D2,S5) SC_in_DS =  6, n1DeeSCEcna =   8 -> 137 for construction bottom/left
  if( SC_in_DS ==  6 && n1DeeSCEcna ==   8 && TypQuad == "BL" )
    {
      ModifiedSCEcha = n1SCEcha;
    }
     
     //======================= ERROR message if ModifiedSCEcha is not correct
  if( ModifiedSCEcha < 1 || ModifiedSCEcha > fEcal->MaxCrysInSC() )
    {
      cout << "! *** ERROR *** > ModifiedSCEcha = " << ModifiedSCEcha
	   << ", SC_in_DS = " << SC_in_DS
	   << ", nSCCons = " << nSCCons
	   << ", n1DeeSCEcna = " << n1DeeSCEcna
	   << ", n1SCEcha = " << n1SCEcha
	   << ", ModifiedSCEcha = " << ModifiedSCEcha
	   << ", TypQuad = " << TypQuad
	   << fTTBELL << endl;
    }

 
  return ModifiedSCEcha;
}
// end of ModifiedSCEchaForNotConnectedSCs(...)

//======================================================================================
//
//                          ViewHistime: evolution in time
//
//======================================================================================

//======================================================================================
//
//                          ViewHistime: time evolution
//
//======================================================================================
void TEcnaHistos::ViewHistime(const TString& list_of_run_file_name, 
			      const Int_t&  StexStin_A, const Int_t& i0StinEcha,
			      const TString& HistoCode,  const TString& opt_plot_arg)
{
  //Histogram of the quantities as a function of time (several runs)

  TString opt_plot  = opt_plot_arg;
  TString HistoType = fCnaParHistos->GetHistoType(HistoCode);

  if( opt_plot_arg == "ONLYONE" ){opt_plot = fOnlyOnePlot;}
  if( opt_plot_arg == "SEVERAL" ){opt_plot = fSeveralPlot;}
  if( opt_plot_arg == "SAMEONE" ){opt_plot = fSameOnePlot;}

  Int_t OKHisto = 0;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Canvas already closed in option SAME %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Int_t xCanvasExists = 1; // a priori ==> SAME plot                                  //   (ViewHistime)
  if( opt_plot != fOnlyOnePlot && GetMemoFlag(HistoCode, opt_plot) == "Busy" )
    {
      TVirtualPad* main_subpad = 0; 
      //---------------- Call to ActivePad
      main_subpad = ActivePad(HistoCode.Data(), opt_plot.Data());  // => return 0 if canvas has been closed
      if( main_subpad == 0 )
	{
	  cout << "*TEcnaHistos::ViewHistime(...)> WARNING ===> Canvas has been closed in option SAME or SAME n."
	       << endl
	       << "                               Please, restart with a new canvas."
	       << fTTBELL << endl;
	  
	  ReInitCanvas(HistoCode, opt_plot);
	  xCanvasExists = 0;
	}
    }
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  //%%%%%%%%%%%%%%%%%%%%% Change of X variable in option SAME n with no proj %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Int_t SameXVarMemo = 1;   //  a priori ==> SAME n option: X variable OK                     (ViewHistime)
  if( !( HistoType == "Proj" || HistoType == "SampProj" || HistoType == "H1BasicProj" || HistoType == "EvolProj" ) )
    {
      TString XVarHisto = fCnaParHistos->GetXVarHisto(HistoCode.Data(), fFlagSubDet.Data(), fFapStexNumber);
      TString YVarHisto = fCnaParHistos->GetYVarHisto(HistoCode.Data(), fFlagSubDet.Data(), fFapStexNumber);

      if( (opt_plot == fSameOnePlot ) && GetMemoFlag(HistoCode, opt_plot) == "Free" )
	{
	  SetXVarMemo(HistoCode, opt_plot, XVarHisto);  SetYVarMemo(HistoCode, opt_plot, YVarHisto); SameXVarMemo = 1;
	}
      if( (opt_plot == fSameOnePlot ) && GetMemoFlag(HistoCode, opt_plot) == "Busy" )
	{
	  TString XVariableMemo = GetXVarFromMemo(HistoCode, opt_plot);
	  TString YVariableMemo = GetYVarFromMemo(HistoCode, opt_plot);
	  
	  if( XVarHisto != XVariableMemo )
	    {
	      cout << "!TEcnaHistos::ViewHistime(...)> *** ERROR *** ===> X coordinate changed in option SAME n." << endl
		   << "                               Present  X = " << XVarHisto << endl
		   << "                               Present  Y = " << YVarHisto << endl
		   << "                               Previous X = " << XVariableMemo << endl
		   << "                               Previous Y = " << YVariableMemo 
		   << fTTBELL << endl;
	      SameXVarMemo = 0;
	    }
	  else
	    {SetYVarMemo(HistoCode, opt_plot, YVarHisto);}
	}
    }
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  //%%%%%%%%%%%%%%%%%%%%%%% Change of Y variable in option SAME n with proj %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  Int_t SameYVarMemo = 1;   //  a priori ==> SAME n option: Y variable OK                     (ViewHistime)
  if(  HistoType == "Proj" || HistoType == "SampProj" || HistoType == "H1BasicProj" || HistoType == "EvolProj" )
    {
      TString XVarHisto = fCnaParHistos->GetXVarHisto(HistoCode.Data(), fFlagSubDet.Data(), fFapStexNumber);
      TString YVarHisto = fCnaParHistos->GetYVarHisto(HistoCode.Data(), fFlagSubDet.Data(), fFapStexNumber);

      if( (opt_plot == fSameOnePlot ) && GetMemoFlag(HistoCode, opt_plot) == "Free" )
	{
	  SetYVarMemo(HistoCode, opt_plot, YVarHisto);  SetYVarMemo(HistoCode, opt_plot, YVarHisto); SameYVarMemo = 1;
	}
      if( (opt_plot == fSameOnePlot ) && GetMemoFlag(HistoCode, opt_plot) == "Busy" )
	{
	  TString XVariableMemo = GetXVarFromMemo(HistoCode, opt_plot);
	  TString YVariableMemo = GetYVarFromMemo(HistoCode, opt_plot);
	  
	  if( YVarHisto != YVariableMemo )
	    {
	      cout << "!TEcnaHistos::ViewHistime(...)> *** ERROR *** ===> Y coordinate changed in option SAME n." << endl
		   << "                               Present  X = " << XVarHisto << endl
		   << "                               Present  Y = " << YVarHisto << endl
		   << "                               Previous X = " << XVariableMemo << endl
		   << "                               Previous Y = " << YVariableMemo 
		   << fTTBELL << endl;
	      SameYVarMemo = 0;
	    }
	  else
	    {SetYVarMemo(HistoCode, opt_plot, YVarHisto);}
	}
    }
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if( xCanvasExists == 1 && SameXVarMemo == 1 && SameYVarMemo == 1 ){OKHisto = 1;}

  //======================== Histime accepted
  if( OKHisto == 1 )
    {
      // fMyRootFile->PrintNoComment();

      fCnaParHistos->SetColorPalette(fFlagColPal);

      //................................. Init YMin and YMax of histo         //   (ViewHistime)
      if((opt_plot == fOnlyOnePlot) ||
	 (opt_plot == fSeveralPlot && GetMemoFlag(HistoCode, opt_plot) == "Free") ||
	 (opt_plot == fSameOnePlot && GetMemoFlag(HistoCode, opt_plot) == "Free"))
	{SetYminMemoFromPreviousMemo(HistoCode); SetYmaxMemoFromPreviousMemo(HistoCode);}

      //........ GetHistoryRunListParameters(...) : performs the allocation of the array fT1DRunNumber[]
      //         at first call of the present method ViewHistime
      //         increments the number of read file (fNbOfListFileEvolXXX) for option SAME
      //         and read the values fT1DRunNumber[0 to max] from the file list_of_run_file_name
      //         return the number of runs in the list of the file
      //............... Get the run parameters

      Int_t nb_of_runs_in_list = GetHistoryRunListParameters(list_of_run_file_name.Data(), HistoCode);

      if( nb_of_runs_in_list > 0 )
	{
	  //..............................  prepa x axis: time in hours
	  //Double_t sec_in_day   = (Double_t)86400.;          //===> (number of seconds in a day)
	  Double_t margin_frame_xaxis = (Double_t)25.;       //===> margin in x coordinates 

	  Double_t thstart_evol = (Double_t)0.;
	  Double_t thstop_evol  = (Double_t)0.;

	  Int_t* exist_indic  = new Int_t[nb_of_runs_in_list];    fCnew++;

	  //===================================== FIRST LOOP BEGINNING ===================================
	  //-------------------------------------------------------------------------------- (ViewHistime)
	  //
	  //     FIRST LOOP: read the "HistoryRunList" file. Check the existence of the runs
	  //                 and determine the number of existing runs.
	  //
	  //--------------------------------------------------------------------------------

	  fNbOfExistingRuns = (Int_t)0;

	  if( fFapStexNumber > 0 )
	    {
	      for(Int_t i_run = 0; i_run < nb_of_runs_in_list; i_run++)
		{
		  exist_indic[i_run] = 0;
		  // ==> set the attribute value relative to the run (fFapRunNumber)
		  SetRunNumberFromList(i_run, nb_of_runs_in_list);

		  fMyRootFile->PrintNoComment();
		  fMyRootFile->FileParameters(fFapAnaType.Data(),   fFapNbOfSamples,
					      fT1DRunNumber[i_run], fFapFirstReqEvtNumber,
					      fFapLastReqEvtNumber, fFapReqNbOfEvts,
					      fFapStexNumber,       fCfgResultsRootFilePath.Data());
		  
		  if ( fMyRootFile->LookAtRootFile() == kTRUE )             //   (ViewHistime, 1rst loop)
		    {
		      fStatusFileFound = kTRUE;

		      //------ At first HistoryRunList file: set fStartEvol... and fStopEvol... quantities
		      if( GetListFileNumber(HistoCode) == 1 )
			{
			  if( fNbOfExistingRuns == 0 ) 
			    {  
			      // start time of the first existing run of the list
			      fStartEvolTime = fMyRootFile->GetStartTime();
			      fStartEvolDate = fMyRootFile->GetStartDate();
			      fStartEvolRun  = fT1DRunNumber[i_run];
			      // start time of the last existing run of the list
			      // (in case of only one existing run in the list)
			      fStopEvolTime = fMyRootFile->GetStartTime(); 
			      fStopEvolDate = fMyRootFile->GetStartDate();
			      fStopEvolRun  = fT1DRunNumber[i_run];
			    }
			  else
			    {
			      // start time of the last existing run of the list
			      fStopEvolTime = fMyRootFile->GetStartTime();
			      fStopEvolDate = fMyRootFile->GetStartDate();
			      fStopEvolRun  = fT1DRunNumber[i_run];
			    }
			}
		      //---- set flag of run existence and increase number of existing runs 
		      //     (for the first HistoryRunList file)
		      exist_indic[i_run] = 1;
		      fNbOfExistingRuns++;
		    } // end of if ( fMyRootFile->LookAtRootFile() == kTRUE )
		  else
		    {
		      fStatusFileFound = kFALSE;

		      cout << "!TEcnaHistos::ViewHistime(...)> *ERROR* =====> "
			   << " ROOT file not found for run " << fT1DRunNumber[i_run]
			   << fTTBELL << endl << endl;
		    }
		} // end of for(Int_t i_run = 0; i_run < nb_of_runs_in_list; i_run++)

	      //===================================== FIRST LOOP END ===========================     (ViewHistime) 
	      if( fNbOfExistingRuns > 0 )
		{
		  //-------------------- recover the array after removing non existing ROOT files
		  Int_t i_existing_run = (Int_t)0;

		  for( Int_t i_run = 0; i_run < nb_of_runs_in_list;  i_run++)
		    {
		      if( exist_indic[i_run] == 1 )
			{
			  fT1DRunNumber[i_existing_run] = fT1DRunNumber[i_run];
			  i_existing_run++;
			}
		    }	 

		  //---------------------- Get start and stop time values to set the axis limits        (ViewHistime)

		  thstart_evol = (Double_t)fStartEvolTime;
		  thstop_evol  = (Double_t)fStopEvolTime;

		  Double_t xinf_lim = thstart_evol-(thstop_evol-thstart_evol)/margin_frame_xaxis;
		  Double_t xsup_lim = thstop_evol +(thstop_evol-thstart_evol)/margin_frame_xaxis;

		  Axis_t xinf_his = (Axis_t)(xinf_lim);
		  Axis_t xsup_his = (Axis_t)(xsup_lim);

		  //............................. i0StexEcha, i0Sample
		  Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(StexStin_A, i0StinEcha);
		  Int_t i0Sample = 0;
	 
		  Double_t* time_coordx  = new Double_t[fNbOfExistingRuns];   fCnew++;
		  Double_t* hval_coordy  = new Double_t[fNbOfExistingRuns];   fCnew++;
	 
		  //........... Set values to -1 

		  for( Int_t i_run = 0; i_run < fNbOfExistingRuns;  i_run++)
		    {
		      time_coordx[i_run] = (Double_t)(-1);
		      hval_coordy[i_run] = (Double_t)(-1);
		    }	 
 
		  //========================== SECOND LOOP BEGINNING =====================================
		  //----------------------------------------------------------------------- (ViewHistime)
		  //
		  //      SECOND LOOP OVER THE EXISTING RUNS : FILL THE GRAPH COORDINATES
		  //
		  //-----------------------------------------------------------------------
		  for (Int_t i_run = 0; i_run < fNbOfExistingRuns; i_run++)
		    {
		      // => set the attribute value relative to the run (fFapRunNumber)
		      SetRunNumberFromList(i_run, fNbOfExistingRuns);

		      fMyRootFile->PrintNoComment();
		      fMyRootFile->FileParameters(fFapAnaType.Data(),   fFapNbOfSamples,
						  fT1DRunNumber[i_run], fFapFirstReqEvtNumber,
						  fFapLastReqEvtNumber, fFapReqNbOfEvts,
						  fFapStexNumber,       fCfgResultsRootFilePath.Data());
	  
		      if ( fMyRootFile->LookAtRootFile() == kTRUE )    //  (ViewHistime, 2nd loop) 
			{
			  fStatusFileFound = kTRUE;

			  Bool_t ok_view_histo = GetOkViewHisto(fMyRootFile, StexStin_A, i0StinEcha, i0Sample, HistoCode);

			  //............... F I L L   G R A P H   C O O R D I N A T E S   (ViewHistime)		  
			  if( ok_view_histo == kTRUE )
			    {
			      //................................................. x coordinate
			      time_t xStartTime = fMyRootFile->GetStartTime();
			      Double_t thstart  = (Double_t)xStartTime;
			      time_coordx[i_run] = (Double_t)(thstart - xinf_lim);
			      //................................................. y coordinate
			      TVectorD read_histo(fEcal->MaxCrysEcnaInStex());
			      for(Int_t i=0; i<fEcal->MaxCrysEcnaInStex(); i++){read_histo(i)=(Double_t)0.;}
	      		     
			      if(HistoCode == "H_Ped_Date" || HistoCode == "H_Ped_RuDs")
				{read_histo = fMyRootFile->ReadPedestals(fEcal->MaxCrysEcnaInStex());}
			      if(HistoCode == "H_TNo_Date" || HistoCode == "H_TNo_RuDs")
				{read_histo = fMyRootFile->ReadTotalNoise(fEcal->MaxCrysEcnaInStex());}
			      if(HistoCode == "H_MCs_Date" || HistoCode == "H_MCs_RuDs")
				{read_histo = fMyRootFile->ReadMeanCorrelationsBetweenSamples(fEcal->MaxCrysEcnaInStex());}
	      		     
			      if(HistoCode == "H_LFN_Date" || HistoCode == "H_LFN_RuDs")
				{read_histo = fMyRootFile->ReadLowFrequencyNoise(fEcal->MaxCrysEcnaInStex());}
			      if(HistoCode == "H_HFN_Date" || HistoCode == "H_HFN_RuDs")
				{read_histo = fMyRootFile->ReadHighFrequencyNoise(fEcal->MaxCrysEcnaInStex());}
			      if(HistoCode == "H_SCs_Date" || HistoCode == "H_SCs_RuDs")
				{read_histo = fMyRootFile->ReadSigmaOfCorrelationsBetweenSamples(fEcal->MaxCrysEcnaInStex());}
			      hval_coordy[i_run] = (Double_t)read_histo(i0StexEcha);
			    }
			  else
			    {
			      cout << "!TEcnaHistos::ViewHistime(...)> Histo not available. "
				   << fTTBELL << endl;
			    }
			} // end of if ( fMyRootFile->LookAtRootFile() == kTRUE )
		      else
			{
			  fStatusFileFound = kFALSE;
			}
		    }
		  //========================== END OF SECOND LOOP ===========================================

		  //.................................................................... SCALE x and y
		  Int_t opt_scale_x = fOptScaleLinx;
		  if (fFlagScaleX == "LIN" ){opt_scale_x = fOptScaleLinx;}
		  if (fFlagScaleX == "LOG" ){opt_scale_x = fOptScaleLogx;}

		  Int_t opt_scale_y = fOptScaleLiny;
		  if (fFlagScaleY == "LIN" ){opt_scale_y = fOptScaleLiny;}
		  if (fFlagScaleY == "LOG" ){opt_scale_y = fOptScaleLogy;}

		  //------------------------------------------------- G R A P H    (ViewHistime)
		  TGraph* g_graph0 = new TGraph(fNbOfExistingRuns, time_coordx, hval_coordy); fCnewRoot++;
		  if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot)
		    {g_graph0->SetTitle(fCnaParHistos->GetQuantityName(HistoCode));}
		  if( opt_plot == fSameOnePlot )
		    {g_graph0->SetTitle(";");} 

		  //... Ymin and ymax default values will be taken if fFlagUserHistoMin/Max = "OFF"
		  //    (and if "Free" for "SAME" and "SAME n" options)
		  if((opt_plot == fOnlyOnePlot) ||
		     (opt_plot == fSeveralPlot && GetMemoFlag(HistoCode, opt_plot) == "Free") ||
		     (opt_plot == fSameOnePlot && GetMemoFlag(HistoCode, opt_plot) == "Free") )
		    {
		      SetYminMemoFromValue(HistoCode.Data(), fCnaParHistos->GetYminDefaultValue(HistoCode.Data()));
		      SetYmaxMemoFromValue(HistoCode.Data(), fCnaParHistos->GetYmaxDefaultValue(HistoCode.Data()));
		    }
	      
		  //................................ Put min max values       (ViewHistime)
		  //.......... default if flag not set to "ON"
		  //SetYminMemoFromValue(HistoCode.Data(), fCnaParHistos->GetYminDefaultValue(HistoCode.Data()));
		  //SetYmaxMemoFromValue(HistoCode.Data(), fCnaParHistos->GetYmaxDefaultValue(HistoCode.Data()));

		  g_graph0->Set(fNbOfExistingRuns);
		  Double_t graph_ymin =
		    GetYminFromGraphFrameAndMarginValue(g_graph0, fCnaParHistos->GetMarginAutoMinMax());
		  Double_t graph_ymax =
		    GetYmaxFromGraphFrameAndMarginValue(g_graph0, fCnaParHistos->GetMarginAutoMinMax());

		  //---------------------------------- G R A P H   P L O T  ---------------------------- (ViewHistime)
		  if( HistoType == "Evol" )
		    {
		      //----------------- G R A P H  Y M I N / Y M A X    M A N A G E M E N T
		      if((opt_plot == fOnlyOnePlot) ||
			 (opt_plot == fSeveralPlot && GetMemoFlag(HistoCode, opt_plot) == "Free") ||
			 (opt_plot == fSameOnePlot && GetMemoFlag(HistoCode, opt_plot) == "Free") )
			{
			  if( opt_plot == fSameOnePlot ){fHistoCodeFirst = HistoCode;} // registration of first HistoCode

			  if( fUserHistoMin >= fUserHistoMax ){fFlagUserHistoMin = "AUTO"; fFlagUserHistoMax = "AUTO";}
			  //.......... user's value if flag set to "ON"
			  if( fFlagUserHistoMin == "ON" )
			    {SetYminMemoFromValue(HistoCode.Data(), fUserHistoMin); fFlagUserHistoMin = "OFF";}
			  if( fFlagUserHistoMax == "ON" )
			    {SetYmaxMemoFromValue(HistoCode.Data(), fUserHistoMax); fFlagUserHistoMax = "OFF";}
			  //................................. automatic min and/or max
			  if( fFlagUserHistoMin == "AUTO" )
			    {SetYminMemoFromValue(HistoCode.Data(), graph_ymin); fFlagUserHistoMin = "OFF";}
			  if( fFlagUserHistoMax == "AUTO" )
			    {SetYmaxMemoFromValue(HistoCode.Data(), graph_ymax); fFlagUserHistoMax = "OFF";}
		      
			  //................................. Init Ymin and Ymax for graph
			  SetYminMemoFromPreviousMemo(HistoCode);
			  SetYmaxMemoFromPreviousMemo(HistoCode);
			}
		      //--- Set ymin and ymax to the first HistoCode values for option SAME n
		      if( opt_plot == fSameOnePlot && GetMemoFlag(HistoCode, opt_plot) == "Busy" )		  
			{
			  Double_t ymin = GetYminValueFromMemo(fHistoCodeFirst.Data());
			  SetYminMemoFromValue(HistoCode.Data(), ymin);
		      
			  Double_t ymax = GetYmaxValueFromMemo(fHistoCodeFirst.Data());
			  SetYmaxMemoFromValue(HistoCode.Data(), ymax);
			}
		  
		      //..... graph set ymin and ymax and consequently margin at top of the plot  
		      Int_t  xFlagAutoYsupMargin = SetGraphFrameYminYmaxFromMemo(g_graph0, HistoCode);
		  
		      HistimePlot(g_graph0,         xinf_his,         xsup_his,
				  HistoCode.Data(), HistoType.Data(),
				  StexStin_A,       i0StinEcha,       i0Sample,
				  opt_scale_x,      opt_scale_y,      opt_plot, xFlagAutoYsupMargin);
		      //  g_graph0->Delete();   fCdeleteRoot++;  // *===> NE PAS DELETER LE GRAPH SINON CA EFFACE TOUT!
		  
		      //--- Recover ymin and ymax from user's values in option SAME n
		      if( opt_plot == fSameOnePlot && GetMemoFlag(HistoCode, opt_plot) == "Busy" )		  
			{
			  SetYminMemoFromValue(HistoCode.Data(), fUserHistoMin);
			  SetYmaxMemoFromValue(HistoCode.Data(), fUserHistoMax);
			}
		    }
	      
		  //---------- H I S T O    Y    P R O J E C T I O N     P L O T  ---------------------- (ViewHistime)
	      
		  //======  G R A P H   P R O J   X I N F / X S U P   M A N A G E M E N T  =======  (ViewHistime)
		  //
		  //  must be done before booking because of the x <-> y permutation in case of "Proj"
		  //
		  //-----------------------------------------------------------------------------------------
		  //
		  //        CASE:    HistoType == "Proj"   OR   HistoType == "SampProj"
		  //
		  //                 Xinf and Xsup must be calculated from ymin and ymax
		  //                 of the direct graph
		  //
		  //-----------------------------------------------------------------------------------------
	      
		  if( HistoType == "EvolProj" )
		    {
		      Int_t HisSizeEvolProj = fNbBinsProj;
		      TVectorD histo_for_plot(HisSizeEvolProj);
		      for(Int_t i=0; i<HisSizeEvolProj; i++){histo_for_plot[i]=(Double_t)0.;}

		      //graph_ymin = GetYminValueFromMemo(HistoCode.Data());
		      //graph_ymax = GetYmaxValueFromMemo(HistoCode.Data());

		      TString HistoCodi = HistoCode;     // HistoCodi = direct histo

		      if( HistoCode == "H_Ped_RuDs" ){HistoCodi = "H_Ped_Date";}
		      if( HistoCode == "H_TNo_RuDs" ){HistoCodi = "H_TNo_Date";}
		      if( HistoCode == "H_LFN_RuDs" ){HistoCodi = "H_LFN_Date";}
		      if( HistoCode == "H_HFN_RuDs" ){HistoCodi = "H_HFN_Date";}
		      if( HistoCode == "H_MCs_RuDs" ){HistoCodi = "H_MCs_Date";}
		      if( HistoCode == "H_SCs_RuDs" ){HistoCodi = "H_SCs_Date";}	      

		      if( fUserHistoMin >= fUserHistoMax ){fFlagUserHistoMin = "AUTO"; fFlagUserHistoMax = "AUTO";}

		      //--------------------------------------------------------------------------- (ViewHistime)
		      //
		      //    fOnlyOnePlot => compute Xinf and Xsup at each time
		      //    fSeveralPlot => compute Xinf and Xsup once when flag = "Free" for each HistoCode
		      //    fSameOnePlot => compute Xinf and Xsup once
		      //
		      //--------------------------------------------------------------------------------------
		      if( (opt_plot == fOnlyOnePlot) ||
			  ( (opt_plot == fSeveralPlot && GetMemoFlag(HistoCode, fSeveralPlot) == "Free" ) ||
			    (opt_plot == fSameOnePlot && GetMemoFlag(HistoCode, fSameOnePlot) == "Free" ) ) )
			{
			  Double_t XinfProj =(Double_t)0;
			  Double_t XsupProj =(Double_t)0;

			  //...................................................................... (ViewHistime)
			  if( fFlagUserHistoMin == "AUTO" || fFlagUserHistoMax == "AUTO" )
			    {
			      //... Get direct graph ymin and/or ymax and keep them as xinf and xsup
			      //    in memo for the plotted histo 
			      XinfProj = fUserHistoMin;
			      XsupProj = fUserHistoMax;
			      if( fFlagUserHistoMin == "AUTO" ){XinfProj = GetYminValueFromMemo(HistoCodi.Data());}
			      if( fFlagUserHistoMax == "AUTO" ){XsupProj = GetYmaxValueFromMemo(HistoCodi.Data());}
			    } // end of  if( fFlagUserHistoMin == "AUTO" || fFlagUserHistoMax == "AUTO" )
			  else
			    {
			      if( fFlagUserHistoMin == "OFF" )
				{
				  SetYminMemoFromValue(HistoCode.Data(),
						       fCnaParHistos->GetYminDefaultValue(HistoCode.Data()));
				  XinfProj = GetYminValueFromMemo(HistoCode.Data());
				}

			      if( fFlagUserHistoMax == "OFF" )
				{
				  SetYmaxMemoFromValue(HistoCode.Data(),
						       fCnaParHistos->GetYmaxDefaultValue(HistoCode.Data()));
				  XsupProj = GetYmaxValueFromMemo(HistoCode.Data());	  
				}
			      if( fFlagUserHistoMin == "ON" ){XinfProj = fUserHistoMin;}
			      if( fFlagUserHistoMax == "ON" ){XsupProj = fUserHistoMax;}
			    }

			  if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot )
			    {
			      SetXinfMemoFromValue(HistoCode.Data(), XinfProj);
			      SetXsupMemoFromValue(HistoCode.Data(), XsupProj);			  
			    }
			  else
			    {
			      SetXinfMemoFromValue(XinfProj);
			      SetXsupMemoFromValue(XsupProj);
			    }
			} // end of if( (opt_plot == fOnlyOnePlot) || 
		      // (opt_plot == fSeveralPlot  && GetMemoFlag(HistoCode, opt_plot) == "Free") ||
		      // (opt_plot == fSameOnePlot  && GetMemoFlag(HistoCode, opt_plot) == "Free") )

		      Double_t cXinf = (Double_t)0.;
		      Double_t cXsup = (Double_t)0.;

		      //.......... Set Xinf and Xsup at each time because of simultaneous SAME options    (ViewHistime)
		      if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot )
			{
			  cXinf = GetXinfValueFromMemo(HistoCode.Data());
			  cXsup = GetXsupValueFromMemo(HistoCode.Data());
			}
		      else
			{
			  cXinf = GetXinfValueFromMemo();
			  cXsup = GetXsupValueFromMemo();
			}
		      //....... In case of only one run: in order to have cXinf < cXsup for "EvolProj" plot
		      if( cXinf >= cXsup ){cXinf -= 1.; cXsup +=1.;}

		      //..............................  histogram booking (ViewHisto)
		      Axis_t xinf_his = cXinf;  // ancillary variables since no const in arguments of TH1D
		      Axis_t xsup_his = cXsup;

		      TString TitleHisto = ";";
		      if( opt_plot != fSameOnePlot )
			{TitleHisto = fCnaParHistos->GetQuantityName(HistoCode.Data());}

		      //........ fill array histo_for_plot from hval_coordy                    (ViewHistime)
		      for(Int_t i_run=0; i_run<fNbOfExistingRuns; i_run++)
			{
			  Double_t XFromYGraph = hval_coordy[i_run];
			  Double_t binXProjY = (Double_t)HisSizeEvolProj*(XFromYGraph - cXinf)/(cXsup - cXinf);
			  Int_t ibinXProjY = (Int_t)binXProjY;
			  if( ibinXProjY >= 0 && ibinXProjY<HisSizeEvolProj ){histo_for_plot[ibinXProjY]++;}
			}

		      TH1D* h_his_evol_proj = new TH1D("histevolproj", TitleHisto.Data(),
						       HisSizeEvolProj, xinf_his, xsup_his); fCnewRoot++;

		      h_his_evol_proj->Reset();

		      //.... direct histogram filling                                                 (ViewHistime)
		      for(Int_t i=0; i<HisSizeEvolProj; i++)
			{
			  Double_t yi = (Double_t)i/(Double_t)HisSizeEvolProj*(cXsup-cXinf) + cXinf;
			  Double_t his_val = (Double_t)histo_for_plot[i];
			  h_his_evol_proj->Fill(yi, his_val);
			}

		      //------- H I S T O   P R O J    Y M I N / Y M A X    M A N A G E M E N T
		      if( fUserHistoMin >= fUserHistoMax ){fFlagUserHistoMin = "AUTO"; fFlagUserHistoMax = "AUTO";}
		      //.......... user's value if flag set to "ON"
		      if( fFlagUserHistoMin == "ON" )
			{SetYminMemoFromValue(HistoCode.Data(), fUserHistoMin); fFlagUserHistoMin = "OFF";}
		      if( fFlagUserHistoMax == "ON" )
			{SetYmaxMemoFromValue(HistoCode.Data(), fUserHistoMax); fFlagUserHistoMax = "OFF";}
		      //................................. automatic min and/or max
		      if( fFlagUserHistoMin == "AUTO" )
			{SetYminMemoFromValue(HistoCode.Data(), graph_ymin); fFlagUserHistoMin = "OFF";}
		      if( fFlagUserHistoMax == "AUTO" )
			{SetYmaxMemoFromValue(HistoCode.Data(), graph_ymax); fFlagUserHistoMax = "OFF";}

		      //................................. Init Ymin and Ymax for graph
		      SetYminMemoFromPreviousMemo(HistoCode);
		      SetYmaxMemoFromPreviousMemo(HistoCode);

		      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		      //.......... Find maximum in case of Proj (LIN scale only) // PS: EvolProj => in ViewHistime
		      if( fFlagScaleY == "LIN" )
		      	{
		      	  SetYmaxMemoFromValue
		      	    (HistoCode.Data(),
		      	     GetYmaxFromHistoFrameAndMarginValue(h_his_evol_proj, fCnaParHistos->GetMarginAutoMinMax()));
		      	}
		      
		      //--- Set ymin and ymax to the first HistoCode values for option SAME n
		      if( opt_plot == fSameOnePlot && GetMemoFlag(HistoCode, opt_plot) == "Busy" )		  
			{
			  Double_t ymin = GetYminValueFromMemo(fHistoCodeFirst.Data());
			  SetYminMemoFromValue(HistoCode.Data(), ymin);
			  
			  Double_t ymax = GetYmaxValueFromMemo(fHistoCodeFirst.Data());
			  SetYmaxMemoFromValue(HistoCode.Data(), ymax);
			}
		      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		      
		      //..... graph set ymin and ymax and consequently margin at top of the plot  
		      Int_t  xFlagAutoYsupMargin = SetGraphFrameYminYmaxFromMemo(g_graph0, HistoCode);
		      Int_t  arg_AlreadyRead = 0;

		      HistoPlot(h_his_evol_proj,  HisSizeEvolProj,
				xinf_his,         xsup_his,
				HistoCode.Data(), HistoType.Data(),
				StexStin_A,       i0StinEcha,       i0Sample,
				opt_scale_x,      opt_scale_y,      opt_plot, arg_AlreadyRead,
				xFlagAutoYsupMargin);

		      h_his_evol_proj->Delete();   h_his_evol_proj = 0;                         fCdeleteRoot++;
		      //*===> deleter l'histo sinon "Replacing existing histo (potential memory leak)" a l'execution

		    } // end of if( HistoType == "EvolProj" )
		  //---------------------------------------------------------------------------------- (ViewHistime)

		  delete [] time_coordx;   time_coordx = 0;      fCdelete++;   
		  delete [] hval_coordy;   hval_coordy = 0;      fCdelete++;  
		}
	      else
		{
		  cout << "!TEcnaHistos::ViewHistime(...)> The list of runs in file: " << list_of_run_file_name 
		       << " has " << nb_of_runs_in_list << " run numbers" << endl
		       << " but none of them correspond to an existing ROOT file."
		       << fTTBELL << endl;
		}
	    } // end of if( fFapStexNumber > 0 )
	  else
	    {
	      cout << "!TEcnaHistos::ViewHistime(...)> *ERROR* =====> "
		   << fFapStexName << " number = " << fFapStexNumber << ". "
		   << fFapStexName << " number must be in range [1," << fEcal->MaxStexInStas() << "] ";
	      if( fFlagSubDet == "EB" ){cout << " (or [-18,+18])";}
		cout << fTTBELL << endl;
	    }
	  delete [] exist_indic;  exist_indic = 0;         fCdelete++;
	} // end of if( nb_of_runs_in_list > 0 )
      else
	{
	  if( nb_of_runs_in_list == 0 )
	    {
	      cout << "!TEcnaHistos::ViewHistime(...)> The list of runs in file: " << list_of_run_file_name
		   << " is empty !" << fTTBELL << endl;
	    }
	  if( nb_of_runs_in_list < 0 )
	    {
	      cout << "!TEcnaHistos::ViewHistime(...)> " << list_of_run_file_name
		   << ": file not found in directory: " << fCfgHistoryRunListFilePath.Data() << fTTBELL << endl;
	    }
	}
    }  // end of if( OKHisto == 1 )
} // end of ViewHistime

//------------------------------------------------------------------------------------
//
//      GetHistoryRunListParameters(...), AllocArraysForEvol(), GetListFileNumber(...)
//
//------------------------------------------------------------------------------------

Int_t TEcnaHistos::GetHistoryRunListParameters(const TString& list_of_run_file_name, const TString& HistoCode)
{
  // Build the array of run numbers from the list-of-runs .ascii file.
  // Return the list size
  // *=====> list_of_run_file_name is the name of the ASCII file containing the list of the runs
  //
  // SYNTAX OF THE FILE:
  //
  // HistoryRunList_EB_SM6_Analysis_1.ascii    <- 1rst line: comment (file name for example)
  // 73677                                 <- 2nd  line and others: run numbers (empty lines accepted)
  // 73688          
  // 73689
  //
  // 73690
  // 73692 
  //
  // In option SAME (of TEcnaHistos), several lists of runs can be called and these lists can have
  // DIFFERENT sizes (here the "size" is the number of runs of the list). In addition,
  // some runs in some lists may not exist in reality. So, we must adopt a convention which is
  // the following: the number of runs corresponds to the number of EXISTING runs
  // of the FIRST read list. Let be N1 this number.
  // If another list has more runs than N1 runs, we read only the first N1 runs.
  // If another list has less runs than N1 runs, we read all the runs of this list. 
  //
  //--------------------------------------------------------------------------------------------------

  Int_t nb_of_runs_in_list = 0;
  
  //========= immediate return if file name is an empty string
  if( list_of_run_file_name.Data() == '\0' )
    {
      cout << "!TEcnaHistos::GetHistoryRunListParameters(...)> *** ERROR *** =====> "
	   << " EMPTY STRING for list of run file name." << fTTBELL << endl;
    }
  else
    {
      // ===> increase the HistoryRunList file number
      if ( HistoCode == "H_Ped_Date" ){fNbOfListFileH_Ped_Date++;}
      if ( HistoCode == "H_TNo_Date" ){fNbOfListFileH_TNo_Date++;}
      if ( HistoCode == "H_MCs_Date" ){fNbOfListFileH_MCs_Date++;}
      if ( HistoCode == "H_LFN_Date" ){fNbOfListFileH_LFN_Date++;}
      if ( HistoCode == "H_HFN_Date" ){fNbOfListFileH_HFN_Date++;}
      if ( HistoCode == "H_SCs_Date" ){fNbOfListFileH_SCs_Date++;}

      if ( HistoCode == "H_Ped_RuDs" ){fNbOfListFileH_Ped_RuDs++;}
      if ( HistoCode == "H_TNo_RuDs" ){fNbOfListFileH_TNo_RuDs++;}
      if ( HistoCode == "H_MCs_RuDs" ){fNbOfListFileH_MCs_RuDs++;}
      if ( HistoCode == "H_LFN_RuDs" ){fNbOfListFileH_LFN_RuDs++;}
      if ( HistoCode == "H_HFN_RuDs" ){fNbOfListFileH_HFN_RuDs++;}
      if ( HistoCode == "H_SCs_RuDs" ){fNbOfListFileH_SCs_RuDs++;}
      
      fFapFileRuns = list_of_run_file_name.Data();    // (short name)
      
      //........... Add the path to the file name           ( GetHistoryRunListParameters )
      TString xFileNameRunList = list_of_run_file_name.Data();
      const Text_t *t_file_name = (const Text_t *)xFileNameRunList.Data();

      //.............. replace the string "$HOME" by the true $HOME path
      if(fCfgHistoryRunListFilePath.BeginsWith("$HOME"))
	{
	  fCfgHistoryRunListFilePath.Remove(0,5);
	  const Text_t *t_file_nohome = (const Text_t *)fCfgHistoryRunListFilePath.Data(); //  /scratch0/cna/...
	  
	  TString home_path = gSystem->Getenv("HOME");
	  fCfgHistoryRunListFilePath = home_path;             //  /afs/cern.ch/u/USER
	  fCfgHistoryRunListFilePath.Append(t_file_nohome);   //  /afs/cern.ch/u/USER/scratch0/cna/...
	}
     
      xFileNameRunList = fCfgHistoryRunListFilePath.Data();
      
      xFileNameRunList.Append('/');
      xFileNameRunList.Append(t_file_name);
      
      fFcin_f.open(xFileNameRunList.Data());

      //.......................................           ( GetHistoryRunListParameters )
      if( fFcin_f.fail() == kFALSE )
	{   
	  //...................................... first reading to get the number of runs in the list
	  fFcin_f.clear();
	  string xHeadComment;
	  fFcin_f >> xHeadComment;
	  Int_t cRunNumber;
	  Int_t list_size_read = 0;
	  
	  while( !fFcin_f.eof() ){fFcin_f >> cRunNumber; list_size_read++;}
	  fFapNbOfRuns = list_size_read - 1;

	  //...................................... second reading to get the run numbers

	  //====== Return to the beginning of the file =====
	  fFcin_f.clear();
	  fFcin_f.seekg(0, ios::beg);
	  //================================================

	  string yHeadComment;
	  fFcin_f >> yHeadComment;

	  //....................... Set fFapMaxNbOfRuns to -1 at first call (first read file)
	  //
	  //                        fNbOfListFileEvolXXX is initialized to 0 in Init()
	  //                        It is incremented once here above
	  //                        So, at first call fNbOfListFileEvolXXX = 1
	  //                        then fFapMaxNbOfRuns = -1
	  //..........................................................................   (GetHistoryRunListParameters)
	  if( (HistoCode == "H_Ped_Date" && fNbOfListFileH_Ped_Date == 1) || 
	      (HistoCode == "H_TNo_Date" && fNbOfListFileH_TNo_Date == 1) ||
	      (HistoCode == "H_MCs_Date" && fNbOfListFileH_MCs_Date == 1) ||
	      (HistoCode == "H_LFN_Date" && fNbOfListFileH_LFN_Date == 1) || 
	      (HistoCode == "H_HFN_Date" && fNbOfListFileH_HFN_Date == 1) ||
	      (HistoCode == "H_SCs_Date" && fNbOfListFileH_SCs_Date == 1) ||
	      (HistoCode == "H_Ped_RuDs" && fNbOfListFileH_Ped_RuDs == 1) || 
	      (HistoCode == "H_TNo_RuDs" && fNbOfListFileH_TNo_RuDs == 1) ||
	      (HistoCode == "H_MCs_RuDs" && fNbOfListFileH_MCs_RuDs == 1) ||
	      (HistoCode == "H_LFN_RuDs" && fNbOfListFileH_LFN_RuDs == 1) || 
	      (HistoCode == "H_HFN_RuDs" && fNbOfListFileH_HFN_RuDs == 1) ||
	      (HistoCode == "H_SCs_RuDs" && fNbOfListFileH_SCs_RuDs == 1)){fFapMaxNbOfRuns = -1;}
	  
	  // first call: fFapMaxNbOfRuns = fFapNbOfRuns = nb of run from the first reading 
	  if( fFapMaxNbOfRuns == -1 ){fFapMaxNbOfRuns = fFapNbOfRuns;}
	  // next calls: fFapNbOfRuns must not be greater than fFapMaxNbOfRuns found at first time
	  else{if( fFapNbOfRuns > fFapMaxNbOfRuns ){fFapNbOfRuns = fFapMaxNbOfRuns;}}

	  // Allocation and initialization of the array fT1DRunNumber[].
	  //................. check maximum value for allocation
	  if( fFapMaxNbOfRuns > fCnaParHistos->MaxNbOfRunsInLists() )
	    {
	      cout << "TEcnaHistos::GetHistoryRunListParameters(...)> Max number of runs in HistoryRunList = "
		   << fFapMaxNbOfRuns
		   << " too large, forced to parameter TEcnaParHistos->fMaxNbOfRunsInLists value (= "
		   << fCnaParHistos->MaxNbOfRunsInLists()
		   << "). Please, set this parameter to a larger value than " << fFapMaxNbOfRuns
		   << fTTBELL << endl;
	      fFapMaxNbOfRuns = fCnaParHistos->MaxNbOfRunsInLists();
	    }
	  //................................. Alloc of the array and init
	  if( fT1DRunNumber == 0 )
	    {
	      if( fFapMaxNbOfRuns > 0 )
		{
		  fT1DRunNumber = new Int_t[fFapMaxNbOfRuns];               fCnew++;
		}
	      else
		{
		  cout << "!TEcnaHistos::GetHistoryRunListParameters(...)> *** ERROR *** =====> fFapMaxNbOfRuns = "
		       << fFapMaxNbOfRuns << ". Forced to 1." << fTTBELL << endl;
		  fFapMaxNbOfRuns = 1;
		  fT1DRunNumber = new Int_t[fFapMaxNbOfRuns];               fCnew++;
		}
	    }


	  //.................................... Init the list of runs	  
	  for ( Int_t i_run = 0; i_run < fFapMaxNbOfRuns; i_run++ ){fT1DRunNumber[i_run] = -1;}
	  //.................................... read the list of runs
	  for (Int_t i_list = 0; i_list < fFapNbOfRuns; i_list++)
	    {
	      fFcin_f >> cRunNumber;
	      fT1DRunNumber[i_list] = cRunNumber;
	    }
	  //........................................           ( GetHistoryRunListParameters )
	  nb_of_runs_in_list = fFapNbOfRuns;
	  fFcin_f.close();
	}
      else
	{
	  fFcin_f.clear();
	  cout << "!TEcnaHistos::GetHistoryRunListParameters(...)> *** ERROR *** =====> "
	       << xFileNameRunList.Data() << " : file not found." << fTTBELL << endl;
	  nb_of_runs_in_list = -1;
	}
    }
  return nb_of_runs_in_list;
}
//   end of GetHistoryRunListParameters(...)

//------------------------------------------------------------------------------------------------

Int_t TEcnaHistos::GetListFileNumber(const TString& HistoCode)
{
// Get the number of the read list file
  
  Int_t number = 0;

  if ( HistoCode == "H_Ped_Date"){number = fNbOfListFileH_Ped_Date;}
  if ( HistoCode == "H_TNo_Date"){number = fNbOfListFileH_TNo_Date;}
  if ( HistoCode == "H_MCs_Date"){number = fNbOfListFileH_MCs_Date;}
  if ( HistoCode == "H_LFN_Date"){number = fNbOfListFileH_LFN_Date;}
  if ( HistoCode == "H_HFN_Date"){number = fNbOfListFileH_HFN_Date;}
  if ( HistoCode == "H_SCs_Date"){number = fNbOfListFileH_SCs_Date;}
  if ( HistoCode == "H_Ped_RuDs"){number = fNbOfListFileH_Ped_RuDs;}
  if ( HistoCode == "H_TNo_RuDs"){number = fNbOfListFileH_TNo_RuDs;}
  if ( HistoCode == "H_MCs_RuDs"){number = fNbOfListFileH_MCs_RuDs;}
  if ( HistoCode == "H_LFN_RuDs"){number = fNbOfListFileH_LFN_RuDs;}
  if ( HistoCode == "H_HFN_RuDs"){number = fNbOfListFileH_HFN_RuDs;}
  if ( HistoCode == "H_SCs_RuDs"){number = fNbOfListFileH_SCs_RuDs;}
  return number;
}

//--------------------------------------------------------------------------------------------------
void TEcnaHistos::SetRunNumberFromList(const Int_t& xArgIndexRun, const Int_t& MaxNbOfRuns)
{
  // Set run number for the xArgIndexRun_th run in the list of runs (evolution plots)
  //     The array fT1DRunNumber[] have been obtained from a previous call
  //     to GetHistoryRunListParameters(xFileNameRunList, HistoCode)
  
  if( xArgIndexRun >= 0 && xArgIndexRun < MaxNbOfRuns)
    {
      fFapRunNumber  = fT1DRunNumber[xArgIndexRun];
      if( xArgIndexRun == 0 ){InitSpecParBeforeFileReading();} // SpecPar = Special Parameters (dates, times, run types)
    }
  else
    {
      cout << "!TEcnaHistos::SetRunNumberFromList(...)> **** ERROR **** Run index out of range in list of runs. xArgIndexRun = "
	   << xArgIndexRun << " (MaxNbOfRuns = "<< MaxNbOfRuns << ")" << endl;
    }
}

//--------------------------------------------------------------------------------------------------
void TEcnaHistos::InitSpecParBeforeFileReading()
{
  // Init parameters that will be set by reading the info which are in the results ROOT file
  // SpecPar = Special Parameters (dates, times, run types)

  Int_t MaxCar = fgMaxCar;
  fStartDate.Resize(MaxCar);
  fStartDate = "(date not found)";

  MaxCar = fgMaxCar;
  fStopDate.Resize(MaxCar);
  fStopDate  = "(date not found)";

  fStartTime = (time_t)0;
  fStopTime  = (time_t)0;

  fRunType   = "(run type not found)";

} // ------------- ( end of InitSpecParBeforeFileReading() ) -------------

//======================================================================================
//
//           C O M M E N T S / I N F O S     P A V E S      M E T H O D S 
//
//======================================================================================

Bool_t TEcnaHistos::GetOkViewHisto(TEcnaRead*    aMyRootFile,
				   const Int_t&  StexStin_A, const Int_t& i0StinEcha, const Int_t& i0Sample,
				   const TString& HistoCode)
{
// Check possibility to plot the histo

  Bool_t ok_view_histo = kFALSE;

  TString HistoType = fCnaParHistos->GetHistoType(HistoCode);

  TString root_file_name = aMyRootFile->GetRootFileNameShort();

  TVectorD  vStin(fEcal->MaxStinEcnaInStex());
  for(Int_t i=0; i<fEcal->MaxStinEcnaInStex() ; i++){vStin(i)=(Double_t)0.;}
  vStin = aMyRootFile->ReadStinNumbers(fEcal->MaxStinEcnaInStex());

  if( aMyRootFile->DataExist() == kTRUE )
    {
      fStatusDataExist = kTRUE;

      Int_t Stin_ok = 0;
      for (Int_t index_Stin = 0; index_Stin < fEcal->MaxStinEcnaInStex(); index_Stin++)
	{
	  if ( vStin(index_Stin) == StexStin_A ){Stin_ok++;};
	}
      
      //.............................................. ok_view   
      Int_t ok_view    = 1;
      
      if( !( HistoType == "Global" || HistoType == "Proj" )  )
	{
	  if( Stin_ok != 1)
	  {
	    Int_t StinNumber = StexStin_A;
	    if( fFlagSubDet == "EE" )
	      {StinNumber = fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fFapStexNumber, StexStin_A);}
	    cout << "!TEcnaHistos::GetOkViewHisto(...)> *ERROR* =====> " << "File: " << root_file_name
		 << ", " << fFapStinName.Data() << " "
		 << StinNumber
		 << " not found. Available numbers = ";
	    for(Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++)
	      {
		if( vStin(i) > 0 )
		  {
		    if( fFlagSubDet == "EB" ){cout << vStin(i) << ", ";}
		    if( fFlagSubDet == "EE" )
		      {cout << fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fFapStexNumber, (Int_t)vStin(i)) << ", ";}
		  }
	      }
	    cout << fTTBELL << endl;
	    ok_view = -1;
	  }
	  else 
	    { 
	      ok_view = 1;
	    }
	}
      
      //.............................................. ok_max_elt   ( GetOkViewHisto(...) )
      Int_t ok_max_elt  = 1;
      
      if( ( ( (HistoType == "H1Basic") || (HistoType == "Evol") || (HistoType == "EvolProj") )
	    && (i0StinEcha >= 0) && (i0StinEcha<fEcal->MaxCrysInStin()) 
	    && (i0Sample  >= 0) && (i0Sample <fFapNbOfSamples ) ) ||
	  !( (HistoType == "H1Basic") || (HistoType == "Evol") || (HistoType == "EvolProj") ))
	{ok_max_elt = 1;} 
      else
	{
	  Int_t Choffset = 0;
	  if( fFlagSubDet == "EE" ){Choffset = 1;}
	  if( ( (HistoType == "H1Basic") || (HistoType == "Evol") || (HistoType == "EvolProj") )
	      && !( (i0StinEcha >= 0) && (i0StinEcha<fEcal->MaxCrysInStin()) ) )
	    {cout << "!TEcnaHistos::GetOkViewHisto(...)> *ERROR* =====> " << "File: " << root_file_name
		  << ". Wrong channel number. Value = " << i0StinEcha << " (required range: [" << Choffset << ", "
		  << fEcal->MaxCrysInStin()-1+Choffset << "] )"
		  << fTTBELL << endl;}
	  if( (HistoCode == "D_Adc_EvDs" || HistoCode == "D_Adc_EvNb") &&
	      !((i0Sample >= 0) && (i0Sample <fFapNbOfSamples)) )
	    {cout << "!TEcnaHistos::GetOkViewHisto(...)> *ERROR* =====> " << "File: " << root_file_name
		  << ". Wrong sample index. Value = " << i0Sample << " (required range: [0, "
		  << fFapNbOfSamples-1 << "] )"
		  << fTTBELL << endl;}
	  ok_max_elt = -1;
	}
      
      if( (ok_view == 1) && (ok_max_elt == 1) )
	{
	  ok_view_histo = kTRUE;
	}
      else
	{
	  cout << "!TEcnaHistos::GetOkViewHisto(...)> At least one ERROR has been detected. ok_view = " << ok_view
	       << ", ok_max_elt = " << ok_max_elt << fTTBELL << endl;
	}
    }
  else
    {
      fStatusDataExist = kFALSE;

      cout << "!TEcnaHistos::GetOkViewHisto(...)> No data in ROOT file "
	   << ", aMyRootFile->DataExist() = " << aMyRootFile->DataExist() << fTTBELL << endl;
    }
  return ok_view_histo;
}
//..............................................................................................

Int_t TEcnaHistos::SetHistoFrameYminYmaxFromMemo(TH1D* h_his0, const TString& HistoCode)
{
// Set min and max according to HistoCode
  
  // if Ymin = Ymax (or Ymin > Ymax): nothing done here
  // return xFlagAutoYsupMargin = 1
  //
  // if Ymin < Ymax: min and max calculated by h_his0->SetMinimum() and h_his0->SetMaximum()
  // return xFlagAutoYsupMargin = 0

  Int_t xFlagAutoYsupMargin = 1;                             //  (SetHistoFrameYminYmaxFromMemo)
  
  if(HistoCode == "D_NOE_ChNb"){ 
    if(fD_NOE_ChNbYmin < fD_NOE_ChNbYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_NOE_ChNbYmin);  h_his0->SetMaximum(fD_NOE_ChNbYmax);}}
  
  if(HistoCode == "D_NOE_ChDs"){
    if(fD_NOE_ChDsYmin < fD_NOE_ChDsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_NOE_ChDsYmin); h_his0->SetMaximum(fD_NOE_ChDsYmax);}}
  
  if(HistoCode == "D_Ped_ChNb"){
    if(fD_Ped_ChNbYmin < fD_Ped_ChNbYmax){xFlagAutoYsupMargin = 0;
	h_his0->SetMinimum(fD_Ped_ChNbYmin);  h_his0->SetMaximum(fD_Ped_ChNbYmax);}}
  
  if(HistoCode == "D_Ped_ChDs"){
    if(fD_Ped_ChDsYmin < fD_Ped_ChDsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_Ped_ChDsYmin); h_his0->SetMaximum(fD_Ped_ChDsYmax);}}
  
  if(HistoCode == "D_TNo_ChNb"){
    if(fD_TNo_ChNbYmin < fD_TNo_ChNbYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_TNo_ChNbYmin); h_his0->SetMaximum(fD_TNo_ChNbYmax);}}
  
  if(HistoCode == "D_TNo_ChDs"){
    if(fD_TNo_ChDsYmin < fD_TNo_ChDsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_TNo_ChDsYmin); h_his0->SetMaximum(fD_TNo_ChDsYmax);}}
  
  if(HistoCode == "D_MCs_ChNb"){
    if(fD_MCs_ChNbYmin < fD_MCs_ChNbYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_MCs_ChNbYmin); h_his0->SetMaximum(fD_MCs_ChNbYmax);}}
  
  if(HistoCode == "D_MCs_ChDs"){
    if(fD_MCs_ChDsYmin < fD_MCs_ChDsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_MCs_ChDsYmin); h_his0->SetMaximum(fD_MCs_ChDsYmax);}}
  
  if(HistoCode == "D_LFN_ChNb"){
    if(fD_LFN_ChNbYmin < fD_LFN_ChNbYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_LFN_ChNbYmin); h_his0->SetMaximum(fD_LFN_ChNbYmax);}}
  
  if(HistoCode == "D_LFN_ChDs"){
    if(fD_LFN_ChDsYmin < fD_LFN_ChDsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_LFN_ChDsYmin); h_his0->SetMaximum(fD_LFN_ChDsYmax);}}
  
  if(HistoCode == "D_HFN_ChNb"){
    if(fD_HFN_ChNbYmin < fD_HFN_ChNbYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_HFN_ChNbYmin); h_his0->SetMaximum(fD_HFN_ChNbYmax);}}
  
  if(HistoCode == "D_HFN_ChDs"){
    if(fD_HFN_ChDsYmin < fD_HFN_ChDsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_HFN_ChDsYmin); h_his0->SetMaximum(fD_HFN_ChDsYmax);}}

  if(HistoCode == "D_SCs_ChNb"){
    if(fD_SCs_ChNbYmin < fD_SCs_ChNbYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_SCs_ChNbYmin); h_his0->SetMaximum(fD_SCs_ChNbYmax);}}
  
  if(HistoCode == "D_SCs_ChDs"){
    if(fD_SCs_ChDsYmin < fD_SCs_ChDsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_SCs_ChDsYmin); h_his0->SetMaximum(fD_SCs_ChDsYmax);}}
  
  if(HistoCode == "D_MSp_SpNb"){
    if(fD_MSp_SpNbYmin < fD_MSp_SpNbYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_MSp_SpNbYmin); h_his0->SetMaximum(fD_MSp_SpNbYmax);}}
   
  if(HistoCode == "D_MSp_SpDs"){
    if(fD_MSp_SpDsYmin < fD_MSp_SpDsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_MSp_SpDsYmin); h_his0->SetMaximum(fD_MSp_SpDsYmax);}}
   
  if(HistoCode == "D_SSp_SpNb"){
    if(fD_SSp_SpNbYmin < fD_SSp_SpNbYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_SSp_SpNbYmin); h_his0->SetMaximum(fD_SSp_SpNbYmax);}}
   
  if(HistoCode == "D_SSp_SpDs"){
    if(fD_SSp_SpDsYmin < fD_SSp_SpDsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_SSp_SpDsYmin); h_his0->SetMaximum(fD_SSp_SpDsYmax);}}
  
  if(HistoCode == "D_Adc_EvNb"){
    if(fD_Adc_EvNbYmin < fD_Adc_EvNbYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_Adc_EvNbYmin); h_his0->SetMaximum(fD_Adc_EvNbYmax);}}

  if(HistoCode == "D_Adc_EvDs"){
    if(fD_Adc_EvDsYmin < fD_Adc_EvDsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fD_Adc_EvDsYmin); h_his0->SetMaximum(fD_Adc_EvDsYmax);}}

  if(HistoCode == "H2CorccInStins"){
    if(fH2CorccInStinsYmin < fH2CorccInStinsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fH2CorccInStinsYmin); h_his0->SetMaximum(fH2CorccInStinsYmax);}}

  if(HistoCode == "H2LFccMosMatrix"){
    if(fH2LFccMosMatrixYmin < fH2LFccMosMatrixYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fH2LFccMosMatrixYmin); h_his0->SetMaximum(fH2LFccMosMatrixYmax);}}

  if(HistoCode == "H2HFccMosMatrix"){
    if(fH2HFccMosMatrixYmin < fH2HFccMosMatrixYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fH2HFccMosMatrixYmin); h_his0->SetMaximum(fH2HFccMosMatrixYmax);}}

  if(HistoCode == "H_Ped_RuDs"){
    if(fH_Ped_RuDsYmin < fH_Ped_RuDsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fH_Ped_RuDsYmin); h_his0->SetMaximum(fH_Ped_RuDsYmax);}}

  if(HistoCode == "H_TNo_RuDs"){
    if(fH_TNo_RuDsYmin < fH_TNo_RuDsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fH_TNo_RuDsYmin); h_his0->SetMaximum(fH_TNo_RuDsYmax);}}

  if(HistoCode == "H_MCs_RuDs"){
    if(fH_MCs_RuDsYmin < fH_MCs_RuDsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fH_MCs_RuDsYmin); h_his0->SetMaximum(fH_MCs_RuDsYmax);}}

  if(HistoCode == "H_LFN_RuDs"){
    if(fH_LFN_RuDsYmin < fH_LFN_RuDsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fH_LFN_RuDsYmin); h_his0->SetMaximum(fH_LFN_RuDsYmax);}}

  if(HistoCode == "H_HFN_RuDs"){
    if(fH_HFN_RuDsYmin < fH_HFN_RuDsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fH_HFN_RuDsYmin); h_his0->SetMaximum(fH_HFN_RuDsYmax);}}

  if(HistoCode == "H_SCs_RuDs"){
    if(fH_SCs_RuDsYmin < fH_SCs_RuDsYmax){xFlagAutoYsupMargin = 0;
    h_his0->SetMinimum(fH_SCs_RuDsYmin); h_his0->SetMaximum(fH_SCs_RuDsYmax);}}

  return xFlagAutoYsupMargin;  
} // end of  SetHistoFrameYminYmaxFromMemo

Int_t TEcnaHistos::SetGraphFrameYminYmaxFromMemo(TGraph* g_graph0, const TString& HistoCode)
{
// Set min and max according to HistoCode
  
  Int_t xFlagAutoYsupMargin = 1;    // DEFAULT: 1 = min and max calulated by ROOT, 0 = by this code 

  if(HistoCode == "H_Ped_Date"){
    if(fH_Ped_DateYmin < fH_Ped_DateYmax){xFlagAutoYsupMargin = 0;
    g_graph0->SetMinimum(fH_Ped_DateYmin); g_graph0->SetMaximum(fH_Ped_DateYmax);}}

  if(HistoCode == "H_TNo_Date"){
    if(fH_TNo_DateYmin < fH_TNo_DateYmax){xFlagAutoYsupMargin = 0;
    g_graph0->SetMinimum(fH_TNo_DateYmin); g_graph0->SetMaximum(fH_TNo_DateYmax);}}

  if(HistoCode == "H_MCs_Date"){
    if(fH_MCs_DateYmin < fH_MCs_DateYmax){xFlagAutoYsupMargin = 0;
    g_graph0->SetMinimum(fH_MCs_DateYmin); g_graph0->SetMaximum(fH_MCs_DateYmax);}}

  if(HistoCode == "H_LFN_Date"){
    if(fH_LFN_DateYmin < fH_LFN_DateYmax){xFlagAutoYsupMargin = 0;
    g_graph0->SetMinimum(fH_LFN_DateYmin); g_graph0->SetMaximum(fH_LFN_DateYmax);}}

  if(HistoCode == "H_HFN_Date"){
    if(fH_HFN_DateYmin < fH_HFN_DateYmax){xFlagAutoYsupMargin = 0;
    g_graph0->SetMinimum(fH_HFN_DateYmin); g_graph0->SetMaximum(fH_HFN_DateYmax);}}

  if(HistoCode == "H_SCs_Date"){
    if(fH_SCs_DateYmin < fH_SCs_DateYmax){xFlagAutoYsupMargin = 0;
    g_graph0->SetMinimum(fH_SCs_DateYmin); g_graph0->SetMaximum(fH_SCs_DateYmax);}}

  return xFlagAutoYsupMargin;  
} // end of SetGraphFrameYminYmaxFromMemo

//..............................................................................................
Double_t TEcnaHistos::GetYminFromHistoFrameAndMarginValue(TH1D* h_his0, const Double_t margin_factor)
{
//Calculation for automatic minimum with margin
  Double_t minproj = h_his0->GetMinimum();
  Double_t maxproj = h_his0->GetMaximum();
  minproj = minproj - (maxproj-minproj)*margin_factor;
  return minproj;
}

Double_t TEcnaHistos::GetYmaxFromHistoFrameAndMarginValue(TH1D* h_his0, const Double_t margin_factor)
{
//Calculation for automatic maximum with margin
  Double_t minproj = h_his0->GetMinimum();
  Double_t maxproj = h_his0->GetMaximum();
  maxproj = maxproj + (maxproj-minproj)*margin_factor;
  return maxproj;
}

Double_t TEcnaHistos::GetYminFromGraphFrameAndMarginValue(TGraph* g_graph0, const Double_t margin_factor)
{
//Calculation for automatic minimum with margin
  Double_t graph_ymin = g_graph0->GetY()[0];
  for(Int_t i=1; i<g_graph0->GetN(); i++)
    {if( g_graph0->GetY()[i] < graph_ymin ){graph_ymin = g_graph0->GetY()[i];}}

  Double_t graph_ymax = g_graph0->GetY()[0];
  for(Int_t i=1; i<g_graph0->GetN(); i++)
    {if( g_graph0->GetY()[i] > graph_ymax ){graph_ymax = g_graph0->GetY()[i];}}

  graph_ymin = graph_ymin - (graph_ymax-graph_ymin)*margin_factor;
  return graph_ymin;
}
Double_t TEcnaHistos::GetYmaxFromGraphFrameAndMarginValue(TGraph* g_graph0, const Double_t margin_factor)
{
//Calculation for automatic maximum with margin
  Double_t graph_ymin = g_graph0->GetY()[0];
  for(Int_t i=1; i<g_graph0->GetN(); i++)
    {if( g_graph0->GetY()[i] < graph_ymin ){graph_ymin = g_graph0->GetY()[i];}}

  Double_t graph_ymax = g_graph0->GetY()[0];
  for(Int_t i=1; i<g_graph0->GetN(); i++)
    {if( g_graph0->GetY()[i] > graph_ymax ){graph_ymax = g_graph0->GetY()[i];}}

  graph_ymax = graph_ymax + (graph_ymax-graph_ymin)*margin_factor;
  return graph_ymax;
}
//----------------------------------------------- HistoPlot

void TEcnaHistos::HistoPlot(TH1D* h_his0,               const Int_t&   HisSize,  
			    const Axis_t& xinf_his,     const Axis_t&  xsup_his,
			    const TString& HistoCode,    const TString&  HistoType,
			    const Int_t&  StexStin_A,   const Int_t&   i0StinEcha,  const Int_t& i0Sample, 
			    const Int_t&  opt_scale_x,  const Int_t&   opt_scale_y,
			    const TString& opt_plot,     const Int_t&   arg_AlreadyRead,
			    const Int_t&  xFlagAutoYsupMargin)
{
  // Plot 1D histogram

  UInt_t canv_w = fCnaParHistos->SetCanvasWidth(HistoCode.Data(), opt_plot);
  UInt_t canv_h = fCnaParHistos->SetCanvasHeight(HistoCode.Data(), opt_plot);

  TString QuantityName = "                                ";
  Int_t MaxCar = fgMaxCar;
  QuantityName.Resize(MaxCar);
  QuantityName = fCnaParHistos->GetQuantityName(HistoCode.Data());


  if( arg_AlreadyRead == 0 || arg_AlreadyRead == 1 )
    {
      SetHistoPresentation(h_his0, HistoType.Data(), opt_plot);    // (gStyle parameters)
      //.................................................. prepa paves commentaires (HistoPlot)
      SetAllPavesViewHisto(HistoCode.Data(), StexStin_A, i0StinEcha, i0Sample, opt_plot.Data(), arg_AlreadyRead);
    }
  
  //..................................................... Canvas name (HistoPlot) 
  TString canvas_name = SetCanvasName(HistoCode.Data(), opt_scale_x, opt_scale_y,
				      opt_plot.Data(),  arg_AlreadyRead, 
				      StexStin_A,       i0StinEcha,  i0Sample);
  //..................................................... Canvas allocations (HistoPlot)
  TCanvas* MainCanvas = 0;

  if(opt_plot == fOnlyOnePlot && (arg_AlreadyRead == 0 || arg_AlreadyRead == 1 ) )
    {
      MainCanvas = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h);   fCnewRoot++;
      fCurrentPad = gPad; fCurrentCanvas = MainCanvas; fCurrentCanvasName = canvas_name.Data();
    }
  
  if( opt_plot == fSeveralPlot ||  opt_plot == fSameOnePlot)
    {if(GetMemoFlag(HistoCode, opt_plot) == "Free")
      {MainCanvas = CreateCanvas(HistoCode, opt_plot, canvas_name, canv_w, canv_h);
      fCurrentPad = gPad; fCurrentCanvas = MainCanvas; fCurrentCanvasName = canvas_name.Data();}}

  // cout << "*TEcnaHistos::HistoPlot(...)> Plot is displayed on canvas ----> " << canvas_name.Data() << endl;

  //--------------- EE => SC for construction, EB => Xtal in SM (default: Stin ECNA number, i0StinEcha)
  Int_t Stex_StinCons = StexStin_A;   // Stex_StinCons = Tower for EB, SC for construction for EE
  Int_t n1StexCrys    = i0StinEcha+1; // n1StexCrys = Crys in SM for EB, ECNA

  if( StexStin_A >= 1 && StexStin_A <= fEcal->MaxStinEcnaInStex() )
    {
      if( fFlagSubDet == "EB" )
	{n1StexCrys  = fEcalNumbering->Get1StexCrysFrom1StexStinAnd0StinEcha(StexStin_A, i0StinEcha, fFapStexNumber);}
      if( fFlagSubDet == "EE" )
	{Stex_StinCons = fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fFapStexNumber, StexStin_A);}
    }

  //------ String for StexNumber ( to display "EB" or "EE" if Stex = 0 )
  TString sFapStexNumber = StexNumberToString(fFapStexNumber);

  //============================================================================= (HistoPlot)
  //
  //     1st  OPERATIONS:  Pave texts preparation and first Draw.
  //                       SetParametersCanvas
  //                       Set Memo Flags.
  //                       Set main_subpad and main_pavtxt
  //
  //=============================================================================
  TVirtualPad* main_subpad = 0; // main_subpad: Pad for the histo
  TPaveText*   main_pavtxt = 0; // Pave for the "Several Changing" parameters (options SAME and SAME n)

  Int_t xMemoPlotSame = 1; // a priori ==> SAME plot

  //========================================= Option ONLYONE    (HistoPlot)
  if( opt_plot == fOnlyOnePlot )
    {
      if( arg_AlreadyRead == 0 || arg_AlreadyRead == 1 )
	{
	  //.................................... Draw titles and paves (pad = main canvas)
	  if( fPavComGeneralTitle != 0 ){fPavComGeneralTitle->Draw();}
	  fPavComStex->Draw();
	  if( !( HistoType == "Global" || HistoType == "Proj" ) ){fPavComStin->Draw(); fPavComXtal->Draw();}

	  if( HistoType == "EvolProj" )
	    {
	      fPavComEvolRuns->Draw();
	      fPavComEvolNbOfEvtsAna->Draw();
	    }
	  else
	    {
	      fPavComAnaRun->Draw();
	      fPavComNbOfEvts->Draw();
	    }

	  Double_t x_low = fCnaParHistos->BoxLeftX("bottom_left_box")    - 0.005;
	  Double_t x_up  = fCnaParHistos->BoxRightX("bottom_right_box")  + 0.005;
	  Double_t y_low = fCnaParHistos->BoxTopY("bottom_right_box")    + 0.005;
	  Double_t y_up  = fCnaParHistos->BoxBottomY("top_right_box_EB") - 0.005;
	  Color_t  fond_pad = fCnaParHistos->ColorDefinition("blanc");

	  Double_t x_margin = x_low;
	  Double_t y_margin = y_low;
	  MainCanvas->Divide( 1,  1, x_margin, y_margin, fond_pad);
	  //           Divide(nx, ny, x_margin, y_margin,    color);

	  gPad->cd(1);
	  main_subpad = gPad;
	  main_subpad->SetPad(x_low, y_low, x_up, y_up);
	
	  xMemoPlotSame = 0;
	}
      if (arg_AlreadyRead > 1 )
	{main_subpad = fCurrentPad;}

    } // end of if(opt_plot == fOnlyOnePlot && (arg_AlreadyRead == 0 || arg_AlreadyRead == 1 ) )

  //========================================= Options SAME and SAME n  (HistoPlot)
  if( (opt_plot == fSeveralPlot) || (opt_plot == fSameOnePlot) )
    {
      //..................... First call in options SAME and SAME n
      if( GetMemoFlag(HistoCode, opt_plot) == "Free" )
	{
	  //Call to SetParametersPavTxt
	  //fPavTxt<HISTOCODE> = fPavComSeveralChanging;  => come from SetAllPavesViewHisto
	  SetParametersPavTxt(HistoCode, opt_plot);

	  //---------------- Call to ActivePavTxt
	  // main_pavtxt = fPavTxt<HISTOCODE>;} (return main_pavtxt)
	  main_pavtxt = ActivePavTxt(HistoCode.Data(), opt_plot.Data());

	  //---------------------------- Set texts for pave "several changing", options SAME and SAME n
	  if( fPavComGeneralTitle != 0 ){fPavComGeneralTitle->Draw();}

	  main_pavtxt->SetTextAlign(fTextPaveAlign);
	  main_pavtxt->SetTextFont(fTextPaveFont);
	  main_pavtxt->SetBorderSize(fTextBorderSize);
	  Float_t cTextPaveSize  = 0.025;
	  if( HistoType == "H1Basic" || HistoType == "SampProj" || HistoType == "H1BasicProj" ||
	      HistoType == "Proj"    || HistoType == "EvolProj"  )
	    {cTextPaveSize = 0.025;}
	  main_pavtxt->SetTextSize(cTextPaveSize);

	  char* f_in = new char[fgMaxCar];                            fCnew++;

	  //------------------------------------------------------------ titles pave "several" (HistoPlot)
	  TString DecalStexName = "";
	  if( fFlagSubDet == "EB" ){DecalStexName = " ";}
	  TString DecalStinName = "";
	  if( fFlagSubDet == "EE" ){DecalStinName = "   ";}

	  TString sStexOrStasName = "";
	  if( fFapStexNumber == 0 ){sStexOrStasName = "  ";}
	  if( fFapStexNumber != 0 ){sStexOrStasName = fFapStexName;}

	  if( opt_plot == fSeveralPlot || opt_plot == fSameOnePlot )
	    {
	      if( HistoType == "SampGlobal" )
		{sprintf(f_in, "Analysis   Samp   RUN# (run type            )  Evts range  Nb Evts   %s%s %s%s %s %s Sample",
			 DecalStexName.Data(), sStexOrStasName.Data(),
			 DecalStinName.Data(), fFapStinName.Data(), fFapXtalName.Data(), fFapEchaName.Data());}
	      if( HistoType == "SampProj" )
		{sprintf(f_in, "Analysis   Samp   RUN# (run type            )  Evts range  Nb Evts   %s%s %s%s %s %s Sample",
			 DecalStexName.Data(), sStexOrStasName.Data(),
			 DecalStinName.Data(), fFapStinName.Data(), fFapXtalName.Data(), fFapEchaName.Data());}
	      if( HistoType == "H1Basic" || HistoType == "H1BasicProj" )
		{sprintf(f_in, "Analysis   Samp   RUN# (run type            )  Evts range  Nb Evts   %s%s %s%s %s %s",
			 DecalStexName.Data(), sStexOrStasName.Data(),
			 DecalStinName.Data(), fFapStinName.Data(), fFapXtalName.Data(), fFapEchaName.Data());}
	      if((HistoType == "Global") || (HistoType == "Proj") )
		{sprintf(f_in, "Analysis   Samp   RUN# (run type            )  Evts range  Nb Evts   %s%s",
			 DecalStexName.Data(), sStexOrStasName.Data());}

	      if( HistoType == "EvolProj" )
		{sprintf(f_in, "Analysis   Samp   Evts range  Nb Evts   %s%s  %s%s %s %s",
			 DecalStexName.Data(), sStexOrStasName.Data(),
			 DecalStinName.Data(), fFapStinName.Data(), fFapXtalName.Data(), fFapEchaName.Data());}
	    }

	  TText* ttit = main_pavtxt->AddText(f_in);
	  ttit->SetTextColor(fCnaParHistos->ColorDefinition("noir"));
	  
	  //------------------------------------------------------------ values pave "several"  (HistoPlot)

	  //.................................... option SAME n only  (HistoPlot)
	  if( opt_plot == fSameOnePlot)
	    {
	      if( (HistoType == "Global") || (HistoType == "Proj") || (HistoType == "H1BasicProj") )
		{
		  sprintf(f_in, "%-10s 1-%2d%7d (%-20s) %5d-%5d  %7d  %4s %-25s",
			  fFapAnaType.Data(), fFapNbOfSamples, fFapRunNumber, fRunType.Data(),
			  fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts,
			  sFapStexNumber.Data(), QuantityName.Data());
		}

	      if( HistoType == "EvolProj" )
		{
		  sprintf(f_in, "%-10s 1-%2d %5d-%5d  %7d %5s%6d%7d%7d %-25s",
			  fFapAnaType.Data(), fFapNbOfSamples, fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts,
			  sFapStexNumber.Data(), Stex_StinCons, n1StexCrys, i0StinEcha, QuantityName.Data());
		}
	      
	    } // end of if for option SAME n only

	  //..................................... option SAME (HistoPlot)
	  if( opt_plot == fSeveralPlot )
	    {
	      Int_t kSample = i0Sample+1;  // Sample number range = [1,n<=10]

	      if( HistoType == "SampGlobal" )
		{
		  sprintf(f_in, "%-10s 1-%2d%7d (%-20s) %5d-%5d  %7d  %4s%6d%5d%5d%6d",
			  fFapAnaType.Data(), fFapNbOfSamples, fFapRunNumber, fRunType.Data(),
			  fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts,sFapStexNumber.Data(),
			  Stex_StinCons, n1StexCrys, i0StinEcha, kSample);
		}
	      if( HistoType == "SampProj"  )
		{
		  sprintf(f_in, "%-10s 1-%2d%7d (%-20s) %5d-%5d  %7d  %4s%6d%5d%5d%6d",
			  fFapAnaType.Data(), fFapNbOfSamples, fFapRunNumber, fRunType.Data(),
			  fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts, sFapStexNumber.Data(),
			  Stex_StinCons, n1StexCrys, i0StinEcha, kSample);
		}	      
	      if( HistoType == "H1Basic" || HistoType == "H1BasicProj" )
		{
		  sprintf(f_in, "%-10s 1-%2d%7d (%-20s) %5d-%5d  %7d  %4s%6d%5d%5d",
			  fFapAnaType.Data(), fFapNbOfSamples, fFapRunNumber, fRunType.Data(),
			  fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts, sFapStexNumber.Data(),
			  Stex_StinCons, n1StexCrys, i0StinEcha);
		}	      
	      if( (HistoType == "Global") || (HistoType == "Proj")  )
		{
		  sprintf(f_in, "%-10s 1-%2d%7d (%-20s) %5d-%5d  %7d  %4s",
			  fFapAnaType.Data(), fFapNbOfSamples, fFapRunNumber,  fRunType.Data(),
			  fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts, sFapStexNumber.Data());
		}
	      
	      if( HistoType == "EvolProj" )
		{
		  sprintf(f_in, "%-10s 1-%2d  %5d-%5d  %7d  %4s%7d%5d%5d",
			  fFapAnaType.Data(), fFapNbOfSamples,
			  fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts, sFapStexNumber.Data(), 
			  Stex_StinCons, n1StexCrys, i0StinEcha);
		}
	    } 
	  
	  TText* tt = main_pavtxt->AddText(f_in);
	  tt->SetTextColor(GetViewHistoColor(HistoCode, opt_plot));
	  
	  delete [] f_in;   f_in = 0;                                          fCdelete++;

	  //---------------- Draw the "several changing" pave with its text in the Canvas (AT FIRST TIME)
	  main_pavtxt->Draw();
	  //---------------- Draw evol run pave if "EvolProj" (AT FIRST TIME)  
	  if( HistoType == "EvolProj" ){fPavComEvolRuns->Draw();}

	  //---------------- Call to SetParametersCanvas
	  //fImp<HISTOCODE> = (TCanvasImp*)fCanv<HISTOCODE>->GetCanvasImp();
	  //fCanv<HISTOCODE>->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  //fPad<HISTOCODE> = gPad;
	  //fMemoPlot<HISTOCODE> = 1;          =======>  set MemoFlag to "Buzy"
	  //fMemoColor<HISTOCODE> = 0;
	  SetParametersCanvas(HistoCode, opt_plot);	  

	  //---------------- Set xMemoPlotSame to 0
	  xMemoPlotSame = 0;
	} // end of if( GetMemoFlag(HistoCode, opt_plot) == "Free" )    (HistoPlot)

      //.......... First and further calls in options SAME and SAME n (fMemoPlot<HISTOCODE> = 1)
      if(GetMemoFlag(HistoCode, opt_plot) == "Busy")
	{
	  //---------------- Call to ActivePavTxt
	  // main_pavtxt = fPavTxt<HISTOCODE>;} (return main_pavtxt)
	  main_pavtxt = ActivePavTxt(HistoCode.Data(), opt_plot.Data());
	  
	  //---------------- Call to ActivePad
	  main_subpad = ActivePad(HistoCode.Data(), opt_plot.Data());  // => return 0 if canvas has been closed

	  //---------------- Recover pointer to the current canvas
	  MainCanvas = GetCurrentCanvas(HistoCode.Data(), opt_plot.Data());
	}
    } // end of if( (opt_plot == fSeveralPlot) || (opt_plot == fSameOnePlot) )


  //============================================================================= (HistoPlot)
  //
  //     2nd  OPERATIONS: Write and Draw the parameter values in the
  //                      "several changing" pave (options SAME and SAME n)
  //                      and Draw Histo
  //=============================================================================
  if(main_subpad != 0)
    { 
      if( (opt_plot == fSeveralPlot) || (opt_plot == fSameOnePlot) )
	{
	  //------------------------------------------------------------ values	 
	  if(xMemoPlotSame != 0)
	    {
	      // main_pavtxt = fPavComSeveralChanging = fPavTxt<HISTOCODE>
	      main_pavtxt->SetTextAlign(fTextPaveAlign);
	      main_pavtxt->SetTextFont(fTextPaveFont);
	      main_pavtxt->SetBorderSize(fTextBorderSize);
	      Float_t cTextPaveSize  = 0.025;
	      if( HistoType == "H1Basic" || HistoType == "SampProj"
		  || HistoType == "Proj" || HistoType == "EvolProj" || HistoType == "H1BasicProj" )
		{cTextPaveSize = 0.025;}
	      main_pavtxt->SetTextSize(cTextPaveSize);
	      
	      char* f_in = new char[fgMaxCar];                            fCnew++;
	      
	      if( opt_plot == fSameOnePlot )
		{
		  if( (HistoType == "Global") || (HistoType == "Proj") )
		    {
		      sprintf(f_in, "%-10s 1-%2d%7d (%-20s) %5d-%5d  %7d  %4s %-25s",
			      fFapAnaType.Data(), fFapNbOfSamples, fFapRunNumber, fRunType.Data(),
			      fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts,
			      sFapStexNumber.Data(), QuantityName.Data());
		    }

		  if( HistoType == "EvolProj" )
		    {
		      sprintf(f_in, "%-10s 1-%2d %5d-%5d  %7d %5s%6d%7d%7d %-25s",
			      fFapAnaType.Data(), fFapNbOfSamples,
			      fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts, sFapStexNumber.Data(),
			      Stex_StinCons, n1StexCrys, i0StinEcha, QuantityName.Data());
		    }
		}

	      if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot )
		{
		  Int_t kSample = i0Sample+1;  // Sample number range = [1,n<=10] (HistoPlot)

		  if(HistoType == "SampGlobal" )
		    {
		      sprintf(f_in, "%-10s 1-%2d%7d (%-20s) %5d-%5d  %7d  %4s%6d%5d%5d%6d",
			      fFapAnaType.Data(), fFapNbOfSamples, fFapRunNumber, fRunType.Data(),
			      fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts, sFapStexNumber.Data(),
			      Stex_StinCons, n1StexCrys, i0StinEcha, kSample);
		    }
		  if( HistoType == "SampProj" )
		    {
		      sprintf(f_in, "%-10s 1-%2d%7d (%-20s) %5d-%5d  %7d  %4s%6d%5d%5d%6d",
			      fFapAnaType.Data(), fFapNbOfSamples, fFapRunNumber, fRunType.Data(),
			      fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts, sFapStexNumber.Data(),
			      Stex_StinCons, n1StexCrys, i0StinEcha, kSample);
		    }		  
		  if( HistoType == "H1Basic" || HistoType == "H1BasicProj")
		    {
		      sprintf(f_in, "%-10s 1-%2d%7d (%-20s) %5d-%5d  %7d  %4s%6d%5d%5d",
			      fFapAnaType.Data(), fFapNbOfSamples, fFapRunNumber, fRunType.Data(),
			      fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts, sFapStexNumber.Data(),
			      Stex_StinCons, n1StexCrys, i0StinEcha);
		    }
		  if( (HistoType == "Global") || (HistoType == "Proj") )
		    {
		      sprintf(f_in, "%-10s 1-%2d%7d (%-20s) %5d-%5d  %7d  %4s",
			      fFapAnaType.Data(), fFapNbOfSamples, fFapRunNumber, fRunType.Data(),
			      fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts, sFapStexNumber.Data());
		    }
		  
		  if( HistoType == "EvolProj" )
		    {
		      sprintf(f_in, "%-10s 1-%2d  %5d-%5d  %7d  %4s%7d%5d%5d",
			      fFapAnaType.Data(), fFapNbOfSamples,
			      fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts,
			      sFapStexNumber.Data(), Stex_StinCons, n1StexCrys, i0StinEcha);
		    }
		}
	       
	      TText *tt = main_pavtxt->AddText(f_in);
	      tt->SetTextColor(GetViewHistoColor(HistoCode, opt_plot));
	      MainCanvas->cd(); gStyle->SetOptDate(0);
	      main_pavtxt->Draw();

	      delete [] f_in;   f_in = 0;                      fCdelete++;
	    }
	  
	  main_subpad->cd();
	  Double_t x_low = fCnaParHistos->BoxLeftX("bottom_left_box")   - 0.005;
	  Double_t x_up  = fCnaParHistos->BoxRightX("bottom_right_box") + 0.005;
	  Double_t y_low = fCnaParHistos->BoxTopY("several_plots_box")  + 0.005;
	  Double_t y_up  = fCnaParHistos->BoxBottomY("general_comment") - 0.005;
	  if( opt_plot == fSameOnePlot && HistoType == "EvolProj" )
	    {y_up = fCnaParHistos->BoxBottomY("top_left_box_EB") - 0.005;}
	  main_subpad->SetPad(x_low, y_low, x_up, y_up);	  
	} // end of if( (opt_plot == fSeveralPlot) || (opt_plot == fSameOnePlot) )

      //............................................ Style	(HistoPlot)
      SetViewHistoColors(h_his0, HistoCode.Data(), opt_plot, arg_AlreadyRead);

      //................................. Set axis titles
      TString axis_x_var_name = SetHistoXAxisTitle(HistoCode);
      TString axis_y_var_name = SetHistoYAxisTitle(HistoCode);
      h_his0->GetXaxis()->SetTitle(axis_x_var_name);
      h_his0->GetYaxis()->SetTitle(axis_y_var_name);

      Int_t lin_scale = 0;
      Int_t log_scale = 1;
      
      if(opt_scale_x == fOptScaleLinx){gPad->SetLogx(lin_scale);}
      if(opt_scale_x == fOptScaleLogx){gPad->SetLogx(log_scale);}
      if(opt_scale_y == fOptScaleLiny){gPad->SetLogy(lin_scale);}
      if(opt_scale_y == fOptScaleLogy){gPad->SetLogy(log_scale);}	      

      //---------------------------------------------------------------- Draw histo	(HistoPlot)
      if(opt_plot == fOnlyOnePlot && arg_AlreadyRead == 0 ){h_his0->DrawCopy();}
      if(opt_plot == fOnlyOnePlot && arg_AlreadyRead == 1 ){h_his0->DrawCopy();}
      if(opt_plot == fOnlyOnePlot && arg_AlreadyRead >  1 ){h_his0->DrawCopy("AHSAME");}

      if(opt_plot == fSeveralPlot || opt_plot == fSameOnePlot)
	{
	  if(xMemoPlotSame == 0){h_his0->DrawCopy();}
	  if(xMemoPlotSame != 0){h_his0->DrawCopy("SAME");}
	}
      //----------------------------------------------------------------

      //.................... Horizontal line at y=0	(HistoPlot)
      if( !(  HistoCode == "D_Adc_EvDs" || HistoCode == "D_Adc_EvNb" ||
	      HistoType == "Proj"       || HistoType == "SampProj" ||
	      HistoType == "EvolProj"   || HistoType == "H1BasicProj"  ) &&
	  !( HistoType == "H1Basic" && arg_AlreadyRead == 0 ) )
	{
	  Double_t yinf = h_his0->GetMinimum();
	  Double_t ysup = h_his0->GetMaximum();
	  if( yinf <= (Double_t)0. && ysup >= (Double_t)0. )
	    {TLine* lin =  new TLine(0.,0.,(Double_t)HisSize, 0.);   fCnewRoot++;
	    lin->Draw();}
	}

      if( ( opt_plot == fOnlyOnePlot )
	  || ( (opt_plot == fSeveralPlot || opt_plot == fSameOnePlot) && xMemoPlotSame == 0 ) )
	{
	  Double_t yinf = (Double_t)h_his0->GetMinimum();
	  Double_t ysup = (Double_t)h_his0->GetMaximum();

	  if(xFlagAutoYsupMargin == 1)
	    {
	      if( yinf >= ysup ){yinf = (Double_t)0.; ysup += ysup;}  // ROOT default if ymin >= ymax
	      Double_t MaxMarginFactor = (Double_t)0.05;    // frame top line = 5% above the maximum
	      ysup += (ysup-yinf)*MaxMarginFactor;       // ROOT default if ymin < ymax
	    }

	  char* f_in = new char[fgMaxCar];               fCnew++;

	  //.................... Vertical lines for Data sectors (EE Global plot only)
	  if( fFlagSubDet == "EE" && fFapStexNumber == 0 )
	    {
	      //............................................................. Data Sectors	(HistoPlot)
	      Color_t coul_DS = fCnaParHistos->ColorDefinition("noir");
	      Int_t DeeOffset = 0;
	      for(Int_t n1Dee=1; n1Dee<=4; n1Dee++)
		{
		  if( n1Dee == 4 ){DeeOffset = 0;}
		  if( n1Dee == 3 ){DeeOffset =   fEcal->MaxSCForConsInDee();}   // 149
		  if( n1Dee == 2 ){DeeOffset = 2*fEcal->MaxSCForConsInDee();}   // 446
		  if( n1Dee == 1 ){DeeOffset = 3*fEcal->MaxSCForConsInDee();}   // 595

		  Double_t ydee = ysup + (ysup-yinf)/(Double_t)15.;
		  Double_t xBinDee = DeeOffset + fEcal->MaxSCForConsInDee()/(Double_t)2.;
		  sprintf( f_in, "D%d", n1Dee );
		  TText *text_Dee = new TText(xBinDee, ydee, f_in);  fCnewRoot++;
		  text_Dee->SetTextColor(coul_DS);
		  text_Dee->SetTextSize((Double_t)0.04);
		  text_Dee->Draw("SAME");

		  Double_t DSLabelOffset = (Double_t)12;

		  for(Int_t i=1; i<5; i++)
		    {
		      Int_t iDS = 0;
		      if( n1Dee == 1 ){iDS = i;}
		      if( n1Dee == 2 ){iDS = i+4;}
		      if( n1Dee == 3 ){iDS = i+5;}
		      if( n1Dee == 4 ){iDS = i+1;}

		      Double_t xBinDS = DeeOffset + (Double_t)GetDSOffset(n1Dee, iDS)/fEcal->MaxCrysInSC();
		      TLine* lin_DS = new TLine(xBinDS, yinf, xBinDS, ysup);   fCnewRoot++;
		      lin_DS->SetLineColor(coul_DS);
		      lin_DS->SetLineWidth(1);
		      lin_DS->SetLineStyle(2);
		      lin_DS->Draw();

		      if( n1Dee == 2 && i == 4 )
			{
			  TLine* lin_DSp = new TLine(DeeOffset, yinf, DeeOffset, ysup);   fCnewRoot++;
			  lin_DSp->SetLineColor(coul_DS);
			  lin_DSp->SetLineWidth(1);
			  lin_DSp->SetLineStyle(2);
			  lin_DSp->Draw();
			}

		      Double_t yds = ysup + (ysup-yinf)/(Double_t)50.;
		      Double_t xBinDSp = xBinDS + DSLabelOffset;
		      Int_t nDS = iDS;

		      sprintf( f_in, "S%d", nDS );
		      TText *text_DS = new TText(xBinDSp, yds, f_in);  fCnewRoot++;
		      text_DS->SetTextColor(coul_DS);
		      text_DS->SetTextSize((Double_t)0.03);
		      text_DS->Draw("SAME");
		      if( (n1Dee == 4 && i == 1) || (n1Dee == 2 && i == 4) )
			{
			  if(n1Dee == 4){nDS = iDS-1;}
			  if(n1Dee == 2){nDS = iDS+1;}
			  sprintf( f_in, "S%d", nDS );
			  TText *text_DS = new TText(xBinDS-1.75*DSLabelOffset, yds, f_in);  fCnewRoot++;
			  text_DS->SetTextColor(coul_DS);
			  text_DS->SetTextSize((Double_t)0.03);
			  text_DS->Draw("SAME");
			}
		    }
		}
	    }

	  //........... Vertical lines for Data sectors and special SC's (EE only, Dee's Global plots)	(HistoPlot)
	  if( fFlagSubDet == "EE" && fFapStexNumber > 0 )
	    {
	      if( HistoType == "Global" )
		{
		  Double_t ytext = yinf - (ysup-yinf)/8.5;		  
		  //............................................................. Data Sectors
		  Color_t coul_DS = fCnaParHistos->ColorDefinition("noir");
		  for(Int_t i=1; i<5; i++)
		    {
		      Int_t iDS = 0;
		      if( fFapStexNumber == 1 ){iDS = i;}
		      if( fFapStexNumber == 2 ){iDS = i+4;}
		      if( fFapStexNumber == 3 ){iDS = i+5;}
		      if( fFapStexNumber == 4 ){iDS = i+1;}
		      
		      Double_t xBinDS = (Double_t)GetDSOffset(fFapStexNumber, iDS);
		      TLine* lin_DS = new TLine(xBinDS, yinf, xBinDS, ysup);   fCnewRoot++;
		      lin_DS->SetLineColor(coul_DS);
		      lin_DS->SetLineWidth(2);
		      lin_DS->SetLineStyle(2);
		      lin_DS->Draw();
		      Double_t ytextds = ysup + (ysup-yinf)/30.;
		      Double_t xBinDSNumber =
			xBinDS + fEcalNumbering->GetMaxSCInDS(iDS)*fEcal->MaxCrysInSC()/(Double_t)2.25;
		      sprintf( f_in, "S%d", iDS );
		      TText *text_DS = new TText(xBinDSNumber, ytextds, f_in);  fCnewRoot++;
		      text_DS->SetTextColor(coul_DS);
		      text_DS->SetTextSize((Double_t)0.04);
		      text_DS->Draw("SAME");
		      if( ( (fFapStexNumber == 1 || fFapStexNumber == 2 ) && i == 4 ) ||
			  ( (fFapStexNumber == 3 || fFapStexNumber == 4 ) && i == 1 ) )
			{
			  Int_t iDSp = iDS;
			  if( i == 4 ){iDSp = iDS+1;}
			  if( i == 1 ){iDSp = iDS-1;}
			  sprintf( f_in, "S%d", iDSp);
			  Double_t xBinpDSNumber =
			    xBinDSNumber - fEcalNumbering->GetMaxSCInDS(iDS)*fEcal->MaxCrysInSC();
			  TText *text_DSp = new TText(xBinpDSNumber, ytextds, f_in);  fCnewRoot++;
			  text_DSp->SetTextColor(coul_DS);
			  text_DSp->SetTextSize((Double_t)0.04);
			  text_DSp->Draw("SAME");
			}
		    }
		  //.............................................................. Vertical lines for SC's
		  //                                                       Trop serre. A garder en reserve.
		  //for(Int_t i=0; i<fEcal->MaxSCForConsInDee(); i++)
		  //  {
		  //    Double_t xBinSC =(Double_t)(fEcal->MaxCrysInSC()*i);
		  //    TLine* lin_SC = new TLine(xBinSC, yinf, xBinSC, ysup);  fCnewRoot++;
		  //    lin_SC->SetLineColor(coul_DS);
		  //    lin_SC->SetLineStyle(3);
		  //    lin_SC->Draw();
		  //  }
		  //............................................................... Not connected SC's
		  Color_t coul_notconnected = fCnaParHistos->ColorDefinition("bleu_fonce");
		  for(Int_t i=1; i<=fEcal->NumberOfNotConnectedSCs(); i++)
		    {
		      Int_t index = 0;
		      if( fFapStexNumber == 1 || fFapStexNumber == 3 ){index = 2*i - 1;}
		      if( fFapStexNumber == 2 || fFapStexNumber == 4 ){index = 2*i;}
		      //................. display of the not connected SC's numbers (+ vertical line)
		      Double_t xBinNotConnectedSC = NotConnectedSCH1DBin(index);
		      TLine* lin_notconnected =
			new TLine(xBinNotConnectedSC, yinf, xBinNotConnectedSC, ysup);  fCnewRoot++;
		      lin_notconnected->SetLineColor(coul_notconnected);
		      lin_notconnected->SetLineStyle(3);
		      lin_notconnected->Draw();
		      
		      Double_t xBinNotConnectedSCEnd = NotConnectedSCH1DBin(index)+fEcal->MaxCrysInSC();
		      TLine* lin_notconnected_end =
			new TLine(xBinNotConnectedSCEnd, yinf, xBinNotConnectedSCEnd, ysup); fCnewRoot++;
		      lin_notconnected_end->SetLineColor(coul_notconnected);
		      lin_notconnected_end->SetLineStyle(3);
		      lin_notconnected_end->Draw();
		      
		      //sprintf( f_in, "%d", GetNotConnectedSCForConsFromIndex(index) );
		      sprintf( f_in, "%d", GetNotConnectedDSSCFromIndex(index) );
		      TText *text_SC_NotConnected = new TText(xBinNotConnectedSC, ytext, f_in);  fCnewRoot++;
		      text_SC_NotConnected->SetTextAngle((Double_t)45.);
		      text_SC_NotConnected->SetTextColor(coul_notconnected);
		      text_SC_NotConnected->SetTextFont(42);
		      text_SC_NotConnected->SetTextSize((Double_t)0.03);
		      text_SC_NotConnected->Draw("SAME");
		    }
		  //Double_t xtext = xinf_his - (xsup_his-xinf_his)/8.;
		  //Double_t ytextp = yinf - (ysup-yinf)/6.;
		  //sprintf( f_in, "Special SC => ");
		  //TText *text_legend_NotConnected = new TText(xtext, ytext, f_in);  fCnewRoot++;
		  //text_legend_NotConnected->SetTextColor(coul_notconnected);
		  //text_legend_NotConnected->SetTextSize((Double_t)0.03);
		  //text_legend_NotConnected->Draw("SAME");
		  
		  //............................................................... Not complete SC's
		  Color_t coul_notcomplete = fCnaParHistos->ColorDefinition("rouge40");
		  for(Int_t i=1; i<=fEcal->NumberOfNotCompleteSCs(); i++)
		    {
		      Int_t index = 0;
		      if( fFapStexNumber == 1 || fFapStexNumber == 3 ){index = 2*i - 1;}
		      if( fFapStexNumber == 2 || fFapStexNumber == 4 ){index = 2*i;}
		      //................. display of the not complete SC's numbers (+ vertical line)
		      Double_t xBinNotCompleteSC = NotCompleteSCH1DBin(index);
		      TLine* lin_notcomplete =
			new TLine(xBinNotCompleteSC, yinf, xBinNotCompleteSC, ysup);   fCnewRoot++;
		      lin_notcomplete->SetLineColor(coul_notcomplete);
		      lin_notcomplete->SetLineStyle(3);
		      lin_notcomplete->Draw();
		      
		      Double_t xBinNotCompleteSCEnd = NotCompleteSCH1DBin(index)+fEcal->MaxCrysInSC();;
		      TLine* lin_notcomplete_end =
			new TLine(xBinNotCompleteSCEnd, yinf, xBinNotCompleteSCEnd, ysup);   fCnewRoot++;
		      lin_notcomplete_end->SetLineColor(coul_notcomplete);
		      lin_notcomplete_end->SetLineStyle(3);
		      lin_notcomplete_end->Draw();

		      sprintf( f_in, "%d", GetNotCompleteDSSCFromIndex(index) );
		      // sprintf( f_in, "%d", GetNotCompleteSCForConsFromIndex(index) );
		      TText *text_SC_NotComplete = new TText(xBinNotCompleteSC, ytext, f_in);  fCnewRoot++;
		      text_SC_NotComplete->SetTextAngle((Double_t)45.);
		      text_SC_NotComplete->SetTextColor(coul_notcomplete);
		      text_SC_NotComplete->SetTextFont(42);
		      text_SC_NotComplete->SetTextSize((Double_t)0.03);
		      text_SC_NotComplete->Draw("SAME");
		    }
		  //Double_t xtextp = xinf_his + (xsup_his-xinf_his)/15.;
		  //sprintf( f_in, "Not complete SC");
		  //TText *text_legend_NotComplete = new TText(xtextp, ytextp, f_in);  fCnewRoot++;
		  //text_legend_NotComplete->SetTextColor(coul_notcomplete);
		  //text_legend_NotComplete->SetTextSize((Double_t)0.03);
		  //text_legend_NotComplete->Draw("SAME");
		}
	    }
	  delete [] f_in;   f_in = 0;                      fCdelete++;
	} // end of if( ( opt_plot == fOnlyOnePlot )
	  // || ( (opt_plot == fSeveralPlot || opt_plot == fSameOnePlot) && xMemoPlotSame == 0 ) )

      //..............................................Top Axis (HistoPlot)
      Int_t min_value = 0;
      Int_t max_value = 0;
      if(HistoType == "Global")
	{
	  if( fFapStexNumber > 0 )
	    {
	      //.......................... Axis for the Stin numbers and Data sectors (EE) numbers
	      if( fFlagSubDet == "EB" )
		{
		  min_value = 0;
		  max_value = fEcal->MaxStinEcnaInStex() - 1;
		}
	      if( fFlagSubDet == "EE" )
		{
		  if( fFapStexNumber == 1 ){min_value = 1; max_value = 5;}
		  if( fFapStexNumber == 2 ){min_value = 5; max_value = 9;}
		  if( fFapStexNumber == 3 ){min_value = 5; max_value = 9;}
		  if( fFapStexNumber == 4 ){min_value = 1; max_value = 5;}
		}
	    }
	  if( fFapStexNumber == 0 )
	    {
	      //.......................... Axis for the SM (EB) and Dee numbers (EE)
	      if( fFlagSubDet == "EB" )
		{
		  min_value = 0; 
		  max_value = fEcal->MaxStexInStas() - 1;
		}
	      if( fFlagSubDet == "EE" )
		{
		  min_value = 1; 
		  max_value = fEcal->MaxStexInStas();
		}
	    }
	  TopAxisForHistos(h_his0, opt_plot, xMemoPlotSame, min_value, max_value,
			   xFlagAutoYsupMargin, HisSize);
	} // end of if (HistoType == "Global")

      if( !( (HistoType == "H1Basic" || HistoType == "H1BasicProj")
	     && ( arg_AlreadyRead > 1 && arg_AlreadyRead < fEcal->MaxCrysInStin() ) ) )
	{
	  gPad->Update();
	}
    }
  else    // else du if(main_subpad !=0)
    {
      cout << "*TEcnaHistos::HistoPlot(...)> Canvas not found. Previously closed in option SAME."
	   << fTTBELL << endl;

      ReInitCanvas(HistoCode, opt_plot);
      xMemoPlotSame = 0;
    }

  //  delete MainCanvas;                  fCdeleteRoot++;

} // end of HistoPlot


TString TEcnaHistos::StexNumberToString(const Int_t& StexNumber)
{
  // Convert Int_t StexNumber into TString: "StexNumber" if StexNumber in [-18,36]
  // or into TString: "EB" or "EE" if StexNumber = 0. 

  TString sFapStexNumber = "?";
  if( StexNumber ==  -1 ){sFapStexNumber = " -1";}
  if( StexNumber ==  -2 ){sFapStexNumber = " -2";}
  if( StexNumber ==  -3 ){sFapStexNumber = " -3";}
  if( StexNumber ==  -4 ){sFapStexNumber = " -4";}
  if( StexNumber ==  -5 ){sFapStexNumber = " -5";}
  if( StexNumber ==  -6 ){sFapStexNumber = " -6";}
  if( StexNumber ==  -7 ){sFapStexNumber = " -7";}
  if( StexNumber ==  -8 ){sFapStexNumber = " -8";}
  if( StexNumber ==  -9 ){sFapStexNumber = " -9";}
  if( StexNumber == -10 ){sFapStexNumber = "-10";}
  if( StexNumber == -11 ){sFapStexNumber = "-11";}
  if( StexNumber == -12 ){sFapStexNumber = "-12";}
  if( StexNumber == -13 ){sFapStexNumber = "-13";}
  if( StexNumber == -14 ){sFapStexNumber = "-14";}
  if( StexNumber == -15 ){sFapStexNumber = "-15";}
  if( StexNumber == -16 ){sFapStexNumber = "-16";}
  if( StexNumber == -17 ){sFapStexNumber = "-17";}
  if( StexNumber == -18 ){sFapStexNumber = "-18";}
  if( StexNumber ==  0 ){sFapStexNumber = fFlagSubDet;}
  if( StexNumber ==  1 ){sFapStexNumber = "  1";}
  if( StexNumber ==  2 ){sFapStexNumber = "  2";}
  if( StexNumber ==  3 ){sFapStexNumber = "  3";}
  if( StexNumber ==  4 ){sFapStexNumber = "  4";}
  if( StexNumber ==  5 ){sFapStexNumber = "  5";}
  if( StexNumber ==  6 ){sFapStexNumber = "  6";}
  if( StexNumber ==  7 ){sFapStexNumber = "  7";}
  if( StexNumber ==  8 ){sFapStexNumber = "  8";}
  if( StexNumber ==  9 ){sFapStexNumber = "  9";}
  if( StexNumber == 10 ){sFapStexNumber = " 10";}
  if( StexNumber == 11 ){sFapStexNumber = " 11";}
  if( StexNumber == 12 ){sFapStexNumber = " 12";}
  if( StexNumber == 13 ){sFapStexNumber = " 13";}
  if( StexNumber == 14 ){sFapStexNumber = " 14";}
  if( StexNumber == 15 ){sFapStexNumber = " 15";}
  if( StexNumber == 16 ){sFapStexNumber = " 16";}
  if( StexNumber == 17 ){sFapStexNumber = " 17";}
  if( StexNumber == 18 ){sFapStexNumber = " 18";}
  if( StexNumber == 19 ){sFapStexNumber = " -1";}
  if( StexNumber == 20 ){sFapStexNumber = " -2";}
  if( StexNumber == 21 ){sFapStexNumber = " -3";}
  if( StexNumber == 22 ){sFapStexNumber = " -4";}
  if( StexNumber == 23 ){sFapStexNumber = " -5";}
  if( StexNumber == 24 ){sFapStexNumber = " -6";}
  if( StexNumber == 25 ){sFapStexNumber = " -7";}
  if( StexNumber == 26 ){sFapStexNumber = " -8";}
  if( StexNumber == 27 ){sFapStexNumber = " -9";}
  if( StexNumber == 28 ){sFapStexNumber = "-10";}
  if( StexNumber == 29 ){sFapStexNumber = "-11";}
  if( StexNumber == 30 ){sFapStexNumber = "-12";}
  if( StexNumber == 31 ){sFapStexNumber = "-13";}
  if( StexNumber == 32 ){sFapStexNumber = "-14";}
  if( StexNumber == 33 ){sFapStexNumber = "-15";}
  if( StexNumber == 34 ){sFapStexNumber = "-16";}
  if( StexNumber == 35 ){sFapStexNumber = "-17";}
  if( StexNumber == 36 ){sFapStexNumber = "-18";}
  return sFapStexNumber;
}

Double_t TEcnaHistos::NotConnectedSCH1DBin(const Int_t& index)
{
  // gives the x coordinate for the i_th NotConnected SC
  // GetDSOffset(DeeNumber, DataSector) , GetSCOffset(DeeNumber, DataSector, SC_in_DS)

  Double_t xbin = (Double_t)(-1);

  if( index ==  1 ){xbin = GetDSOffset(1,1)+GetSCOffset(1,1, 30);}  // nb_for_cons == 182  (D1,S1) (D3,S9)
  if( index ==  2 ){xbin = GetDSOffset(2,9)+GetSCOffset(2,9, 30);}  // nb_for_cons ==  33  (D2,S9) (D4,S1)

  if( index ==  3 ){xbin = GetDSOffset(1,2)+GetSCOffset(1,2,  3);}  // nb_for_cons == 178  (D1,S2) (D3,S8)
  if( index ==  4 ){xbin = GetDSOffset(2,8)+GetSCOffset(2,8,  3);}  // nb_for_cons ==  29  (D2,S8) (D4,S2)

  if( index ==  5 ){xbin = GetDSOffset(1,2)+GetSCOffset(1,2, 25);}  // nb_for_cons == 207  (D1,S2) (D3,S8)
  if( index ==  6 ){xbin = GetDSOffset(2,8)+GetSCOffset(2,8, 25);}  // nb_for_cons ==  58  (D2,S8) (D4,S2)

  if( index ==  7 ){xbin = GetDSOffset(1,3)+GetSCOffset(1,3, 34);}  // nb_for_cons == 298  (D1,S3) (D3,S7) 
  if( index ==  8 ){xbin = GetDSOffset(2,7)+GetSCOffset(2,7, 34);}  // nb_for_cons == 149  (D2,S7) (D4,S3)

  if( index ==  9 ){xbin = GetDSOffset(1,4)+GetSCOffset(1,4, 14);}  // nb_for_cons == 261  (D1,S4) (D3,S6)
  if( index == 10 ){xbin = GetDSOffset(2,6)+GetSCOffset(2,6, 14);}  // nb_for_cons == 112  (D2,S6) (D4,S4)
  if( index == 11 ){xbin = GetDSOffset(1,4)+GetSCOffset(1,4, 21);}  // nb_for_cons == 268  (D1,S4) (D3,S6)
  if( index == 12 ){xbin = GetDSOffset(2,6)+GetSCOffset(2,6, 21);}  // nb_for_cons == 119  (D2,S6) (D4,S4)

  if( index == 13 ){xbin = GetDSOffset(1,5)+GetSCOffset(1,5, 20);}  // nb_for_cons == 281  (D1,S5) (D3,S5)
  if( index == 14 ){xbin = GetDSOffset(2,5)+GetSCOffset(2,5,  3);}  // nb_for_cons == 132  (D2,S5) (D4,S5)

  return xbin;
}

Double_t TEcnaHistos::NotCompleteSCH1DBin(const Int_t& index)
{
  // gives the x coordinate for the i_th NotConnected SC

  Double_t xbin = (Double_t)(-1);

  if( index ==  1 ){xbin = GetDSOffset(1,1)+GetSCOffset(1,1, 12);}  // nb_for_cons == 161  (D1,S1) (D3,S9)
  if( index ==  2 ){xbin = GetDSOffset(2,9)+GetSCOffset(2,9, 12);}  // nb_for_cons ==  12  (D2,S9) (D4,S1)

  if( index ==  3 ){xbin = GetDSOffset(1,2)+GetSCOffset(1,2, 32);}  // nb_for_cons == 216  (D1,S2) (D3,S8)
  if( index ==  4 ){xbin = GetDSOffset(2,8)+GetSCOffset(2,8, 32);}  // nb_for_cons ==  67  (D2,S8) (D4,S2)

  if( index ==  5 ){xbin = GetDSOffset(1,3)+GetSCOffset(1,3, 10);}  // nb_for_cons == 224  (D1,S3) (D3,S7) 
  if( index ==  6 ){xbin = GetDSOffset(2,7)+GetSCOffset(2,7, 10);}  // nb_for_cons ==  75  (D2,S7) (D4,S3)

  if( index ==  7 ){xbin = GetDSOffset(1,5)+GetSCOffset(1,5, 23);}  // nb_for_cons == 286  (D1,S5) (D3,S5)
  if( index ==  8 ){xbin = GetDSOffset(2,5)+GetSCOffset(2,5,  6);}  // nb_for_cons == 137  (D2,S5) (D4,S5)

  return xbin;
}

Int_t TEcnaHistos::GetNotConnectedSCForConsFromIndex(const Int_t& index)
{

  Int_t SCForCons = 0;
  if( index ==  1 ){SCForCons = 182;} // (D1,S1) (D3,S9)
  if( index ==  2 ){SCForCons =  33;} // (D2,S9) (D4,S1)

  if( index ==  3 ){SCForCons = 178;} // (D1,S2) (D3,S8)
  if( index ==  4 ){SCForCons =  29;} // (D2,S8) (D4,S2)
  if( index ==  5 ){SCForCons = 207;} // (D1,S2) (D3,S8)
  if( index ==  6 ){SCForCons =  58;} // (D2,S8) (D4,S2)

  if( index ==  7 ){SCForCons = 298;} // (D1,S3) (D3,S7) 
  if( index ==  8 ){SCForCons = 149;} // (D2,S7) (D4,S3)

  if( index ==  9 ){SCForCons = 261;} // (D1,S4) (D3,S6)
  if( index == 10 ){SCForCons = 112;} // (D2,S6) (D4,S4)
  if( index == 11 ){SCForCons = 268;} // (D1,S4) (D3,S6)
  if( index == 12 ){SCForCons = 119;} // (D2,S6) (D4,S4)

  if( index == 13 ){SCForCons = 281;} // (D1,S5) (D3,S5)
  if( index == 14 ){SCForCons = 132;} // (D2,S5) (D4,S5)
  return SCForCons;
}

Int_t TEcnaHistos::GetNotConnectedDSSCFromIndex(const Int_t& index)
{

  Int_t DSSC = 0;
  if( index ==  1 ){DSSC =  30;} // (D1,S1) (D3,S9)
  if( index ==  2 ){DSSC =  30;} // (D2,S9) (D4,S1)

  if( index ==  3 ){DSSC =   3;} // (D1,S2) (D3,S8)
  if( index ==  4 ){DSSC =   3;} // (D2,S8) (D4,S2)
  if( index ==  5 ){DSSC =  25;} // (D1,S2) (D3,S8)
  if( index ==  6 ){DSSC =  25;} // (D2,S8) (D4,S2)

  if( index ==  7 ){DSSC =  34;} // (D1,S3) (D3,S7) 
  if( index ==  8 ){DSSC =  34;} // (D2,S7) (D4,S3)

  if( index ==  9 ){DSSC =  14;} // (D1,S4) (D3,S6)
  if( index == 10 ){DSSC =  14;} // (D2,S6) (D4,S4)
  if( index == 11 ){DSSC =  21;} // (D1,S4) (D3,S6)
  if( index == 12 ){DSSC =  21;} // (D2,S6) (D4,S4)

  if( index == 13 ){DSSC =  20;} // (D1,S5) (D3,S5)
  if( index == 14 ){DSSC =   3;} // (D2,S5) (D4,S5)
  return DSSC;
}


Int_t TEcnaHistos::GetNotCompleteSCForConsFromIndex(const Int_t& index)
{

  Int_t DSSC = 0;
  if( index ==  1 ){DSSC =  161;} // (D1,S1) (D3,S9)
  if( index ==  2 ){DSSC =   12;} // (D2,S9) (D4,S1)

  if( index ==  3 ){DSSC =  216;} // (D1,S2) (D3,S8)
  if( index ==  4 ){DSSC =   67;} // (D2,S8) (D4,S2)

  if( index ==  5 ){DSSC =  224;} // (D1,S3) (D3,S7) 
  if( index ==  6 ){DSSC =   75;} // (D2,S7) (D4,S3)

  if( index ==  7 ){DSSC =  286;} // (D1,S5) (D3,S5)
  if( index ==  8 ){DSSC =  137;} // (D2,S5) (D4,S5)
  return DSSC;
}

Int_t TEcnaHistos::GetNotCompleteDSSCFromIndex(const Int_t& index)
{

  Int_t DSSC = 0;
  if( index ==  1 ){DSSC =  12;} // (D1,S1) (D3,S9)
  if( index ==  2 ){DSSC =  12;} // (D2,S9) (D4,S1)

  if( index ==  3 ){DSSC =  32;} // (D1,S2) (D3,S8)
  if( index ==  4 ){DSSC =  32;} // (D2,S8) (D4,S2)

  if( index ==  5 ){DSSC =  10;} // (D1,S3) (D3,S7) 
  if( index ==  6 ){DSSC =  10;} // (D2,S7) (D4,S3)

  if( index ==  7 ){DSSC =  23;} // (D1,S5) (D3,S5)
  if( index ==  8 ){DSSC =   6;} // (D2,S5) (D4,S5)
  return DSSC;
}
//----------------------------------------------- HistimePlot
void TEcnaHistos::HistimePlot(TGraph*       g_graph0, 
			      Axis_t        xinf,        Axis_t        xsup,
			      const TString& HistoCode,   const TString& HistoType,
			      const Int_t&  StexStin_A,  const Int_t&  i0StinEcha, const Int_t& i0Sample, 
			      const Int_t&  opt_scale_x, const Int_t& opt_scale_y,
			      const TString& opt_plot,    const Int_t&  xFlagAutoYsupMargin)
{
  // Plot 1D histogram for evolution in time

  UInt_t canv_w = fCnaParHistos->SetCanvasWidth(HistoCode, opt_plot);
  UInt_t canv_h = fCnaParHistos->SetCanvasHeight(HistoCode, opt_plot);

  SetGraphPresentation(g_graph0, HistoType.Data(), opt_plot.Data());   // (gStyle parameters)}
  
  //...................................................... paves commentaires (HistimePlot)	 
  SetAllPavesViewHisto(HistoCode, StexStin_A, i0StinEcha, i0Sample, opt_plot);
  
  //..................................................... Canvas name (HistimePlot)
  Int_t arg_AlreadyRead = 0;
  TString canvas_name = SetCanvasName(HistoCode.Data(), opt_scale_x, opt_scale_y, opt_plot, arg_AlreadyRead,
				      StexStin_A,       i0StinEcha,  i0Sample);
	  
  //------------------------------------------------ Canvas allocation	(HistimePlot)
  //......................................... declarations canvas et pad
  TCanvas*  MainCanvas = 0;

  if( opt_plot == fOnlyOnePlot )
    {MainCanvas = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w , canv_h);   fCnewRoot++;
    fCurrentPad = gPad; fCurrentCanvas = MainCanvas; fCurrentCanvasName = canvas_name.Data();}
  
  if( opt_plot == fSeveralPlot )
    {
      if(GetMemoFlag(HistoCode, opt_plot) == "Free")
	{
	  MainCanvas = CreateCanvas(HistoCode, opt_plot, canvas_name, canv_w, canv_h);
	  fCurrentPad = gPad; fCurrentCanvas = MainCanvas; fCurrentCanvasName = canvas_name.Data();
	}
    }
  
  if( opt_plot == fSameOnePlot )
    {
      if(GetMemoFlag(HistoCode, opt_plot) == "Free")
	{
	  MainCanvas = CreateCanvas(HistoCode, opt_plot, canvas_name, canv_w, canv_h);
	  fCurrentPad = gPad; fCurrentCanvas = MainCanvas; fCurrentCanvasName = canvas_name.Data();
	}
    }
  
  // cout << "*TEcnaHistos::HistimePlot(...)> Plot is displayed on canvas ----> " << canvas_name.Data() << endl;

  //--------------- EE => SC for construction, EB => Xtal in SM (default: Stin ECNA number, i0StinEcha)
  Int_t Stex_StinCons = StexStin_A;   // Stex_StinCons = Tower for EB, SC for construction for EE
  Int_t n1StexCrys    = i0StinEcha+1; // n1StexCrys = Crys in SM for EB

  if( StexStin_A >= 1 && StexStin_A <= fEcal->MaxStinEcnaInStex() )
    {
      if( fFlagSubDet == "EB" )
	{n1StexCrys  = fEcalNumbering->Get1StexCrysFrom1StexStinAnd0StinEcha(StexStin_A, i0StinEcha, fFapStexNumber);}
      if( fFlagSubDet == "EE" )
	{Stex_StinCons = fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fFapStexNumber, StexStin_A);}
    }

  //------ String for StexNumber ( to display "EB" or "EE" if Stex = 0 )
  TString sFapStexNumber = StexNumberToString(fFapStexNumber);

  //============================================================================= (HistimePlot)
  //
  //     1st  OPERATIONS:  Pave texts preparation and first Draw.
  //                       SetParametersCanvas
  //                       Set Memo Flags.
  //                       Set main_subpad and main_pavtxt
  //
  //=============================================================================
  TVirtualPad* main_subpad = 0;  //      main_subpad: Pad for the histo
  TPaveText*   main_pavtxt = 0;  //      main_pavtxt: pave for changing parameters

  Int_t xMemoPlotSame = 1;   // a priori ==> SAME plot 

  TString QuantityName = fCnaParHistos->GetQuantityName(HistoCode.Data());

  //========================================= Option ONLYONE    (HistimePlot)
  if( opt_plot == fOnlyOnePlot )
    {
      //................................. Draw titles and paves (pad = main canvas)
      if( fPavComGeneralTitle != 0 ){fPavComGeneralTitle->Draw();}
      fPavComStex->Draw();

      if( !( HistoType == "Global"  || HistoType == "Proj" ) )
	{
	  fPavComStin->Draw();
	  fPavComXtal->Draw();
	}
      fPavComEvolNbOfEvtsAna->Draw();
      fPavComEvolRuns->Draw();

      Double_t x_low = fCnaParHistos->BoxLeftX("bottom_left_box")   - 0.005;
      Double_t x_up  = fCnaParHistos->BoxRightX("bottom_right_box") + 0.005;
      Double_t y_low = fCnaParHistos->BoxTopY("bottom_right_box")   + 0.005;
      Double_t y_up  = fCnaParHistos->BoxBottomY("top_left_box_EB") - 0.005;

      Double_t x_margin = x_low;
      Double_t y_margin = y_low;

      Color_t  fond_pad = fCnaParHistos->ColorDefinition("gris18");

      MainCanvas->Divide( 1,  1, x_margin, y_margin, fond_pad);
      //           Divide(nx, ny, x_margin, y_margin,    color);

      gPad->cd(1);
      main_subpad = gPad;
      main_subpad->SetPad(x_low, y_low, x_up, y_up);

      xMemoPlotSame = 0;
    }
  //========================================= Options SAME and SAME n	(HistimePlot)
  if(opt_plot == fSeveralPlot || opt_plot == fSameOnePlot)
    {
      if(GetMemoFlag(HistoCode, opt_plot) == "Free")
	{
	  if( fPavComGeneralTitle != 0 ){fPavComGeneralTitle->Draw();}
	  fPavComSeveralChanging->Draw();

	  fPavComEvolRuns->Draw();

	  if( !( HistoType == "Global"     || HistoType == "Proj"       ||
		 HistoCode == "H_Ped_Date" || HistoCode == "H_TNo_Date" || HistoCode == "H_MCs_Date" ||
		 HistoCode == "H_LFN_Date" || HistoCode == "H_HFN_Date" || HistoCode == "H_SCs_Date") )
	    {
	      fPavComStin->Draw();
	      fPavComXtal->Draw();
	    } 
	  
	  if( !( HistoCode == "H_Ped_Date" || HistoCode == "H_TNo_Date" || HistoCode == "H_MCs_Date" ||
		 HistoCode == "H_LFN_Date" || HistoCode == "H_HFN_Date" || HistoCode == "H_SCs_Date" ) )
	    {
	      fPavComXtal->Draw();
	    } 
	  //Call to SetParametersPavTxt
	  //fPavTxt<HISTOCODE> = fPavComSeveralChanging;  => come from SetAllPavesViewHisto
	  SetParametersPavTxt(HistoCode, opt_plot);
	  
	  //---------------- Call to ActivePavTxt
	  // main_pavtxt = fPavTxt<HISTOCODE>;} (return main_pavtxt)
	  main_pavtxt = ActivePavTxt(HistoCode.Data(), opt_plot.Data());
	  
	  //-------------------- Set texts for pave "several changing", options SAME and SAME n	(HistimePlot)
	  main_pavtxt->SetTextAlign(fTextPaveAlign);
	  main_pavtxt->SetTextFont(fTextPaveFont);
	  main_pavtxt->SetTextSize(fTextPaveSize);
	  main_pavtxt->SetBorderSize(fTextBorderSize);

	  char* f_in = new char[fgMaxCar];                            fCnew++;

	  TString DecalStexName = "";
	  if( fFlagSubDet == "EB" ){DecalStexName = " ";}
	  TString DecalStinName = "";
	  if( fFlagSubDet == "EE" ){DecalStinName = "   ";}

	  TString sStexOrStasName = "";
	  if( fFapStexNumber == 0 ){sStexOrStasName = "  ";}
	  if( fFapStexNumber != 0 ){sStexOrStasName = fFapStexName;}


	  //-----------------------------> HistoType = "EvolProj" => treated in HistoPlot, not here.
	  if(opt_plot == fSeveralPlot)
	    {
	      sprintf(f_in, "Analysis   Samp  Evts range  Nb Evts   %s%s %s%s   %s  %s",
		      DecalStexName.Data(), sStexOrStasName.Data(),
		      DecalStinName.Data(), fFapStinName.Data(), fFapXtalName.Data(), fFapEchaName.Data());
	    }
	  if(opt_plot == fSameOnePlot)
	    {
	      sprintf(f_in, "Analysis   Samp  Evts range  Nb Evts   %s%s %s%s   %s  %s",
		      DecalStexName.Data(), sStexOrStasName.Data(),
		      DecalStinName.Data(), fFapStinName.Data(), fFapXtalName.Data(), fFapEchaName.Data());
	    }

	  //................................................................... (HistimePlot)
	  TText* ttit = main_pavtxt->AddText(f_in);
	  ttit->SetTextColor(fCnaParHistos->ColorDefinition("noir"));
	  
	  if(opt_plot == fSeveralPlot)
	    {
	      sprintf(f_in, "%-10s 1-%2d %5d-%5d  %7d %5s%6d%7d%6d",
		      fFapAnaType.Data(), fFapNbOfSamples,
		      fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts, sFapStexNumber.Data(),
		      Stex_StinCons, n1StexCrys, i0StinEcha);
	    }
	  if(opt_plot == fSameOnePlot)
	    {
	      sprintf(f_in, "%-10s 1-%2d %5d-%5d  %7d %5s%6d%7d%6d %-25s",
		      fFapAnaType.Data(), fFapNbOfSamples,
		      fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts, sFapStexNumber.Data(),
		      Stex_StinCons, n1StexCrys, i0StinEcha, QuantityName.Data());
	    }

	  TText* tt = main_pavtxt->AddText(f_in);
	  tt->SetTextColor(GetViewHistoColor(HistoCode, opt_plot));

	  delete [] f_in;      f_in = 0;                                       fCdelete++;

	  //---------- Draw the "several changing" pave with its text in the Canvas (FIRST TIME)	(HistimePlot)
	  main_pavtxt->Draw();

	  //---------------- Call to SetParametersCanvas
	  //fImp<HISTOCODE> = (TCanvasImp*)fCanv<HISTOCODE>->GetCanvasImp();
	  //fCanv<HISTOCODE>->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  //fPad<HISTOCODE> = gPad;
	  //fMemoPlot<HISTOCODE> = 1;          =======>  set MemoFlag to "Buzy"
	  //fMemoColor<HISTOCODE> = 0;
	  SetParametersCanvas(HistoCode, opt_plot);

	  //---------------- Set xMemoPlotSame to 0
	  xMemoPlotSame = 0;
	}

      //............................ cases fMemoPlotxxx = 1            (HistimePlot)
      if(GetMemoFlag(HistoCode, opt_plot) == "Busy")
	{
	  //---------------- Call to ActivePavTxt
	  // main_pavtxt = fPavTxt<HISTOCODE>;} (return main_pavtxt)
	  main_pavtxt = ActivePavTxt(HistoCode.Data(), opt_plot.Data());
	  
	  //---------------- Call to ActivePad
	  main_subpad = ActivePad(HistoCode.Data(), opt_plot.Data());  // => return 0 if canvas has been closed

	  //---------------- Recover pointer to the current canvas
	  MainCanvas = GetCurrentCanvas(HistoCode.Data(), opt_plot.Data());
	}
    }

  //============================================================================= (HistimePlot)
  //
  //     2nd  OPERATIONS: Write and Draw the parameter values in the
  //                      "several changing" pave (options SAME and SAME n)
  //                      Draw Histo
  //=============================================================================
  if(main_subpad != 0)
    {
      if(opt_plot == fSeveralPlot || opt_plot == fSameOnePlot)
	{
	  if(xMemoPlotSame != 0)
	    {
	      main_pavtxt->SetTextAlign(fTextPaveAlign);
	      main_pavtxt->SetTextFont(fTextPaveFont);
	      main_pavtxt->SetTextSize(fTextPaveSize);
	      main_pavtxt->SetBorderSize(fTextBorderSize);

	      char* f_in = new char[fgMaxCar];                            fCnew++;
	      
	      if(opt_plot == fSeveralPlot )
		{sprintf(f_in, "%-10s 1-%2d %5d-%5d  %7d %5s%6d%7d%6d",
			 fFapAnaType.Data(), fFapNbOfSamples, fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts,
			 sFapStexNumber.Data(), Stex_StinCons, n1StexCrys, i0StinEcha);}
	      if(opt_plot == fSameOnePlot )
		{sprintf(f_in, "%-10s 1-%2d %5d-%5d  %7d %5s%6d%7d%6d %-25s",
			 fFapAnaType.Data(), fFapNbOfSamples, fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts,
			 sFapStexNumber.Data(), Stex_StinCons, n1StexCrys, i0StinEcha, QuantityName.Data());}
	      
	      TText *tt = main_pavtxt->AddText(f_in);
	      tt->SetTextColor(GetViewHistoColor(HistoCode, opt_plot));
	      MainCanvas->cd(); gStyle->SetOptDate(0);
	      main_pavtxt->Draw();

	      delete [] f_in;    f_in = 0;                          fCdelete++;
	    }
	  main_subpad->cd();
	  Double_t x_low = fCnaParHistos->BoxLeftX("bottom_left_box")   - 0.005;
	  Double_t x_up  = fCnaParHistos->BoxRightX("bottom_right_box") + 0.005;
	  Double_t y_low = fCnaParHistos->BoxTopY("several_evol_box")   + 0.005;
	  Double_t y_up  = fCnaParHistos->BoxBottomY("general_comment") - 0.005;
	  if( opt_plot == fSameOnePlot ){y_up = fCnaParHistos->BoxBottomY("top_left_box_EB") - 0.005;}
	  main_subpad->SetPad(x_low, y_low, x_up, y_up); 
	}
	      
      //............................................ Style	(HistimePlot)
      SetViewGraphColors(g_graph0, HistoCode, opt_plot);

      //................................. Set axis titles
      TString axis_x_var_name = SetHistoXAxisTitle(HistoCode);
      TString axis_y_var_name = SetHistoYAxisTitle(HistoCode);
      g_graph0->GetXaxis()->SetTitle(axis_x_var_name);
      g_graph0->GetYaxis()->SetTitle(axis_y_var_name);

      //................................. Xaxis is a time axis
      g_graph0->GetXaxis()->SetTimeDisplay(1);
      g_graph0->GetXaxis()->SetTimeFormat("%d %b-%Hh");
 
      g_graph0->GetXaxis()->SetTimeOffset(xinf);

      Int_t nb_displayed = fCnaParHistos->GetNbOfRunsDisplayed();      // max nb of run numbers displayed

      //...........................................................................	(HistimePlot)
      Int_t liny = 0;
      Int_t logy = 1;
	      
      if(opt_plot == fOnlyOnePlot)
	{
	  fXinf = (Double_t)xinf;
	  fXsup = (Double_t)xsup;
	  fYinf = (Double_t)GetYminValueFromMemo(HistoCode);
	  fYsup = (Double_t)GetYmaxValueFromMemo(HistoCode);

	  gPad->RangeAxis(fXinf, fYinf, fXsup, fYsup);

	  //if(opt_scale_y == fOptScaleLiny){gPad->SetLogy(liny);}
	  if(opt_scale_y == fOptScaleLogy){gPad->SetLogy(logy); g_graph0->Draw("ALP");}

	  if(opt_scale_y == fOptScaleLiny)
	    {
	      gPad->SetLogy(liny);
	      g_graph0->Draw("ALP");
	      Int_t     nb_pts  = g_graph0->GetN();
	      Double_t* coord_x = g_graph0->GetX();
	      Double_t* coord_y = g_graph0->GetY();

	      char* f_in = new char[fgMaxCar];                            fCnew++;

	      //................. display of the run numbers                                         (HistimePlot)
	      Double_t interv_displayed = (coord_x[nb_pts-1] - coord_x[0])/(Double_t)nb_displayed;
	      Double_t last_drawn_coordx = coord_x[0] - 1.5*interv_displayed;

	      for(Int_t i_run=0; i_run<nb_pts; i_run++)
		{
		  if ( (coord_x[i_run] - last_drawn_coordx) > interv_displayed )
		    {
		      Double_t joinYinf = fYinf;
		      Double_t joinYsup = fYsup;
		      if( joinYsup <= joinYinf )
			{
			  joinYinf =
			    GetYminFromGraphFrameAndMarginValue(g_graph0, fCnaParHistos->GetMarginAutoMinMax());
			  joinYsup =
			    GetYmaxFromGraphFrameAndMarginValue(g_graph0, fCnaParHistos->GetMarginAutoMinMax());
			  joinYsup = joinYsup + (joinYsup-joinYinf)/20.;
			}

		      sprintf( f_in, "R%d",  fT1DRunNumber[i_run]);
		      TText *text_run_num = new TText(coord_x[i_run], joinYsup, f_in);  fCnewRoot++;
		      text_run_num->SetTextAngle((Double_t)45.);
		      text_run_num->SetTextSize((Double_t)0.035);
		      text_run_num->Draw("SAME");
		      // delete text_StexStin_num;             fCdeleteRoot++;

		      TLine *jointlign;
		      jointlign = new TLine(coord_x[i_run], joinYsup, coord_x[i_run], coord_y[i_run]); fCnewRoot++;
		      jointlign->SetLineWidth(1);
		      jointlign->SetLineStyle(2);
		      jointlign->Draw("SAME");
		      // delete jointlign;                  fCdeleteRoot++;

		      last_drawn_coordx = coord_x[i_run];                           //        (HistimePlot)
		    }
		}

	      delete [] f_in;      f_in = 0;                                         fCdelete++;

	    }
	  if(opt_scale_y == fOptScaleLogy)
	    {
	      gPad->SetLogy(logy);
	      g_graph0->Draw("ALP");
	    }
	}

      //......................................................................  (HistimePlot)
      if(opt_plot == fSeveralPlot || opt_plot == fSameOnePlot)
	{
	  if(xMemoPlotSame == 0)
	    {
	      if(opt_scale_y == fOptScaleLiny){gPad->SetLogy(liny);}
	      if(opt_scale_y == fOptScaleLogy){gPad->SetLogy(logy);
}
	      g_graph0->Draw("ALP");

	      fXinf = (Double_t)xinf;
	      fXsup = (Double_t)xsup;
	      fYinf = (Double_t)GetYminValueFromMemo(HistoCode);
	      fYsup = (Double_t)GetYmaxValueFromMemo(HistoCode); 

	      gPad->RangeAxis(fXinf, fYinf, fXsup, fYsup);
	    }
	  
	  if(xMemoPlotSame != 0)                                          //        (HistimePlot)
	    {
	      if(opt_scale_y == fOptScaleLiny){gPad->SetLogy(liny);}
	      if(opt_scale_y == fOptScaleLogy){gPad->SetLogy(logy);}

	      g_graph0->Draw("LP");
	    }
	}
      gPad->Update();
    }
  else    // else du if(main_subpad !=0)
    {
      cout << "*TEcnaHistos::HistimePlot(...)> Canvas not found. Previously closed in option SAME."
	   << fTTBELL << endl;

      ReInitCanvas(HistoCode, opt_plot);
      xMemoPlotSame = 0;
    }

  //  delete MainCanvas;                  fCdeleteRoot++;
 
} // end of HistimePlot

//------------------------------------------------------------------------------------------------------
void TEcnaHistos::TopAxisForHistos(TH1D*        h_his0,              const TString& opt_plot,
				   const Int_t& xMemoPlotSame,       const Int_t&  min_value, const Int_t&  max_value,
				   const Int_t& xFlagAutoYsupMargin, const Int_t&  HisSize)
{
// Axis on top of the plot to indicate the Stin numbers

  if( opt_plot == fOnlyOnePlot ||
      ( (opt_plot == fSeveralPlot) && (xMemoPlotSame == 0) ) ||
      ( (opt_plot == fSameOnePlot) && (xMemoPlotSame == 0) ) )
    {   
      Double_t Maxih = (Double_t)h_his0->GetMaximum();
      Double_t Minih = (Double_t)h_his0->GetMinimum();

      if(xFlagAutoYsupMargin == 1)
	{
	  if( Minih >= Maxih ){Minih = (Double_t)0.; Maxih += Maxih;}  // ROOT default if ymin >= ymax
	  Double_t MaxMarginFactor = (Double_t)0.05;    // frame top line = 5% above the maximum
	  Maxih += (Maxih-Minih)*MaxMarginFactor;       // ROOT default if ymin < ymax
	}

      Double_t v_min = min_value;
      Double_t v_max = max_value+(Double_t)1.;
      Double_t v_min_p = v_min+(Double_t)1.;
      Double_t v_max_p = v_max+(Double_t)1.;

      Int_t ndiv = 50207;
      TString opt = "B-";
      Double_t Xbegin = 0.;
      Double_t Xend   = (Double_t)HisSize;
      Double_t ticks  = 0.05;

      if( fFapStexNumber == 0 && fFlagSubDet == "EE" )
	{
	  v_min = 0;
	  v_max = max_value;
	  ndiv = 4;
	  opt = "CSU";                  // first draw axis with ticksize and no label
	  Xbegin = (Double_t)HisSize;
	  Xend = 0.;
	}

      if( fFapStexNumber > 0 && fFlagSubDet == "EE" )
	{
	  ticks = 0;
	  if( fFapStexNumber == 1 )
	    {
	      v_min = min_value;
	      v_max = max_value+0.5;
	      Xbegin = (Double_t)HisSize;
	      Xend   = 0.;
	      opt = "CSU";                // first draw axis with no ticksize and no label
	    }
	  if( fFapStexNumber == 2 )
	    {
	      v_min = min_value+0.5;
	      v_max = max_value+1.;
	      Xbegin = (Double_t)HisSize;
	      Xend   = 0.;
	      opt = "CSU";                // first draw axis with no ticksize and no label
	    }
	  if( fFapStexNumber == 3 )
	    {
	      v_min = min_value+0.5;
	      v_max = max_value+1.;
	      Xbegin = 0.;
	      Xend   = (Double_t)HisSize;
	      opt = "-CSU";                // first draw axis with no ticksize and no label
	    }
	  if( fFapStexNumber == 4 )
	    {
	      v_min = min_value;
	      v_max = max_value+0.5;
	      Xbegin = 0.;
	      Xend   = (Double_t)HisSize;
	      opt = "-CSU";                // first draw axis with no ticksize and no label
	    }
	  v_min -= 1;
	  v_max -= 1;
	  ndiv = 5;
	}

      TGaxis* top_axis_x = 0;

      top_axis_x = new TGaxis(Xbegin, Maxih, Xend, Maxih,
			       v_min, v_max, ndiv, opt, 0.);         fCnewRoot++;

      top_axis_x->SetTickSize(ticks);
      top_axis_x->SetTitleOffset((Float_t)(1.2));
      top_axis_x->SetLabelOffset((Float_t)(0.005));

      TString  x_var_name  = "?";
      Int_t MaxCar = fgMaxCar;
      x_var_name.Resize(MaxCar);
      if( fFapStexNumber >  0 )
	{
	  if( fFlagSubDet == "EB"){x_var_name = "Tower number";}
	  if( fFlagSubDet == "EE")
	    {
	      x_var_name = " ";
	     // x_var_name = "                                                                                                            Data sector"; // don't remove the space characters !
	    }
	}
      if( fFapStexNumber == 0 )
	{
	  if( fFlagSubDet == "EB"){x_var_name = "SM number";}
	  if( fFlagSubDet == "EE"){x_var_name = " ";}
	} 
      top_axis_x->SetTitle(x_var_name);
      top_axis_x->Draw("SAME");
      
      if( fFlagSubDet == "EE" )
	{
	  // redraw axis with ticksize = 0, with labelOffset<0 or >0 and div centered in middle division
	  opt = "-MS";
	  if(fFapStexNumber == 1 || fFapStexNumber == 2 ){opt = "-MS";}
	  if(fFapStexNumber == 3 || fFapStexNumber == 4 ){opt = "MS";} 
	  ndiv = 4;
	  if( fFapStexNumber > 0 ){ndiv = 5;}
	  TGaxis* top_axis_x_bis = 0;
	  top_axis_x_bis = new TGaxis(Xbegin, Maxih, Xend, Maxih,
				      v_min_p, v_max_p, ndiv, opt, 0.);   fCnewRoot++;
	  top_axis_x_bis->SetTickSize(0.);
	  Float_t lab_siz_x = top_axis_x->GetLabelSize();
	  top_axis_x_bis->SetLabelSize(lab_siz_x);
	  top_axis_x_bis->SetLabelOffset(-0.1);

	  top_axis_x_bis->SetLabelOffset((Float_t)(9999.));
	  // if(fFapStexNumber == 1 || fFapStexNumber == 2 ){top_axis_x_bis->SetLabelOffset(-0.07);}
	  // if(fFapStexNumber == 3 || fFapStexNumber == 4 ){top_axis_x_bis->SetLabelOffset(-0.05);}  
	  // if(fFapStexNumber == 0 )
	     //  {top_axis_x_bis->SetLabelOffset((Float_t)(9999.));}  // keep the tick and remove the value
	  top_axis_x_bis->Draw("SAME");
	}
    } 
} // end of TopAxisForHistos

//............................................................................................
void TEcnaHistos::SetAllPavesViewMatrix(const TString&   BetweenWhat,
					const Int_t&    StexStin_A,  const Int_t&  StexStin_B,
					const Int_t&    i0StinEcha)
{
// Put all the paves of a matrix view

  fPavComGeneralTitle = fCnaParHistos->SetPaveGeneralComment(fFlagGeneralTitle);

  fPavComStex = fCnaParHistos->SetPaveStex("standard", fFapStexNumber);
  
  if(BetweenWhat == fLFBetweenChannels || BetweenWhat == fHFBetweenChannels)
    {fPavComStin = fCnaParHistos->SetPaveStinsXY(StexStin_A, StexStin_B);}
  if(BetweenWhat == fBetweenSamples)
    {
      fPavComStin = fCnaParHistos->SetPaveStin(StexStin_A, fFapStexNumber);
      
      if( fFlagSubDet == "EB" )
	{Int_t n1StexCrys = fEcalNumbering->Get1StexCrysFrom1StexStinAnd0StinEcha(StexStin_A, i0StinEcha, fFapStexNumber);
	fPavComXtal = fCnaParHistos->SetPaveCrystal(n1StexCrys, StexStin_A, i0StinEcha);}
      if( fFlagSubDet == "EE" )
	{TString Dir = fEcalNumbering->GetDeeDirViewedFromIP(fFapStexNumber);
	Int_t n1StexCrys = fEcalNumbering->Get1StexCrysFrom1StexStinAnd0StinEcha(StexStin_A, i0StinEcha, fFapStexNumber);
	fPavComXtal = fCnaParHistos->SetPaveCrystal(n1StexCrys, StexStin_A, i0StinEcha);}
    }

  fPavComAnaRun  = fCnaParHistos->SetPaveAnalysisRun(fFapAnaType,  fFapNbOfSamples, fFapRunNumber, fRunType,
						     fFapFirstReqEvtNumber, fFapLastReqEvtNumber, "OneCol");
  fPavComNbOfEvts = fCnaParHistos->SetPaveNbOfEvts(fFapNbOfEvts, fStartDate, fStopDate, "OneCol");
}

void TEcnaHistos::SetAllPavesViewStin(const Int_t& StexStin_A)
{
// Put all the paves of a Stin view

  fPavComGeneralTitle = fCnaParHistos->SetPaveGeneralComment(fFlagGeneralTitle);
  fPavComStex = fCnaParHistos->SetPaveStex("standard", fFapStexNumber);

  fPavComStin   = fCnaParHistos->SetPaveStin(StexStin_A, fFapStexNumber);

  fPavComAnaRun  = fCnaParHistos->SetPaveAnalysisRun(fFapAnaType, fFapNbOfSamples, fFapRunNumber, fRunType,
						     fFapFirstReqEvtNumber, fFapLastReqEvtNumber, "OneCol");
  fPavComNbOfEvts = fCnaParHistos->SetPaveNbOfEvts(fFapNbOfEvts, fStartDate, fStopDate, "OneCol");
}

void TEcnaHistos::SetAllPavesViewStinCrysNb(const Int_t& StexNumber, const Int_t& StexStin_A)
{
// Put all the paves of a crystal numbering Stin view

  fPavComStex  = fCnaParHistos->SetPaveStex("standard", StexNumber);
  fPavComStin  = fCnaParHistos->SetPaveStin(StexStin_A, StexNumber);

  if( fFlagSubDet == "EB")
    {fPavComLVRB   = fCnaParHistos->SetPaveLVRB(StexNumber, StexStin_A);}
  if( fFlagSubDet == "EE")
    {fPavComCxyz   = fCnaParHistos->SetPaveCxyz(StexNumber);}
}

void TEcnaHistos::SetAllPavesViewStex(const TString& chopt, const Int_t& StexNumber)
{
  if( chopt == "Numbering" )
    {
      fCnaParHistos->SetViewHistoStyle("Stex2DEENb");
      gStyle->SetTextColor(fCnaParHistos->ColorDefinition("noir"));
      fPavComStex = fCnaParHistos->SetPaveStex("standStex", StexNumber);
      if( fFlagSubDet == "EE" ){fPavComCxyz = fCnaParHistos->SetPaveCxyz(StexNumber);}
    }
  else
    {
      SetAllPavesViewStex(StexNumber);
    }
}
// end of SetAllPavesViewStex(...,...)

void TEcnaHistos::SetAllPavesViewStex(const Int_t& StexNumber)
{
  gStyle->SetTextColor(fCnaParHistos->ColorDefinition("noir"));
  fPavComGeneralTitle = fCnaParHistos->SetPaveGeneralComment(fFlagGeneralTitle);
  fPavComStex = fCnaParHistos->SetPaveStex("standStex", StexNumber);

  TString opt_pave_nbcol = "OneCol";
  if( fFapStexName == "SM"){opt_pave_nbcol = "TwoCol";}
    
  fPavComAnaRun  = fCnaParHistos->SetPaveAnalysisRun(fFapAnaType, fFapNbOfSamples, fFapRunNumber, fRunType,
						     fFapFirstReqEvtNumber, fFapLastReqEvtNumber,opt_pave_nbcol);
  fPavComNbOfEvts = fCnaParHistos->SetPaveNbOfEvts(fFapNbOfEvts, fStartDate, fStopDate, opt_pave_nbcol); 
}
// end of SetAllPavesViewStex(...)

void TEcnaHistos::SetAllPavesViewStas()
{
  gStyle->SetTextColor(fCnaParHistos->ColorDefinition("noir"));
  fPavComGeneralTitle = fCnaParHistos->SetPaveGeneralComment(fFlagGeneralTitle);
  fPavComStas = fCnaParHistos->SetPaveStas();

  fPavComAnaRun  = fCnaParHistos->SetPaveAnalysisRun(fFapAnaType, fFapNbOfSamples, fFapRunNumber, fRunType,
						     fFapFirstReqEvtNumber, fFapLastReqEvtNumber, "OneCol");
  fPavComNbOfEvts = fCnaParHistos->SetPaveNbOfEvts(fFapNbOfEvts, fStartDate, fStopDate, "OneCol");
}
// end of SetAllPavesViewStas

void TEcnaHistos::SetAllPavesViewHisto(const TString& HistoCode,  const Int_t&  StexStin_A,
				       const Int_t&  i0StinEcha, const Int_t&  i0Sample,  
				       const TString& opt_plot)
{
  Int_t arg_AlreadyRead = 0;
  SetAllPavesViewHisto(HistoCode, StexStin_A, i0StinEcha, i0Sample, opt_plot, arg_AlreadyRead);
}

void TEcnaHistos::SetAllPavesViewHisto(const TString& HistoCode,  const Int_t&  StexStin_A,
				       const Int_t&  i0StinEcha, const Int_t&  i0Sample,  
				       const TString& opt_plot,   const Int_t&  arg_AlreadyRead)
{
// Put all the paves of a histo view according to HistoCode

  gStyle->SetTextColor(fCnaParHistos->ColorDefinition("noir"));

  TString HistoType = fCnaParHistos->GetHistoType(HistoCode.Data());

  fPavComGeneralTitle = fCnaParHistos->SetPaveGeneralComment(fFlagGeneralTitle);

  if(opt_plot == fOnlyOnePlot)
    {      
      if( !( HistoCode == "D_NOE_ChNb" || HistoCode == "D_NOE_ChDs" ||
	     HistoCode == "D_Ped_ChNb" || HistoCode == "D_Ped_ChDs" ||
	     HistoCode == "D_LFN_ChNb" || HistoCode == "D_LFN_ChDs" ||
	     HistoCode == "D_TNo_ChNb" || HistoCode == "D_TNo_ChDs" || 
	     HistoCode == "D_HFN_ChNb" || HistoCode == "D_HFN_ChDs" || 
	     HistoCode == "D_MCs_ChNb" || HistoCode == "D_MCs_ChDs" || 
	     HistoCode == "D_SCs_ChNb" || HistoCode == "D_SCs_ChDs" ) )
	{
	  fPavComStex = fCnaParHistos->SetPaveStex("standard", fFapStexNumber);
	  fPavComStin = fCnaParHistos->SetPaveStin(StexStin_A, fFapStexNumber);
	}
      else
	{
	  if( HistoCode == "D_NOE_ChNb" ||
	      HistoCode == "D_Ped_ChNb" || HistoCode == "D_TNo_ChNb" ||
	      HistoCode == "D_MCs_ChNb" || HistoCode == "D_LFN_ChNb" ||
	      HistoCode == "D_HFN_ChNb" || HistoCode == "D_SCs_ChNb" )
	    {fPavComStex = fCnaParHistos->SetPaveStex("standGH", fFapStexNumber);}
	  else
	    {fPavComStex = fCnaParHistos->SetPaveStex("standard", fFapStexNumber);}
	}
  //.................................................... (SetAllPavesViewHisto)
     
      if( HistoCode == "H_Ped_Date" || HistoCode == "H_TNo_Date" || HistoCode == "H_MCs_Date" ||
	  HistoCode == "H_LFN_Date" || HistoCode == "H_HFN_Date" || HistoCode == "H_SCs_Date" ||
	  HistoCode == "H_Ped_RuDs" || HistoCode == "H_TNo_RuDs" || HistoCode == "H_MCs_RuDs" ||
	  HistoCode == "H_LFN_RuDs" || HistoCode == "H_HFN_RuDs" || HistoCode == "H_SCs_RuDs" )
	{
	  Int_t n1StexCrys  =
	    fEcalNumbering->Get1StexCrysFrom1StexStinAnd0StinEcha(StexStin_A, i0StinEcha, fFapStexNumber);
	  fPavComXtal = fCnaParHistos->SetPaveCrystal(n1StexCrys, StexStin_A, i0StinEcha);
	}

      if( HistoCode == "D_MSp_SpNb" || HistoCode == "D_SSp_SpNb" ||
	  HistoCode == "D_MSp_SpDs" || HistoCode == "D_SSp_SpDs" )
	{
	  Int_t n1StexCrys  =
	    fEcalNumbering->Get1StexCrysFrom1StexStinAnd0StinEcha(StexStin_A, i0StinEcha, fFapStexNumber);
	  fPavComXtal =
	    fCnaParHistos->SetPaveCrystal(n1StexCrys, StexStin_A, i0StinEcha, arg_AlreadyRead, fPlotAllXtalsInStin);
	}

      if( HistoCode == "D_Adc_EvDs" || HistoCode == "D_Adc_EvNb")
	{
	  Int_t n1StexCrys  =
	    fEcalNumbering->Get1StexCrysFrom1StexStinAnd0StinEcha(StexStin_A, i0StinEcha, fFapStexNumber);
	  fPavComXtal = fCnaParHistos->SetPaveCrystalSample(n1StexCrys, StexStin_A, i0StinEcha, i0Sample);
	}
      
      if( HistoCode == "H_Ped_Date" || HistoCode == "H_TNo_Date" || HistoCode == "H_MCs_Date" ||
	  HistoCode == "H_LFN_Date" || HistoCode == "H_HFN_Date" || HistoCode == "H_SCs_Date" ||
	  HistoCode == "H_Ped_RuDs" || HistoCode == "H_TNo_RuDs" || HistoCode == "H_MCs_RuDs" ||
	  HistoCode == "H_LFN_RuDs" || HistoCode == "H_HFN_RuDs" || HistoCode == "H_SCs_RuDs" )
	{
	  fPavComEvolNbOfEvtsAna =
	    fCnaParHistos->SetPaveEvolNbOfEvtsAna(fFapAnaType, fFapNbOfSamples,
						  fFapFirstReqEvtNumber, fFapLastReqEvtNumber, HistoType);
	  fPavComEvolRuns = fCnaParHistos->SetPaveEvolRuns(fStartEvolRun, fStartEvolDate,
							   fStopEvolRun,  fStopEvolDate, opt_plot, HistoType);
	}
      else
	{
	  fPavComAnaRun = fCnaParHistos->SetPaveAnalysisRun(fFapAnaType, fFapNbOfSamples, fFapRunNumber, fRunType,
							    fFapFirstReqEvtNumber, fFapLastReqEvtNumber, "OneCol");
	  fPavComNbOfEvts = fCnaParHistos->SetPaveNbOfEvts(fFapNbOfEvts, fStartDate, fStopDate, "OneCol");
	}
    }

  //.................................................... (SetAllPavesViewHisto)

  if( opt_plot == fSeveralPlot && GetMemoFlag(HistoCode, opt_plot) == "Free" )
    {
      if( HistoCode == "H_Ped_Date" || HistoCode == "H_TNo_Date" || HistoCode == "H_MCs_Date" ||
	  HistoCode == "H_LFN_Date" || HistoCode == "H_HFN_Date" || HistoCode == "H_SCs_Date" ||
	  HistoCode == "H_Ped_RuDs" || HistoCode == "H_TNo_RuDs" || HistoCode == "H_MCs_RuDs" ||
	  HistoCode == "H_LFN_RuDs" || HistoCode == "H_HFN_RuDs" || HistoCode == "H_SCs_RuDs" )
	{
	  fPavComSeveralChanging = fCnaParHistos->SetOptionSamePaveBorder("sevevol", HistoType);
	  fPavComEvolRuns = fCnaParHistos->SetPaveEvolRuns(fStartEvolRun, fStartEvolDate,
							   fStopEvolRun,  fStopEvolDate, opt_plot, HistoType);
	}
      else
	{
	  fPavComSeveralChanging = fCnaParHistos->SetOptionSamePaveBorder("several", HistoType);
	}
    } 

  if( opt_plot == fSameOnePlot && GetMemoFlag(HistoCode, opt_plot) == "Free" )
    {
      fPavComSeveralChanging = fCnaParHistos->SetOptionSamePaveBorder("several", HistoType);
      fPavComEvolRuns = fCnaParHistos->SetPaveEvolRuns(fStartEvolRun, fStartEvolDate,
						       fStopEvolRun,  fStopEvolDate, opt_plot, HistoType);
    } 

}
// end of SetAllPavesViewHisto

TString TEcnaHistos::SetCanvasName(const TString& HistoCode,
				   const Int_t&  opt_scale_x, const Int_t& opt_scale_y,
				   const TString& opt_plot,    const Int_t& arg_AlreadyRead,
				   const Int_t& StexStin_A,   const Int_t& i0StinEcha,  const Int_t& i0Sample)
{
  //......... Set Canvas name *===> FOR 1D HISTO ONLY 
  //          (for 2D histos, see inside ViewMatrix, ViewStex,...)

  TString canvas_name;
  Int_t MaxCar = fgMaxCar;
  canvas_name.Resize(MaxCar);
  canvas_name = "?";

  char* f_in = new char[fgMaxCar];               fCnew++;

  //......................... name_ opt_plot  (Set Canvas name)
  TString  name_opt_plot;
  MaxCar = fgMaxCar;
  name_opt_plot.Resize(MaxCar);
  name_opt_plot = "?";

  //if(opt_plot == fOnlyOnePlot && arg_AlreadyRead == 0){name_opt_plot = "P0";}  // Only one plot
  //if(opt_plot == fOnlyOnePlot && arg_AlreadyRead == 1){name_opt_plot = "P1";}  // SAME in Stin plot
  //if(opt_plot == fOnlyOnePlot && arg_AlreadyRead >= 1){name_opt_plot = "Pn";}  // SAME in Stin plot

  if( opt_plot == fOnlyOnePlot ){sprintf(f_in,"P%d", arg_AlreadyRead); name_opt_plot = f_in;}

  if(opt_plot == fSeveralPlot)
    {
      name_opt_plot = "SAME_N";
      //...................................... name_same (opt_plot = fSeveralPlot)
      Int_t name_same = -1;
      
      if(HistoCode == "D_NOE_ChNb"){name_same = fCanvSameD_NOE_ChNb;}
      if(HistoCode == "D_NOE_ChDs"){name_same = fCanvSameD_NOE_ChDs;}
      if(HistoCode == "D_Ped_ChNb"){name_same = fCanvSameD_Ped_ChNb;}
      if(HistoCode == "D_Ped_ChDs"){name_same = fCanvSameD_Ped_ChDs;}
      if(HistoCode == "D_TNo_ChNb"){name_same = fCanvSameD_TNo_ChNb;}
      if(HistoCode == "D_TNo_ChDs"){name_same = fCanvSameD_TNo_ChDs;}
      if(HistoCode == "D_MCs_ChNb"){name_same = fCanvSameD_MCs_ChNb;}
      if(HistoCode == "D_MCs_ChDs"){name_same = fCanvSameD_MCs_ChDs;}
      if(HistoCode == "D_LFN_ChNb"){name_same = fCanvSameD_LFN_ChNb;}
      if(HistoCode == "D_LFN_ChDs"){name_same = fCanvSameD_LFN_ChDs;}
      if(HistoCode == "D_HFN_ChNb"){name_same = fCanvSameD_HFN_ChNb;}
      if(HistoCode == "D_HFN_ChDs"){name_same = fCanvSameD_HFN_ChDs;}
      if(HistoCode == "D_SCs_ChNb"){name_same = fCanvSameD_SCs_ChNb;}
      if(HistoCode == "D_SCs_ChDs"){name_same = fCanvSameD_SCs_ChDs;}
      if(HistoCode == "D_MSp_SpNb"){name_same = fCanvSameD_MSp_SpNb;}
      if(HistoCode == "D_MSp_SpDs"){name_same = fCanvSameD_MSp_SpDs;}
      if(HistoCode == "D_SSp_SpNb"){name_same = fCanvSameD_SSp_SpNb;}
      if(HistoCode == "D_SSp_SpDs"){name_same = fCanvSameD_SSp_SpDs;}
      if(HistoCode == "D_Adc_EvDs"){name_same = fCanvSameD_Adc_EvDs;}
      if(HistoCode == "D_Adc_EvNb"){name_same = fCanvSameD_Adc_EvNb;}	  
      if(HistoCode == "H_Ped_Date"){name_same = fCanvSameH_Ped_Date;}
      if(HistoCode == "H_TNo_Date"){name_same = fCanvSameH_TNo_Date;}
      if(HistoCode == "H_MCs_Date"){name_same = fCanvSameH_MCs_Date;}
      if(HistoCode == "H_LFN_Date"){name_same = fCanvSameH_LFN_Date;}
      if(HistoCode == "H_HFN_Date"){name_same = fCanvSameH_HFN_Date;}
      if(HistoCode == "H_SCs_Date"){name_same = fCanvSameH_SCs_Date;}
      if(HistoCode == "H_Ped_RuDs"){name_same = fCanvSameH_Ped_RuDs;}
      if(HistoCode == "H_TNo_RuDs"){name_same = fCanvSameH_TNo_RuDs;}
      if(HistoCode == "H_MCs_RuDs"){name_same = fCanvSameH_MCs_RuDs;}
      if(HistoCode == "H_LFN_RuDs"){name_same = fCanvSameH_LFN_RuDs;}
      if(HistoCode == "H_HFN_RuDs"){name_same = fCanvSameH_HFN_RuDs;}
      if(HistoCode == "H_SCs_RuDs"){name_same = fCanvSameH_SCs_RuDs;}   

      sprintf(f_in,"%d", name_same);
      TString s_name_same = f_in;     
      const Text_t *t_name_same = (const Text_t *)s_name_same.Data();
      name_opt_plot.Append(t_name_same);
    } 
  if(opt_plot == fSameOnePlot)
    {
      name_opt_plot = "SAME_Plus_N";
      //...................................... name_same (opt_plot = fSeveralPlot)
      Int_t name_same = fCanvSameH1SamePlus;
      sprintf(f_in,"%d", name_same);
      TString s_name_same = f_in;     
      const Text_t *t_name_same = (const Text_t *)s_name_same.Data();
      name_opt_plot.Append(t_name_same);
    }

  //......................... name_visu (Set Canvas name)
  TString name_visu;
  MaxCar = fgMaxCar;
  name_visu.Resize(MaxCar);
  name_visu = "";
	  
  TString name_line;
  MaxCar = fgMaxCar;
  name_line.Resize(MaxCar);
  name_line = "Line_";
  TString HistoType = fCnaParHistos->GetHistoType(HistoCode.Data());
  if( HistoType == "Global" && (opt_plot == fSeveralPlot || opt_plot == fSameOnePlot) ){name_line = "Polm_";}

  // if(opt_visu == fOptVisLine){name_line = "Line_";}
  // if(opt_visu == fOptVisPolm){name_line = "Poly_";}

  const Text_t *t_line = (const Text_t *)name_line.Data();
  name_visu.Append(t_line);

  TString name_scale_x;
  MaxCar = fgMaxCar;
  name_scale_x.Resize(MaxCar);
  name_scale_x = "?";
  if(opt_scale_x == fOptScaleLinx){name_scale_x = "LinX_";}
  if(opt_scale_x == fOptScaleLogx){name_scale_x = "LogX_";}
  const Text_t *t_scale_x = (const Text_t *)name_scale_x.Data();
  name_visu.Append(t_scale_x);

  TString name_scale_y;
  MaxCar = fgMaxCar;
  name_scale_y.Resize(MaxCar);
  name_scale_y = "?";
  if(opt_scale_y == fOptScaleLiny){name_scale_y = "LinY";}
  if(opt_scale_y == fOptScaleLogy){name_scale_y = "LogY";}
  const Text_t *t_scale_y = (const Text_t *)name_scale_y.Data();
  name_visu.Append(t_scale_y);

  //...................................... name quantity (Set Canvas name)
  TString  name_quantity;
  MaxCar = fgMaxCar;
  name_quantity.Resize(MaxCar);
  name_quantity = "?";

  if(HistoCode == "D_NOE_ChNb"){name_quantity = "Nb_of_evts_as_func_of_Xtal";}
  if(HistoCode == "D_NOE_ChDs"){name_quantity = "Nb_of_evts_Xtal_distrib";}
  if(HistoCode == "D_Ped_ChNb"){name_quantity = "Pedestals_as_func_of_Xtal";}
  if(HistoCode == "D_Ped_ChDs"){name_quantity = "Pedestals_Xtal_distrib";}
  if(HistoCode == "D_TNo_ChNb"){name_quantity = "Total_Noise_as_func_of_Xtal";}
  if(HistoCode == "D_TNo_ChDs"){name_quantity = "Total_Noise_Xtal_distrib";}
  if(HistoCode == "D_MCs_ChNb"){name_quantity = "Mean_Corss_as_func_of_Xtal";}
  if(HistoCode == "D_MCs_ChDs"){name_quantity = "Mean_Corss_Xtal_distrib";}
  if(HistoCode == "D_LFN_ChNb"){name_quantity = "Low_Fq_Noise_as_func_of_Xtal";}
  if(HistoCode == "D_LFN_ChDs"){name_quantity = "Low_Fq_Noise_Xtal_distrib";}
  if(HistoCode == "D_HFN_ChNb"){name_quantity = "High_Fq_Noise_as_func_of_Xtal";}
  if(HistoCode == "D_HFN_ChDs"){name_quantity = "High_Fq_Noise_Xtal_distrib";}
  if(HistoCode == "D_SCs_ChNb"){name_quantity = "Sigma_Corss_as_func_of_Xtal";}
  if(HistoCode == "D_SCs_ChDs"){name_quantity = "Sigma_Corss_Xtal_distrib";}
  if(HistoCode == "D_MSp_SpNb"){name_quantity = "ExpValue_of_samples";}
  if(HistoCode == "D_MSp_SpDs"){name_quantity = "ExpValue_of_samples_distrib";}
  if(HistoCode == "D_SSp_SpNb"){name_quantity = "Sigma_of_samples";}
  if(HistoCode == "D_SSp_SpDs"){name_quantity = "Sigma_of_samples_distrib";}
  if(HistoCode == "D_Adc_EvDs"){name_quantity = "hevt";}
  if(HistoCode == "D_Adc_EvNb"){name_quantity = "ADC_as_func_of_Event";}	  
  if(HistoCode == "H_Ped_Date"){name_quantity = "Pedestal_history";}
  if(HistoCode == "H_TNo_Date"){name_quantity = "Total_Noise_history";}
  if(HistoCode == "H_MCs_Date"){name_quantity = "Mean_Corss_history";}
  if(HistoCode == "H_LFN_Date"){name_quantity = "Low_Fq_Noise_history";}
  if(HistoCode == "H_HFN_Date"){name_quantity = "High_Fq_Noise_history";}
  if(HistoCode == "H_SCs_Date"){name_quantity = "Sigma_Corss_history";}
  if(HistoCode == "H_Ped_RuDs"){name_quantity = "Pedestal_run_distribution";}
  if(HistoCode == "H_TNo_RuDs"){name_quantity = "Total_Noise_run_distribution";}
  if(HistoCode == "H_MCs_RuDs"){name_quantity = "Mean_Corss_run_distribution";}
  if(HistoCode == "H_LFN_RuDs"){name_quantity = "Low_Fq_Noise_run_distribution";}
  if(HistoCode == "H_HFN_RuDs"){name_quantity = "High_Fq_Noise_run_distribution";}
  if(HistoCode == "H_SCs_RuDs"){name_quantity = "Sigma_Corss_run_distribution";}

  Int_t num_crys = -1;
  if(HistoCode == "D_MSp_SpNb"){num_crys = i0StinEcha;}
  if(HistoCode == "D_MSp_SpDs"){num_crys = i0StinEcha;}
  if(HistoCode == "D_SSp_SpNb"){num_crys = i0StinEcha;}
  if(HistoCode == "D_SSp_SpDs"){num_crys = i0StinEcha;}
  if(HistoCode == "D_Adc_EvDs"){num_crys = i0StinEcha;}
  if(HistoCode == "D_Adc_EvNb"){num_crys = i0StinEcha;}	  
  if(HistoCode == "H_Ped_Date"){num_crys = i0StinEcha;}
  if(HistoCode == "H_TNo_Date"){num_crys = i0StinEcha;}
  if(HistoCode == "H_MCs_Date"){num_crys = i0StinEcha;}
  if(HistoCode == "H_LFN_Date"){num_crys = i0StinEcha;}
  if(HistoCode == "H_HFN_Date"){num_crys = i0StinEcha;}
  if(HistoCode == "H_SCs_Date"){num_crys = i0StinEcha;}
  if(HistoCode == "H_Ped_RuDs"){num_crys = i0StinEcha;}
  if(HistoCode == "H_TNo_RuDs"){num_crys = i0StinEcha;}
  if(HistoCode == "H_MCs_RuDs"){num_crys = i0StinEcha;}
  if(HistoCode == "H_LFN_RuDs"){num_crys = i0StinEcha;}
  if(HistoCode == "H_HFN_RuDs"){num_crys = i0StinEcha;}
  if(HistoCode == "H_SCs_RuDs"){num_crys = i0StinEcha;}

  Int_t num_samp = -1;
  if(HistoCode == "D_Adc_EvDs"){num_samp = i0Sample;}
  if(HistoCode == "D_Adc_EvNb"){num_samp = i0Sample;}

  //........................................................... (Set Canvas name)
  
  if (HistoCode == "D_NOE_ChNb" || HistoCode == "D_NOE_ChDs" || 
      HistoCode == "D_Ped_ChNb" || HistoCode == "D_Ped_ChDs" ||
      HistoCode == "D_TNo_ChNb" || HistoCode == "D_TNo_ChDs" ||
      HistoCode == "D_MCs_ChNb" || HistoCode == "D_MCs_ChDs" ||
      HistoCode == "D_LFN_ChNb" || HistoCode == "D_LFN_ChDs" || 
      HistoCode == "D_HFN_ChNb" || HistoCode == "D_HFN_ChDs" ||
      HistoCode == "D_SCs_ChNb" || HistoCode == "D_SCs_ChDs" )
    {
      sprintf(f_in, "%s_%s_S1_%d_R%d_%d_%d_%d_%s%d_%s_%s",
	      name_quantity.Data(), fFapAnaType.Data(),
	      fFapNbOfSamples, fFapRunNumber, fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts,
	      fFapStexName.Data(), fFapStexNumber,
	      name_opt_plot.Data(), name_visu.Data());
    }
  
  if (HistoCode == "D_MSp_SpNb" || HistoCode == "D_SSp_SpNb" ||
      HistoCode == "H_Ped_Date" || HistoCode == "H_TNo_Date" || HistoCode == "H_MCs_Date" ||
      HistoCode == "H_LFN_Date" || HistoCode == "H_HFN_Date" || HistoCode == "H_SCs_Date" ||
      HistoCode == "D_MSp_SpDs" || HistoCode == "D_SSp_SpDs" ||
      HistoCode == "H_Ped_RuDs" || HistoCode == "H_TNo_RuDs" || HistoCode == "H_MCs_RuDs" ||
      HistoCode == "H_LFN_RuDs" || HistoCode == "H_HFN_RuDs" || HistoCode == "H_SCs_RuDs")
    {
      sprintf(f_in, "%s_%s_S1_%d_R%d_%d_%d_%d_%s%d_%s%d_Xtal%d_%s_%s",
	      name_quantity.Data(), fFapAnaType.Data(), 
	      fFapNbOfSamples, fFapRunNumber, fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts,
	      fFapStexName.Data(), fFapStexNumber, fFapStinName.Data(), StexStin_A, num_crys,
	      name_opt_plot.Data(), name_visu.Data()); 
    }
  
  if (HistoCode == "D_Adc_EvNb" || HistoCode == "D_Adc_EvDs")
    {
      sprintf(f_in, "%s_%s_S1_%d_R%d_%d_%d_%d_%s%d_%s%d_Xtal%d_Samp%d_%s_%s",
	      name_quantity.Data(), fFapAnaType.Data(),
	      fFapNbOfSamples, fFapRunNumber, fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts,
	      fFapStexName.Data(), fFapStexNumber, fFapStinName.Data(), StexStin_A, num_crys, num_samp,
	      name_opt_plot.Data(), name_visu.Data());
    }
  
  canvas_name = f_in;
  delete [] f_in;    f_in = 0;              fCdelete++;
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
//  GetHistoSize, GetHistoValues, SetHistoXAxisTitle,   SetHistoYAxisTitle,
//  GetHistoXinf, GetHistoXsup,   GetHistoNumberOfBins, FillHisto
//
//===========================================================================
Int_t TEcnaHistos::GetHistoSize(const TString& chqcode, const TString& opt_plot_read)
{
// Histo size as a function of the quantity code

// VERY IMPORTANT: in some cases the number of bins must be strictly related to the parameters values
//                 (number of crystals, number of samples, etc...). See below comments "===> ONE BIN BY..."

  Int_t HisSize = fNbBinsProj;   // default value

  //............ histo with sample number as x coordinate => HisSize depends on option "plot" or "read"
  //             because of nb of samples in file: size for plot = 10 even if nb of samples in file < 10
  if( chqcode == "D_MSp_SpNb" ||  chqcode == "D_SSp_SpNb" ||
      chqcode == "D_MSp_SpDs" ||  chqcode == "D_SSp_SpDs" )
    {
      if( opt_plot_read == "read" ){HisSize = fFapNbOfSamples;}
      if( opt_plot_read == "plot" ){HisSize = fEcal->MaxSampADC();}
    }   // ===> ONE BIN BY SAMPLE

  //............ histo with event number as x coordinate.  (==> "D_Adc_EvDs" option: obsolete, to be confirmed)
  if(chqcode == "D_Adc_EvNb" || chqcode == "D_Adc_EvDs"){HisSize = fFapReqNbOfEvts;}   // ===> ONE BIN BY EVENT
  
  //............ 
  if(chqcode == "D_NOE_ChNb" ||
     chqcode == "D_Ped_ChNb" || chqcode == "D_TNo_ChNb" || chqcode == "D_MCs_ChNb" ||
     chqcode == "D_LFN_ChNb" || chqcode == "D_HFN_ChNb" || chqcode == "D_SCs_ChNb" ||
     chqcode == "D_NOE_ChDs" ||
     chqcode == "D_Ped_ChDs" || chqcode == "D_TNo_ChDs" || chqcode == "D_MCs_ChDs" || 
     chqcode == "D_LFN_ChDs" || chqcode == "D_HFN_ChDs" || chqcode == "D_SCs_ChDs" )
    {
      if( fFlagSubDet == "EB" )
	{
	  if( fFapStexNumber >  0 ){HisSize = fEcal->MaxCrysEcnaInStex();}             // ===> ONE BIN BY Xtal
	  if( fFapStexNumber == 0 ){HisSize = fEcal->MaxSMInEB()*fEcal->MaxTowInSM();} // ===> ONE BIN BY Tower
	}
      if( fFlagSubDet == "EE" )
	{
	  if( fFapStexNumber >  0 )
	    {
	      if( opt_plot_read == "read" ){HisSize = fEcal->MaxCrysEcnaInDee();}
	      if( opt_plot_read == "plot" ){HisSize = fEcal->MaxCrysForConsInDee();}
	    }                                                           // ===> ONE BIN BY Xtal
	  if( fFapStexNumber == 0 )
	    {HisSize = fEcal->MaxDeeInEE()*fEcal->MaxSCForConsInDee();} // ===> ONE BIN BY SC
	}
    }

  if( chqcode == "H_Ped_RuDs" || chqcode == "H_TNo_RuDs" || chqcode == "H_MCs_RuDs" || 
      chqcode == "H_LFN_RuDs" || chqcode == "H_HFN_RuDs" || chqcode == "H_SCs_RuDs" )
    {
      HisSize = fNbBinsProj;
    }

  return HisSize;
}

TVectorD TEcnaHistos::GetHistoValues(const TVectorD& arg_read_histo, const Int_t&  arg_AlreadyRead,
				     TEcnaRead*      aMyRootFile,    const TString& HistoCode,
				     const Int_t&    HisSizePlot,    const Int_t&  HisSizeRead,
				     const Int_t&    StexStin_A,     const Int_t&  i0StinEcha,
				     const Int_t&    i0Sample,       Int_t&  i_data_exist)
{
  // Histo values in a TVectorD. i_data_exist entry value = 0. Incremented in this method.

  TVectorD plot_histo(HisSizePlot); for(Int_t i=0; i<HisSizePlot; i++){plot_histo(i)=(Double_t)0.;}

  fStatusDataExist = kFALSE;

  if( arg_AlreadyRead >= 1 )
    {
      //cout << "*TEcnaHistos::GetHistoValues(...)> arg_AlreadyRead = " << arg_AlreadyRead << endl;
      for(Int_t i=0; i<HisSizeRead; i++){plot_histo(i)=arg_read_histo(i);}
      fStatusDataExist = kTRUE; i_data_exist++;
    }

  if( arg_AlreadyRead == 0 )
    {
      //cout << "*TEcnaHistos::GetHistoValues(...)> arg_AlreadyRead = " << arg_AlreadyRead << endl;
      TVectorD read_histo(HisSizeRead); for(Int_t i=0; i<HisSizeRead; i++){read_histo(i)=(Double_t)0.;}

      if( HistoCode == "D_MSp_SpNb" || HistoCode == "D_MSp_SpDs" ||
	  HistoCode == "D_SSp_SpNb" || HistoCode == "D_SSp_SpDs" )
	{
	  //====> For plots as a function of Sample# (read10->plot10, read3->plot10)
	  if( HisSizeRead <= HisSizePlot )
	    {
	      if (HistoCode == "D_MSp_SpNb" || HistoCode == "D_MSp_SpDs" )
		{
		  read_histo = aMyRootFile->ReadSampleMeans(StexStin_A, i0StinEcha, HisSizeRead);
		  if( aMyRootFile->DataExist() == kTRUE ){fStatusDataExist = kTRUE; i_data_exist++;}
		  for(Int_t i=0; i<HisSizeRead; i++){plot_histo(i)=read_histo(i);}
		}
	  
	      if (HistoCode == "D_SSp_SpNb" || HistoCode == "D_SSp_SpDs" )
		{
		  read_histo = aMyRootFile->ReadSampleSigmas(StexStin_A, i0StinEcha, HisSizeRead);
		  if( aMyRootFile->DataExist() == kTRUE ){fStatusDataExist = kTRUE; i_data_exist++;}
		  for(Int_t i=0; i<HisSizeRead; i++){plot_histo(i)=read_histo(i);}
		}
	    }
	  else
	    {
	      cout << "!TEcnaHistos::GetHistoValues(...)> *** ERROR *** > HisSizeRead greater than HisSizePlot"
		   << " for plot as a function of sample#. HisSizeRead = " << HisSizeRead
		   << ", HisSizePlot = " << HisSizePlot << fTTBELL << endl;
	    }
	} // end of if( HistoCode == "D_MSp_SpNb" || HistoCode == "D_SSp_SpNb" " ||
	  //            HistoCode == "D_SSp_SpNb" || HistoCode == "D_SSp_SpDs" )

      if( !(HistoCode == "D_MSp_SpNb" || HistoCode == "D_SSp_SpNb" ||
	    HistoCode == "D_MSp_SpDs" || HistoCode == "D_SSp_SpDs" ) )  // = else of previous if
	{
	  //====> For other plots
	  if( HisSizeRead == HisSizePlot )
	    {
	      //========> for EE, HisSizeRead > HisSizePlot but readEcna#->plotForCons# will be build in the calling method
	      //          HisSizeRead = fEcal->MaxCrysEcnaInStex() (GetHistoValues)

	      if( HistoCode == "D_Adc_EvNb" || HistoCode == "D_Adc_EvDs" )
		{
		  read_histo = aMyRootFile->ReadSampleAdcValues(StexStin_A, i0StinEcha, i0Sample, HisSizeRead);
		  if( aMyRootFile->DataExist() == kTRUE ){fStatusDataExist = kTRUE; i_data_exist++;}
		}

	      if( HistoCode == "D_NOE_ChNb" || HistoCode == "D_NOE_ChDs" )
		{
		  read_histo = aMyRootFile->ReadNumberOfEvents(HisSizeRead);
		  if( aMyRootFile->DataExist() == kTRUE ){fStatusDataExist = kTRUE; i_data_exist++;}
		}
      
	      if( HistoCode == "D_Ped_ChNb" || HistoCode == "D_Ped_ChDs" )
		{
		  read_histo = aMyRootFile->ReadPedestals(HisSizeRead);
		  if( aMyRootFile->DataExist() == kTRUE ){fStatusDataExist = kTRUE; i_data_exist++;}
		}
      
	      //...................................................... (GetHistoValues)
	      if( HistoCode == "D_TNo_ChNb" || HistoCode == "D_TNo_ChDs")
		{
		  read_histo = aMyRootFile->ReadTotalNoise(HisSizeRead);
		  if( aMyRootFile->DataExist() == kTRUE ){fStatusDataExist = kTRUE; i_data_exist++;}
		}
      
	      if( HistoCode == "D_LFN_ChNb" || HistoCode == "D_LFN_ChDs" )
		{
		  read_histo = aMyRootFile->ReadLowFrequencyNoise(HisSizeRead);
		  if( aMyRootFile->DataExist() == kTRUE ){fStatusDataExist = kTRUE; i_data_exist++;}
		
		}
      
	      if( HistoCode == "D_HFN_ChNb" || HistoCode == "D_HFN_ChDs" )
		{
		  read_histo = aMyRootFile->ReadHighFrequencyNoise(HisSizeRead);
		  if( aMyRootFile->DataExist() == kTRUE ){fStatusDataExist = kTRUE; i_data_exist++;}
		
		}

      	      if( HistoCode == "D_MCs_ChNb" || HistoCode == "D_MCs_ChDs" )
		{
		  read_histo = aMyRootFile->ReadMeanCorrelationsBetweenSamples(HisSizeRead);
		  if( aMyRootFile->DataExist() == kTRUE ){fStatusDataExist = kTRUE; i_data_exist++;}
		}

	      if( HistoCode == "D_SCs_ChNb" || HistoCode == "D_SCs_ChDs"  )
		{
		  read_histo = aMyRootFile->ReadSigmaOfCorrelationsBetweenSamples(HisSizeRead);
		  if( aMyRootFile->DataExist() == kTRUE ){fStatusDataExist = kTRUE; i_data_exist++;}
		}

	      for(Int_t i=0; i<HisSizeRead; i++){plot_histo(i)=read_histo(i);}
	    
	    }
	  else
	    {
	      cout << "!TEcnaHistos::GetHistoValues(...)> *** ERROR *** > HisSizeRead not equal to HisSizePlot."
		   << " HisSizeRead = " << HisSizeRead
		   << ", HisSizePlot = " << HisSizePlot << fTTBELL << endl;
	    }
	}  // end of if( !(HistoCode == "D_MSp_SpNb" || HistoCode == "D_SSp_SpNb") )
    }  // end of if( arg_AlreadyRead == 0 )

  if( i_data_exist == 0 )
    {
      cout << "!TEcnaHistos::GetHistoValues(...)> Histo not found." << fTTBELL << endl;
    }

  return plot_histo;
}
//------- (end of GetHistoValues) -------------

TString  TEcnaHistos::SetHistoXAxisTitle(const TString& HistoCode)
{
  // Set histo X axis title

  TString axis_x_var_name;
  
  if(HistoCode == "D_NOE_ChNb" || HistoCode == "D_Ped_ChNb" ||
     HistoCode == "D_TNo_ChNb" || HistoCode == "D_MCs_ChNb" ||
     HistoCode == "D_LFN_ChNb" || HistoCode == "D_HFN_ChNb" ||
     HistoCode == "D_SCs_ChNb" )
    {
      if( fFapStexNumber >  0 )
	{
	  if( fFlagSubDet == "EB" ){axis_x_var_name = "Xtal (electronic channel number)";}
	  if( fFlagSubDet == "EE" ){axis_x_var_name = "Xtal";}
	}
      if( fFapStexNumber ==  0 )
	{
	  if( fFlagSubDet == "EB" ){axis_x_var_name = "Tower number";}
	  if( fFlagSubDet == "EE" ){axis_x_var_name = "SC number";}
	}
    }

  if(HistoCode == "D_NOE_ChDs"){axis_x_var_name = "Number of events";}
  if(HistoCode == "D_Ped_ChDs"){axis_x_var_name = "Pedestal";}
  if(HistoCode == "D_TNo_ChDs"){axis_x_var_name = "Total noise";}
  if(HistoCode == "D_MCs_ChDs"){axis_x_var_name = "Mean cor(s,s')";}
  if(HistoCode == "D_LFN_ChDs"){axis_x_var_name = "Low frequency noise";} 
  if(HistoCode == "D_HFN_ChDs"){axis_x_var_name = "High frequency noise";}
  if(HistoCode == "D_SCs_ChDs"){axis_x_var_name = "Sigmas cor(s,s')";}
  if(HistoCode == "D_MSp_SpNb"){axis_x_var_name = "Sample";}
  if(HistoCode == "D_MSp_SpDs"){axis_x_var_name = "Pedestal";}
  if(HistoCode == "D_SSp_SpNb"){axis_x_var_name = "Sample";}
  if(HistoCode == "D_SSp_SpDs"){axis_x_var_name = "Total noise";}
  if(HistoCode == "D_Adc_EvDs"){axis_x_var_name = "ADC";}
  if(HistoCode == "D_Adc_EvNb"){axis_x_var_name = "Event number";}
  if(HistoCode == "H_Ped_Date" || HistoCode == "H_TNo_Date" || HistoCode == "H_MCs_Date" ||
     HistoCode == "H_LFN_Date" || HistoCode == "H_HFN_Date" || HistoCode == "H_SCs_Date")
    {axis_x_var_name = "Time";}
  if(HistoCode == "H_Ped_RuDs"){axis_x_var_name = "Pedestal";}
  if(HistoCode == "H_TNo_RuDs"){axis_x_var_name = "Total noise";}
  if(HistoCode == "H_MCs_RuDs"){axis_x_var_name = "Mean cor(s,s')";}
  if(HistoCode == "H_LFN_RuDs"){axis_x_var_name = "Low frequency noise";} 
  if(HistoCode == "H_HFN_RuDs"){axis_x_var_name = "High frequency noise";}
  if(HistoCode == "H_SCs_RuDs"){axis_x_var_name = "Sigmas cor(s,s')";} 

  return axis_x_var_name; 
}

TString  TEcnaHistos::SetHistoYAxisTitle(const TString& HistoCode)
{
// Set histo Y axis title

  TString axis_y_var_name;

  if(HistoCode == "D_NOE_ChNb"){axis_y_var_name = "Number of events";}
  if(HistoCode == "D_Ped_ChNb"){axis_y_var_name = "Pedestal";}
  if(HistoCode == "D_TNo_ChNb"){axis_y_var_name = "Total noise";}
  if(HistoCode == "D_MCs_ChNb"){axis_y_var_name = "Mean cor(s,s')";}
  if(HistoCode == "D_LFN_ChNb"){axis_y_var_name = "Low frequency noise";} 
  if(HistoCode == "D_HFN_ChNb"){axis_y_var_name = "High frequency noise";}  
  if(HistoCode == "D_SCs_ChNb"){axis_y_var_name = "Sigma of cor(s,s')";}
  
  if(HistoCode == "D_NOE_ChDs" ||
     HistoCode == "D_Ped_ChDs" || HistoCode == "D_TNo_ChDs" || HistoCode == "D_MCs_ChDs" ||
     HistoCode == "D_LFN_ChDs" || HistoCode == "D_HFN_ChDs" || HistoCode == "D_SCs_ChDs" )  
    {
      if( fFapStexNumber >  0 ){axis_y_var_name = "number of crystals";}
      if( fFapStexNumber == 0 )
	{
	  if( fFlagSubDet == "EB" ){axis_y_var_name = "number of towers";}
	  if( fFlagSubDet == "EE" ){axis_y_var_name = "number of SC's";}
	}
    }

  if(HistoCode == "D_MSp_SpNb"){axis_y_var_name = "Sample mean";}
  if(HistoCode == "D_MSp_SpDs"){axis_y_var_name = "Number of samples";}
  if(HistoCode == "D_SSp_SpNb"){axis_y_var_name = "Sample sigma";}
  if(HistoCode == "D_SSp_SpDs"){axis_y_var_name = "Number of samples";}
  if(HistoCode == "D_Adc_EvNb"){axis_y_var_name = "Sample ADC value";}
  if(HistoCode == "D_Adc_EvDs"){axis_y_var_name = "Number of events";}
  if(HistoCode == "H_Ped_Date"){axis_y_var_name = "Pedestal";}
  if(HistoCode == "H_TNo_Date"){axis_y_var_name = "Total noise";}
  if(HistoCode == "H_MCs_Date"){axis_y_var_name = "Mean cor(s,s')";}
  if(HistoCode == "H_LFN_Date"){axis_y_var_name = "Low frequency noise";}
  if(HistoCode == "H_HFN_Date"){axis_y_var_name = "High frequency noise";}
  if(HistoCode == "H_SCs_Date"){axis_y_var_name = "Sigma cor(s,s')";}

  if(HistoCode == "H_Ped_RuDs" || HistoCode == "H_TNo_RuDs" || HistoCode == "H_MCs_RuDs" ||
     HistoCode == "H_LFN_RuDs" || HistoCode == "H_HFN_RuDs" ||  HistoCode == "H_SCs_RuDs" ) 
    {axis_y_var_name = "number of runs";}

  return axis_y_var_name;
}
//-------------------------------------------------------------------------------
Axis_t TEcnaHistos::GetHistoXinf(const TString& HistoCode, const Int_t& HisSize, const TString& opt_plot)
{
// Set histo Xinf

  Axis_t xinf_his = (Axis_t)0;

  if(HistoCode == "D_NOE_ChNb"){xinf_his = (Axis_t)0.;}
  if(HistoCode == "D_Ped_ChNb"){xinf_his = (Axis_t)0.;}
  if(HistoCode == "D_TNo_ChNb"){xinf_his = (Axis_t)0.;}
  if(HistoCode == "D_MCs_ChNb"){xinf_his = (Axis_t)0.;}
  if(HistoCode == "D_LFN_ChNb"){xinf_his = (Axis_t)0.;} 
  if(HistoCode == "D_HFN_ChNb"){xinf_his = (Axis_t)0.;}  
  if(HistoCode == "D_SCs_ChNb"){xinf_his = (Axis_t)0.;}
  if(HistoCode == "D_MSp_SpNb"){xinf_his = (Axis_t)0.;}
  if(HistoCode == "D_SSp_SpNb"){xinf_his = (Axis_t)0.;}
  if(HistoCode == "D_Adc_EvNb"){xinf_his = (Axis_t)0.;}

  if(HistoCode == "D_NOE_ChDs"){xinf_his = (Axis_t)fD_NOE_ChNbYmin;}  // D_XXX_YYDs = projection of D_XXX_YYNb
  if(HistoCode == "D_Ped_ChDs"){xinf_his = (Axis_t)fD_Ped_ChNbYmin;}
  if(HistoCode == "D_TNo_ChDs"){xinf_his = (Axis_t)fD_TNo_ChNbYmin;}
  if(HistoCode == "D_MCs_ChDs"){xinf_his = (Axis_t)fD_MCs_ChNbYmin;}
  if(HistoCode == "D_LFN_ChDs"){xinf_his = (Axis_t)fD_LFN_ChNbYmin;}
  if(HistoCode == "D_HFN_ChDs"){xinf_his = (Axis_t)fD_HFN_ChNbYmin;}
  if(HistoCode == "D_SCs_ChDs"){xinf_his = (Axis_t)fD_SCs_ChNbYmin;}
  if(HistoCode == "D_MSp_SpDs"){xinf_his = (Axis_t)fD_MSp_SpNbYmin;}
  if(HistoCode == "D_SSp_SpDs"){xinf_his = (Axis_t)fD_SSp_SpNbYmin;}
  if(HistoCode == "D_Adc_EvDs"){xinf_his = (Axis_t)fD_Adc_EvNbYmin;}

  if(HistoCode == "H_Ped_Date"){xinf_his = (Axis_t)0.;}
  if(HistoCode == "H_TNo_Date"){xinf_his = (Axis_t)0.;}
  if(HistoCode == "H_MCs_Date"){xinf_his = (Axis_t)0.;}
  if(HistoCode == "H_LFN_Date"){xinf_his = (Axis_t)0.;}
  if(HistoCode == "H_HFN_Date"){xinf_his = (Axis_t)0.;}
  if(HistoCode == "H_SCs_Date"){xinf_his = (Axis_t)0.;}

  if(HistoCode == "H_Ped_RuDs"){xinf_his = (Axis_t)fH_Ped_RuDsYmin;}
  if(HistoCode == "H_TNo_RuDs"){xinf_his = (Axis_t)fH_TNo_RuDsYmin;}
  if(HistoCode == "H_MCs_RuDs"){xinf_his = (Axis_t)fH_MCs_RuDsYmin;}
  if(HistoCode == "H_LFN_RuDs"){xinf_his = (Axis_t)fH_LFN_RuDsYmin;}
  if(HistoCode == "H_HFN_RuDs"){xinf_his = (Axis_t)fH_HFN_RuDsYmin;}
  if(HistoCode == "H_SCs_RuDs"){xinf_his = (Axis_t)fH_SCs_RuDsYmin;}

  return xinf_his;
}

Axis_t TEcnaHistos::GetHistoXsup(const TString& HistoCode, const Int_t& HisSize, const TString& opt_plot)
{
// Set histo Xsup

  Axis_t xsup_his = (Axis_t)0; 

  if(HistoCode == "D_NOE_ChNb"){xsup_his = (Axis_t)HisSize;}
  if(HistoCode == "D_Ped_ChNb"){xsup_his = (Axis_t)HisSize;}
  if(HistoCode == "D_TNo_ChNb"){xsup_his = (Axis_t)HisSize;}
  if(HistoCode == "D_MCs_ChNb"){xsup_his = (Axis_t)HisSize;}
  if(HistoCode == "D_LFN_ChNb"){xsup_his = (Axis_t)HisSize;} 
  if(HistoCode == "D_HFN_ChNb"){xsup_his = (Axis_t)HisSize;}  
  if(HistoCode == "D_SCs_ChNb"){xsup_his = (Axis_t)HisSize;}
  if(HistoCode == "D_MSp_SpNb"){xsup_his = (Axis_t)HisSize;}
  if(HistoCode == "D_SSp_SpNb"){xsup_his = (Axis_t)HisSize;}
  if(HistoCode == "D_Adc_EvNb"){xsup_his = (Axis_t)(fFapReqNbOfEvts);}

  if(HistoCode == "D_NOE_ChDs"){xsup_his = (Axis_t)fD_NOE_ChNbYmax;}
  if(HistoCode == "D_Ped_ChDs"){xsup_his = (Axis_t)fD_Ped_ChNbYmax;}
  if(HistoCode == "D_TNo_ChDs"){xsup_his = (Axis_t)fD_TNo_ChNbYmax;}
  if(HistoCode == "D_MCs_ChDs"){xsup_his = (Axis_t)fD_MCs_ChNbYmax;}
  if(HistoCode == "D_LFN_ChDs"){xsup_his = (Axis_t)fD_LFN_ChNbYmax;}
  if(HistoCode == "D_HFN_ChDs"){xsup_his = (Axis_t)fD_HFN_ChNbYmax;}
  if(HistoCode == "D_SCs_ChDs"){xsup_his = (Axis_t)fD_SCs_ChNbYmax;}
  if(HistoCode == "D_MSp_SpDs"){xsup_his = (Axis_t)fD_MSp_SpNbYmax;}
  if(HistoCode == "D_SSp_SpDs"){xsup_his = (Axis_t)fD_SSp_SpNbYmax;}
  if(HistoCode == "D_Adc_EvDs"){xsup_his = (Axis_t)fD_Adc_EvNbYmax;}

  if(HistoCode == "H_Ped_Date"){xsup_his = (Axis_t)0.;}
  if(HistoCode == "H_TNo_Date"){xsup_his = (Axis_t)0.;}
  if(HistoCode == "H_MCs_Date"){xsup_his = (Axis_t)0.;}
  if(HistoCode == "H_LFN_Date"){xsup_his = (Axis_t)0.;}
  if(HistoCode == "H_HFN_Date"){xsup_his = (Axis_t)0.;}
  if(HistoCode == "H_SCs_Date"){xsup_his = (Axis_t)0.;}

  if(HistoCode == "H_Ped_RuDs"){xsup_his = (Axis_t)fH_Ped_RuDsYmax;}
  if(HistoCode == "H_TNo_RuDs"){xsup_his = (Axis_t)fH_TNo_RuDsYmax;}
  if(HistoCode == "H_MCs_RuDs"){xsup_his = (Axis_t)fH_MCs_RuDsYmax;}
  if(HistoCode == "H_LFN_RuDs"){xsup_his = (Axis_t)fH_LFN_RuDsYmax;}
  if(HistoCode == "H_HFN_RuDs"){xsup_his = (Axis_t)fH_HFN_RuDsYmax;}
  if(HistoCode == "H_SCs_RuDs"){xsup_his = (Axis_t)fH_SCs_RuDsYmax;}

  return xsup_his;
}
//-----------------------------------------------------------------------------------
Int_t TEcnaHistos::GetHistoNumberOfBins(const TString& HistoCode, const Int_t& HisSize)
{
// Set histo number of bins

  Int_t nb_binx = HisSize;
  TString HistoType = fCnaParHistos->GetHistoType(HistoCode.Data());
  if ( HistoType == "Proj"     ||  HistoType == "SampProj" ||
       HistoType == "EvolProj" || HistoType == "H1BasicProj" )
    {nb_binx = fNbBinsProj;}

  return nb_binx;
}
//-----------------------------------------------------------------------------------
void TEcnaHistos::FillHisto(TH1D* h_his0, const TVectorD& read_histo, const TString& HistoCode,
			    const Int_t&  HisSize)
{
// Fill histo

  h_his0->Reset();

  for(Int_t i=0; i<HisSize; i++)
    {
      Double_t his_val = (Double_t)0;
      Double_t xi      = (Double_t)0;
      //................................................... Basic + Global
      if (HistoCode == "D_NOE_ChNb" || HistoCode == "D_Ped_ChNb" ||
	  HistoCode == "D_LFN_ChNb" || HistoCode == "D_TNo_ChNb" ||
	  HistoCode == "D_HFN_ChNb" || HistoCode == "D_MCs_ChNb" ||
	  HistoCode == "D_SCs_ChNb" || HistoCode == "D_MSp_SpNb" || HistoCode == "D_SSp_SpNb" )
	{
	  xi = (Double_t)i;
	  his_val = (Double_t)read_histo[i];
	  h_his0->Fill(xi, his_val);
	}

      //................................................... D_Adc_EvNb option
      if (HistoCode == "D_Adc_EvNb" )
	{
	  xi = (Double_t)i;
	  his_val = (Double_t)read_histo[i];
	  h_his0->Fill(xi, his_val);
	}
      //................................................... Proj
      if (HistoCode == "D_NOE_ChDs" ||
	  HistoCode == "D_Ped_ChDs" || HistoCode == "D_LFN_ChDs" ||
	  HistoCode == "D_TNo_ChDs" || HistoCode == "D_HFN_ChDs" ||
	  HistoCode == "D_MCs_ChDs" || HistoCode == "D_SCs_ChDs" ||
	  HistoCode == "D_MSp_SpDs" || HistoCode == "D_SSp_SpDs" ||
	  HistoCode == "D_Adc_EvDs" )
	{
	  his_val = (Double_t)read_histo[i];
	  Double_t increment = (Double_t)1;
	  h_his0->Fill(his_val, increment);
	}

      //................................................... EvolProj
      //
      //    *=======> direct Fill in ViewHistime(...)
      //
    }
}

//===========================================================================
//
//          SetXinfMemoFromValue(...), SetXsupMemoFromValue(...)
//          GetXsupValueFromMemo(...), GetXsupValueFromMemo(...)
//
//===========================================================================
void TEcnaHistos::SetXinfMemoFromValue(const TString& HistoCode, const Double_t& value)
{
  if( HistoCode == "D_NOE_ChNb"){fD_NOE_ChNbXinf = value;}
  if( HistoCode == "D_NOE_ChDs"){fD_NOE_ChDsXinf = value;}  
  if( HistoCode == "D_Ped_ChNb"){fD_Ped_ChNbXinf = value;} 
  if( HistoCode == "D_Ped_ChDs"){fD_Ped_ChDsXinf = value;} 
  if( HistoCode == "D_TNo_ChNb"){fD_TNo_ChNbXinf = value;}
  if( HistoCode == "D_TNo_ChDs"){fD_TNo_ChDsXinf = value;} 
  if( HistoCode == "D_MCs_ChNb"){fD_MCs_ChNbXinf = value;} 
  if( HistoCode == "D_MCs_ChDs"){fD_MCs_ChDsXinf = value;} 
  if( HistoCode == "D_LFN_ChNb"){fD_LFN_ChNbXinf = value;} 
  if( HistoCode == "D_LFN_ChDs"){fD_LFN_ChDsXinf = value;}
  if( HistoCode == "D_HFN_ChNb"){fD_HFN_ChNbXinf = value;} 
  if( HistoCode == "D_HFN_ChDs"){fD_HFN_ChDsXinf = value;} 
  if( HistoCode == "D_SCs_ChNb"){fD_SCs_ChNbXinf = value;}
  if( HistoCode == "D_SCs_ChDs"){fD_SCs_ChDsXinf = value;}
  if( HistoCode == "D_MSp_SpNb"){fD_Ped_ChNbXinf = value;}
  if( HistoCode == "D_MSp_SpDs"){fD_Ped_ChDsXinf = value;}
  if( HistoCode == "D_SSp_SpNb"){fD_TNo_ChNbXinf = value;}
  if( HistoCode == "D_SSp_SpDs"){fD_TNo_ChDsXinf = value;}
  if( HistoCode == "D_Adc_EvNb"){fD_Ped_ChNbXinf = value;}
  if( HistoCode == "D_Adc_EvDs"){fD_Adc_EvDsXinf = value;}
  if( HistoCode == "H_Ped_Date"){fH_Ped_DateXinf = value;}
  if( HistoCode == "H_TNo_Date"){fH_TNo_DateXinf = value;}
  if( HistoCode == "H_MCs_Date"){fH_MCs_DateXinf = value;}
  if( HistoCode == "H_LFN_Date"){fH_LFN_DateXinf = value;}
  if( HistoCode == "H_HFN_Date"){fH_HFN_DateXinf = value;}
  if( HistoCode == "H_SCs_Date"){fH_SCs_DateXinf = value;}
  if( HistoCode == "H_Ped_RuDs"){fH_Ped_RuDsXinf = value;}
  if( HistoCode == "H_TNo_RuDs"){fH_TNo_RuDsXinf = value;}
  if( HistoCode == "H_MCs_RuDs"){fH_MCs_RuDsXinf = value;}
  if( HistoCode == "H_LFN_RuDs"){fH_LFN_RuDsXinf = value;}
  if( HistoCode == "H_HFN_RuDs"){fH_HFN_RuDsXinf = value;}
  if( HistoCode == "H_SCs_RuDs"){fH_SCs_RuDsXinf = value;}
}// end of SetXinfMemoFromValue(...)

void TEcnaHistos::SetXinfMemoFromValue(const Double_t& value)
{fH1SameOnePlotXinf = value;}

void TEcnaHistos::SetXsupMemoFromValue(const TString& HistoCode, const Double_t& value)
{
  if( HistoCode == "D_NOE_ChNb"){fD_NOE_ChNbXsup = value;}
  if( HistoCode == "D_NOE_ChDs"){fD_NOE_ChDsXsup = value;}  
  if( HistoCode == "D_Ped_ChNb"){fD_Ped_ChNbXsup = value;} 
  if( HistoCode == "D_Ped_ChDs"){fD_Ped_ChDsXsup = value;} 
  if( HistoCode == "D_TNo_ChNb"){fD_TNo_ChNbXsup = value;}
  if( HistoCode == "D_TNo_ChDs"){fD_TNo_ChDsXsup = value;} 
  if( HistoCode == "D_MCs_ChNb"){fD_MCs_ChNbXsup = value;} 
  if( HistoCode == "D_MCs_ChDs"){fD_MCs_ChDsXsup = value;} 
  if( HistoCode == "D_LFN_ChNb"){fD_LFN_ChNbXsup = value;} 
  if( HistoCode == "D_LFN_ChDs"){fD_LFN_ChDsXsup = value;}
  if( HistoCode == "D_HFN_ChNb"){fD_HFN_ChNbXsup = value;} 
  if( HistoCode == "D_HFN_ChDs"){fD_HFN_ChDsXsup = value;} 
  if( HistoCode == "D_SCs_ChNb"){fD_SCs_ChNbXsup = value;}
  if( HistoCode == "D_SCs_ChDs"){fD_SCs_ChDsXsup = value;}
  if( HistoCode == "D_MSp_SpNb"){fD_Ped_ChNbXsup = value;}
  if( HistoCode == "D_MSp_SpDs"){fD_Ped_ChDsXsup = value;}
  if( HistoCode == "D_SSp_SpNb"){fD_TNo_ChNbXsup = value;}
  if( HistoCode == "D_SSp_SpDs"){fD_TNo_ChDsXsup = value;}
  if( HistoCode == "D_Adc_EvNb"){fD_Ped_ChNbXsup = value;}
  if( HistoCode == "D_Adc_EvDs"){fD_Adc_EvDsXsup = value;}
  if( HistoCode == "H_Ped_Date"){fH_Ped_DateXsup = value;}
  if( HistoCode == "H_TNo_Date"){fH_TNo_DateXsup = value;}
  if( HistoCode == "H_MCs_Date"){fH_MCs_DateXsup = value;}
  if( HistoCode == "H_LFN_Date"){fH_LFN_DateXsup = value;}
  if( HistoCode == "H_HFN_Date"){fH_HFN_DateXsup = value;}
  if( HistoCode == "H_SCs_Date"){fH_SCs_DateXsup = value;}
  if( HistoCode == "H_Ped_RuDs"){fH_Ped_RuDsXsup = value;}
  if( HistoCode == "H_TNo_RuDs"){fH_TNo_RuDsXsup = value;}
  if( HistoCode == "H_MCs_RuDs"){fH_MCs_RuDsXsup = value;}
  if( HistoCode == "H_LFN_RuDs"){fH_LFN_RuDsXsup = value;}
  if( HistoCode == "H_HFN_RuDs"){fH_HFN_RuDsXsup = value;}
  if( HistoCode == "H_SCs_RuDs"){fH_SCs_RuDsXsup = value;}
}// end of SetXsupMemoFromValue(...)

void TEcnaHistos::SetXsupMemoFromValue(const Double_t& value)
{fH1SameOnePlotXsup = value;}

Double_t TEcnaHistos::GetXinfValueFromMemo(const TString& HistoCode)
{
  Double_t val_inf      = (Double_t)0.;

   if( HistoCode == "D_NOE_ChNb"){val_inf = fD_NOE_ChNbXinf;}
   if( HistoCode == "D_NOE_ChDs"){val_inf = fD_NOE_ChDsXinf;}  
   if( HistoCode == "D_Ped_ChNb"){val_inf = fD_Ped_ChNbXinf;} 
   if( HistoCode == "D_Ped_ChDs"){val_inf = fD_Ped_ChDsXinf;} 
   if( HistoCode == "D_TNo_ChNb"){val_inf = fD_TNo_ChNbXinf;}
   if( HistoCode == "D_TNo_ChDs"){val_inf = fD_TNo_ChDsXinf;} 
   if( HistoCode == "D_MCs_ChNb"){val_inf = fD_MCs_ChNbXinf;} 
   if( HistoCode == "D_MCs_ChDs"){val_inf = fD_MCs_ChDsXinf;} 
   if( HistoCode == "D_LFN_ChNb"){val_inf = fD_LFN_ChNbXinf;} 
   if( HistoCode == "D_LFN_ChDs"){val_inf = fD_LFN_ChDsXinf;}
   if( HistoCode == "D_HFN_ChNb"){val_inf = fD_HFN_ChNbXinf;} 
   if( HistoCode == "D_HFN_ChDs"){val_inf = fD_HFN_ChDsXinf;} 
   if( HistoCode == "D_SCs_ChNb"){val_inf = fD_SCs_ChNbXinf;}
   if( HistoCode == "D_SCs_ChDs"){val_inf = fD_SCs_ChDsXinf;}
   if( HistoCode == "D_MSp_SpNb"){val_inf = fD_Ped_ChNbXinf;}
   if( HistoCode == "D_MSp_SpDs"){val_inf = fD_Ped_ChDsXinf;}
   if( HistoCode == "D_SSp_SpNb"){val_inf = fD_TNo_ChNbXinf;}
   if( HistoCode == "D_SSp_SpDs"){val_inf = fD_TNo_ChDsXinf;}
   if( HistoCode == "D_Adc_EvNb"){val_inf = fD_Adc_EvNbXinf;}
   if( HistoCode == "D_Adc_EvDs"){val_inf = fD_Adc_EvDsXinf;}
   if( HistoCode == "H_Ped_Date"){val_inf = fH_Ped_DateXinf;}
   if( HistoCode == "H_TNo_Date"){val_inf = fH_TNo_DateXinf;}
   if( HistoCode == "H_MCs_Date"){val_inf = fH_MCs_DateXinf;}
   if( HistoCode == "H_LFN_Date"){val_inf = fH_LFN_DateXinf;}
   if( HistoCode == "H_HFN_Date"){val_inf = fH_HFN_DateXinf;}
   if( HistoCode == "H_SCs_Date"){val_inf = fH_SCs_DateXinf;}
   if( HistoCode == "H_Ped_RuDs"){val_inf = fH_Ped_RuDsXinf;}
   if( HistoCode == "H_TNo_RuDs"){val_inf = fH_TNo_RuDsXinf;}
   if( HistoCode == "H_MCs_RuDs"){val_inf = fH_MCs_RuDsXinf;}
   if( HistoCode == "H_LFN_RuDs"){val_inf = fH_LFN_RuDsXinf;}
   if( HistoCode == "H_HFN_RuDs"){val_inf = fH_HFN_RuDsXinf;}
   if( HistoCode == "H_SCs_RuDs"){val_inf = fH_SCs_RuDsXinf;}
  return val_inf;
}// end of GetXinfValueFromMemo(...)

Double_t TEcnaHistos::GetXinfValueFromMemo()
{return fH1SameOnePlotXinf;}

Double_t TEcnaHistos::GetXsupValueFromMemo(const TString& HistoCode)
{
  Double_t val_sup      = (Double_t)0.;

   if( HistoCode == "D_NOE_ChNb"){val_sup = fD_NOE_ChNbXsup;}
   if( HistoCode == "D_NOE_ChDs"){val_sup = fD_NOE_ChDsXsup;}  
   if( HistoCode == "D_Ped_ChNb"){val_sup = fD_Ped_ChNbXsup;} 
   if( HistoCode == "D_Ped_ChDs"){val_sup = fD_Ped_ChDsXsup;} 
   if( HistoCode == "D_TNo_ChNb"){val_sup = fD_TNo_ChNbXsup;}
   if( HistoCode == "D_TNo_ChDs"){val_sup = fD_TNo_ChDsXsup;} 
   if( HistoCode == "D_MCs_ChNb"){val_sup = fD_MCs_ChNbXsup;} 
   if( HistoCode == "D_MCs_ChDs"){val_sup = fD_MCs_ChDsXsup;} 
   if( HistoCode == "D_LFN_ChNb"){val_sup = fD_LFN_ChNbXsup;} 
   if( HistoCode == "D_LFN_ChDs"){val_sup = fD_LFN_ChDsXsup;}
   if( HistoCode == "D_HFN_ChNb"){val_sup = fD_HFN_ChNbXsup;} 
   if( HistoCode == "D_HFN_ChDs"){val_sup = fD_HFN_ChDsXsup;} 
   if( HistoCode == "D_SCs_ChNb"){val_sup = fD_SCs_ChNbXsup;}
   if( HistoCode == "D_SCs_ChDs"){val_sup = fD_SCs_ChDsXsup;}
   if( HistoCode == "D_MSp_SpNb"){val_sup = fD_Ped_ChNbXsup;}
   if( HistoCode == "D_MSp_SpDs"){val_sup = fD_Ped_ChDsXsup;}
   if( HistoCode == "D_SSp_SpNb"){val_sup = fD_TNo_ChNbXsup;}
   if( HistoCode == "D_SSp_SpDs"){val_sup = fD_TNo_ChDsXsup;}
   if( HistoCode == "D_Adc_EvNb"){val_sup = fD_Adc_EvNbXsup;}
   if( HistoCode == "D_Adc_EvDs"){val_sup = fD_Adc_EvDsXsup;}
   if( HistoCode == "H_Ped_Date"){val_sup = fH_Ped_DateXsup;}
   if( HistoCode == "H_TNo_Date"){val_sup = fH_TNo_DateXsup;}
   if( HistoCode == "H_MCs_Date"){val_sup = fH_MCs_DateXsup;}
   if( HistoCode == "H_LFN_Date"){val_sup = fH_LFN_DateXsup;}
   if( HistoCode == "H_HFN_Date"){val_sup = fH_HFN_DateXsup;}
   if( HistoCode == "H_SCs_Date"){val_sup = fH_SCs_DateXsup;}
   if( HistoCode == "H_Ped_RuDs"){val_sup = fH_Ped_RuDsXsup;}
   if( HistoCode == "H_TNo_RuDs"){val_sup = fH_TNo_RuDsXsup;}
   if( HistoCode == "H_MCs_RuDs"){val_sup = fH_MCs_RuDsXsup;}
   if( HistoCode == "H_LFN_RuDs"){val_sup = fH_LFN_RuDsXsup;}
   if( HistoCode == "H_HFN_RuDs"){val_sup = fH_HFN_RuDsXsup;}
   if( HistoCode == "H_SCs_RuDs"){val_sup = fH_SCs_RuDsXsup;}
  return val_sup;
}// end of GetXsupValueFromMemo(...)

Double_t TEcnaHistos::GetXsupValueFromMemo()
{return fH1SameOnePlotXsup;}

//-------------------------------------------------------------------------------------------
//
//           SetHistoMin, SetHistoMax, SetAllYminYmaxMemoFromDefaultValues
//
//-------------------------------------------------------------------------------------------
void TEcnaHistos::SetHistoMin(const Double_t& value){fUserHistoMin = value; fFlagUserHistoMin = "ON";}
void TEcnaHistos::SetHistoMax(const Double_t& value){fUserHistoMax = value; fFlagUserHistoMax = "ON";}

void TEcnaHistos::SetHistoMin(){fFlagUserHistoMin = "AUTO";}
void TEcnaHistos::SetHistoMax(){fFlagUserHistoMax = "AUTO";}

void TEcnaHistos::SetAllYminYmaxMemoFromDefaultValues()
{
//.......... Default values for histo min and max

  SetYminMemoFromValue("D_NOE_ChNb", fCnaParHistos->GetYminDefaultValue("D_NOE_ChNb"));
  SetYmaxMemoFromValue("D_NOE_ChNb", fCnaParHistos->GetYmaxDefaultValue("D_NOE_ChNb"));

  SetYminMemoFromValue("D_NOE_ChDs", fCnaParHistos->GetYminDefaultValue("D_NOE_ChDs"));
  SetYmaxMemoFromValue("D_NOE_ChDs", fCnaParHistos->GetYmaxDefaultValue("D_NOE_ChDs"));

  SetYminMemoFromValue("D_Ped_ChNb", fCnaParHistos->GetYminDefaultValue("D_Ped_ChNb"));
  SetYmaxMemoFromValue("D_Ped_ChNb", fCnaParHistos->GetYmaxDefaultValue("D_Ped_ChNb"));

  SetYminMemoFromValue("D_Ped_ChDs", fCnaParHistos->GetYminDefaultValue("D_Ped_ChDs"));
  SetYmaxMemoFromValue("D_Ped_ChDs", fCnaParHistos->GetYmaxDefaultValue("D_Ped_ChDs"));

  SetYminMemoFromValue("D_TNo_ChNb", fCnaParHistos->GetYminDefaultValue("D_TNo_ChNb"));
  SetYmaxMemoFromValue("D_TNo_ChNb", fCnaParHistos->GetYmaxDefaultValue("D_TNo_ChNb"));

  SetYminMemoFromValue("D_TNo_ChDs", fCnaParHistos->GetYminDefaultValue("D_TNo_ChDs"));
  SetYmaxMemoFromValue("D_TNo_ChDs", fCnaParHistos->GetYmaxDefaultValue("D_TNo_ChDs"));

  SetYminMemoFromValue("D_MCs_ChNb", fCnaParHistos->GetYminDefaultValue("D_MCs_ChNb"));
  SetYmaxMemoFromValue("D_MCs_ChNb", fCnaParHistos->GetYmaxDefaultValue("D_MCs_ChNb"));

  SetYminMemoFromValue("D_MCs_ChDs", fCnaParHistos->GetYminDefaultValue("D_MCs_ChDs"));
  SetYmaxMemoFromValue("D_MCs_ChDs", fCnaParHistos->GetYmaxDefaultValue("D_MCs_ChDs"));

  SetYminMemoFromValue("D_LFN_ChNb", fCnaParHistos->GetYminDefaultValue("D_LFN_ChNb"));
  SetYmaxMemoFromValue("D_LFN_ChNb", fCnaParHistos->GetYmaxDefaultValue("D_LFN_ChNb"));

  SetYminMemoFromValue("D_LFN_ChDs", fCnaParHistos->GetYminDefaultValue("D_LFN_ChDs"));
  SetYmaxMemoFromValue("D_LFN_ChDs", fCnaParHistos->GetYmaxDefaultValue("D_LFN_ChDs"));

  SetYminMemoFromValue("D_HFN_ChNb", fCnaParHistos->GetYminDefaultValue("D_HFN_ChNb"));
  SetYmaxMemoFromValue("D_HFN_ChNb", fCnaParHistos->GetYmaxDefaultValue("D_HFN_ChNb"));

  SetYminMemoFromValue("D_HFN_ChDs", fCnaParHistos->GetYminDefaultValue("D_HFN_ChDs"));
  SetYmaxMemoFromValue("D_HFN_ChDs", fCnaParHistos->GetYmaxDefaultValue("D_HFN_ChDs"));

  SetYminMemoFromValue("D_SCs_ChNb", fCnaParHistos->GetYminDefaultValue("D_SCs_ChNb"));
  SetYmaxMemoFromValue("D_SCs_ChNb", fCnaParHistos->GetYmaxDefaultValue("D_SCs_ChNb"));

  SetYminMemoFromValue("D_SCs_ChDs", fCnaParHistos->GetYminDefaultValue("D_SCs_ChDs"));
  SetYmaxMemoFromValue("D_SCs_ChDs", fCnaParHistos->GetYmaxDefaultValue("D_SCs_ChDs"));

  SetYminMemoFromValue("D_MSp_SpNb", fCnaParHistos->GetYminDefaultValue("D_MSp_SpNb"));
  SetYmaxMemoFromValue("D_MSp_SpNb", fCnaParHistos->GetYmaxDefaultValue("D_MSp_SpNb"));

  SetYminMemoFromValue("D_MSp_SpDs", fCnaParHistos->GetYminDefaultValue("D_MSp_SpDs"));
  SetYmaxMemoFromValue("D_MSp_SpDs", fCnaParHistos->GetYmaxDefaultValue("D_MSp_SpDs"));

  SetYminMemoFromValue("D_SSp_SpNb", fCnaParHistos->GetYminDefaultValue("D_SSp_SpNb"));
  SetYmaxMemoFromValue("D_SSp_SpNb", fCnaParHistos->GetYmaxDefaultValue("D_SSp_SpNb"));

  SetYminMemoFromValue("D_SSp_SpDs", fCnaParHistos->GetYminDefaultValue("D_SSp_SpDs"));
  SetYmaxMemoFromValue("D_SSp_SpDs", fCnaParHistos->GetYmaxDefaultValue("D_SSp_SpDs"));

  SetYminMemoFromValue("D_Adc_EvDs", fCnaParHistos->GetYminDefaultValue("D_Adc_EvDs"));
  SetYmaxMemoFromValue("D_Adc_EvDs", fCnaParHistos->GetYmaxDefaultValue("D_Adc_EvDs"));

  SetYminMemoFromValue("D_Adc_EvNb", fCnaParHistos->GetYminDefaultValue("D_Adc_EvNb"));
  SetYmaxMemoFromValue("D_Adc_EvNb", fCnaParHistos->GetYmaxDefaultValue("D_Adc_EvNb"));

  SetYminMemoFromValue("H_Ped_Date", fCnaParHistos->GetYminDefaultValue("H_Ped_Date"));
  SetYmaxMemoFromValue("H_Ped_Date", fCnaParHistos->GetYmaxDefaultValue("H_Ped_Date"));

  SetYminMemoFromValue("H_TNo_Date", fCnaParHistos->GetYminDefaultValue("H_TNo_Date"));
  SetYmaxMemoFromValue("H_TNo_Date", fCnaParHistos->GetYmaxDefaultValue("H_TNo_Date"));

  SetYminMemoFromValue("H_LFN_Date", fCnaParHistos->GetYminDefaultValue("H_LFN_Date"));
  SetYmaxMemoFromValue("H_LFN_Date", fCnaParHistos->GetYmaxDefaultValue("H_LFN_Date"));

  SetYminMemoFromValue("H_HFN_Date", fCnaParHistos->GetYminDefaultValue("H_HFN_Date"));
  SetYmaxMemoFromValue("H_HFN_Date", fCnaParHistos->GetYmaxDefaultValue("H_HFN_Date"));

  SetYminMemoFromValue("H_MCs_Date", fCnaParHistos->GetYminDefaultValue("H_MCs_Date"));
  SetYmaxMemoFromValue("H_MCs_Date", fCnaParHistos->GetYmaxDefaultValue("H_MCs_Date"));

  SetYminMemoFromValue("H_SCs_Date", fCnaParHistos->GetYminDefaultValue("H_SCs_Date"));
  SetYmaxMemoFromValue("H_SCs_Date", fCnaParHistos->GetYmaxDefaultValue("H_SCs_Date"));

  SetYminMemoFromValue("H_Ped_RuDs", fCnaParHistos->GetYminDefaultValue("H_Ped_RuDs"));
  SetYmaxMemoFromValue("H_Ped_RuDs", fCnaParHistos->GetYmaxDefaultValue("H_Ped_RuDs"));

  SetYminMemoFromValue("H_TNo_RuDs", fCnaParHistos->GetYminDefaultValue("H_TNo_RuDs"));
  SetYmaxMemoFromValue("H_TNo_RuDs", fCnaParHistos->GetYmaxDefaultValue("H_TNo_RuDs"));

  SetYminMemoFromValue("H_LFN_RuDs", fCnaParHistos->GetYminDefaultValue("H_LFN_RuDs"));
  SetYmaxMemoFromValue("H_LFN_RuDs", fCnaParHistos->GetYmaxDefaultValue("H_LFN_RuDs"));

  SetYminMemoFromValue("H_HFN_RuDs", fCnaParHistos->GetYminDefaultValue("H_HFN_RuDs"));
  SetYmaxMemoFromValue("H_HFN_RuDs", fCnaParHistos->GetYmaxDefaultValue("H_HFN_RuDs"));

  SetYminMemoFromValue("H_MCs_RuDs", fCnaParHistos->GetYminDefaultValue("H_MCs_RuDs"));
  SetYmaxMemoFromValue("H_MCs_RuDs", fCnaParHistos->GetYmaxDefaultValue("H_MCs_RuDs"));

  SetYminMemoFromValue("H_SCs_RuDs", fCnaParHistos->GetYminDefaultValue("H_SCs_RuDs"));
  SetYmaxMemoFromValue("H_SCs_RuDs", fCnaParHistos->GetYmaxDefaultValue("H_SCs_RuDs"));

  SetYminMemoFromValue("H2LFccMosMatrix", fCnaParHistos->GetYminDefaultValue("H2LFccMosMatrix"));
  SetYmaxMemoFromValue("H2LFccMosMatrix", fCnaParHistos->GetYmaxDefaultValue("H2LFccMosMatrix"));

  SetYminMemoFromValue("H2HFccMosMatrix", fCnaParHistos->GetYminDefaultValue("H2HFccMosMatrix"));
  SetYmaxMemoFromValue("H2HFccMosMatrix", fCnaParHistos->GetYmaxDefaultValue("H2HFccMosMatrix"));

  SetYminMemoFromValue("H2CorccInStins",  fCnaParHistos->GetYminDefaultValue("H2CorccInStins"));
  SetYmaxMemoFromValue("H2CorccInStins",  fCnaParHistos->GetYmaxDefaultValue("H2CorccInStins"));

  //........... set user's min and max flags to "OFF" and values to -1 and +1 (just to have fUserHistoMin < fUserHistoMax)
  fUserHistoMin = -1.; fFlagUserHistoMin = "OFF";
  fUserHistoMax =  1.; fFlagUserHistoMax = "OFF";
} // end of SetAllYminYmaxMemoFromDefaultValues()

//===========================================================================
//
//          SetYminMemoFromValue(...), SetYmaxMemoFromValue(...)
//          GetYminValueFromMemo(...), GetYmaxValueFromMemo(...)
//
//===========================================================================
void TEcnaHistos::SetYminMemoFromValue(const TString& HistoCode, const Double_t& value)
{
  if( HistoCode == "D_NOE_ChNb" ){fD_NOE_ChNbYmin = value;}
  if( HistoCode == "D_NOE_ChDs" ){fD_NOE_ChDsYmin = value;}  
  if( HistoCode == "D_Ped_ChNb" ){fD_Ped_ChNbYmin = value;} 
  if( HistoCode == "D_Ped_ChDs" ){fD_Ped_ChDsYmin = value;} 
  if( HistoCode == "D_TNo_ChNb" ){fD_TNo_ChNbYmin = value;}
  if( HistoCode == "D_TNo_ChDs" ){fD_TNo_ChDsYmin = value;} 
  if( HistoCode == "D_MCs_ChNb" ){fD_MCs_ChNbYmin = value;} 
  if( HistoCode == "D_MCs_ChDs" ){fD_MCs_ChDsYmin = value;} 
  if( HistoCode == "D_LFN_ChNb" ){fD_LFN_ChNbYmin = value;} 
  if( HistoCode == "D_LFN_ChDs" ){fD_LFN_ChDsYmin = value;}
  if( HistoCode == "D_HFN_ChNb" ){fD_HFN_ChNbYmin = value;} 
  if( HistoCode == "D_HFN_ChDs" ){fD_HFN_ChDsYmin = value;} 
  if( HistoCode == "D_SCs_ChNb" ){fD_SCs_ChNbYmin = value;}
  if( HistoCode == "D_SCs_ChDs" ){fD_SCs_ChDsYmin = value;}
  if( HistoCode == "D_MSp_SpNb" ){fD_Ped_ChNbYmin = value;}
  if( HistoCode == "D_MSp_SpDs" ){fD_Ped_ChDsYmin = value;}
  if( HistoCode == "D_SSp_SpNb" ){fD_TNo_ChNbYmin = value;}
  if( HistoCode == "D_SSp_SpDs" ){fD_TNo_ChDsYmin = value;}
  if( HistoCode == "D_Adc_EvNb" ){fD_Ped_ChNbYmin = value;}
  if( HistoCode == "D_Adc_EvDs" ){fD_Adc_EvDsYmin = value;}
  if( HistoCode == "H_Ped_Date" ){fH_Ped_DateYmin = value;}
  if( HistoCode == "H_TNo_Date" ){fH_TNo_DateYmin = value;}
  if( HistoCode == "H_MCs_Date" ){fH_MCs_DateYmin = value;}
  if( HistoCode == "H_LFN_Date" ){fH_LFN_DateYmin = value;}
  if( HistoCode == "H_HFN_Date" ){fH_HFN_DateYmin = value;}
  if( HistoCode == "H_SCs_Date" ){fH_SCs_DateYmin = value;}
  if( HistoCode == "H_Ped_RuDs" ){fH_Ped_RuDsYmin = value;}
  if( HistoCode == "H_TNo_RuDs" ){fH_TNo_RuDsYmin = value;}
  if( HistoCode == "H_MCs_RuDs" ){fH_MCs_RuDsYmin = value;}
  if( HistoCode == "H_LFN_RuDs" ){fH_LFN_RuDsYmin = value;}
  if( HistoCode == "H_HFN_RuDs" ){fH_HFN_RuDsYmin = value;}
  if( HistoCode == "H_SCs_RuDs" ){fH_SCs_RuDsYmin = value;}
  if( HistoCode == "H2LFccMosMatrix" ){fH2LFccMosMatrixYmin = value;}
  if( HistoCode == "H2HFccMosMatrix" ){fH2HFccMosMatrixYmin = value;}
  if( HistoCode == "H2CorccInStins"  ){fH2CorccInStinsYmin  = value;}
}// end of SetYminMemoFromValue(...)

void TEcnaHistos::SetYmaxMemoFromValue(const TString& HistoCode, const Double_t& value)
{
  if( HistoCode == "D_NOE_ChNb" ){fD_NOE_ChNbYmax = value;}
  if( HistoCode == "D_NOE_ChDs" ){fD_NOE_ChDsYmax = value;}  
  if( HistoCode == "D_Ped_ChNb" ){fD_Ped_ChNbYmax = value;} 
  if( HistoCode == "D_Ped_ChDs" ){fD_Ped_ChDsYmax = value;} 
  if( HistoCode == "D_TNo_ChNb" ){fD_TNo_ChNbYmax = value;}
  if( HistoCode == "D_TNo_ChDs" ){fD_TNo_ChDsYmax = value;} 
  if( HistoCode == "D_MCs_ChNb" ){fD_MCs_ChNbYmax = value;} 
  if( HistoCode == "D_MCs_ChDs" ){fD_MCs_ChDsYmax = value;} 
  if( HistoCode == "D_LFN_ChNb" ){fD_LFN_ChNbYmax = value;} 
  if( HistoCode == "D_LFN_ChDs" ){fD_LFN_ChDsYmax = value;}
  if( HistoCode == "D_HFN_ChNb" ){fD_HFN_ChNbYmax = value;} 
  if( HistoCode == "D_HFN_ChDs" ){fD_HFN_ChDsYmax = value;} 
  if( HistoCode == "D_SCs_ChNb" ){fD_SCs_ChNbYmax = value;}
  if( HistoCode == "D_SCs_ChDs" ){fD_SCs_ChDsYmax = value;}
  if( HistoCode == "D_MSp_SpNb" ){fD_Ped_ChNbYmax = value;}
  if( HistoCode == "D_MSp_SpDs" ){fD_Ped_ChDsYmax = value;}
  if( HistoCode == "D_SSp_SpNb" ){fD_TNo_ChNbYmax = value;}
  if( HistoCode == "D_SSp_SpDs" ){fD_TNo_ChDsYmax = value;}
  if( HistoCode == "D_Adc_EvNb" ){fD_Ped_ChNbYmax = value;}
  if( HistoCode == "D_Adc_EvDs" ){fD_Ped_ChDsYmax = value;}
  if( HistoCode == "H_Ped_Date" ){fH_Ped_DateYmax = value;}
  if( HistoCode == "H_TNo_Date" ){fH_TNo_DateYmax = value;}
  if( HistoCode == "H_MCs_Date" ){fH_MCs_DateYmax = value;}
  if( HistoCode == "H_LFN_Date" ){fH_LFN_DateYmax = value;}
  if( HistoCode == "H_HFN_Date" ){fH_HFN_DateYmax = value;}
  if( HistoCode == "H_SCs_Date" ){fH_SCs_DateYmax = value;}
  if( HistoCode == "H_Ped_RuDs" ){fH_Ped_RuDsYmax = value;}
  if( HistoCode == "H_TNo_RuDs" ){fH_TNo_RuDsYmax = value;}
  if( HistoCode == "H_MCs_RuDs" ){fH_MCs_RuDsYmax = value;}
  if( HistoCode == "H_LFN_RuDs" ){fH_LFN_RuDsYmax = value;}
  if( HistoCode == "H_HFN_RuDs" ){fH_HFN_RuDsYmax = value;}
  if( HistoCode == "H_SCs_RuDs" ){fH_SCs_RuDsYmax = value;}
  if( HistoCode == "H2LFccMosMatrix" ){fH2LFccMosMatrixYmax = value;}
  if( HistoCode == "H2HFccMosMatrix" ){fH2HFccMosMatrixYmax = value;}
  if( HistoCode == "H2CorccInStins"  ){fH2CorccInStinsYmax  = value;}
}// end of SetYmaxMemoFromValue(...)

Double_t TEcnaHistos::GetYminValueFromMemo(const TString& HistoCode)
{
  Double_t val_min      = (Double_t)0.;
  Double_t val_min_proj = (Double_t)0.1;

   if( HistoCode == "D_NOE_ChNb" ){val_min = fD_NOE_ChNbYmin;}
   if( HistoCode == "D_NOE_ChDs" ){val_min = val_min_proj;}  
   if( HistoCode == "D_Ped_ChNb" ){val_min = fD_Ped_ChNbYmin;} 
   if( HistoCode == "D_Ped_ChDs" ){val_min = val_min_proj;} 
   if( HistoCode == "D_TNo_ChNb" ){val_min = fD_TNo_ChNbYmin;}
   if( HistoCode == "D_TNo_ChDs" ){val_min = val_min_proj;} 
   if( HistoCode == "D_MCs_ChNb" ){val_min = fD_MCs_ChNbYmin;} 
   if( HistoCode == "D_MCs_ChDs" ){val_min = val_min_proj;} 
   if( HistoCode == "D_LFN_ChNb" ){val_min = fD_LFN_ChNbYmin;} 
   if( HistoCode == "D_LFN_ChDs" ){val_min = val_min_proj;}
   if( HistoCode == "D_HFN_ChNb" ){val_min = fD_HFN_ChNbYmin;} 
   if( HistoCode == "D_HFN_ChDs" ){val_min = val_min_proj;} 
   if( HistoCode == "D_SCs_ChNb" ){val_min = fD_SCs_ChNbYmin;}
   if( HistoCode == "D_SCs_ChDs" ){val_min = val_min_proj;}
   if( HistoCode == "D_MSp_SpNb" ){val_min = fD_Ped_ChNbYmin;}
   if( HistoCode == "D_MSp_SpDs" ){val_min = val_min_proj;}
   if( HistoCode == "D_SSp_SpNb" ){val_min = fD_TNo_ChNbYmin;}
   if( HistoCode == "D_SSp_SpDs" ){val_min = val_min_proj;}
   if( HistoCode == "D_Adc_EvNb" ){val_min = fD_Ped_ChNbYmin;}
   if( HistoCode == "D_Adc_EvDs" ){val_min = val_min_proj;}
   if( HistoCode == "H_Ped_Date" ){val_min = fH_Ped_DateYmin;}
   if( HistoCode == "H_TNo_Date" ){val_min = fH_TNo_DateYmin;}
   if( HistoCode == "H_MCs_Date" ){val_min = fH_MCs_DateYmin;}
   if( HistoCode == "H_LFN_Date" ){val_min = fH_LFN_DateYmin;}
   if( HistoCode == "H_HFN_Date" ){val_min = fH_HFN_DateYmin;}
   if( HistoCode == "H_SCs_Date" ){val_min = fH_SCs_DateYmin;}
   if( HistoCode == "H_Ped_RuDs" ){val_min = fH_Ped_RuDsYmin;}
   if( HistoCode == "H_TNo_RuDs" ){val_min = fH_TNo_RuDsYmin;}
   if( HistoCode == "H_MCs_RuDs" ){val_min = fH_MCs_RuDsYmin;}
   if( HistoCode == "H_LFN_RuDs" ){val_min = fH_LFN_RuDsYmin;}
   if( HistoCode == "H_HFN_RuDs" ){val_min = fH_HFN_RuDsYmin;}
   if( HistoCode == "H_SCs_RuDs" ){val_min = fH_SCs_RuDsYmin;}
   if( HistoCode == "H2LFccMosMatrix" ){val_min = fH2LFccMosMatrixYmin;}
   if( HistoCode == "H2HFccMosMatrix" ){val_min = fH2HFccMosMatrixYmin;}
   if( HistoCode == "H2CorccInStins"  ){val_min = fH2CorccInStinsYmin;}
  return val_min;
}// end of GetYminValueFromMemo(...)

Double_t TEcnaHistos::GetYmaxValueFromMemo(const TString& HistoCode)
{
  Double_t val_max      = (Double_t)0.;
  Double_t val_max_proj = (Double_t)2000.;

   if( HistoCode == "D_NOE_ChNb" ){val_max = fD_NOE_ChNbYmax;}
   if( HistoCode == "D_NOE_ChDs" ){val_max = val_max_proj;} 
   if( HistoCode == "D_Ped_ChNb" ){val_max = fD_Ped_ChNbYmax;} 
   if( HistoCode == "D_Ped_ChDs" ){val_max = val_max_proj;}  
   if( HistoCode == "D_TNo_ChNb" ){val_max = fD_TNo_ChNbYmax;}   
   if( HistoCode == "D_TNo_ChDs" ){val_max = val_max_proj;} 
   if( HistoCode == "D_MCs_ChNb" ){val_max = fD_MCs_ChNbYmax;}
   if( HistoCode == "D_MCs_ChDs" ){val_max = val_max_proj;} 
   if( HistoCode == "D_LFN_ChNb" ){val_max = fD_LFN_ChNbYmax;} 
   if( HistoCode == "D_LFN_ChDs" ){val_max = val_max_proj;} 
   if( HistoCode == "D_HFN_ChNb" ){val_max = fD_HFN_ChNbYmax;}  
   if( HistoCode == "D_HFN_ChDs" ){val_max = val_max_proj;} 
   if( HistoCode == "D_SCs_ChNb" ){val_max = fD_SCs_ChNbYmax;} 
   if( HistoCode == "D_SCs_ChDs" ){val_max = val_max_proj;}
   if( HistoCode == "D_MSp_SpNb" ){val_max = fD_Ped_ChNbYmax;}
   if( HistoCode == "D_MSp_SpDs" ){val_max = val_max_proj;}
   if( HistoCode == "D_SSp_SpNb" ){val_max = fD_TNo_ChNbYmax;}
   if( HistoCode == "D_SSp_SpDs" ){val_max = val_max_proj;}
   if( HistoCode == "D_Adc_EvNb" ){val_max = fD_Ped_ChNbYmax;}
   if( HistoCode == "D_Adc_EvDs" ){val_max = val_max_proj;}
   if( HistoCode == "H_Ped_Date" ){val_max = fH_Ped_DateYmax;}
   if( HistoCode == "H_TNo_Date" ){val_max = fH_TNo_DateYmax;}
   if( HistoCode == "H_MCs_Date" ){val_max = fH_MCs_DateYmax;}
   if( HistoCode == "H_LFN_Date" ){val_max = fH_LFN_DateYmax;}
   if( HistoCode == "H_HFN_Date" ){val_max = fH_HFN_DateYmax;}
   if( HistoCode == "H_SCs_Date" ){val_max = fH_SCs_DateYmax;}
   if( HistoCode == "H_Ped_RuDs" ){val_max = fH_Ped_RuDsYmax;}
   if( HistoCode == "H_TNo_RuDs" ){val_max = fH_TNo_RuDsYmax;}
   if( HistoCode == "H_MCs_RuDs" ){val_max = fH_MCs_RuDsYmax;}
   if( HistoCode == "H_LFN_RuDs" ){val_max = fH_LFN_RuDsYmax;}
   if( HistoCode == "H_HFN_RuDs" ){val_max = fH_HFN_RuDsYmax;}
   if( HistoCode == "H_SCs_RuDs" ){val_max = fH_SCs_RuDsYmax;}
   if( HistoCode == "H2LFccMosMatrix" ){val_max = fH2LFccMosMatrixYmax;}
   if( HistoCode == "H2HFccMosMatrix" ){val_max = fH2HFccMosMatrixYmax;}
   if( HistoCode == "H2CorccInStins"  ){val_max = fH2CorccInStinsYmax;}
  return val_max;
}// end of GetYmaxValueFromMemo(...)

void TEcnaHistos::SetYminMemoFromPreviousMemo(const TString&  HistoCode)
{
// InitQuantity Ymin

  if( HistoCode == "D_NOE_ChNb" ){fD_NOE_ChNbYmin = GetYminValueFromMemo("D_NOE_ChNb");}
  if( HistoCode == "D_NOE_ChDs" ){fD_NOE_ChDsYmin = GetYminValueFromMemo("D_NOE_ChDs");}
  if( HistoCode == "D_Ped_ChNb" ){fD_Ped_ChNbYmin = GetYminValueFromMemo("D_Ped_ChNb");}
  if( HistoCode == "D_Ped_ChDs" ){fD_Ped_ChDsYmin = GetYminValueFromMemo("D_Ped_ChDs");}
  if( HistoCode == "D_TNo_ChNb" ){fD_TNo_ChNbYmin = GetYminValueFromMemo("D_TNo_ChNb");}
  if( HistoCode == "D_TNo_ChDs" ){fD_TNo_ChDsYmin = GetYminValueFromMemo("D_TNo_ChDs");}
  if( HistoCode == "D_MCs_ChNb" ){fD_MCs_ChNbYmin = GetYminValueFromMemo("D_MCs_ChNb");}
  if( HistoCode == "D_MCs_ChDs" ){fD_MCs_ChDsYmin = GetYminValueFromMemo("D_MCs_ChDs");}
  if( HistoCode == "D_LFN_ChNb" ){fD_LFN_ChNbYmin = GetYminValueFromMemo("D_LFN_ChNb");}
  if( HistoCode == "D_LFN_ChDs" ){fD_LFN_ChDsYmin = GetYminValueFromMemo("D_LFN_ChDs");}
  if( HistoCode == "D_HFN_ChNb" ){fD_HFN_ChNbYmin = GetYminValueFromMemo("D_HFN_ChNb");}
  if( HistoCode == "D_HFN_ChDs" ){fD_HFN_ChDsYmin = GetYminValueFromMemo("D_HFN_ChDs");}
  if( HistoCode == "D_SCs_ChNb" ){fD_SCs_ChNbYmin = GetYminValueFromMemo("D_SCs_ChNb");}
  if( HistoCode == "D_SCs_ChDs" ){fD_SCs_ChDsYmin = GetYminValueFromMemo("D_SCs_ChDs");}
  if( HistoCode == "D_MSp_SpNb" ){fD_MSp_SpNbYmin = GetYminValueFromMemo("D_MSp_SpNb");}
  if( HistoCode == "D_MSp_SpDs" ){fD_MSp_SpDsYmin = GetYminValueFromMemo("D_MSp_SpDs");}
  if( HistoCode == "D_SSp_SpNb" ){fD_SSp_SpNbYmin = GetYminValueFromMemo("D_SSp_SpNb");}
  if( HistoCode == "D_SSp_SpDs" ){fD_SSp_SpDsYmin = GetYminValueFromMemo("D_SSp_SpDs");}
  if( HistoCode == "D_Adc_EvNb" ){fD_Adc_EvNbYmin = GetYminValueFromMemo("D_Adc_EvNb");}
  if( HistoCode == "D_Adc_EvDs" ){fD_Adc_EvDsYmin = GetYminValueFromMemo("D_Adc_EvDs");}
  if( HistoCode == "H_Ped_Date" ){fH_Ped_DateYmin = GetYminValueFromMemo("H_Ped_Date");}
  if( HistoCode == "H_TNo_Date" ){fH_TNo_DateYmin = GetYminValueFromMemo("H_TNo_Date");}
  if( HistoCode == "H_MCs_Date" ){fH_MCs_DateYmin = GetYminValueFromMemo("H_MCs_Date");}
  if( HistoCode == "H_LFN_Date" ){fH_LFN_DateYmin = GetYminValueFromMemo("H_LFN_Date");}
  if( HistoCode == "H_HFN_Date" ){fH_HFN_DateYmin = GetYminValueFromMemo("H_HFN_Date");}
  if( HistoCode == "H_SCs_Date" ){fH_SCs_DateYmin = GetYminValueFromMemo("H_SCs_Date");}
  if( HistoCode == "H_Ped_RuDs" ){fH_Ped_RuDsYmin = GetYminValueFromMemo("H_Ped_RuDs");}
  if( HistoCode == "H_TNo_RuDs" ){fH_TNo_RuDsYmin = GetYminValueFromMemo("H_TNo_RuDs");}
  if( HistoCode == "H_MCs_RuDs" ){fH_MCs_RuDsYmin = GetYminValueFromMemo("H_MCs_RuDs");}
  if( HistoCode == "H_LFN_RuDs" ){fH_LFN_RuDsYmin = GetYminValueFromMemo("H_LFN_RuDs");}
  if( HistoCode == "H_HFN_RuDs" ){fH_HFN_RuDsYmin = GetYminValueFromMemo("H_HFN_RuDs");}
  if( HistoCode == "H_SCs_RuDs" ){fH_SCs_RuDsYmin = GetYminValueFromMemo("H_SCs_RuDs");}
  if( HistoCode == "H2LFccMosMatrix" ){fH2LFccMosMatrixYmin = GetYminValueFromMemo("H2LFccMosMatrix");}
  if( HistoCode == "H2HFccMosMatrix" ){fH2HFccMosMatrixYmin = GetYminValueFromMemo("H2HFccMosMatrix");}
  if( HistoCode == "H2CorccInStins"  ){fH2CorccInStinsYmin  = GetYminValueFromMemo("H2CorccInStins");}
}// end of SetYminMemoFromPreviousMemo(...)

void TEcnaHistos::SetYmaxMemoFromPreviousMemo(const TString&  HistoCode)
{
// InitQuantity Ymax

  if( HistoCode == "D_NOE_ChNb" ){fD_NOE_ChNbYmax = GetYmaxValueFromMemo("D_NOE_ChNb");}
  if( HistoCode == "D_NOE_ChDs" ){fD_NOE_ChDsYmax = GetYmaxValueFromMemo("D_NOE_ChDs");}
  if( HistoCode == "D_Ped_ChNb" ){fD_Ped_ChNbYmax = GetYmaxValueFromMemo("D_Ped_ChNb");}
  if( HistoCode == "D_Ped_ChDs" ){fD_Ped_ChDsYmax = GetYmaxValueFromMemo("D_Ped_ChDs");}
  if( HistoCode == "D_TNo_ChNb" ){fD_TNo_ChNbYmax = GetYmaxValueFromMemo("D_TNo_ChNb");}
  if( HistoCode == "D_TNo_ChDs" ){fD_TNo_ChDsYmax = GetYmaxValueFromMemo("D_TNo_ChDs");}
  if( HistoCode == "D_MCs_ChNb" ){fD_MCs_ChNbYmax = GetYmaxValueFromMemo("D_MCs_ChNb");}
  if( HistoCode == "D_MCs_ChDs" ){fD_MCs_ChDsYmax = GetYmaxValueFromMemo("D_MCs_ChDs");}
  if( HistoCode == "D_LFN_ChNb" ){fD_LFN_ChNbYmax = GetYmaxValueFromMemo("D_LFN_ChNb");}
  if( HistoCode == "D_LFN_ChDs" ){fD_LFN_ChDsYmax = GetYmaxValueFromMemo("D_LFN_ChDs");}
  if( HistoCode == "D_HFN_ChNb" ){fD_HFN_ChNbYmax = GetYmaxValueFromMemo("D_HFN_ChNb");}
  if( HistoCode == "D_HFN_ChDs" ){fD_HFN_ChDsYmax = GetYmaxValueFromMemo("D_HFN_ChDs");}
  if( HistoCode == "D_SCs_ChNb" ){fD_SCs_ChNbYmax = GetYmaxValueFromMemo("D_SCs_ChNb");}
  if( HistoCode == "D_SCs_ChDs" ){fD_SCs_ChDsYmax = GetYmaxValueFromMemo("D_SCs_ChDs");}
  if( HistoCode == "D_MSp_SpNb" ){fD_MSp_SpNbYmax = GetYmaxValueFromMemo("D_MSp_SpNb");}
  if( HistoCode == "D_MSp_SpDs" ){fD_MSp_SpDsYmax = GetYmaxValueFromMemo("D_MSp_SpDs");}
  if( HistoCode == "D_SSp_SpNb" ){fD_SSp_SpNbYmax = GetYmaxValueFromMemo("D_SSp_SpNb");}
  if( HistoCode == "D_SSp_SpDs" ){fD_SSp_SpDsYmax = GetYmaxValueFromMemo("D_SSp_SpDs");}
  if( HistoCode == "D_Adc_EvNb" ){fD_Adc_EvNbYmax = GetYmaxValueFromMemo("D_Adc_EvNb");}
  if( HistoCode == "D_Adc_EvDs" ){fD_Adc_EvDsYmax = GetYmaxValueFromMemo("D_Adc_EvDs");}
  if( HistoCode == "H_Ped_Date" ){fH_Ped_DateYmax = GetYmaxValueFromMemo("H_Ped_Date");}
  if( HistoCode == "H_TNo_Date" ){fH_TNo_DateYmax = GetYmaxValueFromMemo("H_TNo_Date");}
  if( HistoCode == "H_MCs_Date" ){fH_MCs_DateYmax = GetYmaxValueFromMemo("H_MCs_Date");}
  if( HistoCode == "H_LFN_Date" ){fH_LFN_DateYmax = GetYmaxValueFromMemo("H_LFN_Date");}
  if( HistoCode == "H_HFN_Date" ){fH_HFN_DateYmax = GetYmaxValueFromMemo("H_HFN_Date");}
  if( HistoCode == "H_SCs_Date" ){fH_SCs_DateYmax = GetYmaxValueFromMemo("H_SCs_Date");}
  if( HistoCode == "H_Ped_RuDs" ){fH_Ped_RuDsYmax = GetYmaxValueFromMemo("H_Ped_RuDs");}
  if( HistoCode == "H_TNo_RuDs" ){fH_TNo_RuDsYmax = GetYmaxValueFromMemo("H_TNo_RuDs");}
  if( HistoCode == "H_MCs_RuDs" ){fH_MCs_RuDsYmax = GetYmaxValueFromMemo("H_MCs_RuDs");}
  if( HistoCode == "H_LFN_RuDs" ){fH_LFN_RuDsYmax = GetYmaxValueFromMemo("H_LFN_RuDs");}
  if( HistoCode == "H_HFN_RuDs" ){fH_HFN_RuDsYmax = GetYmaxValueFromMemo("H_HFN_RuDs");}
  if( HistoCode == "H_SCs_RuDs" ){fH_SCs_RuDsYmax = GetYmaxValueFromMemo("H_SCs_RuDs");}
  if( HistoCode == "H2LFccMosMatrix" ){fH2LFccMosMatrixYmax = GetYmaxValueFromMemo("H2LFccMosMatrix");}
  if( HistoCode == "H2HFccMosMatrix" ){fH2HFccMosMatrixYmax = GetYmaxValueFromMemo("H2HFccMosMatrix");}
  if( HistoCode == "H2CorccInStins"  ){fH2CorccInStinsYmax  = GetYmaxValueFromMemo("H2CorccInStins");}
}// end of SetYmaxMemoFromPreviousMemo(...)

//------------------------------------------------------------------------------------------------------
void TEcnaHistos::SetXVarMemo(const TString& HistoCode, const TString& opt_plot, const TString& xvar)
{

  if( opt_plot == fSameOnePlot ){fXMemoH1SamePlus = xvar;}

  if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot )
    {
      if( HistoCode == "D_NOE_ChNb"){fXMemoD_NOE_ChNb = xvar;}
      if( HistoCode == "D_NOE_ChDs"){fXMemoD_NOE_ChDs = xvar;}  
      if( HistoCode == "D_Ped_ChNb"){fXMemoD_Ped_ChNb = xvar;} 
      if( HistoCode == "D_Ped_ChDs"){fXMemoD_Ped_ChDs = xvar;} 
      if( HistoCode == "D_TNo_ChNb"){fXMemoD_TNo_ChNb = xvar;}
      if( HistoCode == "D_TNo_ChDs"){fXMemoD_TNo_ChDs = xvar;} 
      if( HistoCode == "D_MCs_ChNb"){fXMemoD_MCs_ChNb = xvar;} 
      if( HistoCode == "D_MCs_ChDs"){fXMemoD_MCs_ChDs = xvar;} 
      if( HistoCode == "D_LFN_ChNb"){fXMemoD_LFN_ChNb = xvar;} 
      if( HistoCode == "D_LFN_ChDs"){fXMemoD_LFN_ChDs = xvar;}
      if( HistoCode == "D_HFN_ChNb"){fXMemoD_HFN_ChNb = xvar;} 
      if( HistoCode == "D_HFN_ChDs"){fXMemoD_HFN_ChDs = xvar;} 
      if( HistoCode == "D_SCs_ChNb"){fXMemoD_SCs_ChNb = xvar;}
      if( HistoCode == "D_SCs_ChDs"){fXMemoD_SCs_ChDs = xvar;}
      if( HistoCode == "D_MSp_SpNb"){fXMemoD_MSp_SpNb = xvar;}
      if( HistoCode == "D_MSp_SpDs"){fXMemoD_MSp_SpDs = xvar;}
      if( HistoCode == "D_SSp_SpNb"){fXMemoD_SSp_SpNb = xvar;}
      if( HistoCode == "D_SSp_SpDs"){fXMemoD_SSp_SpDs = xvar;}
      if( HistoCode == "D_Adc_EvNb"){fXMemoD_Adc_EvNb = xvar;}
      if( HistoCode == "D_Adc_EvDs"){fXMemoD_Adc_EvDs = xvar;}
      if( HistoCode == "H_Ped_Date"){fXMemoH_Ped_Date = xvar;}
      if( HistoCode == "H_TNo_Date"){fXMemoH_TNo_Date = xvar;}
      if( HistoCode == "H_MCs_Date"){fXMemoH_MCs_Date = xvar;}
      if( HistoCode == "H_LFN_Date"){fXMemoH_LFN_Date = xvar;}
      if( HistoCode == "H_HFN_Date"){fXMemoH_HFN_Date = xvar;}
      if( HistoCode == "H_SCs_Date"){fXMemoH_SCs_Date = xvar;}
      if( HistoCode == "H_Ped_RuDs"){fXMemoH_Ped_RuDs = xvar;}
      if( HistoCode == "H_TNo_RuDs"){fXMemoH_TNo_RuDs = xvar;}
      if( HistoCode == "H_MCs_RuDs"){fXMemoH_MCs_RuDs = xvar;}
      if( HistoCode == "H_LFN_RuDs"){fXMemoH_LFN_RuDs = xvar;}
      if( HistoCode == "H_HFN_RuDs"){fXMemoH_HFN_RuDs = xvar;}
      if( HistoCode == "H_SCs_RuDs"){fXMemoH_SCs_RuDs = xvar;}
    }
}// end of SetXVarMemo(...)

TString TEcnaHistos::GetXVarFromMemo(const TString& HistoCode, const TString& opt_plot)
{
  TString xvar = "(xvar not found)";
  
  if( opt_plot == fSameOnePlot ){xvar = fXMemoH1SamePlus;}

  if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot )
    {
      if( HistoCode == "D_NOE_ChNb"){xvar = fXMemoD_NOE_ChNb;}
      if( HistoCode == "D_NOE_ChDs"){xvar = fXMemoD_NOE_ChDs;}  
      if( HistoCode == "D_Ped_ChNb"){xvar = fXMemoD_Ped_ChNb;} 
      if( HistoCode == "D_Ped_ChDs"){xvar = fXMemoD_Ped_ChDs;} 
      if( HistoCode == "D_TNo_ChNb"){xvar = fXMemoD_TNo_ChNb;}
      if( HistoCode == "D_TNo_ChDs"){xvar = fXMemoD_TNo_ChDs;} 
      if( HistoCode == "D_MCs_ChNb"){xvar = fXMemoD_MCs_ChNb;} 
      if( HistoCode == "D_MCs_ChDs"){xvar = fXMemoD_MCs_ChDs;} 
      if( HistoCode == "D_LFN_ChNb"){xvar = fXMemoD_LFN_ChNb;} 
      if( HistoCode == "D_LFN_ChDs"){xvar = fXMemoD_LFN_ChDs;}
      if( HistoCode == "D_HFN_ChNb"){xvar = fXMemoD_HFN_ChNb;} 
      if( HistoCode == "D_HFN_ChDs"){xvar = fXMemoD_HFN_ChDs;} 
      if( HistoCode == "D_SCs_ChNb"){xvar = fXMemoD_SCs_ChNb;}
      if( HistoCode == "D_SCs_ChDs"){xvar = fXMemoD_SCs_ChDs;}
      if( HistoCode == "D_MSp_SpNb"){xvar = fXMemoD_MSp_SpNb;}
      if( HistoCode == "D_MSp_SpDs"){xvar = fXMemoD_MSp_SpDs;}
      if( HistoCode == "D_SSp_SpNb"){xvar = fXMemoD_SSp_SpNb;}
      if( HistoCode == "D_SSp_SpDs"){xvar = fXMemoD_SSp_SpDs;}
      if( HistoCode == "D_Adc_EvNb"){xvar = fXMemoD_Adc_EvNb;}
      if( HistoCode == "D_Adc_EvDs"){xvar = fXMemoD_Adc_EvDs;}
      if( HistoCode == "H_Ped_Date"){xvar = fXMemoH_Ped_Date;}
      if( HistoCode == "H_TNo_Date"){xvar = fXMemoH_TNo_Date;}
      if( HistoCode == "H_MCs_Date"){xvar = fXMemoH_MCs_Date;}
      if( HistoCode == "H_LFN_Date"){xvar = fXMemoH_LFN_Date;}
      if( HistoCode == "H_HFN_Date"){xvar = fXMemoH_HFN_Date;}
      if( HistoCode == "H_SCs_Date"){xvar = fXMemoH_SCs_Date;}
      if( HistoCode == "H_Ped_RuDs"){xvar = fXMemoH_Ped_RuDs;}
      if( HistoCode == "H_TNo_RuDs"){xvar = fXMemoH_TNo_RuDs;}
      if( HistoCode == "H_MCs_RuDs"){xvar = fXMemoH_MCs_RuDs;}
      if( HistoCode == "H_LFN_RuDs"){xvar = fXMemoH_LFN_RuDs;}
      if( HistoCode == "H_HFN_RuDs"){xvar = fXMemoH_HFN_RuDs;}
      if( HistoCode == "H_SCs_RuDs"){xvar = fXMemoH_SCs_RuDs;}
    }
  return xvar;
}// end of GetXVarFromMemo(...)


void TEcnaHistos::SetYVarMemo(const TString& HistoCode, const TString& opt_plot, const TString& yvar)
{
  if( opt_plot == fSameOnePlot ){fYMemoH1SamePlus = yvar;}

  if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot )
    {
      if( HistoCode == "D_NOE_ChNb"){fYMemoD_NOE_ChNb = yvar;}
      if( HistoCode == "D_NOE_ChDs"){fYMemoD_NOE_ChDs = yvar;}  
      if( HistoCode == "D_Ped_ChNb"){fYMemoD_Ped_ChNb = yvar;} 
      if( HistoCode == "D_Ped_ChDs"){fYMemoD_Ped_ChDs = yvar;} 
      if( HistoCode == "D_TNo_ChNb"){fYMemoD_TNo_ChNb = yvar;}
      if( HistoCode == "D_TNo_ChDs"){fYMemoD_TNo_ChDs = yvar;} 
      if( HistoCode == "D_MCs_ChNb"){fYMemoD_MCs_ChNb = yvar;} 
      if( HistoCode == "D_MCs_ChDs"){fYMemoD_MCs_ChDs = yvar;} 
      if( HistoCode == "D_LFN_ChNb"){fYMemoD_LFN_ChNb = yvar;} 
      if( HistoCode == "D_LFN_ChDs"){fYMemoD_LFN_ChDs = yvar;}
      if( HistoCode == "D_HFN_ChNb"){fYMemoD_HFN_ChNb = yvar;} 
      if( HistoCode == "D_HFN_ChDs"){fYMemoD_HFN_ChDs = yvar;} 
      if( HistoCode == "D_SCs_ChNb"){fYMemoD_SCs_ChNb = yvar;}
      if( HistoCode == "D_SCs_ChDs"){fYMemoD_SCs_ChDs = yvar;}
      if( HistoCode == "D_MSp_SpNb"){fYMemoD_MSp_SpNb = yvar;}
      if( HistoCode == "D_MSp_SpDs"){fYMemoD_MSp_SpDs = yvar;}
      if( HistoCode == "D_SSp_SpNb"){fYMemoD_SSp_SpNb = yvar;}
      if( HistoCode == "D_Adc_EvDs"){fYMemoD_Adc_EvDs = yvar;}
      if( HistoCode == "D_SSp_SpDs"){fYMemoD_SSp_SpDs = yvar;}
      if( HistoCode == "D_Adc_EvNb"){fYMemoD_Adc_EvNb = yvar;}
      if( HistoCode == "H_Ped_Date"){fYMemoH_Ped_Date = yvar;}
      if( HistoCode == "H_TNo_Date"){fYMemoH_TNo_Date = yvar;}
      if( HistoCode == "H_MCs_Date"){fYMemoH_MCs_Date = yvar;}
      if( HistoCode == "H_LFN_Date"){fYMemoH_LFN_Date = yvar;}
      if( HistoCode == "H_HFN_Date"){fYMemoH_HFN_Date = yvar;}
      if( HistoCode == "H_SCs_Date"){fYMemoH_SCs_Date = yvar;}
      if( HistoCode == "H_Ped_RuDs"){fYMemoH_Ped_RuDs = yvar;}
      if( HistoCode == "H_TNo_RuDs"){fYMemoH_TNo_RuDs = yvar;}
      if( HistoCode == "H_MCs_RuDs"){fYMemoH_MCs_RuDs = yvar;}
      if( HistoCode == "H_LFN_RuDs"){fYMemoH_LFN_RuDs = yvar;}
      if( HistoCode == "H_HFN_RuDs"){fYMemoH_HFN_RuDs = yvar;}
      if( HistoCode == "H_SCs_RuDs"){fYMemoH_SCs_RuDs = yvar;}
    }
}// end of SetYVarMemo(...)

TString TEcnaHistos::GetYVarFromMemo(const TString& HistoCode, const TString& opt_plot)
{
  TString yvar = "(yvar not found)";

  if( opt_plot == fSameOnePlot ){yvar = fYMemoH1SamePlus;}

  if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot )
    {  
      if( HistoCode == "D_NOE_ChNb"){yvar = fYMemoD_NOE_ChNb;}
      if( HistoCode == "D_NOE_ChDs"){yvar = fYMemoD_NOE_ChDs;}
      if( HistoCode == "D_Ped_ChNb"){yvar = fYMemoD_Ped_ChNb;}
      if( HistoCode == "D_Ped_ChDs"){yvar = fYMemoD_Ped_ChDs;}
      if( HistoCode == "D_TNo_ChNb"){yvar = fYMemoD_TNo_ChNb;}
      if( HistoCode == "D_TNo_ChDs"){yvar = fYMemoD_TNo_ChDs;}
      if( HistoCode == "D_MCs_ChNb"){yvar = fYMemoD_MCs_ChNb;}
      if( HistoCode == "D_MCs_ChDs"){yvar = fYMemoD_MCs_ChDs;}
      if( HistoCode == "D_LFN_ChNb"){yvar = fYMemoD_LFN_ChNb;}
      if( HistoCode == "D_LFN_ChDs"){yvar = fYMemoD_LFN_ChDs;}
      if( HistoCode == "D_HFN_ChNb"){yvar = fYMemoD_HFN_ChNb;}
      if( HistoCode == "D_HFN_ChDs"){yvar = fYMemoD_HFN_ChDs;}
      if( HistoCode == "D_SCs_ChNb"){yvar = fYMemoD_SCs_ChNb;}
      if( HistoCode == "D_SCs_ChDs"){yvar = fYMemoD_SCs_ChDs;}
      if( HistoCode == "D_MSp_SpNb"){yvar = fYMemoD_MSp_SpNb;}
      if( HistoCode == "D_MSp_SpDs"){yvar = fYMemoD_MSp_SpDs;}
      if( HistoCode == "D_SSp_SpNb"){yvar = fYMemoD_SSp_SpNb;}
      if( HistoCode == "D_SSp_SpDs"){yvar = fYMemoD_SSp_SpDs;}
      if( HistoCode == "D_Adc_EvNb"){yvar = fYMemoD_Adc_EvNb;}
      if( HistoCode == "D_Adc_EvDs"){yvar = fYMemoD_Adc_EvDs;}
      if( HistoCode == "H_Ped_Date"){yvar = fYMemoH_Ped_Date;}
      if( HistoCode == "H_TNo_Date"){yvar = fYMemoH_TNo_Date;}
      if( HistoCode == "H_MCs_Date"){yvar = fYMemoH_MCs_Date;}
      if( HistoCode == "H_LFN_Date"){yvar = fYMemoH_LFN_Date;}
      if( HistoCode == "H_HFN_Date"){yvar = fYMemoH_HFN_Date;}
      if( HistoCode == "H_SCs_Date"){yvar = fYMemoH_SCs_Date;}
      if( HistoCode == "H_Ped_RuDs"){yvar = fYMemoH_Ped_RuDs;}
      if( HistoCode == "H_TNo_RuDs"){yvar = fYMemoH_TNo_RuDs;}
      if( HistoCode == "H_MCs_RuDs"){yvar = fYMemoH_MCs_RuDs;}
      if( HistoCode == "H_LFN_RuDs"){yvar = fYMemoH_LFN_RuDs;}
      if( HistoCode == "H_HFN_RuDs"){yvar = fYMemoH_HFN_RuDs;}
      if( HistoCode == "H_SCs_RuDs"){yvar = fYMemoH_SCs_RuDs;}
    }
  return yvar;
}// end of GetYVarFromMemo(...)

void TEcnaHistos::SetNbBinsMemo(const TString& HistoCode, const TString& opt_plot, const Int_t& nb_bins)
{

  if( opt_plot == fSameOnePlot ){fNbBinsMemoH1SamePlus = nb_bins;}

  if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot )
    {
      if( HistoCode == "D_NOE_ChNb"){fNbBinsMemoD_NOE_ChNb = nb_bins;}
      if( HistoCode == "D_NOE_ChDs"){fNbBinsMemoD_NOE_ChDs = nb_bins;}  
      if( HistoCode == "D_Ped_ChNb"){fNbBinsMemoD_Ped_ChNb = nb_bins;} 
      if( HistoCode == "D_Ped_ChDs"){fNbBinsMemoD_Ped_ChDs = nb_bins;} 
      if( HistoCode == "D_TNo_ChNb"){fNbBinsMemoD_TNo_ChNb = nb_bins;}
      if( HistoCode == "D_TNo_ChDs"){fNbBinsMemoD_TNo_ChDs = nb_bins;} 
      if( HistoCode == "D_MCs_ChNb"){fNbBinsMemoD_MCs_ChNb = nb_bins;} 
      if( HistoCode == "D_MCs_ChDs"){fNbBinsMemoD_MCs_ChDs = nb_bins;} 
      if( HistoCode == "D_LFN_ChNb"){fNbBinsMemoD_LFN_ChNb = nb_bins;} 
      if( HistoCode == "D_LFN_ChDs"){fNbBinsMemoD_LFN_ChDs = nb_bins;}
      if( HistoCode == "D_HFN_ChNb"){fNbBinsMemoD_HFN_ChNb = nb_bins;} 
      if( HistoCode == "D_HFN_ChDs"){fNbBinsMemoD_HFN_ChDs = nb_bins;} 
      if( HistoCode == "D_SCs_ChNb"){fNbBinsMemoD_SCs_ChNb = nb_bins;}
      if( HistoCode == "D_SCs_ChDs"){fNbBinsMemoD_SCs_ChDs = nb_bins;}
      if( HistoCode == "D_MSp_SpNb"){fNbBinsMemoD_MSp_SpNb = nb_bins;}
      if( HistoCode == "D_MSp_SpDs"){fNbBinsMemoD_MSp_SpDs = nb_bins;}
      if( HistoCode == "D_SSp_SpNb"){fNbBinsMemoD_SSp_SpNb = nb_bins;}
      if( HistoCode == "D_SSp_SpDs"){fNbBinsMemoD_SSp_SpDs = nb_bins;}
      if( HistoCode == "D_Adc_EvNb"){fNbBinsMemoD_Adc_EvNb = nb_bins;}
      if( HistoCode == "D_Adc_EvDs"){fNbBinsMemoD_Adc_EvDs = nb_bins;}
      if( HistoCode == "H_Ped_Date"){fNbBinsMemoH_Ped_Date = nb_bins;}
      if( HistoCode == "H_TNo_Date"){fNbBinsMemoH_TNo_Date = nb_bins;}
      if( HistoCode == "H_MCs_Date"){fNbBinsMemoH_MCs_Date = nb_bins;}
      if( HistoCode == "H_LFN_Date"){fNbBinsMemoH_LFN_Date = nb_bins;}
      if( HistoCode == "H_HFN_Date"){fNbBinsMemoH_HFN_Date = nb_bins;}
      if( HistoCode == "H_SCs_Date"){fNbBinsMemoH_SCs_Date = nb_bins;}
      if( HistoCode == "H_Ped_RuDs"){fNbBinsMemoH_Ped_RuDs = nb_bins;}
      if( HistoCode == "H_TNo_RuDs"){fNbBinsMemoH_TNo_RuDs = nb_bins;}
      if( HistoCode == "H_MCs_RuDs"){fNbBinsMemoH_MCs_RuDs = nb_bins;}
      if( HistoCode == "H_LFN_RuDs"){fNbBinsMemoH_LFN_RuDs = nb_bins;}
      if( HistoCode == "H_HFN_RuDs"){fNbBinsMemoH_HFN_RuDs = nb_bins;}
      if( HistoCode == "H_SCs_RuDs"){fNbBinsMemoH_SCs_RuDs = nb_bins;}
    }
}// end of SetNbBinsMemo(...)

Int_t TEcnaHistos::GetNbBinsFromMemo(const TString& HistoCode, const TString& opt_plot)
{
  Int_t nb_bins = 0;

  if( opt_plot == fSameOnePlot ){nb_bins = fNbBinsMemoH1SamePlus;}

  if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot )
    {
      if( HistoCode == "D_NOE_ChNb"){nb_bins = fNbBinsMemoD_NOE_ChNb;}
      if( HistoCode == "D_NOE_ChDs"){nb_bins = fNbBinsMemoD_NOE_ChDs;}  
      if( HistoCode == "D_Ped_ChNb"){nb_bins = fNbBinsMemoD_Ped_ChNb;} 
      if( HistoCode == "D_Ped_ChDs"){nb_bins = fNbBinsMemoD_Ped_ChDs;} 
      if( HistoCode == "D_TNo_ChNb"){nb_bins = fNbBinsMemoD_TNo_ChNb;}
      if( HistoCode == "D_TNo_ChDs"){nb_bins = fNbBinsMemoD_TNo_ChDs;} 
      if( HistoCode == "D_MCs_ChNb"){nb_bins = fNbBinsMemoD_MCs_ChNb;} 
      if( HistoCode == "D_MCs_ChDs"){nb_bins = fNbBinsMemoD_MCs_ChDs;} 
      if( HistoCode == "D_LFN_ChNb"){nb_bins = fNbBinsMemoD_LFN_ChNb;} 
      if( HistoCode == "D_LFN_ChDs"){nb_bins = fNbBinsMemoD_LFN_ChDs;}
      if( HistoCode == "D_HFN_ChNb"){nb_bins = fNbBinsMemoD_HFN_ChNb;} 
      if( HistoCode == "D_HFN_ChDs"){nb_bins = fNbBinsMemoD_HFN_ChDs;} 
      if( HistoCode == "D_SCs_ChNb"){nb_bins = fNbBinsMemoD_SCs_ChNb;}
      if( HistoCode == "D_SCs_ChDs"){nb_bins = fNbBinsMemoD_SCs_ChDs;}
      if( HistoCode == "D_MSp_SpNb"){nb_bins = fNbBinsMemoD_MSp_SpNb;}
      if( HistoCode == "D_MSp_SpDs"){nb_bins = fNbBinsMemoD_MSp_SpDs;}
      if( HistoCode == "D_SSp_SpNb"){nb_bins = fNbBinsMemoD_SSp_SpNb;}
      if( HistoCode == "D_SSp_SpDs"){nb_bins = fNbBinsMemoD_SSp_SpDs;}
      if( HistoCode == "D_Adc_EvNb"){nb_bins = fNbBinsMemoD_Adc_EvNb;}
      if( HistoCode == "D_Adc_EvDs"){nb_bins = fNbBinsMemoD_Adc_EvDs;}
      if( HistoCode == "H_Ped_Date"){nb_bins = fNbBinsMemoH_Ped_Date;}
      if( HistoCode == "H_TNo_Date"){nb_bins = fNbBinsMemoH_TNo_Date;}
      if( HistoCode == "H_MCs_Date"){nb_bins = fNbBinsMemoH_MCs_Date;}
      if( HistoCode == "H_LFN_Date"){nb_bins = fNbBinsMemoH_LFN_Date;}
      if( HistoCode == "H_HFN_Date"){nb_bins = fNbBinsMemoH_HFN_Date;}
      if( HistoCode == "H_SCs_Date"){nb_bins = fNbBinsMemoH_SCs_Date;}
      if( HistoCode == "H_Ped_RuDs"){nb_bins = fNbBinsMemoH_Ped_RuDs;}
      if( HistoCode == "H_TNo_RuDs"){nb_bins = fNbBinsMemoH_TNo_RuDs;}
      if( HistoCode == "H_MCs_RuDs"){nb_bins = fNbBinsMemoH_MCs_RuDs;}
      if( HistoCode == "H_LFN_RuDs"){nb_bins = fNbBinsMemoH_LFN_RuDs;}
      if( HistoCode == "H_HFN_RuDs"){nb_bins = fNbBinsMemoH_HFN_RuDs;}
      if( HistoCode == "H_SCs_RuDs"){nb_bins = fNbBinsMemoH_SCs_RuDs;}
    }
  return nb_bins;
}// end of GetNbBinsFromMemo(...)

TString TEcnaHistos::GetMemoFlag(const TString& opt_plot)
{
  TString memo_flag;
  Int_t MaxCar = fgMaxCar;
  memo_flag.Resize(MaxCar);
  memo_flag = "(no memo_flag info)";

  Int_t memo_flag_number = -1;

  if( opt_plot == fSameOnePlot ){memo_flag_number = fMemoPlotH1SamePlus;}

  if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot )
    {
      memo_flag_number = fMemoPlotD_TNo_ChDs+fMemoPlotD_MCs_ChDs
	+fMemoPlotD_LFN_ChDs+fMemoPlotD_HFN_ChDs+fMemoPlotD_SCs_ChDs;
    }

  if(memo_flag_number == 0){memo_flag = "Free";}
  if(memo_flag_number >= 1){memo_flag = "Busy";}

  return memo_flag;
}

TString TEcnaHistos::GetMemoFlag(const TString& HistoCode, const TString& opt_plot)
{
// Get Memo Flag

  TString memo_flag;
  Int_t MaxCar = fgMaxCar;
  memo_flag.Resize(MaxCar);
  memo_flag = "(no memo_flag info)";

  Int_t memo_flag_number = -1;

  if( opt_plot == fSameOnePlot ){memo_flag_number = fMemoPlotH1SamePlus;}

  if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot )
    {
      if(HistoCode == "D_NOE_ChNb"){memo_flag_number = fMemoPlotD_NOE_ChNb;}
      if(HistoCode == "D_NOE_ChDs"){memo_flag_number = fMemoPlotD_NOE_ChDs;}
      if(HistoCode == "D_Ped_ChNb"){memo_flag_number = fMemoPlotD_Ped_ChNb;}
      if(HistoCode == "D_Ped_ChDs"){memo_flag_number = fMemoPlotD_Ped_ChDs;}
      if(HistoCode == "D_TNo_ChNb"){memo_flag_number = fMemoPlotD_TNo_ChNb;}
      if(HistoCode == "D_TNo_ChDs"){memo_flag_number = fMemoPlotD_TNo_ChDs;}
      if(HistoCode == "D_MCs_ChNb"){memo_flag_number = fMemoPlotD_MCs_ChNb;}
      if(HistoCode == "D_MCs_ChDs"){memo_flag_number = fMemoPlotD_MCs_ChDs;}
      if(HistoCode == "D_LFN_ChNb"){memo_flag_number = fMemoPlotD_LFN_ChNb;}
      if(HistoCode == "D_LFN_ChDs"){memo_flag_number = fMemoPlotD_LFN_ChDs;} 
      if(HistoCode == "D_HFN_ChNb"){memo_flag_number = fMemoPlotD_HFN_ChNb;} 
      if(HistoCode == "D_HFN_ChDs"){memo_flag_number = fMemoPlotD_HFN_ChDs;}
      if(HistoCode == "D_SCs_ChNb"){memo_flag_number = fMemoPlotD_SCs_ChNb;}
      if(HistoCode == "D_SCs_ChDs"){memo_flag_number = fMemoPlotD_SCs_ChDs;}
      if(HistoCode == "D_MSp_SpNb"){memo_flag_number = fMemoPlotD_MSp_SpNb;}
      if(HistoCode == "D_MSp_SpDs"){memo_flag_number = fMemoPlotD_MSp_SpDs;}
      if(HistoCode == "D_SSp_SpNb"){memo_flag_number = fMemoPlotD_SSp_SpNb;}
      if(HistoCode == "D_SSp_SpDs"){memo_flag_number = fMemoPlotD_SSp_SpDs;}
      if(HistoCode == "D_Adc_EvNb"){memo_flag_number = fMemoPlotD_Adc_EvNb;}
      if(HistoCode == "D_Adc_EvDs"){memo_flag_number = fMemoPlotD_Adc_EvDs;}
      if(HistoCode == "H_Ped_Date"){memo_flag_number = fMemoPlotH_Ped_Date;}
      if(HistoCode == "H_TNo_Date"){memo_flag_number = fMemoPlotH_TNo_Date;}
      if(HistoCode == "H_MCs_Date"){memo_flag_number = fMemoPlotH_MCs_Date;}
      if(HistoCode == "H_LFN_Date"){memo_flag_number = fMemoPlotH_LFN_Date;}
      if(HistoCode == "H_HFN_Date"){memo_flag_number = fMemoPlotH_HFN_Date;}
      if(HistoCode == "H_SCs_Date"){memo_flag_number = fMemoPlotH_SCs_Date;}
      if(HistoCode == "H_Ped_RuDs"){memo_flag_number = fMemoPlotH_Ped_RuDs;}
      if(HistoCode == "H_TNo_RuDs"){memo_flag_number = fMemoPlotH_TNo_RuDs;}
      if(HistoCode == "H_MCs_RuDs"){memo_flag_number = fMemoPlotH_MCs_RuDs;}
      if(HistoCode == "H_LFN_RuDs"){memo_flag_number = fMemoPlotH_LFN_RuDs;}
      if(HistoCode == "H_HFN_RuDs"){memo_flag_number = fMemoPlotH_HFN_RuDs;}
      if(HistoCode == "H_SCs_RuDs"){memo_flag_number = fMemoPlotH_SCs_RuDs;}
    }

  if(memo_flag_number == 0){memo_flag = "Free";}
  if(memo_flag_number == 1){memo_flag = "Busy";}

  return memo_flag;
}

TCanvas* TEcnaHistos::CreateCanvas(const TString& HistoCode, const TString& opt_plot, const TString& canvas_name,
				   UInt_t canv_w, UInt_t canv_h)
{
// Create canvas according to HistoCode

  TCanvas* main_canvas = 0;
 
  if( opt_plot == fSameOnePlot )
    {
      fCanvH1SamePlus = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
      main_canvas = fCanvH1SamePlus;
    }
  if( opt_plot == fSeveralPlot || opt_plot == fOnlyOnePlot )
    {
      if(HistoCode == "D_NOE_ChNb"){
	fCanvD_NOE_ChNb = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_NOE_ChNb;}
      if(HistoCode == "D_NOE_ChDs"){
	fCanvD_NOE_ChDs = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_NOE_ChDs;}
      if(HistoCode == "D_Ped_ChNb"){
	fCanvD_Ped_ChNb = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_Ped_ChNb;}
      if(HistoCode == "D_Ped_ChDs"){
	fCanvD_Ped_ChDs = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_Ped_ChDs;}
      if(HistoCode == "D_TNo_ChNb"){
	fCanvD_TNo_ChNb = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_TNo_ChNb;}
      if(HistoCode == "D_TNo_ChDs"){
	fCanvD_TNo_ChDs = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_TNo_ChDs;}
      if(HistoCode == "D_MCs_ChNb"){
	fCanvD_MCs_ChNb = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_MCs_ChNb;}
      if(HistoCode == "D_MCs_ChDs"){
	fCanvD_MCs_ChDs = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_MCs_ChDs;}
      if(HistoCode == "D_LFN_ChNb"){
	fCanvD_LFN_ChNb = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_LFN_ChNb;}
      if(HistoCode == "D_LFN_ChDs"){
	fCanvD_LFN_ChDs = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_LFN_ChDs;}
      if(HistoCode == "D_HFN_ChNb"){
	fCanvD_HFN_ChNb = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_HFN_ChNb;}
      if(HistoCode == "D_HFN_ChDs"){
	fCanvD_HFN_ChDs = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_HFN_ChDs;}
      if(HistoCode == "D_SCs_ChNb"){
	fCanvD_SCs_ChNb = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_SCs_ChNb;}
      if(HistoCode == "D_SCs_ChDs"){
	fCanvD_SCs_ChDs = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_SCs_ChDs;}

      if(HistoCode == "D_MSp_SpNb"        ){
	fCanvD_MSp_SpNb = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_MSp_SpNb;}
      if(HistoCode == "D_MSp_SpDs"        ){
	fCanvD_MSp_SpDs = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_MSp_SpDs;}
      if(HistoCode =="D_SSp_SpNb"      ){
	fCanvD_SSp_SpNb = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_SSp_SpNb;}
      if(HistoCode =="D_SSp_SpDs"      ){
	fCanvD_SSp_SpDs = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_SSp_SpDs;}

      if(HistoCode == "D_Adc_EvNb"){
	fCanvD_Adc_EvNb = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_Adc_EvNb;}
      if(HistoCode == "D_Adc_EvDs"){
	fCanvD_Adc_EvDs = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvD_Adc_EvDs;}

      if(HistoCode == "H_Ped_Date"){
	fCanvH_Ped_Date = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvH_Ped_Date;}
      if(HistoCode == "H_TNo_Date"){
	fCanvH_TNo_Date = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvH_TNo_Date;}
      if(HistoCode == "H_MCs_Date"){
	fCanvH_MCs_Date = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvH_MCs_Date;}
      if(HistoCode == "H_LFN_Date"){
	fCanvH_LFN_Date = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvH_LFN_Date;}
      if(HistoCode == "H_HFN_Date"){
	fCanvH_HFN_Date = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvH_HFN_Date;}
      if(HistoCode == "H_SCs_Date"){
	fCanvH_SCs_Date = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvH_SCs_Date;}

      if(HistoCode == "H_Ped_RuDs"){
	fCanvH_Ped_RuDs = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvH_Ped_RuDs;}
      if(HistoCode == "H_TNo_RuDs"){
	fCanvH_TNo_RuDs = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvH_TNo_RuDs;}
      if(HistoCode == "H_MCs_RuDs"){
	fCanvH_MCs_RuDs = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvH_MCs_RuDs;}
      if(HistoCode == "H_LFN_RuDs"){
	fCanvH_LFN_RuDs = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvH_LFN_RuDs;}
      if(HistoCode == "H_HFN_RuDs"){
	fCanvH_HFN_RuDs = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvH_HFN_RuDs;}
      if(HistoCode == "H_SCs_RuDs"){
	fCanvH_SCs_RuDs = new TCanvas(canvas_name.Data(), canvas_name.Data(), canv_w, canv_h); fCnewRoot++;
	main_canvas = fCanvH_SCs_RuDs;}

    }
  return main_canvas;
}
// end of CreateCanvas

void TEcnaHistos::SetParametersCanvas(const TString& HistoCode, const TString& opt_plot)
{
// Set parameters canvas according to HistoCode
  
  Double_t x_margin_factor = fCnaParHistos->BoxLeftX("bottom_left_box") - 0.005;
  Double_t y_margin_factor = fCnaParHistos->BoxTopY("bottom_right_box") + 0.005;

  if( opt_plot == fSameOnePlot )
    {
      fImpH1SamePlus = (TCanvasImp*)fCanvH1SamePlus->GetCanvasImp();
      fCanvH1SamePlus->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
      fPadH1SamePlus = gPad;   fClosedH1SamePlus = kFALSE;
      fMemoPlotH1SamePlus = 1; fMemoColorH1SamePlus = 0;
    }

  if( opt_plot == fOnlyOnePlot ||  opt_plot == fSeveralPlot)
    {
      if(HistoCode == "D_NOE_ChNb")
	{
	  fImpD_NOE_ChNb = (TCanvasImp*)fCanvD_NOE_ChNb->GetCanvasImp();
	  fCanvD_NOE_ChNb->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_NOE_ChNb = gPad;   fClosedD_NOE_ChNb = kFALSE;
	  fMemoPlotD_NOE_ChNb = 1; fMemoColorD_NOE_ChNb = 0;
	}
      
      if(HistoCode == "D_NOE_ChDs")                                               // (SetParametersCanvas)
	{
	  fImpD_NOE_ChDs = (TCanvasImp*)fCanvD_NOE_ChDs->GetCanvasImp();
	  fCanvD_NOE_ChDs->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_NOE_ChDs = gPad;   fClosedD_NOE_ChDs = kFALSE;
	  fMemoPlotD_NOE_ChDs = 1; fMemoColorD_NOE_ChDs = 0;
	}
      
      if(HistoCode == "D_Ped_ChNb")
	{
	  fImpD_Ped_ChNb = (TCanvasImp*)fCanvD_Ped_ChNb->GetCanvasImp();
	  fCanvD_Ped_ChNb->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_Ped_ChNb = gPad;   fClosedD_Ped_ChNb = kFALSE;
	  fMemoPlotD_Ped_ChNb = 1; fMemoColorD_Ped_ChNb = 0;
	}
      
      if(HistoCode == "D_Ped_ChDs")
	{
	  fImpD_Ped_ChDs = (TCanvasImp*)fCanvD_Ped_ChDs->GetCanvasImp();
	  fCanvD_Ped_ChDs->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_Ped_ChDs = gPad;   fClosedD_Ped_ChDs = kFALSE;
	  fMemoPlotD_Ped_ChDs = 1; fMemoColorD_Ped_ChDs = 0;
	}
      
      if(HistoCode == "D_TNo_ChNb")
	{
	  fImpD_TNo_ChNb = (TCanvasImp*)fCanvD_TNo_ChNb->GetCanvasImp();
	  fCanvD_TNo_ChNb->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_TNo_ChNb = gPad;   fClosedD_TNo_ChNb = kFALSE;
	  fMemoPlotD_TNo_ChNb = 1; fMemoColorD_TNo_ChNb = 0;
	}
      
      if(HistoCode == "D_TNo_ChDs")                                               // (SetParametersCanvas)
	{
	  fImpD_TNo_ChDs = (TCanvasImp*)fCanvD_TNo_ChDs->GetCanvasImp();
	  fCanvD_TNo_ChDs->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_TNo_ChDs = gPad;   fClosedD_TNo_ChDs = kFALSE;
	  fMemoPlotD_TNo_ChDs = 1; fMemoColorD_TNo_ChDs = 0;
	}
      
      if(HistoCode == "D_MCs_ChNb")
	{
	  fImpD_MCs_ChNb = (TCanvasImp*)fCanvD_MCs_ChNb->GetCanvasImp();
	  fCanvD_MCs_ChNb->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_MCs_ChNb = gPad;   fClosedD_MCs_ChNb = kFALSE;
	  fMemoPlotD_MCs_ChNb = 1; fMemoColorD_MCs_ChNb = 0;
	}
      
      if(HistoCode == "D_MCs_ChDs")
	{
	  fImpD_MCs_ChDs = (TCanvasImp*)fCanvD_MCs_ChDs->GetCanvasImp();
	  fCanvD_MCs_ChDs->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_MCs_ChDs = gPad;   fClosedD_MCs_ChDs = kFALSE;
	  fMemoPlotD_MCs_ChDs = 1; fMemoColorD_MCs_ChDs = 0;
	}
      
      if(HistoCode == "D_LFN_ChNb")                                               // (SetParametersCanvas)
	{
	  fImpD_LFN_ChNb = (TCanvasImp*)fCanvD_LFN_ChNb->GetCanvasImp();
	  fCanvD_LFN_ChNb->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_LFN_ChNb = gPad;   fClosedD_LFN_ChNb = kFALSE;
	  fMemoPlotD_LFN_ChNb = 1; fMemoColorD_LFN_ChNb = 0;
	}
      
      if(HistoCode == "D_LFN_ChDs")
	{
	  fImpD_LFN_ChDs = (TCanvasImp*)fCanvD_LFN_ChDs->GetCanvasImp();
	  fCanvD_LFN_ChDs->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_LFN_ChDs = gPad;   fClosedD_LFN_ChDs = kFALSE;
	  fMemoPlotD_LFN_ChDs = 1; fMemoColorD_LFN_ChDs = 0;
	}
      
      if(HistoCode == "D_HFN_ChNb")
	{
	  fImpD_HFN_ChNb = (TCanvasImp*)fCanvD_HFN_ChNb->GetCanvasImp();
	  fCanvD_HFN_ChNb->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_HFN_ChNb = gPad;   fClosedD_HFN_ChNb = kFALSE;
	  fMemoPlotD_HFN_ChNb = 1; fMemoColorD_HFN_ChNb = 0;
	}
      
      if(HistoCode == "D_HFN_ChDs")
	{
	  fImpD_HFN_ChDs = (TCanvasImp*)fCanvD_HFN_ChDs->GetCanvasImp();
	  fCanvD_HFN_ChDs->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_HFN_ChDs = gPad;   fClosedD_HFN_ChDs = kFALSE;
	  fMemoPlotD_HFN_ChDs = 1; fMemoColorD_HFN_ChDs = 0;
	}
      
      if(HistoCode == "D_SCs_ChNb")                                               // (SetParametersCanvas)
	{
	  fImpD_SCs_ChNb = (TCanvasImp*)fCanvD_SCs_ChNb->GetCanvasImp();
	  fCanvD_SCs_ChNb->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_SCs_ChNb = gPad;   fClosedD_SCs_ChNb = kFALSE;
	  fMemoPlotD_SCs_ChNb = 1; fMemoColorD_SCs_ChNb = 0;
	}
      
      if(HistoCode == "D_SCs_ChDs")
	{
	  fImpD_SCs_ChDs = (TCanvasImp*)fCanvD_SCs_ChDs->GetCanvasImp();
	  fCanvD_SCs_ChDs->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_SCs_ChDs = gPad;   fClosedD_SCs_ChDs = kFALSE;
	  fMemoPlotD_SCs_ChDs = 1; fMemoColorD_SCs_ChDs = 0;
	}
      
      if(HistoCode == "D_MSp_SpNb")
	{
	  fImpD_MSp_SpNb = (TCanvasImp*)fCanvD_MSp_SpNb->GetCanvasImp();
	  fCanvD_MSp_SpNb->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_MSp_SpNb = gPad;   fClosedD_MSp_SpNb = kFALSE;
	  fMemoPlotD_MSp_SpNb = 1; fMemoColorD_MSp_SpNb = 0;
	}
      
      if(HistoCode == "D_MSp_SpDs")
	{
	  fImpD_MSp_SpDs = (TCanvasImp*)fCanvD_MSp_SpDs->GetCanvasImp();
	  fCanvD_MSp_SpDs->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_MSp_SpDs = gPad;   fClosedD_MSp_SpDs = kFALSE;
	  fMemoPlotD_MSp_SpDs = 1; fMemoColorD_MSp_SpDs = 0;
	}
      
      if(HistoCode == "D_SSp_SpNb")                                               // (SetParametersCanvas)
	{
	  fImpD_SSp_SpNb = (TCanvasImp*)fCanvD_SSp_SpNb->GetCanvasImp();
	  fCanvD_SSp_SpNb->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_SSp_SpNb = gPad;   fClosedD_SSp_SpNb = kFALSE;
	  fMemoPlotD_SSp_SpNb = 1; fMemoColorD_SSp_SpNb = 0;
	}
      
      if(HistoCode == "D_SSp_SpDs")                                               // (SetParametersCanvas)
	{
	  fImpD_SSp_SpDs = (TCanvasImp*)fCanvD_SSp_SpDs->GetCanvasImp();
	  fCanvD_SSp_SpDs->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_SSp_SpDs = gPad;   fClosedD_SSp_SpDs = kFALSE;
	  fMemoPlotD_SSp_SpDs = 1; fMemoColorD_SSp_SpDs = 0;
	}
      
      if(HistoCode == "D_Adc_EvDs")
	{
	  fImpD_Adc_EvDs = (TCanvasImp*)fCanvD_Adc_EvDs->GetCanvasImp();
	  fCanvD_Adc_EvDs->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_Adc_EvDs = gPad;   fClosedD_Adc_EvDs = kFALSE;
	  fMemoPlotD_Adc_EvDs = 1; fMemoColorD_Adc_EvDs = 0;		  
	}
      
      if(HistoCode == "D_Adc_EvNb")
	{
	  fImpD_Adc_EvNb = (TCanvasImp*)fCanvD_Adc_EvNb->GetCanvasImp();
	  fCanvD_Adc_EvNb->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadD_Adc_EvNb = gPad;   fClosedD_Adc_EvNb = kFALSE;
	  fMemoPlotD_Adc_EvNb = 1; fMemoColorD_Adc_EvNb = 0;
	}
      
      if(HistoCode == "H_Ped_Date")                                               // (SetParametersCanvas)
	{
	  fImpH_Ped_Date = (TCanvasImp*)fCanvH_Ped_Date->GetCanvasImp();
	  fCanvH_Ped_Date->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadH_Ped_Date = gPad;   fClosedH_Ped_Date = kFALSE;
	  fMemoPlotH_Ped_Date = 1; fMemoColorH_Ped_Date = 0;
	}
      if(HistoCode == "H_TNo_Date")
	{
	  fImpH_TNo_Date = (TCanvasImp*)fCanvH_TNo_Date->GetCanvasImp();
	  fCanvH_TNo_Date->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadH_TNo_Date = gPad;   fClosedH_TNo_Date = kFALSE;
	  fMemoPlotH_TNo_Date = 1; fMemoColorH_TNo_Date = 0;
	}
      if(HistoCode == "H_MCs_Date")
	{
	  fImpH_MCs_Date = (TCanvasImp*)fCanvH_MCs_Date->GetCanvasImp();
	  fCanvH_MCs_Date->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadH_MCs_Date = gPad;   fClosedH_MCs_Date = kFALSE;
	  fMemoPlotH_MCs_Date = 1; fMemoColorH_MCs_Date = 0;
	}

      if(HistoCode == "H_LFN_Date")                                               // (SetParametersCanvas)
	{
	  fImpH_LFN_Date = (TCanvasImp*)fCanvH_LFN_Date->GetCanvasImp();
	  fCanvH_LFN_Date->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadH_LFN_Date = gPad;   fClosedH_LFN_Date = kFALSE;
	  fMemoPlotH_LFN_Date = 1; fMemoColorH_LFN_Date = 0;
	}
      if(HistoCode == "H_HFN_Date")
	{
	  fImpH_HFN_Date = (TCanvasImp*)fCanvH_HFN_Date->GetCanvasImp();
	  fCanvH_HFN_Date->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadH_HFN_Date = gPad;   fClosedH_HFN_Date = kFALSE;
	  fMemoPlotH_HFN_Date = 1; fMemoColorH_HFN_Date = 0;
	}
      if(HistoCode == "H_SCs_Date")
	{
	  fImpH_SCs_Date = (TCanvasImp*)fCanvH_SCs_Date->GetCanvasImp();
	  fCanvH_SCs_Date->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadH_SCs_Date = gPad;   fClosedH_SCs_Date = kFALSE;
	  fMemoPlotH_SCs_Date = 1; fMemoColorH_SCs_Date = 0;
	}

      if(HistoCode == "H_Ped_RuDs")                                               // (SetParametersCanvas)
	{
	  fImpH_Ped_RuDs = (TCanvasImp*)fCanvH_Ped_RuDs->GetCanvasImp();
	  fCanvH_Ped_RuDs->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadH_Ped_RuDs = gPad;   fClosedH_Ped_RuDs = kFALSE;
	  fMemoPlotH_Ped_RuDs = 1; fMemoColorH_Ped_RuDs = 0;
	}
      if(HistoCode == "H_TNo_RuDs")
	{
	  fImpH_TNo_RuDs = (TCanvasImp*)fCanvH_TNo_RuDs->GetCanvasImp();
	  fCanvH_TNo_RuDs->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadH_TNo_RuDs = gPad;   fClosedH_TNo_RuDs = kFALSE;
	  fMemoPlotH_TNo_RuDs = 1; fMemoColorH_TNo_RuDs = 0;
	}
      if(HistoCode == "H_MCs_RuDs")
	{
	  fImpH_MCs_RuDs = (TCanvasImp*)fCanvH_MCs_RuDs->GetCanvasImp();
	  fCanvH_MCs_RuDs->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadH_MCs_RuDs = gPad;   fClosedH_MCs_RuDs = kFALSE;
	  fMemoPlotH_MCs_RuDs = 1; fMemoColorH_MCs_RuDs = 0;
	}

      if(HistoCode == "H_LFN_RuDs")                                               // (SetParametersCanvas)
	{
	  fImpH_LFN_RuDs = (TCanvasImp*)fCanvH_LFN_RuDs->GetCanvasImp();
	  fCanvH_LFN_RuDs->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadH_LFN_RuDs = gPad;   fClosedH_LFN_RuDs = kFALSE;
	  fMemoPlotH_LFN_RuDs = 1; fMemoColorH_LFN_RuDs = 0;
	}
      if(HistoCode == "H_HFN_RuDs")
	{
	  fImpH_HFN_RuDs = (TCanvasImp*)fCanvH_HFN_RuDs->GetCanvasImp();
	  fCanvH_HFN_RuDs->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadH_HFN_RuDs = gPad;   fClosedH_HFN_RuDs = kFALSE;
	  fMemoPlotH_HFN_RuDs = 1; fMemoColorH_HFN_RuDs = 0;
	}
      if(HistoCode == "H_SCs_RuDs")
	{
	  fImpH_SCs_RuDs = (TCanvasImp*)fCanvH_SCs_RuDs->GetCanvasImp();
	  fCanvH_SCs_RuDs->Divide(1, 1, x_margin_factor , y_margin_factor); gPad->cd(1);
	  fPadH_SCs_RuDs = gPad;   fClosedH_SCs_RuDs = kFALSE;
	  fMemoPlotH_SCs_RuDs = 1; fMemoColorH_SCs_RuDs = 0;
	}
    }
}
// end of SetParametersCanvas

TCanvas* TEcnaHistos::GetCurrentCanvas(const TString& HistoCode, const TString& opt_plot)
{
  TCanvas* main_canvas = 0;

  if( opt_plot == fSameOnePlot ){main_canvas = fCanvH1SamePlus;}

  if( opt_plot == fOnlyOnePlot ||  opt_plot == fSeveralPlot)
    {
      if(HistoCode == "D_NOE_ChNb"){main_canvas = fCanvD_NOE_ChNb;}
      if(HistoCode == "D_NOE_ChDs"){main_canvas = fCanvD_NOE_ChDs;}
      if(HistoCode == "D_Ped_ChNb"){main_canvas = fCanvD_Ped_ChNb;}
      if(HistoCode == "D_Ped_ChDs"){main_canvas = fCanvD_Ped_ChDs;}
      if(HistoCode == "D_TNo_ChNb"){main_canvas = fCanvD_TNo_ChNb;}
      if(HistoCode == "D_TNo_ChDs"){main_canvas = fCanvD_TNo_ChDs;}
      if(HistoCode == "D_MCs_ChNb"){main_canvas = fCanvD_MCs_ChNb;}
      if(HistoCode == "D_MCs_ChDs"){main_canvas = fCanvD_MCs_ChDs;}
      if(HistoCode == "D_LFN_ChNb"){main_canvas = fCanvD_LFN_ChNb;}
      if(HistoCode == "D_LFN_ChDs"){main_canvas = fCanvD_LFN_ChDs;}
      if(HistoCode == "D_HFN_ChNb"){main_canvas = fCanvD_HFN_ChNb;}
      if(HistoCode == "D_HFN_ChDs"){main_canvas = fCanvD_HFN_ChDs;}
      if(HistoCode == "D_SCs_ChNb"){main_canvas = fCanvD_SCs_ChNb;}
      if(HistoCode == "D_SCs_ChDs"){main_canvas = fCanvD_SCs_ChDs;}
      if(HistoCode == "D_MSp_SpNb"){main_canvas = fCanvD_MSp_SpNb;}
      if(HistoCode == "D_MSp_SpDs"){main_canvas = fCanvD_MSp_SpDs;}
      if(HistoCode == "D_SSp_SpNb"){main_canvas = fCanvD_SSp_SpNb;}
      if(HistoCode == "D_SSp_SpDs"){main_canvas = fCanvD_SSp_SpDs;}
      if(HistoCode == "D_Adc_EvNb"){main_canvas = fCanvD_Adc_EvNb;}
      if(HistoCode == "D_Adc_EvDs"){main_canvas = fCanvD_Adc_EvDs;}
      if(HistoCode == "H_Ped_Date"){main_canvas = fCanvH_Ped_Date;}
      if(HistoCode == "H_TNo_Date"){main_canvas = fCanvH_TNo_Date;}
      if(HistoCode == "H_MCs_Date"){main_canvas = fCanvH_MCs_Date;}
      if(HistoCode == "H_LFN_Date"){main_canvas = fCanvH_LFN_Date;}
      if(HistoCode == "H_HFN_Date"){main_canvas = fCanvH_HFN_Date;}
      if(HistoCode == "H_SCs_Date"){main_canvas = fCanvH_SCs_Date;}
      if(HistoCode == "H_Ped_RuDs"){main_canvas = fCanvH_Ped_RuDs;}
      if(HistoCode == "H_TNo_RuDs"){main_canvas = fCanvH_TNo_RuDs;}
      if(HistoCode == "H_MCs_RuDs"){main_canvas = fCanvH_MCs_RuDs;}
      if(HistoCode == "H_LFN_RuDs"){main_canvas = fCanvH_LFN_RuDs;}
      if(HistoCode == "H_HFN_RuDs"){main_canvas = fCanvH_HFN_RuDs;}
      if(HistoCode == "H_SCs_RuDs"){main_canvas = fCanvH_SCs_RuDs;}
    }
  return main_canvas;
}
// end of GetCurrentCanvas(...)

TCanvas* TEcnaHistos::GetCurrentCanvas(){return fCurrentCanvas;}
TString  TEcnaHistos::GetCurrentCanvasName(){return fCurrentCanvasName;}

void TEcnaHistos::PlotCloneOfCurrentCanvas()
{
  if( fCurrentCanvas != 0)
    {
      if( (TCanvasImp*)fCurrentCanvas->GetCanvasImp() != 0 )
	{
	  (TCanvas*)fCurrentCanvas->DrawClone();
	}
      else
	{
	  cout << "TEcnaHistos::PlotCloneOfCurrentCanvas()> Last canvas has been removed. No clone can be done."
	       << endl << "                                        Please, display the canvas again."
	       << fTTBELL << endl;
	}
    }
  else
    {
      cout << "TEcnaHistos::PlotCloneOfCurrentCanvas()> No canvas has been created. No clone can be done."
	   << fTTBELL << endl;
    }
}

//--------------------------------------------------------------------------------------------
TVirtualPad* TEcnaHistos::ActivePad(const TString& HistoCode, const TString& opt_plot)
{
// Active Pad for Same plot option

  TVirtualPad* main_subpad = 0;

  fCurrentHistoCode = HistoCode;
  fCurrentOptPlot   = opt_plot;

  if( opt_plot == fSameOnePlot )
    {
      fCanvH1SamePlus->
	Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
      if( fClosedH1SamePlus == kFALSE ){main_subpad = fPadH1SamePlus;}   
    }

  if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot )
    {
      if(HistoCode == "D_NOE_ChNb"){
	fCanvD_NOE_ChNb->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_NOE_ChNb == kFALSE ){main_subpad = fPadD_NOE_ChNb;}}
      
      if(HistoCode == "D_NOE_ChDs"){
	fCanvD_NOE_ChDs->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_NOE_ChDs == kFALSE ){main_subpad = fPadD_NOE_ChDs;}}
      
      if(HistoCode == "D_Ped_ChNb"){
	fCanvD_Ped_ChNb->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_Ped_ChNb == kFALSE ){main_subpad = fPadD_Ped_ChNb;}}
      
      if(HistoCode == "D_Ped_ChDs"){
	fCanvD_Ped_ChDs->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_Ped_ChDs == kFALSE ){main_subpad = fPadD_Ped_ChDs;}}
      
      if(HistoCode == "D_TNo_ChNb"){
	fCanvD_TNo_ChNb->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_TNo_ChNb == kFALSE ){main_subpad = fPadD_TNo_ChNb;}}

      if(HistoCode == "D_TNo_ChDs"){
	fCanvD_TNo_ChDs->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_TNo_ChDs == kFALSE ){main_subpad = fPadD_TNo_ChDs;}}
      
      if(HistoCode == "D_MCs_ChNb"){
	fCanvD_MCs_ChNb->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_MCs_ChNb == kFALSE ){main_subpad = fPadD_MCs_ChNb;}}
      
      if(HistoCode == "D_MCs_ChDs"){
	fCanvD_MCs_ChDs->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_MCs_ChDs == kFALSE ){main_subpad = fPadD_MCs_ChDs;}}

      if(HistoCode == "D_LFN_ChNb"){
	fCanvD_LFN_ChNb->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_LFN_ChNb == kFALSE ){main_subpad = fPadD_LFN_ChNb;}}
      
      if(HistoCode == "D_LFN_ChDs"){
	fCanvD_LFN_ChDs->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_LFN_ChDs == kFALSE ){main_subpad = fPadD_LFN_ChDs;}}
      
      if(HistoCode == "D_HFN_ChNb"){
	fCanvD_HFN_ChNb->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_HFN_ChNb == kFALSE ){main_subpad = fPadD_HFN_ChNb;}}
      
      if(HistoCode == "D_HFN_ChDs"){
	fCanvD_HFN_ChDs->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_HFN_ChDs == kFALSE ){main_subpad = fPadD_HFN_ChDs;}}
      
      if(HistoCode == "D_SCs_ChNb"){
	fCanvD_SCs_ChNb->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_SCs_ChNb == kFALSE ){main_subpad = fPadD_SCs_ChNb;}}
      
      if(HistoCode == "D_SCs_ChDs"){
	  fCanvD_SCs_ChDs->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_SCs_ChDs == kFALSE ){main_subpad = fPadD_SCs_ChDs;}}
      
      if(HistoCode == "D_MSp_SpNb"){
	  fCanvD_MSp_SpNb->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_MSp_SpNb == kFALSE ){main_subpad = fPadD_MSp_SpNb;}}
      
      if(HistoCode == "D_MSp_SpDs"){
	fCanvD_MSp_SpDs->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_MSp_SpDs == kFALSE ){main_subpad = fPadD_MSp_SpDs;}}

      if(HistoCode == "D_SSp_SpNb"){
	fCanvD_SSp_SpNb->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_SSp_SpNb == kFALSE ){main_subpad = fPadD_SSp_SpNb;}}
  
      if(HistoCode == "D_SSp_SpDs"){
	fCanvD_SSp_SpDs->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_SSp_SpDs == kFALSE ){main_subpad = fPadD_SSp_SpDs;}}

      if(HistoCode == "D_Adc_EvNb"){
	fCanvD_Adc_EvNb->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_Adc_EvNb == kFALSE ){main_subpad = fPadD_Adc_EvNb;}}
      
      if(HistoCode == "D_Adc_EvDs"){
	fCanvD_Adc_EvDs->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedD_Adc_EvDs == kFALSE ){main_subpad = fPadD_Adc_EvDs;}}

      if(HistoCode == "H_Ped_Date"){
	fCanvH_Ped_Date->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedH_Ped_Date == kFALSE ){main_subpad = fPadH_Ped_Date;}}

      if(HistoCode == "H_TNo_Date"){
	fCanvH_TNo_Date->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedH_TNo_Date == kFALSE ){main_subpad = fPadH_TNo_Date;}}
      
      if(HistoCode == "H_MCs_Date"){
	fCanvH_MCs_Date->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedH_MCs_Date == kFALSE ){main_subpad = fPadH_MCs_Date;}}

      if(HistoCode == "H_LFN_Date"){
	fCanvH_LFN_Date->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedH_LFN_Date == kFALSE ){main_subpad = fPadH_LFN_Date;}}
      
      if(HistoCode == "H_HFN_Date"){
	fCanvH_HFN_Date->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedH_HFN_Date == kFALSE ){main_subpad = fPadH_HFN_Date;}}
      
      if(HistoCode == "H_SCs_Date"){
	fCanvH_SCs_Date->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedH_SCs_Date == kFALSE ){main_subpad = fPadH_SCs_Date;}}

      if(HistoCode == "H_Ped_RuDs"){
	fCanvH_Ped_RuDs->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedH_Ped_RuDs == kFALSE ){main_subpad = fPadH_Ped_RuDs;}}
      
      if(HistoCode == "H_TNo_RuDs"){
	fCanvH_TNo_RuDs->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedH_TNo_RuDs == kFALSE ){main_subpad = fPadH_TNo_RuDs;}}
      
      if(HistoCode == "H_MCs_RuDs"){
	fCanvH_MCs_RuDs->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedH_MCs_RuDs == kFALSE ){main_subpad = fPadH_MCs_RuDs;}}

      if(HistoCode == "H_LFN_RuDs"){
	fCanvH_LFN_RuDs->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedH_LFN_RuDs == kFALSE ){main_subpad = fPadH_LFN_RuDs;}}
      
      if(HistoCode == "H_HFN_RuDs"){
	fCanvH_HFN_RuDs->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedH_HFN_RuDs == kFALSE ){main_subpad = fPadH_HFN_RuDs;}}
      
      if(HistoCode == "H_SCs_RuDs"){
	fCanvH_SCs_RuDs->Connect("Closed()","TEcnaHistos",this,"DoCanvasClosed()");
	if( fClosedH_SCs_RuDs == kFALSE ){main_subpad = fPadH_SCs_RuDs;}}
    }
    
  if( main_subpad == 0 )
    {cout << "*TEcnaHistos::ActivePad(...)> main_subpad = "
	  << main_subpad << ". This canvas has been closed." << endl;}

  return main_subpad;
}
// end of ActivePad

void TEcnaHistos::DoCanvasClosed()
{  
  if( fCurrentOptPlot == fSameOnePlot ){fClosedH1SamePlus = kTRUE;}
  if( fCurrentOptPlot == fOnlyOnePlot || fCurrentOptPlot == fSeveralPlot )
    {
      if(fCurrentHistoCode == "D_NOE_ChNb"){fClosedD_NOE_ChNb = kTRUE;}
      if(fCurrentHistoCode == "D_NOE_ChDs"){fClosedD_NOE_ChDs = kTRUE;}
      if(fCurrentHistoCode == "D_Ped_ChNb"){fClosedD_Ped_ChNb = kTRUE;}
      if(fCurrentHistoCode == "D_Ped_ChDs"){fClosedD_Ped_ChDs = kTRUE;}
      if(fCurrentHistoCode == "D_TNo_ChNb"){fClosedD_TNo_ChNb = kTRUE;}
      if(fCurrentHistoCode == "D_TNo_ChDs"){fClosedD_TNo_ChDs = kTRUE;}
      if(fCurrentHistoCode == "D_MCs_ChNb"){fClosedD_MCs_ChNb = kTRUE;}
      if(fCurrentHistoCode == "D_MCs_ChDs"){fClosedD_MCs_ChDs = kTRUE;}
      if(fCurrentHistoCode == "D_LFN_ChNb"){fClosedD_LFN_ChNb = kTRUE;}
      if(fCurrentHistoCode == "D_LFN_ChDs"){fClosedD_LFN_ChDs = kTRUE;}
      if(fCurrentHistoCode == "D_HFN_ChNb"){fClosedD_HFN_ChNb = kTRUE;}
      if(fCurrentHistoCode == "D_HFN_ChDs"){fClosedD_HFN_ChDs = kTRUE;}
      if(fCurrentHistoCode == "D_SCs_ChNb"){fClosedD_SCs_ChNb = kTRUE;}
      if(fCurrentHistoCode == "D_SCs_ChDs"){fClosedD_SCs_ChDs = kTRUE;}
      if(fCurrentHistoCode == "D_MSp_SpNb"){fClosedD_MSp_SpNb = kTRUE;}
      if(fCurrentHistoCode == "D_MSp_SpDs"){fClosedD_MSp_SpDs = kTRUE;}
      if(fCurrentHistoCode == "D_SSp_SpNb"){fClosedD_SSp_SpNb = kTRUE;}
      if(fCurrentHistoCode == "D_SSp_SpDs"){fClosedD_SSp_SpDs = kTRUE;}
      if(fCurrentHistoCode == "D_Adc_EvNb"){fClosedD_Adc_EvNb = kTRUE;}
      if(fCurrentHistoCode == "D_Adc_EvDs"){fClosedD_Adc_EvDs = kTRUE;}
      if(fCurrentHistoCode == "H_Ped_Date"){fClosedH_Ped_Date = kTRUE;}
      if(fCurrentHistoCode == "H_TNo_Date"){fClosedH_TNo_Date = kTRUE;}
      if(fCurrentHistoCode == "H_MCs_Date"){fClosedH_MCs_Date = kTRUE;}
      if(fCurrentHistoCode == "H_LFN_Date"){fClosedH_LFN_Date = kTRUE;}
      if(fCurrentHistoCode == "H_HFN_Date"){fClosedH_HFN_Date = kTRUE;}
      if(fCurrentHistoCode == "H_SCs_Date"){fClosedH_SCs_Date = kTRUE;}
      if(fCurrentHistoCode == "H_Ped_RuDs"){fClosedH_Ped_RuDs = kTRUE;}
      if(fCurrentHistoCode == "H_TNo_RuDs"){fClosedH_TNo_RuDs = kTRUE;}
      if(fCurrentHistoCode == "H_MCs_RuDs"){fClosedH_MCs_RuDs = kTRUE;}
      if(fCurrentHistoCode == "H_LFN_RuDs"){fClosedH_LFN_RuDs = kTRUE;}
      if(fCurrentHistoCode == "H_HFN_RuDs"){fClosedH_HFN_RuDs = kTRUE;}
      if(fCurrentHistoCode == "H_SCs_RuDs"){fClosedH_SCs_RuDs = kTRUE;}
    }

  fCurrentOptPlot = "NADA";  // to avoid fClosed... = kTRUE if other canvas than those above Closed (i.e. 2D plots)
  fCurrentHistoCode = "NADA";

  cout << "!TEcnaHistos::DoCanvasClosed(...)> WARNING: canvas has been closed." << endl;
}

void TEcnaHistos::SetParametersPavTxt(const TString& HistoCode, const TString& opt_plot)
{
// Set parameters pave "sevearl changing" according to HistoCode

  if( opt_plot == fSameOnePlot ){fPavTxtH1SamePlus = fPavComSeveralChanging;}

  if( opt_plot == fOnlyOnePlot ||  opt_plot == fSeveralPlot)
    {
      if(HistoCode == "D_NOE_ChNb"){fPavTxtD_NOE_ChNb = fPavComSeveralChanging;}
      if(HistoCode == "D_NOE_ChDs"){fPavTxtD_NOE_ChDs = fPavComSeveralChanging;}
      if(HistoCode == "D_Ped_ChNb"){fPavTxtD_Ped_ChNb = fPavComSeveralChanging;}
      if(HistoCode == "D_Ped_ChDs"){fPavTxtD_Ped_ChDs = fPavComSeveralChanging;}
      if(HistoCode == "D_TNo_ChNb"){fPavTxtD_TNo_ChNb = fPavComSeveralChanging;}
      if(HistoCode == "D_TNo_ChDs"){fPavTxtD_TNo_ChDs = fPavComSeveralChanging;}
      if(HistoCode == "D_MCs_ChNb"){fPavTxtD_MCs_ChNb = fPavComSeveralChanging;}
      if(HistoCode == "D_MCs_ChDs"){fPavTxtD_MCs_ChDs = fPavComSeveralChanging;}
      if(HistoCode == "D_LFN_ChNb"){fPavTxtD_LFN_ChNb = fPavComSeveralChanging;}
      if(HistoCode == "D_LFN_ChDs"){fPavTxtD_LFN_ChDs = fPavComSeveralChanging;}
      if(HistoCode == "D_HFN_ChNb"){fPavTxtD_HFN_ChNb = fPavComSeveralChanging;}
      if(HistoCode == "D_HFN_ChDs"){fPavTxtD_HFN_ChDs = fPavComSeveralChanging;}
      if(HistoCode == "D_SCs_ChNb"){fPavTxtD_SCs_ChNb = fPavComSeveralChanging;}
      if(HistoCode == "D_SCs_ChDs"){fPavTxtD_SCs_ChDs = fPavComSeveralChanging;}
      if(HistoCode == "D_MSp_SpNb"){fPavTxtD_MSp_SpNb = fPavComSeveralChanging;}
      if(HistoCode == "D_MSp_SpDs"){fPavTxtD_MSp_SpDs = fPavComSeveralChanging;}
      if(HistoCode == "D_SSp_SpNb"){fPavTxtD_SSp_SpNb = fPavComSeveralChanging;}
      if(HistoCode == "D_SSp_SpDs"){fPavTxtD_SSp_SpDs = fPavComSeveralChanging;}
      if(HistoCode == "D_Adc_EvNb"){fPavTxtD_Adc_EvNb = fPavComSeveralChanging;}
      if(HistoCode == "D_Adc_EvDs"){fPavTxtD_Adc_EvDs = fPavComSeveralChanging;}
      if(HistoCode == "H_Ped_Date"){fPavTxtH_Ped_Date = fPavComSeveralChanging;}
      if(HistoCode == "H_TNo_Date"){fPavTxtH_TNo_Date = fPavComSeveralChanging;}
      if(HistoCode == "H_MCs_Date"){fPavTxtH_MCs_Date = fPavComSeveralChanging;}
      if(HistoCode == "H_LFN_Date"){fPavTxtH_LFN_Date = fPavComSeveralChanging;}
      if(HistoCode == "H_HFN_Date"){fPavTxtH_HFN_Date = fPavComSeveralChanging;}
      if(HistoCode == "H_SCs_Date"){fPavTxtH_SCs_Date = fPavComSeveralChanging;}
      if(HistoCode == "H_Ped_RuDs"){fPavTxtH_Ped_RuDs = fPavComSeveralChanging;}
      if(HistoCode == "H_TNo_RuDs"){fPavTxtH_TNo_RuDs = fPavComSeveralChanging;}
      if(HistoCode == "H_MCs_RuDs"){fPavTxtH_MCs_RuDs = fPavComSeveralChanging;}
      if(HistoCode == "H_LFN_RuDs"){fPavTxtH_LFN_RuDs = fPavComSeveralChanging;}
      if(HistoCode == "H_HFN_RuDs"){fPavTxtH_HFN_RuDs = fPavComSeveralChanging;}
      if(HistoCode == "H_SCs_RuDs"){fPavTxtH_SCs_RuDs = fPavComSeveralChanging;}
    }
}
// end of SetParametersPavTxt


TPaveText* TEcnaHistos::ActivePavTxt(const TString& HistoCode, const TString& opt_plot)
{
  // Active Pad for Same plot option

  TPaveText* main_pavtxt = 0;
  
  if( opt_plot == fSameOnePlot ){main_pavtxt = fPavTxtH1SamePlus;}
  
  if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot)
    {
      if(HistoCode == "D_NOE_ChNb"){main_pavtxt = fPavTxtD_NOE_ChNb;}
      if(HistoCode == "D_NOE_ChDs"){main_pavtxt = fPavTxtD_NOE_ChDs;}
      if(HistoCode == "D_Ped_ChNb"){main_pavtxt = fPavTxtD_Ped_ChNb;}
      if(HistoCode == "D_Ped_ChDs"){main_pavtxt = fPavTxtD_Ped_ChDs;}
      if(HistoCode == "D_TNo_ChNb"){main_pavtxt = fPavTxtD_TNo_ChNb;}
      if(HistoCode == "D_TNo_ChDs"){main_pavtxt = fPavTxtD_TNo_ChDs;}
      if(HistoCode == "D_MCs_ChNb"){main_pavtxt = fPavTxtD_MCs_ChNb;}     
      if(HistoCode == "D_MCs_ChDs"){main_pavtxt = fPavTxtD_MCs_ChDs;}
      if(HistoCode == "D_LFN_ChNb"){main_pavtxt = fPavTxtD_LFN_ChNb;}
      if(HistoCode == "D_LFN_ChDs"){main_pavtxt = fPavTxtD_LFN_ChDs;}
      if(HistoCode == "D_HFN_ChNb"){main_pavtxt = fPavTxtD_HFN_ChNb;}
      if(HistoCode == "D_HFN_ChDs"){main_pavtxt = fPavTxtD_HFN_ChDs;}
      if(HistoCode == "D_SCs_ChNb"){main_pavtxt = fPavTxtD_SCs_ChNb;}
      if(HistoCode == "D_SCs_ChDs"){main_pavtxt = fPavTxtD_SCs_ChDs;}
      if(HistoCode == "D_MSp_SpNb"){main_pavtxt = fPavTxtD_MSp_SpNb;}
      if(HistoCode == "D_MSp_SpDs"){main_pavtxt = fPavTxtD_MSp_SpDs;}
      if(HistoCode == "D_SSp_SpNb"){main_pavtxt = fPavTxtD_SSp_SpNb;}
      if(HistoCode == "D_SSp_SpDs"){main_pavtxt = fPavTxtD_SSp_SpDs;}
      if(HistoCode == "D_Adc_EvNb"){main_pavtxt = fPavTxtD_Adc_EvNb;}
      if(HistoCode == "D_Adc_EvDs"){main_pavtxt = fPavTxtD_Adc_EvDs;}
      if(HistoCode == "H_Ped_Date"){main_pavtxt = fPavTxtH_Ped_Date;}
      if(HistoCode == "H_TNo_Date"){main_pavtxt = fPavTxtH_TNo_Date;}
      if(HistoCode == "H_MCs_Date"){main_pavtxt = fPavTxtH_MCs_Date;}
      if(HistoCode == "H_LFN_Date"){main_pavtxt = fPavTxtH_LFN_Date;}
      if(HistoCode == "H_HFN_Date"){main_pavtxt = fPavTxtH_HFN_Date;}
      if(HistoCode == "H_SCs_Date"){main_pavtxt = fPavTxtH_SCs_Date;}
      if(HistoCode == "H_Ped_RuDs"){main_pavtxt = fPavTxtH_Ped_RuDs;}
      if(HistoCode == "H_TNo_RuDs"){main_pavtxt = fPavTxtH_TNo_RuDs;}
      if(HistoCode == "H_MCs_RuDs"){main_pavtxt = fPavTxtH_MCs_RuDs;}
      if(HistoCode == "H_LFN_RuDs"){main_pavtxt = fPavTxtH_LFN_RuDs;}
      if(HistoCode == "H_HFN_RuDs"){main_pavtxt = fPavTxtH_HFN_RuDs;}
      if(HistoCode == "H_SCs_RuDs"){main_pavtxt = fPavTxtH_SCs_RuDs;}
    }
  
  if( main_pavtxt == 0 )
    {cout << "*TEcnaHistos::ActivePavTxt(...)> ERROR: main_pavtxt = " << main_pavtxt << endl;}

  return main_pavtxt;
}
// end of ActivePavTxt

//void TEcnaHistos::SetViewHistoMarkerAndLine(TH1D* h_his0, const TString& HistoCode, const TString& opt_plot)
//{
//// Set marker style and line style for histo view
//
//  TString HistoType = fCnaParHistos->GetHistoType(HistoCode.Data());
//
//  //............................... Marker
//  h_his0->SetMarkerStyle(1);        // default
//  
//  if( HistoType == "Global" && ( opt_plot == fSeveralPlot || opt_plot == fSameOnePlot ) )
//    {h_his0->SetMarkerStyle(7); }
//  
//  //............................... Line
//  h_his0->SetLineWidth(1);        // default
//
//  if( HistoType == "Global" && ( opt_plot == fSeveralPlot || opt_plot == fSameOnePlot ) )
//    {h_his0->SetLineWidth(0);}
//
//}

void TEcnaHistos::SetViewHistoColors(TH1D* h_his0,           const TString& HistoCode,
				     const TString& opt_plot, const Int_t&  arg_AlreadyRead)
{
// Set colors, fill, marker, line style for histo view

  TString HistoType = fCnaParHistos->GetHistoType(HistoCode.Data());
  if( HistoType == "Global" ){h_his0->SetMarkerStyle(1);}

  Int_t MaxNbOfColors = fCnaParHistos->GetMaxNbOfColors();

  if( opt_plot == fSameOnePlot )
    {
      h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorH1SamePlus));
      h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorH1SamePlus));
      fMemoColorH1SamePlus++;
      if(fMemoColorH1SamePlus>MaxNbOfColors){fMemoColorH1SamePlus = 0;}
    }

  if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot)
    {
      if(HistoCode == "D_NOE_ChNb")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("rose"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_NOE_ChNb));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_NOE_ChNb));
	    fMemoColorD_NOE_ChNb++;
	    if(fMemoColorD_NOE_ChNb>MaxNbOfColors){fMemoColorD_NOE_ChNb = 0;}}
	}
      if(HistoCode == "D_NOE_ChDs")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("rose"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_NOE_ChDs));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_NOE_ChDs));
	   fMemoColorD_NOE_ChDs++;
	    if(fMemoColorD_NOE_ChDs>MaxNbOfColors){fMemoColorD_NOE_ChDs = 0;}}
	}
      if(HistoCode == "D_Ped_ChNb")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("bleu38"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_Ped_ChNb));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_Ped_ChNb));
	    fMemoColorD_Ped_ChNb++;
	    if(fMemoColorD_Ped_ChNb>MaxNbOfColors){fMemoColorD_Ped_ChNb = 0;}}
	}
      if(HistoCode == "D_Ped_ChDs")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("bleu38"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_Ped_ChDs));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_Ped_ChDs));
	    fMemoColorD_Ped_ChDs++;
	    if(fMemoColorD_Ped_ChDs>MaxNbOfColors){fMemoColorD_Ped_ChDs = 0;}}
	}
      if(HistoCode == "D_TNo_ChNb")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("rouge48"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_TNo_ChNb));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_TNo_ChNb));
	    fMemoColorD_TNo_ChNb++;
	    if(fMemoColorD_TNo_ChNb>MaxNbOfColors){fMemoColorD_TNo_ChNb = 0;}}
	}
      if(HistoCode == "D_TNo_ChDs")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("rouge48"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_TNo_ChDs));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_TNo_ChDs));
	    fMemoColorD_TNo_ChDs++;
	    if(fMemoColorD_TNo_ChDs>MaxNbOfColors){fMemoColorD_TNo_ChDs = 0;}}
	}
	      
      if(HistoCode == "D_MCs_ChNb")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("vert31"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_MCs_ChNb));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_MCs_ChNb));
	    fMemoColorD_MCs_ChNb++;
	    if(fMemoColorD_MCs_ChNb>MaxNbOfColors){fMemoColorD_MCs_ChNb = 0;}}
	}
      if(HistoCode == "D_MCs_ChDs")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("vert31"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_MCs_ChDs));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_MCs_ChDs));
	    fMemoColorD_MCs_ChDs++;
	    if(fMemoColorD_MCs_ChDs>MaxNbOfColors){fMemoColorD_MCs_ChDs = 0;}}
	}
      if(HistoCode == "D_LFN_ChNb")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("rouge44"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_LFN_ChNb));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_LFN_ChNb));
	    fMemoColorD_LFN_ChNb++;
	    if(fMemoColorD_LFN_ChNb>MaxNbOfColors){fMemoColorD_LFN_ChNb = 0;}}
	}
      if(HistoCode == "D_LFN_ChDs")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("rouge44"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_LFN_ChDs));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_LFN_ChDs));
	    fMemoColorD_LFN_ChDs++;
	    if(fMemoColorD_LFN_ChDs>MaxNbOfColors){fMemoColorD_LFN_ChDs = 0;}}
	}
      if(HistoCode == "D_HFN_ChNb")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("rouge50"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_HFN_ChNb));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_HFN_ChNb));
	    fMemoColorD_HFN_ChNb++;
	    if(fMemoColorD_HFN_ChNb>MaxNbOfColors){fMemoColorD_HFN_ChNb = 0;}}
	}
      if(HistoCode == "D_HFN_ChDs")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("rouge50"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_HFN_ChDs));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_HFN_ChDs));
	    fMemoColorD_HFN_ChDs++;
	    if(fMemoColorD_HFN_ChDs>MaxNbOfColors){fMemoColorD_HFN_ChDs = 0;}}
	}

      if(HistoCode == "D_SCs_ChNb")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("marron23"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_SCs_ChNb));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_SCs_ChNb));
	    fMemoColorD_SCs_ChNb++;
	    if(fMemoColorD_SCs_ChNb>MaxNbOfColors){fMemoColorD_SCs_ChNb = 0;}}
	}
      if(HistoCode == "D_SCs_ChDs")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("marron23"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_SCs_ChDs));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_SCs_ChDs));
	    fMemoColorD_SCs_ChDs++;
	    if(fMemoColorD_SCs_ChDs>MaxNbOfColors){fMemoColorD_SCs_ChDs = 0;}}
	}
	      
      if(HistoCode == "D_MSp_SpNb")
	{
	  if( (opt_plot == fOnlyOnePlot && arg_AlreadyRead == 0) ||
	      (opt_plot == fOnlyOnePlot && arg_AlreadyRead >= 1 && fPlotAllXtalsInStin == 0 ) )
	    {h_his0->SetFillColor(fCnaParHistos->ColorDefinition("bleu38"));}

	  if( opt_plot == fOnlyOnePlot && arg_AlreadyRead >= 0 && fPlotAllXtalsInStin == 1 )
	    {h_his0->SetFillColor((Color_t)0);}

	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_MSp_SpNb));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_MSp_SpNb));
	    fMemoColorD_MSp_SpNb++;
	    if(fMemoColorD_MSp_SpNb>MaxNbOfColors){fMemoColorD_MSp_SpNb = 0;}}
	}

      if(HistoCode == "D_MSp_SpDs")
	{
	  if( (opt_plot == fOnlyOnePlot && arg_AlreadyRead == 0) ||
	      (opt_plot == fOnlyOnePlot && arg_AlreadyRead >= 1 && fPlotAllXtalsInStin == 0 ) )
	    {h_his0->SetFillColor(fCnaParHistos->ColorDefinition("bleu38"));}

	  if( opt_plot == fOnlyOnePlot && arg_AlreadyRead >= 0 && fPlotAllXtalsInStin == 1 )
	    {h_his0->SetFillColor((Color_t)0);}

	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_MSp_SpDs));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_MSp_SpDs));
	    fMemoColorD_MSp_SpDs++;
	    if(fMemoColorD_MSp_SpDs>MaxNbOfColors){fMemoColorD_MSp_SpDs = 0;}}
	}
	      
      if(HistoCode == "D_SSp_SpNb")
	{
	  if( (opt_plot == fOnlyOnePlot && arg_AlreadyRead == 0) ||
	      (opt_plot == fOnlyOnePlot && arg_AlreadyRead >= 1 && fPlotAllXtalsInStin == 0 ) )
	    {h_his0->SetFillColor(fCnaParHistos->ColorDefinition("rouge50"));}

	  if(opt_plot == fOnlyOnePlot && arg_AlreadyRead >= 0 && fPlotAllXtalsInStin == 1 )
	    {h_his0->SetFillColor((Color_t)0);}

	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_SSp_SpNb));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_SSp_SpNb));
	    fMemoColorD_SSp_SpNb++;
	    if(fMemoColorD_SSp_SpNb>MaxNbOfColors){fMemoColorD_SSp_SpNb = 0;}}
	}

      if(HistoCode == "D_SSp_SpDs")
	{
	  if( (opt_plot == fOnlyOnePlot && arg_AlreadyRead == 0) ||
	      (opt_plot == fOnlyOnePlot && arg_AlreadyRead >= 1 && fPlotAllXtalsInStin == 0 ) )
	    {h_his0->SetFillColor(fCnaParHistos->ColorDefinition("rouge50"));}

	  if(opt_plot == fOnlyOnePlot && arg_AlreadyRead >= 0 && fPlotAllXtalsInStin == 1 )
	    {h_his0->SetFillColor((Color_t)0);}

	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_SSp_SpDs));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_SSp_SpDs));
	    fMemoColorD_SSp_SpDs++;
	    if(fMemoColorD_SSp_SpDs>MaxNbOfColors){fMemoColorD_SSp_SpDs = 0;}}
	}

      if(HistoCode == "D_Adc_EvNb")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("orange42"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_Adc_EvNb));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_Adc_EvNb));
	    fMemoColorD_Adc_EvNb++;
	    if(fMemoColorD_Adc_EvNb>MaxNbOfColors){fMemoColorD_Adc_EvNb = 0;}}
	  gPad->SetGrid(1,0);
	}

      if(HistoCode == "D_Adc_EvDs")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("orange42"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorD_Adc_EvDs));
	    h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorD_Adc_EvDs));
	    fMemoColorD_Adc_EvDs++;
	    if(fMemoColorD_Adc_EvDs>MaxNbOfColors){fMemoColorD_Adc_EvDs = 0;}}
	}

      if(HistoCode == "H_Ped_RuDs")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("bleu38"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorH_Ped_RuDs));
	    h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorH_Ped_RuDs));
	    fMemoColorH_Ped_RuDs++;
	    if(fMemoColorH_Ped_RuDs>MaxNbOfColors){fMemoColorH_Ped_RuDs = 0;}}
	  gPad->SetGrid(1,1);
	}

      if(HistoCode == "H_TNo_RuDs")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("rouge48"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorH_TNo_RuDs));
	    h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorH_TNo_RuDs));
	    fMemoColorH_TNo_RuDs++;
	    if(fMemoColorH_TNo_RuDs>MaxNbOfColors){fMemoColorH_TNo_RuDs = 0;}}
	  gPad->SetGrid(1,1);
	}

      if(HistoCode == "H_MCs_RuDs")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("vert31"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorH_MCs_RuDs));
	    h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorH_MCs_RuDs));
	    fMemoColorH_MCs_RuDs++;
	    if(fMemoColorH_MCs_RuDs>MaxNbOfColors){fMemoColorH_MCs_RuDs = 0;}}
	  gPad->SetGrid(1,1);
	}

      if(HistoCode == "H_LFN_RuDs")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("rouge44"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorH_LFN_RuDs));
	    h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorH_LFN_RuDs));
	    fMemoColorH_LFN_RuDs++;
	    if(fMemoColorH_LFN_RuDs>MaxNbOfColors){fMemoColorH_LFN_RuDs = 0;}}
	  gPad->SetGrid(1,1);
	}

      if(HistoCode == "H_HFN_RuDs")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("rouge50"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorH_HFN_RuDs));
	    h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorH_HFN_RuDs));
	    fMemoColorH_HFN_RuDs++;
	    if(fMemoColorH_HFN_RuDs>MaxNbOfColors){fMemoColorH_HFN_RuDs = 0;}}
	  gPad->SetGrid(1,1);
	}

      if(HistoCode == "H_SCs_RuDs")
	{
	  if(opt_plot == fOnlyOnePlot){h_his0->SetFillColor(fCnaParHistos->ColorDefinition("marron23"));}
	  if(opt_plot == fSeveralPlot )
	    {h_his0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorH_SCs_RuDs));
	    h_his0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorH_SCs_RuDs));
	    fMemoColorH_SCs_RuDs++;
	    if(fMemoColorH_SCs_RuDs>MaxNbOfColors){fMemoColorH_SCs_RuDs = 0;}}
	  gPad->SetGrid(1,1);
	}
    }

  // if(opt_plot == fSeveralPlot || opt_plot == fSameOnePlot){h_his0->SetLineWidth(2);}
}
// end of SetViewHistoColors

void TEcnaHistos::SetViewGraphColors(TGraph* g_graph0, const TString& HistoCode, const TString& opt_plot)
{
// Set colors for histo view

  Int_t MaxNbOfColors = fCnaParHistos->GetMaxNbOfColors();

  if( opt_plot == fSameOnePlot )
    {
      g_graph0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorH1SamePlus));
      g_graph0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorH1SamePlus));
      fMemoColorH1SamePlus++;
      if(fMemoColorH1SamePlus>MaxNbOfColors){fMemoColorH1SamePlus = 0;}
      gPad->SetGrid(1,1);
    }

  if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot)
    {
      if(HistoCode == "H_Ped_Date")
	{
	  if(opt_plot == fOnlyOnePlot){g_graph0->SetMarkerColor(fCnaParHistos->ColorDefinition("bleu38"));}
	  if(opt_plot == fSeveralPlot )
	    {g_graph0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorH_Ped_Date));
	    g_graph0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorH_Ped_Date));
	    fMemoColorH_Ped_Date++;
	    if(fMemoColorH_Ped_Date>MaxNbOfColors){fMemoColorH_Ped_Date = 0;}}
	  gPad->SetGrid(1,1);
	}
      
      if(HistoCode == "H_TNo_Date")
	{
	  if(opt_plot == fOnlyOnePlot){g_graph0->SetMarkerColor(fCnaParHistos->ColorDefinition("rouge48"));}
	  if(opt_plot == fSeveralPlot)
	    {g_graph0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorH_TNo_Date));
	    g_graph0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorH_TNo_Date));
	    fMemoColorH_TNo_Date++;
	    if(fMemoColorH_TNo_Date>MaxNbOfColors){fMemoColorH_TNo_Date = 0;}}
	  gPad->SetGrid(1,1);
	}
      
      if(HistoCode == "H_MCs_Date")
	{
	  if(opt_plot == fOnlyOnePlot){g_graph0->SetMarkerColor(fCnaParHistos->ColorDefinition("vert31"));}
	  if(opt_plot == fSeveralPlot)
	    {g_graph0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorH_MCs_Date));
	    g_graph0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorH_MCs_Date));
	    fMemoColorH_MCs_Date++;
	    if(fMemoColorH_MCs_Date>MaxNbOfColors){fMemoColorH_MCs_Date = 0;}}
	  gPad->SetGrid(1,1);
	}

      if(HistoCode == "H_LFN_Date")
	{
	  if(opt_plot == fOnlyOnePlot){g_graph0->SetMarkerColor(fCnaParHistos->ColorDefinition("bleu38"));}
	  if(opt_plot == fSeveralPlot )
	    {g_graph0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorH_LFN_Date));
	    g_graph0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorH_LFN_Date));
	    fMemoColorH_LFN_Date++;
	    if(fMemoColorH_LFN_Date>MaxNbOfColors){fMemoColorH_LFN_Date = 0;}}
	  gPad->SetGrid(1,1);
	}
      
      if(HistoCode == "H_HFN_Date")
	{
	  if(opt_plot == fOnlyOnePlot){g_graph0->SetMarkerColor(fCnaParHistos->ColorDefinition("rouge48"));}
	  if(opt_plot == fSeveralPlot)
	    {g_graph0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorH_HFN_Date));
	    g_graph0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorH_HFN_Date));
	    fMemoColorH_HFN_Date++;
	    if(fMemoColorH_HFN_Date>MaxNbOfColors){fMemoColorH_HFN_Date = 0;}}
	  gPad->SetGrid(1,1);
	}
      
      if(HistoCode == "H_SCs_Date")
	{
	  if(opt_plot == fOnlyOnePlot){g_graph0->SetMarkerColor(fCnaParHistos->ColorDefinition("vert31"));}
	  if(opt_plot == fSeveralPlot)
	    {g_graph0->SetMarkerColor(fCnaParHistos->ColorTab(fMemoColorH_SCs_Date));
	    g_graph0->SetLineColor(fCnaParHistos->ColorTab(fMemoColorH_SCs_Date));
	    fMemoColorH_SCs_Date++;
	    if(fMemoColorH_SCs_Date>MaxNbOfColors){fMemoColorH_SCs_Date = 0;}}
	  gPad->SetGrid(1,1);
	}
    }
  //if(opt_plot == fSeveralPlot){g_graph0->SetLineWidth(2);}
}
// end of SetViewGraphColors

Color_t TEcnaHistos::GetViewHistoColor(const TString& HistoCode, const TString& opt_plot)
{
  Color_t couleur = fCnaParHistos->ColorDefinition("noir");        // a priori = "noir"

  if( opt_plot == fSameOnePlot ){couleur = fCnaParHistos->ColorTab(fMemoColorH1SamePlus);}

  if( opt_plot == fOnlyOnePlot || opt_plot == fSeveralPlot )
    {
      if(HistoCode == "D_NOE_ChNb"){couleur = fCnaParHistos->ColorTab(fMemoColorD_NOE_ChNb);}
      if(HistoCode == "D_NOE_ChDs"){couleur = fCnaParHistos->ColorTab(fMemoColorD_NOE_ChDs);}
      if(HistoCode == "D_Ped_ChNb"){couleur = fCnaParHistos->ColorTab(fMemoColorD_Ped_ChNb);}
      if(HistoCode == "D_Ped_ChDs"){couleur = fCnaParHistos->ColorTab(fMemoColorD_Ped_ChDs);}
      if(HistoCode == "D_TNo_ChNb"){couleur = fCnaParHistos->ColorTab(fMemoColorD_TNo_ChNb);}
      if(HistoCode == "D_TNo_ChDs"){couleur = fCnaParHistos->ColorTab(fMemoColorD_TNo_ChDs);}
      if(HistoCode == "D_MCs_ChNb"){couleur = fCnaParHistos->ColorTab(fMemoColorD_MCs_ChNb);}
      if(HistoCode == "D_MCs_ChDs"){couleur = fCnaParHistos->ColorTab(fMemoColorD_MCs_ChDs);}
      if(HistoCode == "D_LFN_ChNb"){couleur = fCnaParHistos->ColorTab(fMemoColorD_LFN_ChNb);}
      if(HistoCode == "D_LFN_ChDs"){couleur = fCnaParHistos->ColorTab(fMemoColorD_LFN_ChDs);} 
      if(HistoCode == "D_HFN_ChNb"){couleur = fCnaParHistos->ColorTab(fMemoColorD_HFN_ChNb);} 
      if(HistoCode == "D_HFN_ChDs"){couleur = fCnaParHistos->ColorTab(fMemoColorD_HFN_ChDs);}
      if(HistoCode == "D_SCs_ChNb"){couleur = fCnaParHistos->ColorTab(fMemoColorD_SCs_ChNb);}
      if(HistoCode == "D_SCs_ChDs"){couleur = fCnaParHistos->ColorTab(fMemoColorD_SCs_ChDs);}
      if(HistoCode == "D_MSp_SpNb"){couleur = fCnaParHistos->ColorTab(fMemoColorD_MSp_SpNb);}
      if(HistoCode == "D_MSp_SpDs"){couleur = fCnaParHistos->ColorTab(fMemoColorD_MSp_SpDs);}
      if(HistoCode == "D_SSp_SpNb"){couleur = fCnaParHistos->ColorTab(fMemoColorD_SSp_SpNb);}
      if(HistoCode == "D_SSp_SpDs"){couleur = fCnaParHistos->ColorTab(fMemoColorD_SSp_SpDs);}
      if(HistoCode == "D_Adc_EvNb"){couleur = fCnaParHistos->ColorTab(fMemoColorD_Adc_EvNb);}
      if(HistoCode == "D_Adc_EvDs"){couleur = fCnaParHistos->ColorTab(fMemoColorD_Adc_EvDs);}
      if(HistoCode == "H_Ped_Date"){couleur = fCnaParHistos->ColorTab(fMemoColorH_Ped_Date);}
      if(HistoCode == "H_TNo_Date"){couleur = fCnaParHistos->ColorTab(fMemoColorH_TNo_Date);}
      if(HistoCode == "H_MCs_Date"){couleur = fCnaParHistos->ColorTab(fMemoColorH_MCs_Date);}
      if(HistoCode == "H_LFN_Date"){couleur = fCnaParHistos->ColorTab(fMemoColorH_LFN_Date);}
      if(HistoCode == "H_HFN_Date"){couleur = fCnaParHistos->ColorTab(fMemoColorH_HFN_Date);}
      if(HistoCode == "H_SCs_Date"){couleur = fCnaParHistos->ColorTab(fMemoColorH_SCs_Date);}
      if(HistoCode == "H_Ped_RuDs"){couleur = fCnaParHistos->ColorTab(fMemoColorH_Ped_RuDs);}
      if(HistoCode == "H_TNo_RuDs"){couleur = fCnaParHistos->ColorTab(fMemoColorH_TNo_RuDs);}
      if(HistoCode == "H_MCs_RuDs"){couleur = fCnaParHistos->ColorTab(fMemoColorH_MCs_RuDs);}
      if(HistoCode == "H_LFN_RuDs"){couleur = fCnaParHistos->ColorTab(fMemoColorH_LFN_RuDs);}
      if(HistoCode == "H_HFN_RuDs"){couleur = fCnaParHistos->ColorTab(fMemoColorH_HFN_RuDs);}
      if(HistoCode == "H_SCs_RuDs"){couleur = fCnaParHistos->ColorTab(fMemoColorH_SCs_RuDs);}
    }
  return couleur;
}
// end of GetViewHistoColor

Color_t TEcnaHistos::GetSCColor(const TString& DeeEndcap, const TString& DeeDir, const TString& QuadType)
{
 //gives the SC color for the numbering plots
  TColor* my_color = new TColor();
  Color_t couleur = fCnaParHistos->ColorDefinition("noir");        // default = "noir"

  if( DeeEndcap == "EE+" )
    {
      if ( DeeDir == "right" && QuadType == "top"    ){couleur=fCnaParHistos->ColorDefinition("rouge");}
      if ( DeeDir == "right" && QuadType == "bottom" ){couleur=fCnaParHistos->ColorDefinition("bleu_fonce");}
      if ( DeeDir == "left"  && QuadType == "top"    ){couleur=(Color_t)my_color->GetColor("#006600");}
      if ( DeeDir == "left"  && QuadType == "bottom" ){couleur=(Color_t)my_color->GetColor("#CC3300");}
    }
  if( DeeEndcap == "EE-" )
    {
      if ( DeeDir == "right"  && QuadType == "top"    ){couleur=(Color_t)my_color->GetColor("#008800");}
      if ( DeeDir == "right"  && QuadType == "bottom" ){couleur=(Color_t)my_color->GetColor("#EE5500");}
      if ( DeeDir == "left"   && QuadType == "top"    ){couleur=fCnaParHistos->ColorDefinition("rouge");}
      if ( DeeDir == "left"   && QuadType == "bottom" ){couleur=fCnaParHistos->ColorDefinition("bleu_fonce");}
    }

  return couleur;
}
void TEcnaHistos::SetHistoPresentation(TH1D* histo, const TString& HistoType)
{
  // Set presentation (axis title offsets, title size, label size, etc... 

  fCnaParHistos->SetViewHistoStyle(HistoType.Data());
  fCnaParHistos->SetViewHistoPadMargins(HistoType.Data(), " ");
  fCnaParHistos->SetViewHistoOffsets(histo, HistoType.Data(), " ");
  fCnaParHistos->SetViewHistoStats(histo, HistoType.Data());
}
void TEcnaHistos::SetHistoPresentation(TH1D* histo, const TString& HistoType, const TString& opt_plot)
{
// Set presentation (axis title offsets, title size, label size, etc... 

  fCnaParHistos->SetViewHistoStyle(HistoType.Data());
  fCnaParHistos->SetViewHistoPadMargins(HistoType.Data(), opt_plot.Data());
  fCnaParHistos->SetViewHistoOffsets(histo, HistoType.Data(), opt_plot.Data());
  fCnaParHistos->SetViewHistoStats(histo, HistoType.Data());
}

void TEcnaHistos::SetGraphPresentation(TGraph* graph, const TString& HistoType, const TString& opt_plot)
{
// Set presentation (axis title offsets, title size, label size, etc... 

  fCnaParHistos->SetViewHistoStyle(HistoType.Data());
  fCnaParHistos->SetViewHistoPadMargins(HistoType.Data(), opt_plot);
  fCnaParHistos->SetViewGraphOffsets(graph, HistoType.Data());
  
  //............................... Graph marker
  graph->SetMarkerStyle(1);
  if( HistoType == "Evol" ){graph->SetMarkerStyle(20);}
}

//=====================================================================
//
//                 NewCanvas, ReInitCanvas
//
//=====================================================================
void TEcnaHistos::NewCanvas(const TString& opt_plot)   
{
// ReInit canvas in option SAME n in order to restart a new SAME n plot
// (called by user only for option Same n)

  if( opt_plot == fSameOnePlot )
    {
      fImpH1SamePlus = 0;       fCanvH1SamePlus = 0;
      fPadH1SamePlus = 0;       fMemoPlotH1SamePlus = 0;
      fMemoColorH1SamePlus = 0; fCanvSameH1SamePlus++;
      fPavTxtH1SamePlus = 0;    fClosedH1SamePlus = kFALSE;
    }
  else
    {
      cout << "TEcnaHistos::NewCanvas(...)> *** ERROR *** " << opt_plot.Data() << ": "
	   << "unknown option for NewCanvas. Only " << fSameOnePlot << " option is accepted."
	   << fTTBELL << endl;
    }
}

void TEcnaHistos::ReInitCanvas(const TString& HistoCode, const TString& opt_plot)
{
// ReInit canvas in option SAME and SAME n

  if( opt_plot == fSameOnePlot )
    {
      fImpH1SamePlus = 0;       fCanvH1SamePlus = 0;
      fPadH1SamePlus = 0;       fMemoPlotH1SamePlus = 0;
      fMemoColorH1SamePlus = 0; fCanvSameH1SamePlus++;
      fPavTxtH1SamePlus = 0;    fClosedH1SamePlus = kFALSE;
    }

  if( opt_plot == fOnlyOnePlot ||  opt_plot == fSeveralPlot)
    {
      if(HistoCode == "D_NOE_ChNb")
	{	      
	  fImpD_NOE_ChNb = 0;       fCanvD_NOE_ChNb = 0;
	  fPadD_NOE_ChNb = 0;       fMemoPlotD_NOE_ChNb = 0;
	  fMemoColorD_NOE_ChNb = 0; fCanvSameD_NOE_ChNb++;
	  fPavTxtD_NOE_ChNb = 0;    fClosedD_NOE_ChNb = kFALSE;
	}
      
      if(HistoCode == "D_NOE_ChDs")
	{	      
	  fImpD_NOE_ChDs = 0;       fCanvD_NOE_ChDs = 0;
	  fPadD_NOE_ChDs = 0;       fMemoPlotD_NOE_ChDs = 0;
	  fMemoColorD_NOE_ChDs = 0; fCanvSameD_NOE_ChDs++;
	  fPavTxtD_NOE_ChDs = 0;    fClosedD_NOE_ChDs = kFALSE;
	}
      
      if(HistoCode == "D_Ped_ChNb")                            // (ReInitCanvas)
	{	      
	  fImpD_Ped_ChNb = 0;       fCanvD_Ped_ChNb = 0;
	  fPadD_Ped_ChNb = 0;       fMemoPlotD_Ped_ChNb = 0;
	  fMemoColorD_Ped_ChNb = 0; fCanvSameD_Ped_ChNb++;
	  fPavTxtD_Ped_ChNb = 0;    fClosedD_Ped_ChNb = kFALSE;
	}
      
      if(HistoCode == "D_Ped_ChDs")
	{	      
	  fImpD_Ped_ChDs = 0;       fCanvD_Ped_ChDs = 0;
	  fPadD_Ped_ChDs = 0;       fMemoPlotD_Ped_ChDs = 0;
	  fMemoColorD_Ped_ChDs = 0; fCanvSameD_Ped_ChDs++;
	  fPavTxtD_Ped_ChDs = 0;    fClosedD_Ped_ChDs = kFALSE;
	}
      
      if(HistoCode == "D_TNo_ChNb")
	{	      
	  fImpD_TNo_ChNb = 0;       fCanvD_TNo_ChNb = 0;
	  fPadD_TNo_ChNb = 0;       fMemoPlotD_TNo_ChNb = 0;
	  fMemoColorD_TNo_ChNb = 0; fCanvSameD_TNo_ChNb++;
	  fPavTxtD_TNo_ChNb = 0;    fClosedD_TNo_ChNb = kFALSE;
	}
      
      if(HistoCode == "D_TNo_ChDs") 
	{	      
	  fImpD_TNo_ChDs = 0;       fCanvD_TNo_ChDs = 0;
	  fPadD_TNo_ChDs = 0;       fMemoPlotD_TNo_ChDs = 0;
	  fMemoColorD_TNo_ChDs = 0; fCanvSameD_TNo_ChDs++;
	  fPavTxtD_TNo_ChDs = 0;    fClosedD_TNo_ChDs = kFALSE;
	}
      
      if(HistoCode == "D_MCs_ChNb")                           // (ReInitCanvas)
	{	      
	  fImpD_MCs_ChNb = 0;       fCanvD_MCs_ChNb = 0;
	  fPadD_MCs_ChNb = 0;       fMemoPlotD_MCs_ChNb = 0;
	  fMemoColorD_MCs_ChNb = 0; fCanvSameD_MCs_ChNb++;
	  fPavTxtD_MCs_ChNb = 0;    fClosedD_MCs_ChNb = kFALSE;
	}
      
      if(HistoCode == "D_MCs_ChDs")
	{	      
	  fImpD_MCs_ChDs = 0;       fCanvD_MCs_ChDs = 0;
	  fPadD_MCs_ChDs = 0;       fMemoPlotD_MCs_ChDs = 0;
	  fMemoColorD_MCs_ChDs = 0; fCanvSameD_MCs_ChDs++;
	  fPavTxtD_MCs_ChDs = 0;    fClosedD_MCs_ChDs = kFALSE;
	}

      if(HistoCode == "D_LFN_ChNb")
	{	      
	  fImpD_LFN_ChNb = 0;       fCanvD_LFN_ChNb = 0;
	  fPadD_LFN_ChNb = 0;       fMemoPlotD_LFN_ChNb = 0;
	  fMemoColorD_LFN_ChNb = 0; fCanvSameD_LFN_ChNb++;
	  fPavTxtD_LFN_ChNb = 0;    fClosedD_LFN_ChNb = kFALSE;
	}
      
      if(HistoCode == "D_LFN_ChDs")                            // (ReInitCanvas)
	{	      
	  fImpD_LFN_ChDs = 0;       fCanvD_LFN_ChDs = 0;
	  fPadD_LFN_ChDs= 0;        fMemoPlotD_LFN_ChDs = 0;
	  fMemoColorD_LFN_ChDs = 0; fCanvSameD_LFN_ChDs++;
	  fPavTxtD_LFN_ChDs= 0;     fClosedD_LFN_ChDs = kFALSE;
	}
      
      if(HistoCode == "D_HFN_ChNb")
	{	      
	  fImpD_HFN_ChNb = 0;       fCanvD_HFN_ChNb = 0;
	  fPadD_HFN_ChNb = 0;       fMemoPlotD_HFN_ChNb = 0;
	  fMemoColorD_HFN_ChNb = 0; fCanvSameD_HFN_ChNb++;
	  fPavTxtD_HFN_ChNb = 0;    fClosedD_HFN_ChNb = kFALSE;
	}
      
      if(HistoCode == "D_HFN_ChDs")
	{	      
	  fImpD_HFN_ChDs = 0;       fCanvD_HFN_ChDs = 0;
	  fPadD_HFN_ChDs = 0;       fMemoPlotD_HFN_ChDs = 0;
	  fMemoColorD_HFN_ChDs = 0; fCanvSameD_HFN_ChDs++;
	  fPavTxtD_HFN_ChDs = 0;    fClosedD_HFN_ChDs = kFALSE;
	}
      
      if(HistoCode == "D_SCs_ChNb")
	{	      
	  fImpD_SCs_ChNb = 0;       fCanvD_SCs_ChNb = 0;
	  fPadD_SCs_ChNb = 0;       fMemoPlotD_SCs_ChNb = 0;
	  fMemoColorD_SCs_ChNb = 0; fCanvSameD_SCs_ChNb++;
	  fPavTxtD_SCs_ChNb = 0;    fClosedD_SCs_ChNb = kFALSE;
	}
      
      if(HistoCode == "D_SCs_ChDs")                            // (ReInitCanvas)
	{	      
	  fImpD_SCs_ChDs = 0;       fCanvD_SCs_ChDs = 0;
	  fPadD_SCs_ChDs = 0;       fMemoPlotD_SCs_ChDs = 0;
	  fMemoColorD_SCs_ChDs = 0; fCanvSameD_SCs_ChDs++;
	  fPavTxtD_SCs_ChDs = 0;    fClosedD_SCs_ChDs = kFALSE;
	}
      
      if(HistoCode == "D_MSp_SpNb")
	{	      
	  fImpD_MSp_SpNb = 0;       fCanvD_MSp_SpNb = 0;
	  fPadD_MSp_SpNb = 0;       fMemoPlotD_MSp_SpNb = 0; 
	  fMemoColorD_MSp_SpNb = 0; fCanvSameD_MSp_SpNb++;
	  fPavTxtD_MSp_SpNb = 0;    fClosedD_MSp_SpNb = kFALSE;
	}
            
      if(HistoCode == "D_MSp_SpDs")
	{	      
	  fImpD_MSp_SpDs = 0;       fCanvD_MSp_SpDs = 0;
	  fPadD_MSp_SpDs = 0;       fMemoPlotD_MSp_SpDs = 0; 
	  fMemoColorD_MSp_SpDs = 0; fCanvSameD_MSp_SpDs++;
	  fPavTxtD_MSp_SpDs = 0;    fClosedD_MSp_SpDs = kFALSE;
	}
      
      if(HistoCode == "D_SSp_SpNb")
	{	      
	  fImpD_SSp_SpNb = 0;       fCanvD_SSp_SpNb = 0;
	  fPadD_SSp_SpNb = 0;       fMemoPlotD_SSp_SpNb= 0;
	  fMemoColorD_SSp_SpNb = 0; fCanvSameD_SSp_SpNb++;
	  fPavTxtD_SSp_SpNb = 0;    fClosedD_SSp_SpNb = kFALSE;
	}

      if(HistoCode == "D_SSp_SpDs")
	{	      
	  fImpD_SSp_SpDs = 0;       fCanvD_SSp_SpDs = 0;
	  fPadD_SSp_SpDs = 0;       fMemoPlotD_SSp_SpDs= 0;
	  fMemoColorD_SSp_SpDs = 0; fCanvSameD_SSp_SpDs++;
	  fPavTxtD_SSp_SpDs = 0;    fClosedD_SSp_SpDs = kFALSE;
	}

      if(HistoCode == "D_Adc_EvNb")                            // (ReInitCanvas)
	{	      
	  fImpD_Adc_EvNb = 0;       fCanvD_Adc_EvNb = 0;
	  fPadD_Adc_EvNb = 0;       fMemoPlotD_Adc_EvNb = 0;
	  fMemoColorD_Adc_EvNb = 0; fCanvSameD_Adc_EvNb++;
	  fPavTxtD_Adc_EvNb = 0;    fClosedD_Adc_EvNb = kFALSE;
	}
       
      if(HistoCode == "D_Adc_EvDs")
	{	      
	  fImpD_Adc_EvDs = 0;       fCanvD_Adc_EvDs = 0;
	  fPadD_Adc_EvDs = 0;       fMemoPlotD_Adc_EvDs = 0;
	  fMemoColorD_Adc_EvDs = 0; fCanvSameD_Adc_EvDs++;
	  fPavTxtD_Adc_EvDs = 0;    fClosedD_Adc_EvDs = kFALSE;
	}
          
 
      if(HistoCode == "H_Ped_Date")
	{	      
	  fImpH_Ped_Date = 0;         fCanvH_Ped_Date = 0;
	  fPadH_Ped_Date = 0;         fMemoPlotH_Ped_Date = 0;
	  fMemoColorH_Ped_Date = 0;   fCanvSameH_Ped_Date++;
	  fNbOfListFileH_Ped_Date = 0;fClosedH_Ped_Date = kFALSE;
	}

      if(HistoCode == "H_TNo_Date")
	{	      
	  fImpH_TNo_Date = 0;          fCanvH_TNo_Date = 0;
	  fPadH_TNo_Date = 0;          fMemoPlotH_TNo_Date = 0;
	  fMemoColorH_TNo_Date = 0;    fCanvSameH_TNo_Date++;
	  fNbOfListFileH_TNo_Date = 0; fClosedH_TNo_Date = kFALSE;
	}

      if(HistoCode == "H_MCs_Date")                            // (ReInitCanvas)
	{	      
	  fImpH_MCs_Date = 0;          fCanvH_MCs_Date = 0;
	  fPadH_MCs_Date = 0;          fMemoPlotH_MCs_Date = 0;
	  fMemoColorH_MCs_Date = 0;    fCanvSameH_MCs_Date++;
	  fNbOfListFileH_MCs_Date = 0; fClosedH_MCs_Date = kFALSE;
	}

      
      if(HistoCode == "H_LFN_Date")
	{	      
	  fImpH_LFN_Date = 0;          fCanvH_LFN_Date = 0;
	  fPadH_LFN_Date = 0;          fMemoPlotH_LFN_Date = 0;
	  fMemoColorH_LFN_Date = 0;    fCanvSameH_LFN_Date++;
	  fNbOfListFileH_LFN_Date = 0; fClosedH_LFN_Date = kFALSE;
	}

      if(HistoCode == "H_HFN_Date")
	{	      
	  fImpH_HFN_Date = 0;          fCanvH_HFN_Date = 0;
	  fPadH_HFN_Date = 0;          fMemoPlotH_HFN_Date = 0;
	  fMemoColorH_HFN_Date = 0;    fCanvSameH_HFN_Date++;
	  fNbOfListFileH_HFN_Date = 0; fClosedH_HFN_Date = kFALSE;
	}

      if(HistoCode == "H_SCs_Date")
	{	      
	  fImpH_SCs_Date = 0;          fCanvH_SCs_Date = 0;
	  fPadH_SCs_Date = 0;          fMemoPlotH_SCs_Date = 0;
	  fMemoColorH_SCs_Date = 0;    fCanvSameH_SCs_Date++;
	  fNbOfListFileH_SCs_Date = 0; fClosedH_SCs_Date = kFALSE;
	}

      if(HistoCode == "H_Ped_RuDs")
	{	      
	  fImpH_Ped_RuDs = 0;          fCanvH_Ped_RuDs = 0;
	  fPadH_Ped_RuDs = 0;          fMemoPlotH_Ped_RuDs = 0;
	  fMemoColorH_Ped_RuDs = 0;    fCanvSameH_Ped_RuDs++;
	  fNbOfListFileH_Ped_RuDs = 0; fClosedH_Ped_RuDs = kFALSE;
	}

      if(HistoCode == "H_TNo_RuDs")
	{	      
	  fImpH_TNo_RuDs = 0;          fCanvH_TNo_RuDs = 0;
	  fPadH_TNo_RuDs = 0;          fMemoPlotH_TNo_RuDs = 0;
	  fMemoColorH_TNo_RuDs = 0;    fCanvSameH_TNo_RuDs++;
	  fNbOfListFileH_TNo_RuDs = 0; fClosedH_TNo_RuDs = kFALSE;
	}

      if(HistoCode == "H_MCs_RuDs")                            // (ReInitCanvas)
	{	      
	  fImpH_MCs_RuDs = 0;          fCanvH_MCs_RuDs = 0;
	  fPadH_MCs_RuDs = 0;          fMemoPlotH_MCs_RuDs = 0;
	  fMemoColorH_MCs_RuDs = 0;    fCanvSameH_MCs_RuDs++;
	  fNbOfListFileH_MCs_RuDs = 0; fClosedH_MCs_RuDs = kFALSE;
	}

      
      if(HistoCode == "H_LFN_RuDs")
	{	      
	  fImpH_LFN_RuDs = 0;          fCanvH_LFN_RuDs = 0;
	  fPadH_LFN_RuDs = 0;          fMemoPlotH_LFN_RuDs = 0;
	  fMemoColorH_LFN_RuDs = 0;    fCanvSameH_LFN_RuDs++;
	  fNbOfListFileH_LFN_RuDs = 0; fClosedH_LFN_RuDs = kFALSE;
	}

      if(HistoCode == "H_HFN_RuDs")
	{	      
	  fImpH_HFN_RuDs = 0;          fCanvH_HFN_RuDs = 0;
	  fPadH_HFN_RuDs = 0;          fMemoPlotH_HFN_RuDs = 0;
	  fMemoColorH_HFN_RuDs = 0;    fCanvSameH_HFN_RuDs++;
	  fNbOfListFileH_HFN_RuDs = 0; fClosedH_HFN_RuDs = kFALSE;
	}

      if(HistoCode == "H_SCs_RuDs")
	{	      
	  fImpH_SCs_RuDs = 0;          fCanvH_SCs_RuDs = 0;
	  fPadH_SCs_RuDs = 0;          fMemoPlotH_SCs_RuDs = 0;
	  fMemoColorH_SCs_RuDs = 0;    fCanvSameH_SCs_RuDs++;
	  fNbOfListFileH_SCs_RuDs = 0; fClosedH_SCs_RuDs = kFALSE;
	}
    }
} 
// ------- end of ReInitCanvas(...) ------------

//==========================================================================================
void TEcnaHistos::WriteMatrixAscii(const TString& BetweenWhat,  const TString&   CorOrCov,
				   const Int_t&  StexStinEcna, const Int_t&    MatrixBinIndex,
				   const Int_t&  MatSize,      const TMatrixD& read_matrix)
{   
// write matrix in ascii file

  Int_t ChanNumber = MatrixBinIndex;

  fCnaWrite->RegisterFileParameters(fFapAnaType,    fFapNbOfSamples, 
				    fFapRunNumber,  fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts,
				    fFapStexNumber, fStartDate, fStopDate, fStartTime, fStopTime);

  if( BetweenWhat == fBetweenSamples && CorOrCov == fCorrelationMatrix )
    {
      fCnaWrite->WriteAsciiCorrelationsBetweenSamples(StexStinEcna, ChanNumber, MatSize, read_matrix);
      fAsciiFileName = fCnaWrite->GetAsciiFileName();
    }

  if( BetweenWhat == fBetweenSamples && CorOrCov == fCovarianceMatrix )
    {
      fCnaWrite->WriteAsciiCovariancesBetweenSamples(StexStinEcna, ChanNumber, MatSize, read_matrix);
      fAsciiFileName = fCnaWrite->GetAsciiFileName();
    }
}
//...............................................................................................
void TEcnaHistos::WriteHistoAscii(const TString&   HistoCode, const Int_t& HisSize,
				 const TVectorD& read_histo)
{
// write matrix in ascii file

  fCnaWrite->RegisterFileParameters(fFapAnaType,    fFapNbOfSamples, 
				    fFapRunNumber,  fFapFirstReqEvtNumber, fFapLastReqEvtNumber, fFapReqNbOfEvts,
				    fFapStexNumber, fStartDate, fStopDate,
				    fStartTime,     fStopTime);

  fCnaWrite->WriteAsciiHisto(HistoCode, HisSize, read_histo);
  fAsciiFileName = fCnaWrite->GetAsciiFileName();
}

TString TEcnaHistos::AsciiFileName(){return fAsciiFileName.Data();}

//---------------> messages de rappel pour l'auteur: 
//
//======= A T T E N T I O N ========= A T T E N T I O N ========= A T T E N T I O N ==============!!!!
//      A EVITER ABSOLUMENT quand on est sous TEcnaGui CAR LE cin >> BLOQUE X11
//      puisqu'on n'a pas la main dans la fenetre de compte-rendu de la CNA
//     {Int_t cintoto; cout << "taper 0 pour continuer" << endl; cin >> cintoto;}
//                         *=================================================*
//                         |                                                 |
//++++++++++++++++++++++++|  A T T E N T I O N:  PAS DE TEST "cintoto" ici! |+++++++++++++++++++++!!!!
//                         |                                                 |
//                         *=================================================*
//
// INFO: When "new" fails to allocate the memory for an object, or "new[]" fails to allocate the memory
// for an object array, a std::bad_alloc object is thrown.
// "In GCC, the RTTI mangled name of std::bad_alloc is, I'm guessing, St9bad_alloc."
