//----------Author's Name: B.Fabbro, F.X.Gentit, P.Jarry  DAPNIA/SPP CEA-Saclay
//----------Copyright:Those valid for CEA software
//----------Modified:07/06/2007

#include "Riostream.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaParameters.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEBNumbering.h"

ClassImp(TEBNumbering)
//_____________________________________________________________________________________________
//
//  Building of the numbering for the Ecal Barrel channels
//
//  Convention for the names used here and in the other "TCna" and "TEB" classes:
//
//  Name     Number                     Reference set   Range      Comment
//
//  SMTow  : Tower number               in SuperModule  [1,68]     (phi,eta) progression
//  SMCrys : Crystal number             in SuperModule  [1,1700]   (phi,eta) progression
//  SMEcha : Electronic channel number  in SuperModule  [0,1699]   S shape data reading order 
//   
//  TowEcha: Electronic channel number  in Tower        [0,24]     S shape data reading order  
//

TEBNumbering::TEBNumbering() {
// Constructor without argument: call to method Init()

  Init();
}

TEBNumbering::~TEBNumbering() {
//destructor

  if (fEcal != 0){delete fEcal; fCdelete++;}
  if (fT2dSMCrys  != 0){delete [] fT2dSMCrys;  fCdelete++;}
  if (fT1dSMCrys  != 0){delete [] fT1dSMCrys;  fCdelete++;}
  if (fT1dSMTow   != 0){delete [] fT1dSMTow;   fCdelete++;}
  if (fT1dTowEcha != 0){delete [] fT1dTowEcha; fCdelete++;}
}

void TEBNumbering::Init() {
//Set default values and build crystal numbering table

  //.............................. Initialisations

  fTTBELL = '\007';

  fT2dSMCrys  = 0;
  fT1dSMCrys  = 0;
  fT1dSMTow   = 0;
  fT1dTowEcha = 0;

  fCodeChNumberingLvrbBot = "bottom";
  fCodeChNumberingLvrbTop = "top";

  fEcal = new TEBParameters();   fCnew++;

  fMaxSMInBarPlus  = fEcal->fMaxSMInBarPlus;
  fMaxSMInBarMinus = fEcal->fMaxSMInBarMinus;
  fMaxSMInBarrel   = fEcal->fMaxSMInBarrel;

  fMaxTowEtaInSM = fEcal->fMaxTowEtaInSM;
  fMaxTowPhiInSM = fEcal->fMaxTowPhiInSM;
  fMaxTowInSM    = fEcal->fMaxTowInSM;

  fMaxCrysEtaInTow = fEcal->fMaxCrysEtaInTow;
  fMaxCrysPhiInTow = fEcal->fMaxCrysPhiInTow;
  fMaxCrysInTow    = fEcal->fMaxCrysInTow;

  fMaxCrysEtaInSM = fEcal->fMaxCrysEtaInSM;
  fMaxCrysPhiInSM = fEcal->fMaxCrysPhiInSM;
  fMaxCrysInSM    = fEcal->fMaxCrysInSM;

  fMaxSampADC = fEcal->fMaxSampADC;

  fMaxEvtsInBurstPedRun = fEcal->fMaxEvtsInBurstPedRun;

  //...................................................................
  TCnaParameters* MyParameters = new TCnaParameters();   fCnew++;

  fCodePrintNoComment   = MyParameters->GetCodePrint("NoComment");
  fCodePrintWarnings    = MyParameters->GetCodePrint("Warnings ");
  fCodePrintComments    = MyParameters->GetCodePrint("Comments");
  fCodePrintAllComments = MyParameters->GetCodePrint("AllComments");

  delete MyParameters;                                   fCdelete++;

  fFlagPrint = fCodePrintWarnings;

  //.............................. Call to BuildCrysTable()
  BuildCrysTable();
}
// end of Init()

//====================================================================================================
//
//               SMCrys <-> (SMTow, TowEcha) correspondance table (from Patrick Jarry)
//
//====================================================================================================
void TEBNumbering::BuildCrysTable()
{
// Build the correspondance table: SMCrys <-> (SMTow, TowEcha)
//
//  From CMS Internal Note  "CMS ECAL Barrel channel numbering"
//
//       Name      Number                     Reference set   Range        Comment
//
//       SMTow  : Tower number               in SuperModule  [1,68]      (phi,eta) progression
//       SMCrys : Crystal number             in SuperModule  [1,1700]    (phi,eta) progression
//       SMEcha : Electronic channel number  in SuperModule  [0,1699]    S shape data reading order 
//   
//       TowEcha: Electronic channel number  in Tower        [0,24]      S shape data reading order  
//
//
//   fill the 2D array:  fT2dSMCrys[n_SMTow][n_TowEcha]
//
//   and the 1d arrays:  fT1dSMTow[i_SMCrys] and fT1dTowEcha[i_SMCrys]
//
//-----------------------------------------------------------------------------------------------------

  if ( fT2dSMCrys == 0 )
    {
      Int_t MaxSMTow   = fEcal->MaxTowInSM();
      Int_t MaxTowEcha = fEcal->MaxCrysInTow();
      Int_t MaxSMCrys  = fEcal->MaxCrysInSM();
      
      //................... Allocation CrysNumbersTable
      
      fT2dSMCrys = new Int_t*[MaxSMTow];                         fCnew++;  
      fT1dSMCrys = new  Int_t[MaxSMTow*MaxTowEcha];              fCnew++;   
      for(Int_t i_SMTow = 0 ; i_SMTow < MaxSMTow ; i_SMTow++){
	fT2dSMCrys[i_SMTow] = &fT1dSMCrys[0] + i_SMTow*MaxTowEcha;}
      
      fT1dSMTow   = new Int_t[MaxSMCrys];                        fCnew++;
      fT1dTowEcha = new Int_t[MaxSMCrys];                        fCnew++;

      if(fFlagPrint == fCodePrintAllComments)
      	{
	  cout << "*TEBNumbering::BuilCrysdTable()> Allocation of CrysNumbersTable done."
	       << " Max nb of towers = "    << MaxSMTow
	       << ", max nb of crystals = " << MaxTowEcha << endl;
	}
      
      Int_t m2  = (Int_t)2;
      Int_t m26 = (Int_t)26;

      //      Int_t jch_type[2][26];
      Int_t** jch_type    = new Int_t*[m2];                      fCnew++;  
      Int_t*  jch_type_d1 = new Int_t[m2*m26];                   fCnew++;
      for(Int_t i_m2 = 0 ; i_m2 < m2 ; i_m2++){
	jch_type[i_m2] = &jch_type_d1[0] + i_m2*m26;}

      for(Int_t k=25;k>=21;k--){jch_type[0][k] = 25-k;}  //  k = 25,24,23,22,21 -> jch_type[0][k] = 0,1,2,3,4
      for(Int_t k=16;k<=20;k++){jch_type[0][k] = k-16;}  //  k = 16,17,18,19,20 -> jch_type[0][k] = 0,1,2,3,4
      for(Int_t k=15;k>=11;k--){jch_type[0][k] = 15-k;}  //  k = 15,14,12,13,11 -> jch_type[0][k] = 0,1,2,3,4
      for(Int_t k=6; k<=10;k++){jch_type[0][k] = k-6;}   //  k =  6, 7, 8, 9,10 -> jch_type[0][k] = 0,1,2,3,4
      for(Int_t k=5; k>=1; k--){jch_type[0][k] = 5-k;}   //  k =  5, 4, 3, 2, 1 -> jch_type[0][k] = 0,1,2,3,4
      
      for(Int_t k=1; k<=5; k++){jch_type[1][k] = k-1;}   //  k =  1, 2, 3, 4, 5 -> jch_type[1][k] = 0,1,2,3,4
      for(Int_t k=10;k>=6; k--){jch_type[1][k] = 10-k;}  //  k = 10, 9, 8, 7, 6 -> jch_type[1][k] = 0,1,2,3,4
      for(Int_t k=11;k<=15;k++){jch_type[1][k] = k-11;}  //  k = 11,12,13,14,15 -> jch_type[1][k] = 0,1,2,3,4
      for(Int_t k=20;k>=16;k--){jch_type[1][k] = 20-k;}  //  k = 20,19,18,17,16 -> jch_type[1][k] = 0,1,2,3,4
      for(Int_t k=21;k<=25;k++){jch_type[1][k] = k-21;}  //  k = 21,22,23,24,25 -> jch_type[1][k] = 0,1,2,3,4

      //      Int_t ich_type[2][26]; 
      Int_t** ich_type    = new Int_t*[m2];                      fCnew++;  
      Int_t*  ich_type_d1 = new Int_t[m2*m26];                   fCnew++;
      for(Int_t i_m2 = 0 ; i_m2 < m2 ; i_m2++){
	ich_type[i_m2] = &ich_type_d1[0] + i_m2*m26;}
     
      for(Int_t k=25;k>=21;k--){ich_type[0][k] = 0;}     //  k = 25,24,23,22,21 -> ich_type[0][k] = 0
      for(Int_t k=16;k<=20;k++){ich_type[0][k] = 1;}     //  k = 16,17,18,19,20 -> ich_type[0][k] = 1
      for(Int_t k=15;k>=11;k--){ich_type[0][k] = 2;}     //  k = 15,14,12,13,11 -> ich_type[0][k] = 2
      for(Int_t k=6; k<=10;k++){ich_type[0][k] = 3;}     //  k =  6, 7, 8, 9,10 -> ich_type[0][k] = 3
      for(Int_t k=5; k>=1; k--){ich_type[0][k] = 4;}     //  k =  5, 4, 3, 2, 1 -> ich_type[0][k] = 4
      
      for(Int_t k=1; k<=5; k++){ich_type[1][k] = 0;}     //  k =  1, 2, 3, 4, 5 -> ich_type[1][k] = 0
      for(Int_t k=10;k>=6; k--){ich_type[1][k] = 1;}     //  k = 10, 9, 8, 7, 6 -> ich_type[1][k] = 1
      for(Int_t k=11;k<=15;k++){ich_type[1][k] = 2;}     //  k = 11,12,13,14,15 -> ich_type[1][k] = 2
      for(Int_t k=20;k>=16;k--){ich_type[1][k] = 3;}     //  k = 20,19,18,17,16 -> ich_type[1][k] = 3
      for(Int_t k=21;k<=25;k++){ich_type[1][k] = 4;}     //  k = 21,22,23,24,25 -> ich_type[1][k] = 4
      
      //     Int_t type[17]={0,0,0,1,1, 0,0,1,1, 0,0,1,1, 0,0,1,1};
      Int_t m17 =17;
      Int_t* type = new Int_t[m17];                             fCnew++; 

      //  0 -> LVRB at the bottom, 1 -> LVRB at the top
      type[0] = 0;                   // M1
      type[1] = 0;
      type[2] = 0;
      type[3] = 1;
      type[4] = 1;

      type[5] = 0;                   // M2
      type[6] = 0;
      type[7] = 1;
      type[8] = 1;

      type[9]  = 0;                  // M3
      type[10] = 0;
      type[11] = 1;
      type[12] = 1;

      type[13] = 0;                  // M4
      type[14] = 0;
      type[15] = 1;
      type[16] = 1;

      for(Int_t tow=0; tow<MaxSMTow; tow++)                    //  tow  = 0 to 67   (MaxSMTow = 68)
	{
	  for(Int_t ic=1; ic<=MaxTowEcha; ic++)                //  ic   = 1 to 25   (MaxTowEcha = 25) 
	    {
	      Int_t jtow = tow % 4;                            //  jtow = 0,1,2,3 
	      Int_t itow = tow / 4;                            //  itow = 0 to 16

	      Int_t icrys = itow*5 + ich_type[type[itow]][ic]; //  type[0->16] = 0,1 ,
	                                                       //  ich_type[0->1][1->25] = 0,1,2,3,4
	                                                       //  icrys = 0 to 84  (=> eta)

	      Int_t jcrys = jtow*5 + jch_type[type[itow]][ic]; //  type[0->16] = 0,1 ,
	                                                       //  jch_type[0->1][1->25] = 0,1,2,3,4
	                                                       //  jcrys = 0 to 19  (=> phi)

	      Int_t i_SMCrys = icrys*20+jcrys+1;               //  i_SMCrys = 1 to 1700

	      fT2dSMCrys[tow][ic-1]   = i_SMCrys;
	      fT1dSMTow[i_SMCrys-1]   = tow+1;
	      fT1dTowEcha[i_SMCrys-1] = ic-1;
	    }
	}
      // cout << "#TEBNumbering::TBuildCrysTable()> Crys Table Building done" << endl;

      delete [] jch_type;                       fCdelete++;
      delete [] jch_type_d1;                    fCdelete++;
      delete [] ich_type;                       fCdelete++;
      delete [] ich_type_d1;                    fCdelete++;
      delete [] type;                           fCdelete++;
    }
  else
    {
      // cout << "#TEBNumbering::TBuildCrysTable()> No Building of Crys Table since it is already done " << endl;
    }
}

//===============================================================================
//
//        GetSMCrysFromSMTowAndTowEcha
//        GetSMCrysFromSMEcha
//
//===============================================================================
Int_t TEBNumbering::GetSMCrysFromSMTowAndTowEcha(const Int_t& i_SMTow,
						  const Int_t& i_TowEcha)
{
//get crystal number in SM from tower number in SM
// and from Electronic Channel number in tower

  Int_t SMcrys = 0;

  if( fT2dSMCrys == 0 )
    {
      BuildCrysTable();
      if(fFlagPrint == fCodePrintAllComments)
	{cout << "*TEBNumbering::GetSMCrysFromSMTowAndTowEcha> SMtower: " << i_SMTow
	      << ", Channel: " << i_TowEcha << endl;}
    }
  
  if (i_SMTow >= 1 && i_SMTow <= fEcal->fMaxTowInSM)
    {
      if (i_TowEcha >=0 && i_TowEcha < fEcal->fMaxCrysInTow)
	{
	  SMcrys = fT2dSMCrys[i_SMTow-1][i_TowEcha];
	}
      else
	{
	  SMcrys = -2;   // Electronic Cnannel in Tower out of range 
	  cout << "!TEBNumbering::GetSMCrysFromSMTowAndTowEcha(...)> Electronic Channel in Tower out of range."
	       << " i_TowEcha = " << i_TowEcha << fTTBELL << endl;
	}
    }
  else
    {
      SMcrys = -3;   // Tower number in SM out of range
      cout << "!TEBNumbering::GetSMCrysFromSMTowAndTowEcha(...)> Tower number in SM out of range."
	   << " i_SMTow = " << i_SMTow << fTTBELL << endl;
    }

  return SMcrys;
}

Int_t TEBNumbering::GetSMCrysFromSMEcha(const Int_t& i_SMEcha)
{
//get crystal number in SM from electronic channel number in SM

  Int_t SMcrys = 0;
  
  Int_t i_SMTow    = GetTowEchaFromSMEcha(i_SMEcha);
  Int_t i_TowEcha  = GetSMTowFromSMEcha(i_SMEcha); 

  SMcrys = GetSMCrysFromSMTowAndTowEcha(i_SMTow, i_TowEcha);

  return SMcrys;
}

//===============================================================================
//
//                GetTowEchaFromSMCrys, GetSMTowFromSMCrys
//
//===============================================================================

Int_t TEBNumbering::GetTowEchaFromSMCrys(const Int_t& i_SMCrys)
{
// get Electronic Channel number in Tower from Crystal number in SuperModule

  Int_t i_TowEcha = -1;

  if( i_SMCrys >= 1 && i_SMCrys <= fEcal->fMaxCrysInSM )
    {
      i_TowEcha = fT1dTowEcha[i_SMCrys-1];
    }
  else
    {
      i_TowEcha = -2;
      cout << "!TEBNumbering::GetTowEchaFromSMCrys(...)> Crystal number in SM out of range."
	   << " i_SMCrys = " << i_SMCrys << fTTBELL << endl;
    }
  return i_TowEcha;
}

Int_t TEBNumbering::GetSMTowFromSMCrys(const Int_t& i_SMCrys)
{
// get Tower number in SM from Crystal number in SuperModule

  Int_t i_SMTow = 0;
  
  if( i_SMCrys >= 1 && i_SMCrys <= fEcal->fMaxCrysInSM )
    {
      i_SMTow = fT1dSMTow[i_SMCrys-1];
    }
  else
    {
      i_SMTow = -1;
      cout << "!TEBNumbering::GetSMTowFromSMCrys(...)> Crystal number in SM out of range."
	   << " i_SMCrys = " << i_SMCrys << fTTBELL << endl;
    }
  return i_SMTow;
}

//===============================================================================
//
//          GetTowEchaFromSMEcha
//          GetSMTowFromSMEcha
//
//===============================================================================

Int_t TEBNumbering::GetTowEchaFromSMEcha(const Int_t& i_SMEcha)
{
//get electronic channel number in tower from electronic channel number in SM

  Int_t SMTowerNumber = i_SMEcha/fEcal->MaxCrysInTow()+1;
  Int_t TowerEcha     = i_SMEcha - fEcal->MaxCrysInTow()*(SMTowerNumber-1);

  return TowerEcha;
}

Int_t TEBNumbering::GetSMTowFromSMEcha(const Int_t& i_SMEcha)
{
//get tower number from electronic channel number in SM

  Int_t SMTowerNumber = i_SMEcha/fEcal->MaxCrysInTow()+1;

  return SMTowerNumber;
}

//===========================================================================
//
//                        GetTowerLvrbType
//
//===========================================================================  

TString  TEBNumbering::GetTowerLvrbType(const Int_t& SMtower)
{
//gives the LVRB type of the crystal numbering of tower

  TString type = fCodeChNumberingLvrbTop;   // => default value

  if (SMtower >=  1 && SMtower <= 12){type = fCodeChNumberingLvrbBot;}
  if (SMtower >= 21 && SMtower <= 28){type = fCodeChNumberingLvrbBot;}
  if (SMtower >= 37 && SMtower <= 44){type = fCodeChNumberingLvrbBot;}
  if (SMtower >= 53 && SMtower <= 60){type = fCodeChNumberingLvrbBot;}

  return type;
}

//===========================================================================
//
//                    GetSMHalfBarrel    
//
//===========================================================================  

TString  TEBNumbering::GetSMHalfBarrel(const Int_t& SMNumber)
{
//gives the half-barrel of the super-module ("barrel+" or "barrel-")

  TString type = "barrel+";   // => default value

  if ( SMNumber >=  1                       && SMNumber <= fEcal->MaxSMInBarPlus() ){type = "barrel+";}
  if ( SMNumber >   fEcal->MaxSMInBarPlus() && SMNumber <= fEcal->MaxSMInBarrel()  ){type = "barrel-";}

  return type;
}

//==============================================================================
//
//       GetEta, GetEtaMin, GetEtaMax,  GetIEtaMin, GetIEtaMax
//
//==============================================================================
Double_t TEBNumbering::GetEta(const Int_t& super_module, const Int_t& SMtower,
			       const Int_t& i_TowEcha)
{
//Gives Eta for a given (super_module, SMtower, i_TowEcha)
  
  Double_t eta = (Double_t)0.;

  Int_t max_crys_eta_in_tower = fEcal->MaxCrysEtaInTow();
  Int_t max_tow_eta_in_sm     = fEcal->MaxTowEtaInSM();
  Int_t max_sm_in_barrel      = fEcal->MaxSMInBarrel();

  if ( super_module >= 1 && super_module <= max_sm_in_barrel )
    {
      for (Int_t i_tow_eta = 0; i_tow_eta < max_tow_eta_in_sm; i_tow_eta++)
	{
	  Int_t i_crys_eta_min =    (Int_t)(1 + i_tow_eta*(max_crys_eta_in_tower-1));
	  Int_t i_crys_eta_max =    (Int_t)((i_tow_eta+1)*(max_crys_eta_in_tower-1));
	  Int_t i_crys_eta     =    (Int_t)(i_tow_eta*max_crys_eta_in_tower);
	  Double_t d_echa_eta  = (Double_t)(i_TowEcha/max_crys_eta_in_tower);

	    if ( SMtower >= i_crys_eta_min && SMtower <= i_crys_eta_max )
	      {
		if (GetTowerLvrbType(SMtower) == fCodeChNumberingLvrbTop)
		{eta = (Double_t)(i_crys_eta) + d_echa_eta;}
		if (GetTowerLvrbType(SMtower) == fCodeChNumberingLvrbBot)
		{eta = (Double_t)(i_crys_eta+max_crys_eta_in_tower-1)-d_echa_eta;} 
	      }
	}      
      if ( GetSMHalfBarrel(super_module) == "barrel-" ){eta = - eta;}      
    }
  else
    {
      cout << "TEBNumbering::GetEta(...)> " << super_module << ": invalid super-module number."
	   << fTTBELL << endl; 
    }

  return eta;
}
//-------------------------------------------------------------------------------------
Double_t TEBNumbering::GetEtaMin(const Int_t& super_module, const Int_t& SMtower)
{
//Gives EtaMin for a given Tower

  Int_t max_tow_eta_in_sm     = fEcal->MaxTowEtaInSM();
  Int_t max_crys_eta_in_tower = fEcal->MaxCrysEtaInTow();

  Double_t eta_min = (Double_t)0.;

  for (Int_t i_tow_eta = 0; i_tow_eta < max_tow_eta_in_sm; i_tow_eta++)
    {
      Int_t i_crys_eta_min =    (Int_t)(1 + i_tow_eta*(max_crys_eta_in_tower-1));
      Int_t i_crys_eta_max =    (Int_t)((i_tow_eta+1)*(max_crys_eta_in_tower-1));
      Int_t i_crys_eta     =    (Int_t)(i_tow_eta*max_crys_eta_in_tower);

      if ( SMtower >= i_crys_eta_min && SMtower <= i_crys_eta_max )
	{
	  if (GetSMHalfBarrel(super_module) == "barrel+")
	    {eta_min = (Double_t)i_crys_eta;}
	  if (GetSMHalfBarrel(super_module) == "barrel-")
	    {eta_min = -(Double_t)(i_crys_eta + max_crys_eta_in_tower);}
	}
    }

  return eta_min;
}
//------------------------------------------------------------------------------------
Double_t TEBNumbering::GetEtaMax(const Int_t& super_module, const Int_t& SMtower)
{
//Gives EtaMax for a given Tower

  Int_t max_tow_eta_in_sm     = fEcal->MaxTowEtaInSM();
  Int_t max_crys_eta_in_tower = fEcal->MaxCrysEtaInTow();

  Double_t eta_max = (max_crys_eta_in_tower-1);

  for (Int_t i_tow_eta = 0; i_tow_eta < max_tow_eta_in_sm; i_tow_eta++)
    {
      Int_t i_crys_eta_min =    (Int_t)(1 + i_tow_eta*(max_crys_eta_in_tower-1));
      Int_t i_crys_eta_max =    (Int_t)((i_tow_eta+1)*(max_crys_eta_in_tower-1));
      Int_t i_crys_eta     =    (Int_t)(i_tow_eta*max_crys_eta_in_tower);

      if ( SMtower >= i_crys_eta_min && SMtower <= i_crys_eta_max )
	{
	  if (GetSMHalfBarrel(super_module) == "barrel+")
	    {eta_max = (Double_t)(i_crys_eta + max_crys_eta_in_tower);}
	  if (GetSMHalfBarrel(super_module) == "barrel-")
	    {eta_max = -(Double_t)i_crys_eta;}
	}
    }

  return eta_max;
}

Double_t TEBNumbering::GetIEtaMin(const Int_t& super_module, const Int_t& SMtower)
{
//Gives IEtaMin for a given (super_module, SMtower)

  Double_t i_eta_min = (Int_t)0.; 
   
  if(GetSMHalfBarrel(super_module) == "barrel+")
    {i_eta_min = (Double_t)GetEtaMin(super_module, SMtower)+(Double_t)0.5;}
  if(GetSMHalfBarrel(super_module) == "barrel-")
    {i_eta_min = (Double_t)GetEtaMin(super_module, SMtower)-(Double_t)0.5;}

  return i_eta_min;
}

Double_t TEBNumbering::GetIEtaMax(const Int_t& super_module, const Int_t& SMtower)
{
//Gives IEtaMax for a given (super_module, SMtower)

  Double_t i_eta_max = (Int_t)0.; 
   
  if(GetSMHalfBarrel(super_module) == "barrel+")
    {i_eta_max = (Double_t)GetEtaMax(super_module, SMtower)+(Double_t)0.5;}
  if(GetSMHalfBarrel(super_module) == "barrel-")
    {i_eta_max = (Double_t)GetEtaMax(super_module, SMtower)-(Double_t)0.5;}

  return i_eta_max;
}

Double_t TEBNumbering::GetIEtaMin(const Int_t& super_module)
{
//Gives IEtaMin for a given (super_module)

  Double_t i_eta_min = (Int_t)0.;

  Int_t SMtowerPlus  = (Int_t)1;   
  Int_t SMtowerMinus = (Int_t)68;   

  if( GetSMHalfBarrel(super_module) == "barrel+")
    {i_eta_min = (Double_t)GetIEtaMin(super_module, SMtowerPlus );}
  if( GetSMHalfBarrel(super_module) == "barrel-")
    {i_eta_min = (Double_t)GetIEtaMin(super_module, SMtowerMinus);}

  return i_eta_min;
}

Double_t TEBNumbering::GetIEtaMax(const Int_t& super_module)
{
//Gives IEtaMax for a given (super_module)

  Double_t i_eta_max = (Int_t)0.; 

  Int_t SMtowerPlus  = (Int_t)68;   
  Int_t SMtowerMinus = (Int_t)1;   
   
  if( GetSMHalfBarrel(super_module) == "barrel+")
    {i_eta_max = (Double_t)GetIEtaMax(super_module, SMtowerPlus );}
  if( GetSMHalfBarrel(super_module) == "barrel-")
    {i_eta_max = (Double_t)GetIEtaMax(super_module, SMtowerMinus);}

  return i_eta_max;
}

//==============================================================================
//
//    GetSMCentralPhi, GetPhi, GetPhiMin, GetPhiMax, GetIPhiMin, GetIPhiMax
//
//==============================================================================
Double_t TEBNumbering::GetSMCentralPhi(const Int_t& super_module)
{
//Gives the central phi value of the SuperModule

  Double_t central_phi = (Double_t)20.;    //  DEFAULT = SM1
  
  if (GetSMHalfBarrel(super_module) == "barrel+")
    {
      if( (super_module >= 1) && (super_module <= 18) ){central_phi = (Double_t)20.*super_module;}
      // if( super_module == 18 )                         {central_phi = (Double_t)0.;} 
    }

  if (GetSMHalfBarrel(super_module) == "barrel-")
    {
      if((super_module >= 19)&&(super_module <= 26))
	{central_phi = (Double_t)160+(Double_t)20.*(19-super_module);}
      if((super_module >= 27)&&(super_module <= 36))
	{central_phi = (Double_t)340+(Double_t)20.*(28-super_module);}
    }

  return central_phi;
}

Double_t TEBNumbering::GetPhi(const Int_t& super_module,
			       const Int_t& SMtower, const Int_t& i_TowEcha)
{
//Gives Phi for a given (super_module, SMtower, i_TowEcha)

  Double_t phi = (Double_t)0.;

  Double_t phi_start = GetSMCentralPhi(super_module);

  Int_t max_crys_phi_in_tower = fEcal->MaxCrysPhiInTow();
  Int_t max_sm_in_barrel      = fEcal->MaxSMInBarrel();

  Int_t rest_temp =(Int_t)(SMtower%(max_crys_phi_in_tower-1));

  if ( super_module >= 1 && super_module <= max_sm_in_barrel )
    {
      if (GetTowerLvrbType(SMtower) == fCodeChNumberingLvrbTop)
	{
	  if( rest_temp == 1 ){phi = phi_start -(Double_t)10. + (Double_t)15.;}
	  if( rest_temp == 2 ){phi = phi_start -(Double_t)10. + (Double_t)10.;}
	  if( rest_temp == 3 ){phi = phi_start -(Double_t)10. + (Double_t)5.;}
	  if( rest_temp == 0 ){phi = phi_start -(Double_t)10. + (Double_t)0.;}
	  
	  if( i_TowEcha ==  4 || i_TowEcha ==  5 || i_TowEcha == 14 || i_TowEcha == 15 || i_TowEcha == 24 )
	    {phi = phi+0;}
 
	  if( i_TowEcha ==  3 || i_TowEcha ==  6 || i_TowEcha == 13 || i_TowEcha == 16 || i_TowEcha == 23 )
	    {phi = phi+1;} 

	  if( i_TowEcha ==  2 || i_TowEcha ==  7 || i_TowEcha == 12 || i_TowEcha == 17 || i_TowEcha == 22 )
	    {phi = phi+2;}
 
	  if( i_TowEcha ==  1 || i_TowEcha ==  8 || i_TowEcha == 11 || i_TowEcha == 18 || i_TowEcha == 21 )
	    {phi = phi+3;}
 
	  if( i_TowEcha ==  0 || i_TowEcha ==  9 || i_TowEcha == 10 || i_TowEcha == 19 || i_TowEcha == 20 )
	    {phi = phi+4;} 
	}
      if (GetTowerLvrbType(SMtower) == fCodeChNumberingLvrbBot)
	{
	  if( rest_temp == 1 ) {phi = phi_start -(Double_t)10. + (Double_t)15.;}
	  if( rest_temp == 2 ) {phi = phi_start -(Double_t)10. + (Double_t)10.;}
	  if( rest_temp == 3 ) {phi = phi_start -(Double_t)10. + (Double_t)5.;}
	  if( rest_temp == 0 ) {phi = phi_start -(Double_t)10. + (Double_t)0.;}
	  
	  if( i_TowEcha == 20 || i_TowEcha == 19 || i_TowEcha == 10 || i_TowEcha ==  9 || i_TowEcha ==  0 )
	    {phi = phi+0;}
 
	  if( i_TowEcha == 21 || i_TowEcha == 18 || i_TowEcha == 11 || i_TowEcha ==  8 || i_TowEcha ==  1 )
	    {phi = phi+1;}
 
	  if( i_TowEcha == 22 || i_TowEcha == 17 || i_TowEcha == 12 || i_TowEcha ==  7 || i_TowEcha ==  2 )
	    {phi = phi+2;}

	  if( i_TowEcha == 23 || i_TowEcha == 16 || i_TowEcha == 13 || i_TowEcha ==  6 || i_TowEcha ==  3 )
	    {phi = phi+3;}

	  if( i_TowEcha == 24 || i_TowEcha == 15 || i_TowEcha == 14 || i_TowEcha ==  5 || i_TowEcha ==  4 )
	    {phi = phi+4;} 
	}
    }
  else
    {
      cout << "TEBNumbering::GetPhi(...)> " << super_module << ": invalid super-module number."
	   << fTTBELL << endl; 
    }

  return phi;
}
//-----------------------------------------------------------------------------------------
Double_t TEBNumbering::GetPhiMin(const Int_t& super_module, const Int_t& SMtower)
{
//Gives PhiMin for a given Tower

  Int_t max_crys_phi_in_tower = fEcal->MaxCrysPhiInTow();

  Double_t phi_min = (Double_t)0.;     // DEFAULT
  Double_t phi_start = GetSMCentralPhi(super_module);

  Int_t rest_temp =(Int_t)(SMtower%(max_crys_phi_in_tower-1));

  if(GetSMHalfBarrel(super_module) == "barrel+")
    {
      if ( rest_temp == 1 ) {phi_min = phi_start + (Double_t)5.;}
      if ( rest_temp == 2 ) {phi_min = phi_start + (Double_t)0.;}
      if ( rest_temp == 3 ) {phi_min = phi_start - (Double_t)5.;}
      if ( rest_temp == 0 ) {phi_min = phi_start - (Double_t)10.;}
    }
  if(GetSMHalfBarrel(super_module) == "barrel-")
    {
      if ( rest_temp == 0 ) {phi_min = phi_start + (Double_t)5.;}
      if ( rest_temp == 3 ) {phi_min = phi_start + (Double_t)0.;}
      if ( rest_temp == 2 ) {phi_min = phi_start - (Double_t)5.;}
      if ( rest_temp == 1 ) {phi_min = phi_start - (Double_t)10.;}
    }

  return phi_min;
}
//-----------------------------------------------------------------------------------------
Double_t TEBNumbering::GetPhiMax(const Int_t& super_module, const Int_t& SMtower)
{
//Gives PhiMax for a given Tower

  Int_t max_crys_phi_in_tower = fEcal->MaxCrysPhiInTow();

  Double_t phi_max = (Double_t)20.;     // DEFAULT
  Double_t phi_start = GetSMCentralPhi(super_module);

  Int_t rest_temp =(Int_t)(SMtower%(max_crys_phi_in_tower-1));

  if(GetSMHalfBarrel(super_module) == "barrel+")
    {
      if ( rest_temp == 1 ) {phi_max = phi_start + (Double_t)10.;}
      if ( rest_temp == 2 ) {phi_max = phi_start + (Double_t)5.;}
      if ( rest_temp == 3 ) {phi_max = phi_start - (Double_t)0.;}
      if ( rest_temp == 0 ) {phi_max = phi_start - (Double_t)5.;}
    }

  if(GetSMHalfBarrel(super_module) == "barrel-")
    {
      if ( rest_temp == 0 ) {phi_max = phi_start + (Double_t)10.;}
      if ( rest_temp == 3 ) {phi_max = phi_start + (Double_t)5.;}
      if ( rest_temp == 2 ) {phi_max = phi_start - (Double_t)0.;}
      if ( rest_temp == 1 ) {phi_max = phi_start - (Double_t)5.;}
    }

  return phi_max;
}
//-----------------------------------------------------------------------------------------
Double_t TEBNumbering::GetPhiMin(const Int_t& super_module)
{
//Gives PhiMin for a given SuperModule

  Double_t phi_min = GetSMCentralPhi(super_module) - (Double_t)10.;

  return phi_min;
}
//-----------------------------------------------------------------------------------------
Double_t TEBNumbering::GetPhiMax(const Int_t& super_module)
{
//Gives PhiMax for a given SuperModule

  Double_t phi_max = GetSMCentralPhi(super_module) + (Double_t)10.;

  return phi_max;
}
//-----------------------------------------------------------------------------------------
Double_t TEBNumbering::GetJPhiMin(const Int_t& super_module, const Int_t& SMtower)
{
//Gives JPhiMin for a given Tower

  Double_t j_phi_min = (Double_t)1.;
  Int_t max_crys_phi_in_tower = fEcal->MaxCrysPhiInTow();
  Int_t rest_temp =(Int_t)(SMtower%(max_crys_phi_in_tower-1));
  
  if ( rest_temp == 1 ){j_phi_min = (Double_t) 1. - (Double_t)0.5;}
  if ( rest_temp == 2 ){j_phi_min = (Double_t) 6. - (Double_t)0.5;}
  if ( rest_temp == 3 ){j_phi_min = (Double_t)11. - (Double_t)0.5;}
  if ( rest_temp == 0 ){j_phi_min = (Double_t)16. - (Double_t)0.5;}

 // Double_t j_phi_min = (Double_t)1.;       // DEFAULT

  // if(GetSMHalfBarrel(super_module) == "barrel+")
  //  {j_phi_min = GetPhiMin(super_module, SMtower)-(Double_t)9.5;}
  // if(GetSMHalfBarrel(super_module) == "barrel-")
  //  {j_phi_min = ((Double_t)360.-GetPhiMin(super_module, SMtower))+(Double_t)10.5;}

  return j_phi_min;
}
//-----------------------------------------------------------------------------------------
Double_t TEBNumbering::GetJPhiMax(const Int_t& super_module, const Int_t& SMtower)
{
//Gives JPhiMax for a given Tower

  Double_t j_phi_max = (Double_t)20.;
  Int_t max_crys_phi_in_tower = fEcal->MaxCrysPhiInTow();
  Int_t rest_temp =(Int_t)(SMtower%(max_crys_phi_in_tower-1));
  
  if ( rest_temp == 1 ){j_phi_max = (Double_t) 5. + (Double_t)0.5;}
  if ( rest_temp == 2 ){j_phi_max = (Double_t)10. + (Double_t)0.5;}
  if ( rest_temp == 3 ){j_phi_max = (Double_t)15. + (Double_t)0.5;}
  if ( rest_temp == 0 ){j_phi_max = (Double_t)20. + (Double_t)0.5;}

  // Double_t j_phi_max = (Double_t)360.;    // DEFAULT

  // if(GetSMHalfBarrel(super_module) == "barrel+")
  //   {j_phi_max = GetPhiMax(super_module, SMtower)-(Double_t)9.5;}
  // if(GetSMHalfBarrel(super_module) == "barrel-")
  //  {j_phi_max = ((Double_t)360.-GetPhiMax(super_module, SMtower))+(Double_t)10.5;}

  return j_phi_max;
}

//-----------------------------------------------------------------------------------------
Double_t TEBNumbering::GetJPhiMin(const Int_t& super_module)
{
//Gives JPhiMin for a given SuperModule

  Double_t j_phi_min = (Double_t)1. - (Double_t)0.5;

  // Double_t j_phi_min = (Double_t)1.;       // DEFAULT

  // Int_t SMtowerPlus  = (Int_t)4;
  // Int_t SMtowerMinus = (Int_t)1;

  // if( GetSMHalfBarrel(super_module) == "barrel+"){j_phi_min = GetJPhiMin(super_module, SMtowerPlus );}
  // if( GetSMHalfBarrel(super_module) == "barrel-"){j_phi_min = GetJPhiMin(super_module, SMtowerMinus);}

  return j_phi_min;
}
//-----------------------------------------------------------------------------------------
Double_t TEBNumbering::GetJPhiMax(const Int_t& super_module)
{
//Gives JPhiMax for a given SuperModule

  Double_t j_phi_max = (Double_t)20. + (Double_t)0.5;

  // Double_t j_phi_max = (Double_t)360.;    // DEFAULT

  // Int_t SMtowerPlus  = (Int_t)1;
  // Int_t SMtowerMinus = (Int_t)4;

  // if( GetSMHalfBarrel(super_module) == "barrel+"){j_phi_max = GetJPhiMax(super_module, SMtowerPlus );}
  // if( GetSMHalfBarrel(super_module) == "barrel-"){j_phi_max = GetJPhiMax(super_module, SMtowerMinus);}

  return j_phi_max;
}

//==============================================================================
//
//   GetXDirection, GetYDirection, GetJYDirection
//
//==============================================================================

//------------------------------------------------------------------
TString TEBNumbering::GetXDirection(const Int_t& SMNumber)
{
  TString xdirection = "x";      // DEFAULT

  if( GetSMHalfBarrel(SMNumber) == "barrel+" ){xdirection = "x";}
  if( GetSMHalfBarrel(SMNumber) == "barrel-" ){xdirection = "x";}

  return xdirection;
}
//---------------------------------------------------------
TString TEBNumbering::GetYDirection(const Int_t& SMNumber)
{
  TString ydirection = "-x";      // DEFAULT

  if( GetSMHalfBarrel(SMNumber) == "barrel+" ){ydirection = "-x";}
  if( GetSMHalfBarrel(SMNumber) == "barrel-" ){ydirection = "-x";}

  return ydirection;
}

//---------------------------------------------------------
TString TEBNumbering::GetJYDirection(const Int_t& SMNumber)
{
  TString jydirection = "-x";      // DEFAULT

  if( GetSMHalfBarrel(SMNumber) == "barrel+" ){jydirection = "x";}
  if( GetSMHalfBarrel(SMNumber) == "barrel-" ){jydirection = "-x";}

  return jydirection;
}

//=========================================================================
//
//         METHODS TO SET FLAGD TO PRINT (OR NOT) COMMENTS (DEBUG)
//
//=========================================================================

void  TEBNumbering::PrintComments()
{
// Set flags to authorize printing of some comments concerning initialisations (default)

  fFlagPrint = fCodePrintComments;
  cout << "*TEBNumbering::PrintComments()> Warnings and some comments on init will be printed" << endl;
}

void  TEBNumbering::PrintWarnings()
{
// Set flags to authorize printing of warnings

  fFlagPrint = fCodePrintWarnings;
  cout << "*TEBNumbering::PrintWarnings()> Warnings will be printed" << endl;
}

void  TEBNumbering::PrintAllComments()
{
// Set flags to authorize printing of the comments of all the methods

  fFlagPrint = fCodePrintAllComments;
  cout << "*TEBNumbering::PrintAllComments()> All the comments will be printed" << endl;
}

void  TEBNumbering::PrintNoComment()
{
// Set flags to forbid the printing of all the comments

  fFlagPrint = fCodePrintNoComment;
}
