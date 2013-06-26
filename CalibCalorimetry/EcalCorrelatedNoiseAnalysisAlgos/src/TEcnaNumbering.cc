//----------Author's Name: B.Fabbro, F.X.Gentit + EB table from P.Jarry  DSM/IRFU/SPP CEA-Saclay
//----------Copyright:Those valid for CEA software
//----------Modified:30/06/2011

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaNumbering.h"

//--------------------------------------
//  TEcnaNumbering.cc
//  Class creation: 30 September 2005
//  Documentation: see TEcnaNumbering.h
//--------------------------------------

ClassImp(TEcnaNumbering)
//-------------------------------------------------------------------------------------------------------
//
//  Building of the numbering for the Ecal channels (EB and EE)
//
//  Convention for the names used here and in the other "TEcna" classes:
//
//  Name     Number                      Reference set   Range      Comment
//
//  SMTow  : Tower number                in SuperModule  [1,68]     (phi,eta) progression
//  SMCrys : Crystal number              in SuperModule  [1,1700]   (phi,eta) progression
//  SMEcha : Electronic channel number   in SuperModule  [0,1699]   S shape data reading order    
//  TowEcha: Electronic channel number   in Tower        [0,24]     S shape data reading order  
//
//  DeeSC  : super-crystal(SC) number    in Dee          [1,200]    (IY,IX) progression
//  DeeCrys: Crystal number              in Dee matrix   [1,5000]   (IY,IX) progression
//  DeeEcha: Electronic channel number   in Dee matrix   [0,4999]   (IY,IX) progression (starting from 0)
//  SCEcha : Electronic channel number   in SC           [1,25]     Crystal numbering for construction 
//-------------------------------------------------------------------------------------------------------

//====================================== constructors =========================================
TEcnaNumbering::TEcnaNumbering() {
// Constructor without argument: call to method Init()

//  cout << "[Info Management] CLASS: TEcnaNumbering.    CREATE OBJECT: this = " << this << endl;

  Init();
}

TEcnaNumbering::TEcnaNumbering(TEcnaObject* pObjectManager, const TString& SubDet) {
// Constructor with argument: call to methods Init() and SetEcalSubDetector(const TString&)

 // cout << "[Info Management] CLASS: TEcnaNumbering.    CREATE OBJECT: this = " << this << endl;

  Init();
  Long_t i_this = (Long_t)this;
  pObjectManager->RegisterPointer("TEcnaNumbering", i_this);

  //............................ fEcal  => to be changed in fParEcal
  fEcal = 0;
  Long_t iParEcal = pObjectManager->GetPointerValue("TEcnaParEcal");
  if( iParEcal == 0 )
    {fEcal = new TEcnaParEcal(pObjectManager, SubDet.Data()); /*fCnew++*/}
  else
    {fEcal = (TEcnaParEcal*)iParEcal;}

  SetEcalSubDetector(SubDet.Data());
}


TEcnaNumbering::TEcnaNumbering(const TString& SubDet, const TEcnaParEcal* pEcal) {
// Constructor with argument: call to methods Init() and SetEcalSubDetector(const TString&)

 // cout << "[Info Management] CLASS: TEcnaNumbering.    CREATE OBJECT: this = " << this << endl;

  Init();
  SetEcalSubDetector(SubDet.Data(), pEcal);
}

//====================================== destructor =========================================
TEcnaNumbering::~TEcnaNumbering() {
//destructor

  //if (fEcal   != 0){delete fEcal;   fCdelete++;}

  //....................... Barrel
  if (fT2dSMCrys  != 0){delete [] fT2dSMCrys;  fCdelete++;}
  if (fT1dSMCrys  != 0){delete [] fT1dSMCrys;  fCdelete++;}
  if (fT1dSMTow   != 0){delete [] fT1dSMTow;   fCdelete++;}
  if (fT1dTowEcha != 0){delete [] fT1dTowEcha; fCdelete++;}

  //....................... Endcap
  if (fT3dDeeCrys     != 0){delete [] fT3dDeeCrys;     fCdelete++;}
  if (fT2dDeeCrys     != 0){delete [] fT2dDeeCrys;     fCdelete++;}
  if (fT1dDeeCrys     != 0){delete [] fT1dDeeCrys;     fCdelete++;}
  if (fT2dDeeSC       != 0){delete [] fT2dDeeSC;       fCdelete++;}
  if (fT1dDeeSC       != 0){delete [] fT1dDeeSC;       fCdelete++;}
  if (fT2dSCEcha      != 0){delete [] fT2dSCEcha;      fCdelete++;}
  if (fT1dSCEcha      != 0){delete [] fT1dSCEcha;      fCdelete++;}
  if (fT2d_jch_JY     != 0){delete [] fT2d_jch_JY;     fCdelete++;}
  if (fT1d_jch_JY     != 0){delete [] fT1d_jch_JY;     fCdelete++;}
  if (fT2d_ich_IX     != 0){delete [] fT2d_ich_IX;     fCdelete++;}
  if (fT1d_ich_IX     != 0){delete [] fT1d_ich_IX;     fCdelete++;}
  if (fT2d_DS         != 0){delete [] fT2d_DS;         fCdelete++;}
  if (fT1d_DS         != 0){delete [] fT1d_DS;         fCdelete++;}
  if (fT2d_DSSC       != 0){delete [] fT2d_DSSC;       fCdelete++;}
  if (fT1d_DSSC       != 0){delete [] fT1d_DSSC;       fCdelete++;}
  if (fT2d_DeeSCCons  != 0){delete [] fT2d_DeeSCCons;  fCdelete++;}
  if (fT1d_DeeSCCons  != 0){delete [] fT1d_DeeSCCons;  fCdelete++;}
  if (fT2d_RecovDeeSC != 0){delete [] fT2d_RecovDeeSC; fCdelete++;}
  if (fT1d_RecovDeeSC != 0){delete [] fT1d_RecovDeeSC; fCdelete++;}

 // cout << "[Info Management] CLASS: TEcnaNumbering.    DESTROY OBJECT: this = " << this << endl;

}
//------------------------------------------------------------- Init()
void TEcnaNumbering::Init()
{
//Set default values and build crystal numbering table

  //.............................. Initialisations
  fTTBELL  = '\007';
  fgMaxCar = 512;

  //....................... Barrel
  fT2dSMCrys  = 0;
  fT1dSMCrys  = 0;
  fT1dSMTow   = 0;
  fT1dTowEcha = 0;

  fCodeChNumberingLvrbBot = "bottom";
  fCodeChNumberingLvrbTop = "top";

  //....................... Endcap
  fT3dDeeCrys     = 0;
  fT2dDeeCrys     = 0;
  fT1dDeeCrys     = 0;
  fT2dDeeSC       = 0;
  fT1dDeeSC       = 0;
  fT2dSCEcha      = 0;
  fT1dSCEcha      = 0;
  fT2d_jch_JY     = 0;
  fT1d_jch_JY     = 0;
  fT2d_ich_IX     = 0;
  fT1d_ich_IX     = 0;
  fT2d_DS         = 0;
  fT1d_DS         = 0;
  fT2d_DSSC       = 0;
  fT1d_DSSC       = 0;
  fT2d_DeeSCCons  = 0;
  fT1d_DeeSCCons  = 0;
  fT2d_RecovDeeSC = 0;
  fT1d_RecovDeeSC = 0;

  fCodeChNumberingITP1Bot = "bottom";   // ==> Type 1 Interface plate  IPT1 (a faire)
  fCodeChNumberingITP2Top = "top";      // ==> Type 2 Interface plate  IPT2 (a faire)

  //------------------ Init pointers on the CNA objects
  fEcal = 0;
}
// end of Init()
//------------------------------------------------------------- SetEcalSubDetector(...)
void TEcnaNumbering::SetEcalSubDetector(const TString& SubDet, const TEcnaParEcal* pEcal){
//Set the current subdetector flag and the current subdetector parameters

  fEcal = 0;
  if( pEcal == 0 )
    {fEcal = new TEcnaParEcal(SubDet.Data());  /*fCnew++*/ ;}
  else
    {fEcal = (TEcnaParEcal*)pEcal;}

  Int_t MaxCar = fgMaxCar;
  fFlagSubDet.Resize(MaxCar);
  fFlagSubDet = fEcal->GetEcalSubDetector();      // fFlagSubDet = "EB" or "EE"

  if( fFlagSubDet == "EB" ){BuildBarrelCrysTable();}
  if( fFlagSubDet == "EE" ){BuildEndcapCrysTable(); BuildEndcapSCTable();}
}

void TEcnaNumbering::SetEcalSubDetector(const TString& SubDet){
//Set the current subdetector flag and the current subdetector parameters

  Int_t MaxCar = fgMaxCar;
  fFlagSubDet.Resize(MaxCar);
  fFlagSubDet = fEcal->GetEcalSubDetector();      // fFlagSubDet = "EB" or "EE"

  if( fFlagSubDet == "EB" ){BuildBarrelCrysTable();}
  if( fFlagSubDet == "EE" ){BuildEndcapCrysTable(); BuildEndcapSCTable();}
}

//====================================================================================================
//
//
//                                  B   A   R   R   E   L 
//
//
//====================================================================================================
//
//               SMCrys <-> (SMTow, TowEcha) correspondance table (from Patrick Jarry)
//
//====================================================================================================
void TEcnaNumbering::BuildBarrelCrysTable()
{
// Build the correspondance table: SMCrys <-> (SMTow, TowEcha) for the ECAL BARREL
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
      
      //................... Allocation and Init CrysNumbersTable
      
      fT2dSMCrys = new Int_t*[MaxSMTow];                         fCnew++;  
      fT1dSMCrys = new  Int_t[MaxSMTow*MaxTowEcha];              fCnew++;   
      for(Int_t i_SMTow = 0 ; i_SMTow < MaxSMTow ; i_SMTow++){
	fT2dSMCrys[i_SMTow] = &fT1dSMCrys[0] + i_SMTow*MaxTowEcha;}
      for(Int_t i=0; i<MaxSMTow; i++)
	{for(Int_t j=0; j<MaxTowEcha; j++){fT2dSMCrys[i][j]=0;}}      

      fT1dSMTow   = new Int_t[MaxSMCrys];                        fCnew++;
      for(Int_t i=0; i<MaxSMCrys; i++){fT1dSMTow[i]=0;}

      fT1dTowEcha = new Int_t[MaxSMCrys];                        fCnew++;
      for(Int_t i=0; i<MaxSMCrys; i++){fT1dTowEcha[i]=0;}

      //........................ Build table      
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

	      Int_t n1SMCrys = icrys*20+jcrys+1;               //  n1SMCrys = 1 to 1700

	      fT2dSMCrys[tow][ic-1]   = n1SMCrys;      // fT2dSMCrys[]  : range = [1,1700]
	      fT1dSMTow[n1SMCrys-1]   = tow+1;         // fT1dSMTow[]   : range = [1,68]
	      fT1dTowEcha[n1SMCrys-1] = ic-1;          // fT1dTowEcha[] : range = [0,24]
	    }
	}
      // cout << "#TEcnaNumbering::TBuildBarrelCrysTable()> Crys Table Building done" << endl;

      delete [] jch_type;                       fCdelete++;
      delete [] jch_type_d1;                    fCdelete++;
      delete [] ich_type;                       fCdelete++;
      delete [] ich_type_d1;                    fCdelete++;
      delete [] type;                           fCdelete++;
    }
  else
    {
      // cout << "#TEcnaNumbering::TBuildBarrelCrysTable()> No Building of Crys Table since it is already done." << endl;
    }
}

//===============================================================================
//
//        GetSMCrysFrom1SMTowAnd0TowEcha
//        GetSMCrysFromSMEcha
//
//===============================================================================
Int_t TEcnaNumbering::Get1SMCrysFrom1SMTowAnd0TowEcha(const Int_t& n1SMTow,
						      const Int_t& i0TowEcha)
{
//get crystal number in SM from tower number in SM
// and from Electronic Channel number in tower

  Int_t n1SMCrys = 0;

  if( fT2dSMCrys == 0 ){BuildBarrelCrysTable();}
  
  if (n1SMTow >= 1 && n1SMTow <= fEcal->MaxTowInSM())
    {
      if (i0TowEcha >=0 && i0TowEcha < fEcal->MaxCrysInTow())
	{
	  n1SMCrys = fT2dSMCrys[n1SMTow-1][i0TowEcha];
	}
      else
	{
	  n1SMCrys = -2;   // Electronic Cnannel in Tower out of range 
	  cout << "!TEcnaNumbering::Get1SMCrysFrom1SMTowAnd0TowEcha(...)> Electronic Channel in Tower out of range."
	       << " i0TowEcha = " << i0TowEcha << "(n1SMTow = " << n1SMTow << ")" << fTTBELL << endl;
	}
    }
  else
    {
      n1SMCrys = -3;   // Tower number in SM out of range
      cout << "!TEcnaNumbering::Get1SMCrysFrom1SMTowAnd0TowEcha(...)> Tower number in SM out of range."
	   << " n1SMTow = " << n1SMTow << "(i0TowEcha = " << i0TowEcha << ")" << fTTBELL << endl;
    }

  return n1SMCrys;   // Range = [1,1700]
}

//===============================================================================
//
//                Get0TowEchaFrom1SMCrys, Get1SMTowFrom1SMCrys
//
//===============================================================================

Int_t TEcnaNumbering::Get0TowEchaFrom1SMCrys(const Int_t& n1SMCrys)
{
// get Electronic Channel number in Tower from Crystal number in SuperModule

  Int_t i0TowEcha = -1;

  if( n1SMCrys >= 1 && n1SMCrys <= fEcal->MaxCrysInSM() )
    {
      i0TowEcha = fT1dTowEcha[n1SMCrys-1];
    }
  else
    {
      i0TowEcha = -2;
      cout << "!TEcnaNumbering::Get0TowEchaFrom1SMCrys(...)> Crystal number in SM out of range."
	   << " n1SMCrys = " << n1SMCrys << fTTBELL << endl;
    }
  return i0TowEcha;   // range = [0,24]
}

Int_t TEcnaNumbering::Get1SMTowFrom1SMCrys(const Int_t& n1SMCrys)
{
// get Tower number in SM (range [1,68]) from Crystal number in SuperModule (range [1,1700])

  Int_t n1SMtox = 0;
  
  if( n1SMCrys >= 1 && n1SMCrys <= fEcal->MaxCrysInSM() )
    {
      n1SMtox = fT1dSMTow[n1SMCrys-1];
    }
  else
    {
      n1SMtox = -1;
      cout << "!TEcnaNumbering::Get1SMTowFrom1SMCrys(...)> Crystal number in SM out of range."
	   << " n1SMCrys = " << n1SMCrys << fTTBELL << endl;
    }
  return n1SMtox;   // range = [1,68]
}

//===============================================================================
//
//          Get0TowEchaFrom0SMEcha
//          Get1SMTowFrom0SMEcha
//
//===============================================================================

Int_t TEcnaNumbering::Get0TowEchaFrom0SMEcha(const Int_t& i0SMEcha)
{
//get electronic channel number in tower from electronic channel number in SM

  Int_t n1SMTow = i0SMEcha/fEcal->MaxCrysInTow()+1;
  Int_t i0TowEcha = i0SMEcha - fEcal->MaxCrysInTow()*(n1SMTow-1);

  return i0TowEcha;   // range = [0,24]
}

Int_t TEcnaNumbering::Get1SMTowFrom0SMEcha(const Int_t& i0SMEcha)
{
//get tower number from electronic channel number in SM

  Int_t n1SMTow = i0SMEcha/fEcal->MaxCrysInTow()+1;

  return n1SMTow;  // range = [1,68]
}

Int_t TEcnaNumbering::Get0SMEchaFrom1SMTowAnd0TowEcha(const Int_t& n1SMTow, const Int_t& i0TowEcha)
{
//get tower number from electronic channel number in SM

  Int_t i0SMEcha = (n1SMTow-1)*fEcal->MaxCrysInTow()+i0TowEcha;

  return i0SMEcha;
}
//===========================================================================
//
//                        GetHashedNumberFromIEtaAndIPhi
//
//===========================================================================
Int_t TEcnaNumbering::GetHashedNumberFromIEtaAndIPhi(const Int_t& IEta, const Int_t& IPhi)
{
  Int_t Hashed = 0;

  if( IEta > 0 ){Hashed = (85 + IEta - 1)*360 + IPhi - 1;}
  if( IEta < 0 ){Hashed = (85 + IEta)*360     + IPhi - 1;}

  return Hashed;
}

Int_t TEcnaNumbering::GetIEtaFromHashed(const Int_t& Hashed,  const Int_t& SMNumber)
{
  Int_t IEta = 0;

  if( GetSMHalfBarrel(SMNumber) == "EB+" ){IEta = Hashed/360 - 85 + 1;}
  if( GetSMHalfBarrel(SMNumber) == "EB-" ){IEta = 85 + Hashed/360;}

  return IEta;
}

Int_t TEcnaNumbering::GetIPhiFromHashed(const Int_t& Hashed)
{
  Int_t IPhi = Hashed%360 + 1;

  return IPhi;
}
//===========================================================================
//
//                        GetTowerLvrbType
//
//===========================================================================
TString  TEcnaNumbering::GetStinLvrbType(const Int_t& n1SMTow)
{
  TString lvrb_type = GetTowerLvrbType(n1SMTow);
  return lvrb_type;
}
TString  TEcnaNumbering::GetTowerLvrbType(const Int_t& n1SMTow)
{
//gives the LVRB type of the crystal numbering of tower

  TString type = fCodeChNumberingLvrbTop;   // => default value

  if (n1SMTow >=  1 && n1SMTow <= 12){type = fCodeChNumberingLvrbBot;}
  if (n1SMTow >= 21 && n1SMTow <= 28){type = fCodeChNumberingLvrbBot;}
  if (n1SMTow >= 37 && n1SMTow <= 44){type = fCodeChNumberingLvrbBot;}
  if (n1SMTow >= 53 && n1SMTow <= 60){type = fCodeChNumberingLvrbBot;}

  return type;
}

//==============================================================================
//
//       GetEta, GetEtaMin, GetEtaMax,  GetIEtaMin, GetIEtaMax
//
//==============================================================================
Double_t TEcnaNumbering::GetEta(const Int_t& n1EBSM, const Int_t& n1SMTow,
				const Int_t& i0TowEcha)
{
//Gives Eta for a given (n1EBSM, n1SMTow, i0TowEcha)
  
  Double_t eta = (Double_t)0.;

  Int_t max_crys_eta_in_tower = fEcal->MaxCrysEtaInTow();
  Int_t max_tow_eta_in_sm     = fEcal->MaxTowEtaInSM();
  Int_t max_sm_in_barrel      = fEcal->MaxSMInEB();

  if ( n1EBSM >= 1 && n1EBSM <= max_sm_in_barrel )
    {
      for (Int_t i_sm_tow_eta = 0; i_sm_tow_eta < max_tow_eta_in_sm; i_sm_tow_eta++)
	{
	  Int_t i_crys_eta_min = (Int_t)(1 + i_sm_tow_eta*(max_crys_eta_in_tower-1));
	  Int_t i_crys_eta_max = (Int_t)((i_sm_tow_eta+1)*(max_crys_eta_in_tower-1));
	  Int_t i_crys_eta     = (Int_t)(i_sm_tow_eta*max_crys_eta_in_tower);
	  // = 0,..,16 -> last xtal in eta for the previous tower
	  Double_t d_echa_eta  = (Double_t)(i0TowEcha/max_crys_eta_in_tower);    // = 0,1,2,3,4

	    if ( n1SMTow >= i_crys_eta_min && n1SMTow <= i_crys_eta_max )
	      {
		if (GetTowerLvrbType(n1SMTow) == fCodeChNumberingLvrbTop)
		{eta = (Double_t)(i_crys_eta) + d_echa_eta + 1;}
		if (GetTowerLvrbType(n1SMTow) == fCodeChNumberingLvrbBot)
		{eta = (Double_t)(i_crys_eta+max_crys_eta_in_tower)-d_echa_eta;}
	      }
	}      
      if ( GetSMHalfBarrel(n1EBSM) == "EB-" ){eta = - eta;}      
    }
  else
    {
      cout << "TEcnaNumbering::GetEta(...)> SM = " << n1EBSM
	   << ". Out of range (range = [1," << fEcal->MaxSMInEB() << "])" 
	   << fTTBELL << endl; 
    }
  return eta;
}
//-------------------------------------------------------------------------------------
Double_t TEcnaNumbering::GetEtaMin(const Int_t& n1EBSM, const Int_t& n1SMTow)
{
//Gives EtaMin for a given Tower

  Int_t max_tow_eta_in_sm     = fEcal->MaxTowEtaInSM();
  Int_t max_crys_eta_in_tower = fEcal->MaxCrysEtaInTow();

  Double_t eta_min = (Double_t)0.;

  for (Int_t i_sm_tow_eta = 0; i_sm_tow_eta < max_tow_eta_in_sm; i_sm_tow_eta++)
    {
      Int_t i_crys_eta_min =    (Int_t)(1 + i_sm_tow_eta*(max_crys_eta_in_tower-1));
      Int_t i_crys_eta_max =    (Int_t)((i_sm_tow_eta+1)*(max_crys_eta_in_tower-1));
      Int_t i_crys_eta     =    (Int_t)(i_sm_tow_eta*max_crys_eta_in_tower);

      if ( n1SMTow >= i_crys_eta_min && n1SMTow <= i_crys_eta_max )
	{
	  if (GetSMHalfBarrel(n1EBSM) == "EB+")
	    {eta_min = (Double_t)i_crys_eta;}
	  if (GetSMHalfBarrel(n1EBSM) == "EB-")
	    {eta_min = -(Double_t)(i_crys_eta + max_crys_eta_in_tower);}
	}
    }
  return eta_min;
}
//------------------------------------------------------------------------------------
Double_t TEcnaNumbering::GetEtaMax(const Int_t& n1EBSM, const Int_t& n1SMTow)
{
//Gives EtaMax for a given Tower

  Int_t max_tow_eta_in_sm     = fEcal->MaxTowEtaInSM();
  Int_t max_crys_eta_in_tower = fEcal->MaxCrysEtaInTow();

  Double_t eta_max = (max_crys_eta_in_tower-1);

  for (Int_t i_sm_tow_eta = 0; i_sm_tow_eta < max_tow_eta_in_sm; i_sm_tow_eta++)
    {
      Int_t i_crys_eta_min = (Int_t)(1 + i_sm_tow_eta*(max_crys_eta_in_tower-1));
      Int_t i_crys_eta_max = (Int_t)((i_sm_tow_eta+1)*(max_crys_eta_in_tower-1));
      Int_t i_crys_eta     = (Int_t)(i_sm_tow_eta*max_crys_eta_in_tower);

      if ( n1SMTow >= i_crys_eta_min && n1SMTow <= i_crys_eta_max )
	{
	  if (GetSMHalfBarrel(n1EBSM) == "EB+")
	    {eta_max = (Double_t)(i_crys_eta + max_crys_eta_in_tower);}
	  if (GetSMHalfBarrel(n1EBSM) == "EB-")
	    {eta_max = -(Double_t)i_crys_eta;}
	}
    }

  return eta_max;
}

Double_t TEcnaNumbering::GetIEtaMin(const Int_t& n1EBSM, const Int_t& n1SMTow)
{
//Gives IEtaMin for a given (n1EBSM, n1SMTow)

  Double_t i_eta_min = (Int_t)0.; 
   
  if(GetSMHalfBarrel(n1EBSM) == "EB+")
    {i_eta_min = (Double_t)GetEtaMin(n1EBSM, n1SMTow)+(Double_t)0.5;}
  if(GetSMHalfBarrel(n1EBSM) == "EB-")
    {i_eta_min = (Double_t)GetEtaMin(n1EBSM, n1SMTow)-(Double_t)0.5;}

  return i_eta_min;
}

Double_t TEcnaNumbering::GetIEtaMax(const Int_t& n1EBSM, const Int_t& n1SMTow)
{
//Gives IEtaMax for a given (n1EBSM, n1SMTow)

  Double_t i_eta_max = (Int_t)0.; 
   
  if(GetSMHalfBarrel(n1EBSM) == "EB+")
    {i_eta_max = (Double_t)GetEtaMax(n1EBSM, n1SMTow)+(Double_t)0.5;}
  if(GetSMHalfBarrel(n1EBSM) == "EB-")
    {i_eta_max = (Double_t)GetEtaMax(n1EBSM, n1SMTow)-(Double_t)0.5;}

  return i_eta_max;
}

Double_t TEcnaNumbering::GetIEtaMin(const Int_t& n1EBSM)
{
//Gives IEtaMin for a given (n1EBSM)

  Double_t i_eta_min = (Int_t)0.;

  Int_t n1SMTowPlus  = (Int_t)1;   
  Int_t n1SMTowMinus = (Int_t)fEcal->MaxTowInSM();

  if( GetSMHalfBarrel(n1EBSM) == "EB+" )
    {i_eta_min = (Double_t)GetIEtaMin(n1EBSM, n1SMTowPlus );}
  if( GetSMHalfBarrel(n1EBSM) == "EB-" )
    {i_eta_min = (Double_t)GetIEtaMin(n1EBSM, n1SMTowMinus);}

  return i_eta_min;
}

Double_t TEcnaNumbering::GetIEtaMax(const Int_t& n1EBSM)
{
//Gives IEtaMax for a given (n1EBSM)

  Double_t i_eta_max = (Int_t)0.; 

  Int_t n1SMTowPlus  = (Int_t)fEcal->MaxTowInSM();   
  Int_t n1SMTowMinus = (Int_t)1;   
   
  if( GetSMHalfBarrel(n1EBSM) == "EB+")
    {i_eta_max = (Double_t)GetIEtaMax(n1EBSM, n1SMTowPlus );}
  if( GetSMHalfBarrel(n1EBSM) == "EB-")
    {i_eta_max = (Double_t)GetIEtaMax(n1EBSM, n1SMTowMinus);}

  return i_eta_max;
}

//==============================================================================
//
//    GetSMCentralPhi, GetPhiInSM, GetPhi,
//    GetPhiMin, GetPhiMax, GetIPhiMin, GetIPhiMax
//
//==============================================================================
Double_t TEcnaNumbering::GetSMCentralPhi(const Int_t& n1EBSM)
{
//Gives the central phi value of the SuperModule

  Double_t central_phi = (Double_t)10.;    //  DEFAULT = SM1
  
  if (GetSMHalfBarrel(n1EBSM) == "EB+"){central_phi = 10. + (Double_t)20.*(n1EBSM-1);}
  if (GetSMHalfBarrel(n1EBSM) == "EB-"){central_phi = 10. + (Double_t)20.*(n1EBSM-19);}

  return central_phi;
}
//------------------------------------------------------------------------------------
Double_t TEcnaNumbering::GetPhiInSM(const Int_t& n1EBSM,
				    const Int_t& n1SMTow, const Int_t& i0TowEcha)
{
//Gives Phi for a given (n1EBSM, n1SMTow, i0TowEcha)

  Double_t phi_in_SM = (Double_t)0.;

  Int_t rest_temp =(Int_t)(n1SMTow%(fEcal->MaxCrysPhiInTow()-1));  // "Phi" of the tower = 1,2,3,0 

  if ( n1EBSM >= 1 && n1EBSM <= fEcal->MaxSMInEB() )
    {
      if( rest_temp == 1 ){phi_in_SM = (Double_t)15.;}
      if( rest_temp == 2 ){phi_in_SM = (Double_t)10.;}
      if( rest_temp == 3 ){phi_in_SM = (Double_t)5.;}
      if( rest_temp == 0 ){phi_in_SM = (Double_t)0.;}
      
      if (GetTowerLvrbType(n1SMTow) == fCodeChNumberingLvrbTop)
	{ 
	  if( i0TowEcha ==  4 || i0TowEcha ==  5 || i0TowEcha == 14 || i0TowEcha == 15 || i0TowEcha == 24 )
	    {phi_in_SM = phi_in_SM + 0;}
	  
	  if( i0TowEcha ==  3 || i0TowEcha ==  6 || i0TowEcha == 13 || i0TowEcha == 16 || i0TowEcha == 23 )
	    {phi_in_SM = phi_in_SM + 1;} 
	  
	  if( i0TowEcha ==  2 || i0TowEcha ==  7 || i0TowEcha == 12 || i0TowEcha == 17 || i0TowEcha == 22 )
	    {phi_in_SM = phi_in_SM + 2;}
	  
	  if( i0TowEcha ==  1 || i0TowEcha ==  8 || i0TowEcha == 11 || i0TowEcha == 18 || i0TowEcha == 21 )
	    {phi_in_SM = phi_in_SM + 3;}
	  
	  if( i0TowEcha ==  0 || i0TowEcha ==  9 || i0TowEcha == 10 || i0TowEcha == 19 || i0TowEcha == 20 )
	    {phi_in_SM = phi_in_SM + 4;} 
	}
      if (GetTowerLvrbType(n1SMTow) == fCodeChNumberingLvrbBot)
	{
	  if( i0TowEcha == 20 || i0TowEcha == 19 || i0TowEcha == 10 || i0TowEcha ==  9 || i0TowEcha ==  0 )
	    {phi_in_SM = phi_in_SM + 0;}
	  
	  if( i0TowEcha == 21 || i0TowEcha == 18 || i0TowEcha == 11 || i0TowEcha ==  8 || i0TowEcha ==  1 )
	    {phi_in_SM = phi_in_SM + 1;}
 
	  if( i0TowEcha == 22 || i0TowEcha == 17 || i0TowEcha == 12 || i0TowEcha ==  7 || i0TowEcha ==  2 )
	    {phi_in_SM = phi_in_SM + 2;}

	  if( i0TowEcha == 23 || i0TowEcha == 16 || i0TowEcha == 13 || i0TowEcha ==  6 || i0TowEcha ==  3 )
	    {phi_in_SM = phi_in_SM + 3;}

	  if( i0TowEcha == 24 || i0TowEcha == 15 || i0TowEcha == 14 || i0TowEcha ==  5 || i0TowEcha ==  4 )
	    {phi_in_SM = phi_in_SM + 4;} 
	  }
    }
  else
    {
      cout << "TEcnaNumbering::GetPhiInSM(...)> SM = " << n1EBSM
	   << ". Out of range (range = [1," << fEcal->MaxSMInEB() << "])" 
	   << fTTBELL << endl; 
    }
  phi_in_SM = 20 - phi_in_SM;
  return phi_in_SM;
}
//---------------------------------------------------------------------------------------
Double_t TEcnaNumbering::GetPhi(const Int_t& n1EBSM,
				const Int_t& n1SMTow, const Int_t& i0TowEcha)
{
//Gives Phi for a given (n1EBSM, n1SMTow, i0TowEcha)
  
  Double_t phi = (Double_t)0.;

  if ( n1EBSM >= 1 && n1EBSM <= fEcal->MaxSMInEB() )
    {
      Double_t phiInSM   = GetPhiInSM(n1EBSM, n1SMTow, i0TowEcha);
      Double_t phi_start = GetSMCentralPhi(n1EBSM);
 
      phi = 20 - phiInSM + phi_start -(Double_t)10.;
    }
  else
    {
      cout << "TEcnaNumbering::GetPhi(...)> SM = " << n1EBSM
	   << ". Out of range (range = [1," << fEcal->MaxSMInEB() << "])" 
	   << fTTBELL << endl; 
    }
  return phi;
}

//-----------------------------------------------------------------------------------------
Double_t TEcnaNumbering::GetPhiMin(const Int_t& n1EBSM, const Int_t& n1SMTow)
{
//Gives PhiMin for a given Tower

  Int_t max_crys_phi_in_tower = fEcal->MaxCrysPhiInTow();

  Double_t phi_min = (Double_t)0.;     // DEFAULT
  Double_t phi_start = GetSMCentralPhi(n1EBSM);

  Int_t rest_temp =(Int_t)(n1SMTow%(max_crys_phi_in_tower-1));

  if(GetSMHalfBarrel(n1EBSM) == "EB+")
    {
      if ( rest_temp == 1 ) {phi_min = phi_start + (Double_t)5.;}
      if ( rest_temp == 2 ) {phi_min = phi_start + (Double_t)0.;}
      if ( rest_temp == 3 ) {phi_min = phi_start - (Double_t)5.;}
      if ( rest_temp == 0 ) {phi_min = phi_start - (Double_t)10.;}
    }
  if(GetSMHalfBarrel(n1EBSM) == "EB-")
    {
      if ( rest_temp == 0 ) {phi_min = phi_start + (Double_t)5.;}
      if ( rest_temp == 3 ) {phi_min = phi_start + (Double_t)0.;}
      if ( rest_temp == 2 ) {phi_min = phi_start - (Double_t)5.;}
      if ( rest_temp == 1 ) {phi_min = phi_start - (Double_t)10.;}
    }
  return phi_min;
}
//-----------------------------------------------------------------------------------------
Double_t TEcnaNumbering::GetPhiMax(const Int_t& n1EBSM, const Int_t& n1SMTow)
{
//Gives PhiMax for a given Tower

  Int_t max_crys_phi_in_tower = fEcal->MaxCrysPhiInTow();

  Double_t phi_max = (Double_t)20.;     // DEFAULT
  Double_t phi_start = GetSMCentralPhi(n1EBSM);

  Int_t rest_temp =(Int_t)(n1SMTow%(max_crys_phi_in_tower-1));

  if(GetSMHalfBarrel(n1EBSM) == "EB+")
    {
      if ( rest_temp == 1 ) {phi_max = phi_start + (Double_t)10.;}
      if ( rest_temp == 2 ) {phi_max = phi_start + (Double_t)5.;}
      if ( rest_temp == 3 ) {phi_max = phi_start - (Double_t)0.;}
      if ( rest_temp == 0 ) {phi_max = phi_start - (Double_t)5.;}
    }

  if(GetSMHalfBarrel(n1EBSM) == "EB-")
    {
      if ( rest_temp == 0 ) {phi_max = phi_start + (Double_t)10.;}
      if ( rest_temp == 3 ) {phi_max = phi_start + (Double_t)5.;}
      if ( rest_temp == 2 ) {phi_max = phi_start - (Double_t)0.;}
      if ( rest_temp == 1 ) {phi_max = phi_start - (Double_t)5.;}
    }

  return phi_max;
}
//-----------------------------------------------------------------------------------------
Double_t TEcnaNumbering::GetPhiMin(const Int_t& n1EBSM)
{
//Gives PhiMin for a given SuperModule

  Double_t phi_min = GetSMCentralPhi(n1EBSM) - (Double_t)10.;

  return phi_min;
}
//-----------------------------------------------------------------------------------------
Double_t TEcnaNumbering::GetPhiMax(const Int_t& n1EBSM)
{
//Gives PhiMax for a given SuperModule

  Double_t phi_max = GetSMCentralPhi(n1EBSM) + (Double_t)10.;

  return phi_max;
}
//-----------------------------------------------------------------------------------------
Double_t TEcnaNumbering::GetJPhiMin(const Int_t& n1EBSM, const Int_t& n1SMTow)
{
//Gives JPhiMin for a given Tower

  Double_t j_phi_min = (Double_t)1.;
  Int_t max_crys_phi_in_tower = fEcal->MaxCrysPhiInTow();
  Int_t rest_temp =(Int_t)(n1SMTow%(max_crys_phi_in_tower-1));
  
  if ( rest_temp == 1 ){j_phi_min = (Double_t) 1. - (Double_t)0.5;}
  if ( rest_temp == 2 ){j_phi_min = (Double_t) 6. - (Double_t)0.5;}
  if ( rest_temp == 3 ){j_phi_min = (Double_t)11. - (Double_t)0.5;}
  if ( rest_temp == 0 ){j_phi_min = (Double_t)16. - (Double_t)0.5;}

  return j_phi_min;
}
//-----------------------------------------------------------------------------------------
Double_t TEcnaNumbering::GetJPhiMax(const Int_t& n1EBSM, const Int_t& n1SMTow)
{
//Gives JPhiMax for a given Tower

  Double_t j_phi_max = (Double_t)20.;
  Int_t max_crys_phi_in_tower = fEcal->MaxCrysPhiInTow();
  Int_t rest_temp =(Int_t)(n1SMTow%(max_crys_phi_in_tower-1));
  
  if ( rest_temp == 1 ){j_phi_max = (Double_t) 5. + (Double_t)0.5;}
  if ( rest_temp == 2 ){j_phi_max = (Double_t)10. + (Double_t)0.5;}
  if ( rest_temp == 3 ){j_phi_max = (Double_t)15. + (Double_t)0.5;}
  if ( rest_temp == 0 ){j_phi_max = (Double_t)20. + (Double_t)0.5;}

  return j_phi_max;
}

//-----------------------------------------------------------------------------------------
Double_t TEcnaNumbering::GetJPhiMin(const Int_t& n1EBSM)
{
//Gives JPhiMin for a given SuperModule

  Double_t j_phi_min = (Double_t)1. - (Double_t)0.5;

  return j_phi_min;
}
//-----------------------------------------------------------------------------------------
Double_t TEcnaNumbering::GetJPhiMax(const Int_t& n1EBSM)
{
//Gives JPhiMax for a given SuperModule

  Double_t j_phi_max = (Double_t)20. + (Double_t)0.5;

  return j_phi_max;
}

//==============================================================================
//
//   GetXDirectionEB, GetYDirectionEB, GetJYDirectionEB
//
//==============================================================================

//------------------------------------------------------------------
TString TEcnaNumbering::GetXDirectionEB(const Int_t& SMNumber)
{
  TString xdirection = "x";      // DEFAULT

  if( GetSMHalfBarrel(SMNumber) == "EB+" ){xdirection = "x";}
  if( GetSMHalfBarrel(SMNumber) == "EB-" ){xdirection = "x";}

  return xdirection;
}
//---------------------------------------------------------
TString TEcnaNumbering::GetYDirectionEB(const Int_t& SMNumber)
{
  TString ydirection = "-x";      // DEFAULT

  if( GetSMHalfBarrel(SMNumber) == "EB+" ){ydirection = "-x";}
  if( GetSMHalfBarrel(SMNumber) == "EB-" ){ydirection = "-x";}

  return ydirection;
}

//---------------------------------------------------------
TString TEcnaNumbering::GetJYDirectionEB(const Int_t& SMNumber)
{
  TString jydirection = "-x";      // DEFAULT

  if( GetSMHalfBarrel(SMNumber) == "EB+" ){jydirection = "x";}
  if( GetSMHalfBarrel(SMNumber) == "EB-" ){jydirection = "-x";}

  return jydirection;
}

//============================================================================
TString  TEcnaNumbering::GetSMHalfBarrel(const Int_t& SMNumber)
{
//gives the half-barrel of the SM ("EB+" or "EB-")

  TString type = "EB-";   // => default value

  if ( SMNumber >=  1                      && SMNumber <= fEcal->MaxSMInEBPlus() ){type = "EB+";}
  if ( SMNumber >   fEcal->MaxSMInEBPlus() && SMNumber <= fEcal->MaxSMInEB()     ){type = "EB-";}

  return type;
}

Int_t TEcnaNumbering::PlusMinusSMNumber(const Int_t& PlusSMNumber)
{
  Int_t PMSMNumber = PlusSMNumber;
  if( PlusSMNumber > fEcal->MaxSMPhiInEB() ){PMSMNumber = - PlusSMNumber + fEcal->MaxSMPhiInEB();}
  return PMSMNumber;
}
//====================================================================================================
//
//
//                                  E   N   D   C   A   P 
//
//
//====================================================================================================
//
//               DeeCrys <-> (DeeSC, SCEcha) correspondance table
//               (from the barrel table given by Patrick Jarry)
//
//====================================================================================================
void TEcnaNumbering::BuildEndcapCrysTable()
{
// Build the correspondance table: DeeCrys <-> (DeeSC, SCEcha) for the ECAL ENDCAP
//
//  From CMS Internal Note  "CMS ECAL Endcap channel numbering"
//
//       Name      Number                     Reference set  Range          Comment
//
//       DeeSC   : Super-Crystal (SC) number    in Dee      [1,200]   (IY,IX) progression
//       DeeCrys : Crystal number               in Dee      [1,5000]  (IY,IX) progression
//       DeeEcha : Electronic channel number    in Dee      [0,4999]  Crystal numbering for construction 
//   
//       SCEcha  : Electronic channel number    in SC       [1,25]    Crystal numbering for construction  
//
//
//   fill the 3D array:  fT3dDeeCrys[n_DeeSC][n_SCEcha][2]
//
//   and the 2d arrays:  fT2dDeeSC[n_DeeCrys][2] and fT2dSCEcha[n_DeeCrys][2]
//
//------------------------------------------------------------------------------------------------------

  if ( fT3dDeeCrys == 0 )
    {
      Int_t MaxDeeSC   = fEcal->MaxSCEcnaInDee();    // fEcal->MaxSCEcnaInDee() = 200
      Int_t MaxSCEcha  = fEcal->MaxCrysInSC();
      Int_t MaxDeeCrys = fEcal->MaxCrysEcnaInDee();

      Int_t MaxDirections = 2;    //  (2 directions: left and right)

      //................... Allocation and Init CrysNumbersTable
      fT3dDeeCrys = new Int_t**[MaxDeeSC];                         fCnew++;       
      fT2dDeeCrys = new  Int_t*[MaxDeeSC*MaxSCEcha];               fCnew++;  
      fT1dDeeCrys = new   Int_t[MaxDeeSC*MaxSCEcha*MaxDirections]; fCnew++;
      
      for(Int_t i_DeeSC = 0; i_DeeSC < MaxDeeSC; i_DeeSC++){
	fT3dDeeCrys[i_DeeSC] = &fT2dDeeCrys[0] + i_DeeSC*MaxSCEcha;
	for(Int_t i_SCEcha = 0; i_SCEcha < MaxSCEcha; i_SCEcha++){
	  fT2dDeeCrys[i_DeeSC*MaxSCEcha + i_SCEcha] = &fT1dDeeCrys[0]
	    + (i_DeeSC*MaxSCEcha + i_SCEcha)*MaxDirections;}}
      for(Int_t i=0; i<MaxDeeSC; i++)
	{for(Int_t j=0; j<MaxSCEcha; j++)
	  {for(Int_t k=0; k<MaxDirections; k++){fT3dDeeCrys[i][j][k]=0;}}}      

      fT2dDeeSC  = new Int_t*[MaxDeeCrys];                         fCnew++;
      fT1dDeeSC  = new  Int_t[MaxDeeCrys*MaxDirections];           fCnew++;
      for(Int_t i_DeeCrys = 0 ; i_DeeCrys < MaxDeeCrys ; i_DeeCrys++){
	fT2dDeeSC[i_DeeCrys] = &fT1dDeeSC[0] + i_DeeCrys*MaxDirections;}
      for(Int_t i=0; i<MaxDeeCrys; i++)
	{for(Int_t j=0; j<MaxDirections; j++){fT2dDeeSC[i][j]=0;}}

      fT2dSCEcha = new Int_t*[MaxDeeCrys];                       fCnew++;
      fT1dSCEcha = new  Int_t[MaxDeeCrys*MaxDirections];                    fCnew++;   
      for(Int_t i_DeeCrys = 0 ; i_DeeCrys < MaxDeeCrys ; i_DeeCrys++){
	fT2dSCEcha[i_DeeCrys] = &fT1dSCEcha[0] + i_DeeCrys*MaxDirections;}
      for(Int_t i=0; i<MaxDeeCrys; i++)
	{for(Int_t j=0; j<MaxDirections; j++){fT2dSCEcha[i][j]=0;}}      

      //................................ Build table
      Int_t MaxTyp    = (Int_t)4;
      Int_t MaxCrysP1 = (Int_t)(fEcal->MaxCrysInSC()+1);

      //      Int_t fT2d_jch_JY[type][Echa+1]:  JY in the SC as a function of type and Echa+1
      fT2d_jch_JY    = new Int_t*[MaxTyp];                      fCnew++;  
      fT1d_jch_JY = new Int_t[MaxTyp*MaxCrysP1];             fCnew++;
      for(Int_t i_MaxTyp = 0 ; i_MaxTyp < MaxTyp ; i_MaxTyp++){
	fT2d_jch_JY[i_MaxTyp] = &fT1d_jch_JY[0] + i_MaxTyp*MaxCrysP1;}

      // type: 0=(top/right), 1=(top/left), 2=(bottom/left), 3=(bottom/right),  

      //................. top/right
      for(Int_t k= 5;k>= 1;k--){fT2d_jch_JY[0][k] = 4;}     //  k =  5, 4, 3, 2, 1 -> fT2d_jch_JY[0][k] = 4
      for(Int_t k=10;k>= 6;k--){fT2d_jch_JY[0][k] = 3;}     //  k = 10, 9, 8, 7, 6 -> fT2d_jch_JY[0][k] = 3
      for(Int_t k=15;k>=11;k--){fT2d_jch_JY[0][k] = 2;}     //  k = 15,14,13,12,11 -> fT2d_jch_JY[0][k] = 2
      for(Int_t k=20;k>=16;k--){fT2d_jch_JY[0][k] = 1;}     //  k = 20,19,18,17,16 -> fT2d_jch_JY[0][k] = 1
      for(Int_t k=25;k>=21;k--){fT2d_jch_JY[0][k] = 0;}     //  k = 25,24,23,22,21 -> fT2d_jch_JY[0][k] = 0
      //................. top/left
      for(Int_t k= 5;k>= 1;k--){fT2d_jch_JY[1][k] =  5-k;}  //  k =  5, 4, 3, 2, 1 -> fT2d_jch_JY[1][k] = 0,1,2,3,4
      for(Int_t k=10;k>= 6;k--){fT2d_jch_JY[1][k] = 10-k;}  //  k = 10, 9, 8, 7, 6 -> fT2d_jch_JY[1][k] = 0,1,2,3,4
      for(Int_t k=15;k>=11;k--){fT2d_jch_JY[1][k] = 15-k;}  //  k = 15,14,13,12,11 -> fT2d_jch_JY[1][k] = 0,1,2,3,4
      for(Int_t k=20;k>=16;k--){fT2d_jch_JY[1][k] = 20-k;}  //  k = 20,19,18,17,16 -> fT2d_jch_JY[1][k] = 0,1,2,3,4
      for(Int_t k=25;k>=21;k--){fT2d_jch_JY[1][k] = 25-k;}  //  k = 25,24,23,22,21 -> fT2d_jch_JY[1][k] = 0,1,2,3,4
      //................. bottom/left
      for(Int_t k= 1;k<=5; k++){fT2d_jch_JY[2][k] = 0;}     //  k =  1, 2, 3, 4, 5 -> fT2d_jch_JY[2][k] = 0
      for(Int_t k= 6;k<=10;k++){fT2d_jch_JY[2][k] = 1;}     //  k =  6, 7, 8, 9,10 -> fT2d_jch_JY[2][k] = 1
      for(Int_t k=11;k<=15;k++){fT2d_jch_JY[2][k] = 2;}     //  k = 11,12,13,14,15 -> fT2d_jch_JY[2][k] = 2
      for(Int_t k=16;k<=20;k++){fT2d_jch_JY[2][k] = 3;}     //  k = 16,17,18,19,20 -> fT2d_jch_JY[2][k] = 3
      for(Int_t k=21;k<=25;k++){fT2d_jch_JY[2][k] = 4;}     //  k = 21,22,23,24,25 -> fT2d_jch_JY[2][k] = 4
      //................. bottom/right
      for(Int_t k= 1;k<=5; k++){fT2d_jch_JY[3][k] = k-1;}   //  k =  1, 2, 3, 4, 5 -> fT2d_jch_JY[3][k] = 0,1,2,3,4
      for(Int_t k= 6;k<=10;k++){fT2d_jch_JY[3][k] = k-6;}   //  k =  6, 7, 8, 9,10 -> fT2d_jch_JY[3][k] = 0,1,2,3,4
      for(Int_t k=11;k<=15;k++){fT2d_jch_JY[3][k] = k-11;}  //  k = 11,12,13,14,15 -> fT2d_jch_JY[3][k] = 0,1,2,3,4
      for(Int_t k=16;k<=20;k++){fT2d_jch_JY[3][k] = k-16;}  //  k = 16,17,18,19,20 -> fT2d_jch_JY[3][k] = 0,1,2,3,4
      for(Int_t k=21;k<=25;k++){fT2d_jch_JY[3][k] = k-21;}  //  k = 21,22,23,24,25 -> fT2d_jch_JY[3][k] = 0,1,2,3,4

      //      Int_t fT2d_ich_IX[type][Echa+1]:  IX in the SC as a function of type and Echa+1
      fT2d_ich_IX = new Int_t*[MaxTyp];                      fCnew++;  
      fT1d_ich_IX = new Int_t[MaxTyp*MaxCrysP1];             fCnew++;
      for(Int_t i_MaxTyp = 0 ; i_MaxTyp < MaxTyp ; i_MaxTyp++){
	fT2d_ich_IX[i_MaxTyp] = &fT1d_ich_IX[0] + i_MaxTyp*MaxCrysP1;}

      //................. top/right
      for(Int_t k= 5;k>= 1;k--){fT2d_ich_IX[0][k] =  5-k;}  //  k =  5, 4, 3, 2, 1 -> fT2d_ich_IX[0][k] = 0,1,2,3,4
      for(Int_t k=10;k>= 6;k--){fT2d_ich_IX[0][k] = 10-k;}  //  k = 10, 9, 8, 7, 6 -> fT2d_ich_IX[0][k] = 0,1,2,3,4
      for(Int_t k=15;k>=11;k--){fT2d_ich_IX[0][k] = 15-k;}  //  k = 15,14,13,12,11 -> fT2d_ich_IX[0][k] = 0,1,2,3,4
      for(Int_t k=20;k>=16;k--){fT2d_ich_IX[0][k] = 20-k;}  //  k = 20,19,18,17,16 -> fT2d_ich_IX[0][k] = 0,1,2,3,4
      for(Int_t k=25;k>=21;k--){fT2d_ich_IX[0][k] = 25-k;}  //  k = 25,24,23,22,21 -> fT2d_ich_IX[0][k] = 0,1,2,3,4
      //................. top/left      
      for(Int_t k= 5;k>= 1;k--){fT2d_ich_IX[1][k] = 4;}     //  k =  5, 4, 3, 2, 1 -> fT2d_ich_IX[1][k] = 4
      for(Int_t k=10;k>= 6;k--){fT2d_ich_IX[1][k] = 3;}     //  k = 10, 9, 8, 7, 6 -> fT2d_ich_IX[1][k] = 3
      for(Int_t k=15;k>=11;k--){fT2d_ich_IX[1][k] = 2;}     //  k = 15,14,13,12,11 -> fT2d_ich_IX[1][k] = 2
      for(Int_t k=20;k>=16;k--){fT2d_ich_IX[1][k] = 1;}     //  k = 20,19,18,17,16 -> fT2d_ich_IX[1][k] = 1
      for(Int_t k=25;k>=21;k--){fT2d_ich_IX[1][k] = 0;}     //  k = 25,24,23,22,21 -> fT2d_ich_IX[1][k] = 0
      //................. bottom/left
      for(Int_t k=1; k<=5; k++){fT2d_ich_IX[2][k] =  5-k;}  //  k =  1, 2, 3, 4, 5 -> fT2d_ich_IX[2][k] = 0,1,2,3,4
      for(Int_t k=6; k<=10;k++){fT2d_ich_IX[2][k] = 10-k;}  //  k =  6, 7, 8, 9,10 -> fT2d_ich_IX[2][k] = 0,1,2,3,4
      for(Int_t k=11;k<=15;k++){fT2d_ich_IX[2][k] = 15-k;}  //  k = 11,12,13,14,15 -> fT2d_ich_IX[2][k] = 0,1,2,3,4
      for(Int_t k=16;k<=20;k++){fT2d_ich_IX[2][k] = 20-k;}  //  k = 16,17,18,19,20 -> fT2d_ich_IX[2][k] = 0,1,2,3,4
      for(Int_t k=21;k<=25;k++){fT2d_ich_IX[2][k] = 25-k;}  //  k = 21,22,23,24,25 -> fT2d_ich_IX[2][k] = 0,1,2,3,4
      //................. bottom/right      
      for(Int_t k= 1;k<= 5;k++){fT2d_ich_IX[3][k] = 4;}     //  k =  1, 2, 3, 4, 5 -> fT2d_ich_IX[3][k] = 4
      for(Int_t k= 6;k<=10;k++){fT2d_ich_IX[3][k] = 3;}     //  k =  6, 7, 8, 9,10 -> fT2d_ich_IX[3][k] = 3
      for(Int_t k=11;k<=15;k++){fT2d_ich_IX[3][k] = 2;}     //  k = 11,12,13,14,15 -> fT2d_ich_IX[3][k] = 2
      for(Int_t k=16;k<=20;k++){fT2d_ich_IX[3][k] = 1;}     //  k = 16,17,18,19,20 -> fT2d_ich_IX[3][k] = 1
      for(Int_t k=21;k<=25;k++){fT2d_ich_IX[3][k] = 0;}     //  k = 21,22,23,24,25 -> fT2d_ich_IX[3][k] = 0

      //............................................ type
      Int_t  Nb_DeeSC_JY  = fEcal->MaxSCIYInDee();      
      Int_t** type    = new Int_t*[Nb_DeeSC_JY];                   fCnew++; 
      Int_t*  type_d1 = new Int_t[Nb_DeeSC_JY*MaxDirections];      fCnew++;  
      for(Int_t i_DeeSC_JY = 0 ; i_DeeSC_JY < Nb_DeeSC_JY ; i_DeeSC_JY++){
	type[i_DeeSC_JY] = &type_d1[0] + i_DeeSC_JY*MaxDirections;}

      //  bottom = (0,9), top  = (10,19)
      //  right  = 0    , left = 1
      //  type = Quadrant number - 1

      type[10][0] = 0;                   // Q1 top right
      type[11][0] = 0;
      type[12][0] = 0;
      type[13][0] = 0;
      type[14][0] = 0;
      type[15][0] = 0;
      type[16][0] = 0;
      type[17][0] = 0;
      type[18][0] = 0;
      type[19][0] = 0;
 
      type[10][1] = 1;                   // Q2 top left    
      type[11][1] = 1;
      type[12][1] = 1;
      type[13][1] = 1;
      type[14][1] = 1;
      type[15][1] = 1;
      type[16][1] = 1;
      type[17][1] = 1;
      type[18][1] = 1;
      type[19][1] = 1;

      type[ 0][1] = 2;                   // Q3 : bottom left 
      type[ 1][1] = 2;
      type[ 2][1] = 2;
      type[ 3][1] = 2;
      type[ 4][1] = 2;
      type[ 5][1] = 2;
      type[ 6][1] = 2;
      type[ 7][1] = 2;
      type[ 8][1] = 2;
      type[ 9][1] = 2;
 
      type[ 0][0] = 3;                   // Q4 : bottom right 
      type[ 1][0] = 3;
      type[ 2][0] = 3;
      type[ 3][0] = 3;
      type[ 4][0] = 3;
      type[ 5][0] = 3;
      type[ 6][0] = 3;
      type[ 7][0] = 3;
      type[ 8][0] = 3;
      type[ 9][0] = 3;

      Int_t Nb_SCCrys_IX = fEcal->MaxCrysIXInSC();
      Int_t Nb_SCCrys_JY = fEcal->MaxCrysIYInSC();

      for(Int_t kSC=0; kSC<MaxDeeSC; kSC++)                   //  kSC  = 0 to 199   (MaxDeeSC = 200)
	{
	  for(Int_t n_Echa=1; n_Echa<=MaxSCEcha; n_Echa++)       //  n_Echa   = 1 to 25    (MaxSCEcha = 25) 
	    {
	      for(Int_t idir=0; idir<2; idir++)
		{
		  Int_t ikSC = kSC / Nb_DeeSC_JY;                 //  ikSC = 0 to 9
		  Int_t jkSC = kSC % Nb_DeeSC_JY;                 //  jkSC = 0,1,2,..,19 
		  
		  Int_t icrys = ikSC*Nb_SCCrys_IX + fT2d_ich_IX[type[jkSC][idir]][n_Echa];
		  //  type[0->9][1->2] = 0,1,2,3
		  //  fT2d_ich_IX[0->3][1->25] = 0,1,2,3,4
		  //  icrys = 0 to 49  (=> IX)
		  
		  Int_t jcrys = jkSC*Nb_SCCrys_JY + fT2d_jch_JY[type[jkSC][idir]][n_Echa];
		  //  type[0->9][1->2] = 0,1,2,3
		  //  fT2d_jch_JY[0->3][1->25] = 0,1,2,3,4
		  //  jcrys = 0 to 99  (=> IY)
		  
		  Int_t n_DeeCrys = icrys*Nb_DeeSC_JY*Nb_SCCrys_JY+jcrys+1;    //  n_DeeCrys = 1 to 5000
		  
		  fT3dDeeCrys[kSC][n_Echa-1][idir] = n_DeeCrys;   // fT3dDeeCrys[][][] : range = [1,5000]
		  fT2dDeeSC[n_DeeCrys-1][idir]     = kSC+1;       // fT2dDeeSC[][]     : range = [1,200]
		  fT2dSCEcha[n_DeeCrys-1][idir]    = n_Echa;      // fT2dSCEcha[][]    : range = [1,25]
		}
	    }
	}
      // cout << "#TEcnaNumbering::TBuildEndcapCrysTable()> Crys Table Building done" << endl;

      delete [] type;                           fCdelete++;
      delete [] type_d1;                        fCdelete++;
    }
  else
    {
      // cout << "#TEcnaNumbering::TBuildEndcapCrysTable()> No Building of Crys Table since it is already done " << endl;
    }
}

void TEcnaNumbering::BuildEndcapSCTable()
{
// Build the correspondance table: (Dee, DeeSC) <-> (DS , DSSC) for the ECAL ENDCAP
//
//  From CMS Internal Note  "CMS ECAL Endcap channel numbering"
//
//       Name       Number                     Reference set    Range     Comment
//
//       Dee      : Dee number                                  [1,4]
//       DeeSC    : Super-Crystal (SC) number  in Dee           [1,200]   (IY,IX) progression
//       DS       : Data Sector number         in EE + or -     [1,9]     (IY,IX) progression
//       DSSC     : Super-Crystal (SC) number  in Data Sector   [1,32]    Crystal numbering in data sector
//       DeeSCCons: Super-Crystal (SC) number  for construction [1,297]                      
// 
//   fill the 2d arrays:  fT2d_DS[4][200], fT2d_DSSC[4][200] and fT2d_DeeSCCons[4][200]
//
//------------------------------------------------------------------------------------------------------

//.................. Allocations and Init
  Int_t MaxEEDee       = fEcal->MaxDeeInEE();
  Int_t MaxDeeSC       = fEcal->MaxSCEcnaInDee();
  Int_t MaxEESCForCons = 2*fEcal->MaxSCForConsInDee();

  fT2d_DS = new Int_t*[MaxEEDee];                       fCnew++;    // = DS[Dee - 1, CNA_SCInDee - 1]
  fT1d_DS = new  Int_t[MaxEEDee*MaxDeeSC];              fCnew++;
  for(Int_t i_DeeCrys = 0 ; i_DeeCrys < MaxEEDee ; i_DeeCrys++){
    fT2d_DS[i_DeeCrys] = &fT1d_DS[0] + i_DeeCrys*MaxDeeSC;}
  for(Int_t i=0; i<MaxEEDee; i++)
    {for(Int_t j=0; j<MaxDeeSC; j++){fT2d_DS[i][j]=0;}}

  fT2d_DSSC = new Int_t*[MaxEEDee];                     fCnew++;    // = SCInDS[Dee - 1, CNA_SCInDee - 1]  
  fT1d_DSSC = new  Int_t[MaxEEDee*MaxDeeSC];            fCnew++;
  for(Int_t i_DeeCrys = 0 ; i_DeeCrys < MaxEEDee ; i_DeeCrys++){
    fT2d_DSSC[i_DeeCrys] = &fT1d_DSSC[0] + i_DeeCrys*MaxDeeSC;}
  for(Int_t i=0; i<MaxEEDee; i++)
    {for(Int_t j=0; j<MaxDeeSC; j++){fT2d_DSSC[i][j]=0;}}

  fT2d_DeeSCCons = new Int_t*[MaxEEDee];                fCnew++;    // = SCConsInDee[Dee - 1, CNA_SCInDee - 1]
  fT1d_DeeSCCons = new  Int_t[MaxEEDee*MaxDeeSC];       fCnew++;
  for(Int_t i_DeeCrys = 0 ; i_DeeCrys < MaxEEDee ; i_DeeCrys++){
    fT2d_DeeSCCons[i_DeeCrys] = &fT1d_DeeSCCons[0] + i_DeeCrys*MaxDeeSC;}
  for(Int_t i=0; i<MaxEEDee; i++)
    {for(Int_t j=0; j<MaxDeeSC; j++){fT2d_DeeSCCons[i][j]=0;}}

  fT2d_RecovDeeSC = new Int_t*[MaxEEDee];               fCnew++;    // = CNA_SCInDee[Dee - 1, SCConsInDee - 1]
  fT1d_RecovDeeSC = new  Int_t[MaxEEDee*MaxEESCForCons];  fCnew++;
  for(Int_t i_DeeCrys = 0 ; i_DeeCrys < MaxEEDee ; i_DeeCrys++){
    fT2d_RecovDeeSC[i_DeeCrys] = &fT1d_RecovDeeSC[0] + i_DeeCrys*MaxEESCForCons;}
  for(Int_t i=0; i<MaxEEDee; i++)
    {for(Int_t j=0; j<MaxEESCForCons; j++){fT2d_RecovDeeSC[i][j]=0;}}

  //.............................. Data sector (DS) numbers: fT2d_DS[][]
  //                               isc = ECNA numbers

  Int_t ids = 0;

  //........... (D1,S1)=(D2,S9)=(D3,S9)=(D4,S1)
  for(Int_t dee = 1; dee<=4; dee++)
    {
      if( dee == 1 || dee == 4 ){ids = 1;}
      if( dee == 2 || dee == 3 ){ids = 9;}
      for(Int_t isc= 13; isc<= 20; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc= 33; isc<= 40; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc= 54; isc<= 60; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc= 75; isc<= 79; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc= 96; isc<= 99; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc=118; isc<=119; isc++)fT2d_DS[dee-1][isc-1] = ids;
    }
  //........... (D1,S2)=(D2,S8)=(D3,S8)=(D4,S2)
  for(Int_t dee = 1; dee<=4; dee++)
    {
      if( dee == 1 || dee == 4 ){ids = 2;}
      if( dee == 2 || dee == 3 ){ids = 8;}
      for(Int_t isc= 32; isc<= 32; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc= 51; isc<= 53; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc= 72; isc<= 74; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc= 92; isc<= 95; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc=112; isc<=117; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc=132; isc<=138; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc=152; isc<=157; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc=173; isc<=176; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc=193; isc<=193; isc++)fT2d_DS[dee-1][isc-1] = ids;
    }
  //........... (D1,S3)=(D2,S7)=(D3,S7)=(D4,S3)
  for(Int_t dee = 1; dee<=4; dee++)
    {
      if( dee == 1 || dee == 4 ){ids = 3;}
      if( dee == 2 || dee == 3 ){ids = 7;}
      for(Int_t isc= 50; isc<= 50; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc= 69; isc<= 71; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc= 88; isc<= 91; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc=108; isc<=111; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc=127; isc<=131; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc=147; isc<=151; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc=166; isc<=172; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc=188; isc<=192; isc++)fT2d_DS[dee-1][isc-1] = ids;
    }
  //........... (D1,S4)=(D2,S6)=(D3,S6)=(D4,S4)
  for(Int_t dee = 1; dee<=4; dee++)
    {
      if( dee == 1 || dee == 4 ){ids = 4;}
      if( dee == 2 || dee == 3 ){ids = 6;}
      for(Int_t isc= 27; isc<= 29; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc= 44; isc<= 49; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc= 62; isc<= 68; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc= 82; isc<= 87; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc=102; isc<=107; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc=123; isc<=126; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc=144; isc<=146; isc++)fT2d_DS[dee-1][isc-1] = ids;
      for(Int_t isc=165; isc<=165; isc++)fT2d_DS[dee-1][isc-1] = ids;
    }
  //........... (D1,S5)=(D2,S5)=(D3,S5)=(D4,S5)
  for(Int_t dee = 1; dee<=4; dee++)
    {
      for(Int_t isc=  1; isc<=  8; isc++)fT2d_DS[dee-1][isc-1] = 5;
      for(Int_t isc= 21; isc<= 26; isc++)fT2d_DS[dee-1][isc-1] = 5;
      for(Int_t isc= 41; isc<= 43; isc++)fT2d_DS[dee-1][isc-1] = 5;
    }
  
  //............................................ SC numbers in Data Sectors: fT2d_DSSC[][]
  //                                             fT2d_DSSC[dee-1][SC ECNA number - 1] = SC number in DS;
  for(Int_t dee=1; dee<=4; dee++)
    {
      for(Int_t isc=1; isc<=MaxDeeSC; isc++)
	{fT2d_DSSC[dee-1][isc-1] = -1;}
      //.......................................... S1 (D1,D4), S9 (D2,D3) ; 33 SC's
      fT2d_DSSC[dee-1][13-1] = 12;
      fT2d_DSSC[dee-1][14-1] = 11;
      fT2d_DSSC[dee-1][15-1] = 10;
      fT2d_DSSC[dee-1][16-1] =  9;
      fT2d_DSSC[dee-1][17-1] =  4;
      fT2d_DSSC[dee-1][18-1] =  3;
      fT2d_DSSC[dee-1][19-1] =  2;
      fT2d_DSSC[dee-1][20-1] =  1;

      fT2d_DSSC[dee-1][33-1] = 16;
      fT2d_DSSC[dee-1][34-1] = 15;
      fT2d_DSSC[dee-1][35-1] = 14;
      fT2d_DSSC[dee-1][36-1] = 13;
      fT2d_DSSC[dee-1][37-1] =  8;
      fT2d_DSSC[dee-1][38-1] =  7;
      fT2d_DSSC[dee-1][39-1] =  6;
      fT2d_DSSC[dee-1][40-1] =  5;

      fT2d_DSSC[dee-1][54-1] = 33;
      fT2d_DSSC[dee-1][55-1] = 31;
      fT2d_DSSC[dee-1][56-1] = 27;
      fT2d_DSSC[dee-1][57-1] = 24;
      fT2d_DSSC[dee-1][58-1] = 20;
      fT2d_DSSC[dee-1][59-1] = 17;
      fT2d_DSSC[dee-1][60-1] = 30;  //  (182a, 33a for construction)
      
      fT2d_DSSC[dee-1][75-1] = 32;
      fT2d_DSSC[dee-1][76-1] = 28;
      fT2d_DSSC[dee-1][77-1] = 25;
      fT2d_DSSC[dee-1][78-1] = 21;
      fT2d_DSSC[dee-1][79-1] = 18;
      
      fT2d_DSSC[dee-1][96-1] = 29;
      fT2d_DSSC[dee-1][97-1] = 26;
      fT2d_DSSC[dee-1][98-1] = 22;
      fT2d_DSSC[dee-1][99-1] = 19;

      fT2d_DSSC[dee-1][118-1] = 23;
      fT2d_DSSC[dee-1][119-1] = 30;  //  (182b, 33b for construction)

      //.......................................... S2 (D1,D4), S8(D2,D3) ; 32 SC's
      fT2d_DSSC[dee-1][32-1] = 25;  // also 3;  // ( (207c, 58c) also (178c, 29c) for construction)

      fT2d_DSSC[dee-1][51-1] = 32;
      fT2d_DSSC[dee-1][52-1] = 26;
      fT2d_DSSC[dee-1][53-1] = 18;

      fT2d_DSSC[dee-1][72-1] = 27;
      fT2d_DSSC[dee-1][73-1] = 19;
      fT2d_DSSC[dee-1][74-1] = 12;
      
      fT2d_DSSC[dee-1][92-1] = 28;
      fT2d_DSSC[dee-1][93-1] = 20;
      fT2d_DSSC[dee-1][94-1] = 13;
      fT2d_DSSC[dee-1][95-1] =  7;

      fT2d_DSSC[dee-1][112-1] = 29;
      fT2d_DSSC[dee-1][113-1] = 21;
      fT2d_DSSC[dee-1][114-1] = 14;
      fT2d_DSSC[dee-1][115-1] =  8;
      fT2d_DSSC[dee-1][116-1] =  4;
      fT2d_DSSC[dee-1][117-1] =  1;

      fT2d_DSSC[dee-1][132-1] = 30;
      fT2d_DSSC[dee-1][133-1] = 22;
      fT2d_DSSC[dee-1][134-1] = 15;
      fT2d_DSSC[dee-1][135-1] =  9;
      fT2d_DSSC[dee-1][136-1] =  5;
      fT2d_DSSC[dee-1][137-1] =  2;
      fT2d_DSSC[dee-1][138-1] =  3;  // (178a, 29a for construction)

      fT2d_DSSC[dee-1][152-1] = 31;
      fT2d_DSSC[dee-1][153-1] = 23;
      fT2d_DSSC[dee-1][154-1] = 16;
      fT2d_DSSC[dee-1][155-1] = 10;
      fT2d_DSSC[dee-1][156-1] =  6;
      fT2d_DSSC[dee-1][157-1] =  3;  // (178b, 29b for construction)

      fT2d_DSSC[dee-1][173-1] = 24;
      fT2d_DSSC[dee-1][174-1] = 17;
      fT2d_DSSC[dee-1][175-1] = 11;
      fT2d_DSSC[dee-1][176-1] = 25;  // (207a, 58a for construction)

      fT2d_DSSC[dee-1][193-1] = 25;  // (207b, 58b for construction)

      //.......................................... S3 (D1,D4), S7 (D2,D3)  ; 34 SC's
      fT2d_DSSC[dee-1][50-1] = 10;

      fT2d_DSSC[dee-1][69-1] = 18;
      fT2d_DSSC[dee-1][70-1] = 11;
      fT2d_DSSC[dee-1][71-1] =  3;

      fT2d_DSSC[dee-1][88-1] = 25;
      fT2d_DSSC[dee-1][89-1] = 19;
      fT2d_DSSC[dee-1][90-1] = 12;
      fT2d_DSSC[dee-1][91-1] =  4;

      fT2d_DSSC[dee-1][108-1] = 26;
      fT2d_DSSC[dee-1][109-1] = 20;
      fT2d_DSSC[dee-1][110-1] = 13;
      fT2d_DSSC[dee-1][111-1] =  5;

      fT2d_DSSC[dee-1][127-1] = 31;
      fT2d_DSSC[dee-1][128-1] = 27;
      fT2d_DSSC[dee-1][129-1] = 21;
      fT2d_DSSC[dee-1][130-1] = 14;
      fT2d_DSSC[dee-1][131-1] =  6;

      fT2d_DSSC[dee-1][147-1] = 32;
      fT2d_DSSC[dee-1][148-1] = 28;
      fT2d_DSSC[dee-1][149-1] = 22;
      fT2d_DSSC[dee-1][150-1] = 15;
      fT2d_DSSC[dee-1][151-1] =  7;

      fT2d_DSSC[dee-1][166-1] = 33;
      fT2d_DSSC[dee-1][167-1] = 30;
      fT2d_DSSC[dee-1][168-1] = 29;
      fT2d_DSSC[dee-1][169-1] = 23;
      fT2d_DSSC[dee-1][170-1] = 16;
      fT2d_DSSC[dee-1][171-1] =  8;
      fT2d_DSSC[dee-1][172-1] =  1;

      fT2d_DSSC[dee-1][188-1] = 34; // (298a, 149a for construction)
      fT2d_DSSC[dee-1][189-1] = 24;
      fT2d_DSSC[dee-1][190-1] = 17;
      fT2d_DSSC[dee-1][191-1] =  9;
      fT2d_DSSC[dee-1][192-1] =  2;

      //.......................................... S4 (D1,D4), S6 (D2,D3) ; 33 SC's
      fT2d_DSSC[dee-1][27-1] = 33;
      fT2d_DSSC[dee-1][28-1] = 32;
      fT2d_DSSC[dee-1][29-1] = 14; // also 21;  //  ( (261a, 112a) also (268a, 119a) for construction)

      fT2d_DSSC[dee-1][44-1] = 22;
      fT2d_DSSC[dee-1][45-1] = 15;
      fT2d_DSSC[dee-1][46-1] =  8;
      fT2d_DSSC[dee-1][47-1] =  4;
      fT2d_DSSC[dee-1][48-1] =  2;
      fT2d_DSSC[dee-1][49-1] =  1;

      fT2d_DSSC[dee-1][62-1] = 29;
      fT2d_DSSC[dee-1][63-1] = 28;
      fT2d_DSSC[dee-1][64-1] = 23;
      fT2d_DSSC[dee-1][65-1] = 16;
      fT2d_DSSC[dee-1][66-1] =  9;
      fT2d_DSSC[dee-1][67-1] =  5;
      fT2d_DSSC[dee-1][68-1] =  3;

      fT2d_DSSC[dee-1][82-1] = 31;
      fT2d_DSSC[dee-1][83-1] = 30;
      fT2d_DSSC[dee-1][84-1] = 24;
      fT2d_DSSC[dee-1][85-1] = 17;
      fT2d_DSSC[dee-1][86-1] = 10;
      fT2d_DSSC[dee-1][87-1] =  6;

      fT2d_DSSC[dee-1][102-1] = 21;  //   (268c, 119c for construction)
      fT2d_DSSC[dee-1][103-1] = 27;
      fT2d_DSSC[dee-1][104-1] = 25;
      fT2d_DSSC[dee-1][105-1] = 18;
      fT2d_DSSC[dee-1][106-1] = 11;
      fT2d_DSSC[dee-1][107-1] =  7;

      fT2d_DSSC[dee-1][123-1] = 21;  //   (268b, 119b for construction)
      fT2d_DSSC[dee-1][124-1] = 26;
      fT2d_DSSC[dee-1][125-1] = 19;
      fT2d_DSSC[dee-1][126-1] = 12;

      fT2d_DSSC[dee-1][144-1] = 14;  //   (261c, 112c for construction)
      fT2d_DSSC[dee-1][145-1] = 20;
      fT2d_DSSC[dee-1][146-1] = 13;

      fT2d_DSSC[dee-1][165-1] = 14;  //   (261b, 112b for construction)

      //.......................................... S5   (2*17 = 34 SC's)
      if(dee == 1 || dee == 3)
	{
	  fT2d_DSSC[dee-1][1-1] = 34;
	  fT2d_DSSC[dee-1][2-1] = 33;
	  fT2d_DSSC[dee-1][3-1] = 32;
	  fT2d_DSSC[dee-1][4-1] = 31;
	  fT2d_DSSC[dee-1][5-1] = 26;
	  fT2d_DSSC[dee-1][6-1] = 25;
	  fT2d_DSSC[dee-1][7-1] = 24;
	  fT2d_DSSC[dee-1][8-1] = 23;

	  fT2d_DSSC[dee-1][21-1] = 30;
	  fT2d_DSSC[dee-1][22-1] = 29;
	  fT2d_DSSC[dee-1][23-1] = 28;
	  fT2d_DSSC[dee-1][24-1] = 27;
	  fT2d_DSSC[dee-1][25-1] = 22;
	  fT2d_DSSC[dee-1][26-1] = 21;

	  fT2d_DSSC[dee-1][41-1] = 20;  //   (281a for construction)
	  fT2d_DSSC[dee-1][42-1] = 19;
	  fT2d_DSSC[dee-1][43-1] = 18;
	}

      if(dee == 2 || dee == 4)
	{
	  fT2d_DSSC[dee-1][1-1] = 17;
	  fT2d_DSSC[dee-1][2-1] = 16;
	  fT2d_DSSC[dee-1][3-1] = 15;
	  fT2d_DSSC[dee-1][4-1] = 14;
	  fT2d_DSSC[dee-1][5-1] =  9;
	  fT2d_DSSC[dee-1][6-1] =  8;
	  fT2d_DSSC[dee-1][7-1] =  7;
	  fT2d_DSSC[dee-1][8-1] =  6;

	  fT2d_DSSC[dee-1][21-1] = 13;
	  fT2d_DSSC[dee-1][22-1] = 12;
	  fT2d_DSSC[dee-1][23-1] = 11;
	  fT2d_DSSC[dee-1][24-1] = 10;
	  fT2d_DSSC[dee-1][25-1] =  5;
	  fT2d_DSSC[dee-1][26-1] =  4;

	  fT2d_DSSC[dee-1][41-1] = 3;  //   (132a for construction)
	  fT2d_DSSC[dee-1][42-1] = 2;
	  fT2d_DSSC[dee-1][43-1] = 1;
	}
    }
  //............................... Numbers for construction: fT2d_DeeSCCons[][]
  //                                fT2d_DeeSCCons[dee-1][SC ECNA number - 1] = SC number for construction;

  //............................... init to -1
  for(Int_t dee=1; dee<=4; dee++)
    {for(Int_t isc=1; isc<=MaxDeeSC; isc++)
	{fT2d_DeeSCCons[dee-1][isc-1] = -1;}}

  for(Int_t i_dee_type=1; i_dee_type<=2; i_dee_type++)
    {
      Int_t dee = -1;
      if( i_dee_type == 1 ){dee = 1;}
      if( i_dee_type == 2 ){dee = 3;}
      
      //.......................................... (D1,S1 or D3,S9) AND (D2,S9 or D4,S1)
      //  number in comment = fT2d_DSSC[dee-1][SC ECNA nb - 1] (= SC number in DS)
      fT2d_DeeSCCons[dee-1][13-1] = 161; fT2d_DeeSCCons[dee][13-1] = 12; // 12;
      fT2d_DeeSCCons[dee-1][14-1] = 160; fT2d_DeeSCCons[dee][14-1] = 11; // 11;
      fT2d_DeeSCCons[dee-1][15-1] = 159; fT2d_DeeSCCons[dee][15-1] = 10; // 10;
      fT2d_DeeSCCons[dee-1][16-1] = 158; fT2d_DeeSCCons[dee][16-1] =  9; //  9;
      fT2d_DeeSCCons[dee-1][17-1] = 153; fT2d_DeeSCCons[dee][17-1] =  4; //  4;
      fT2d_DeeSCCons[dee-1][18-1] = 152; fT2d_DeeSCCons[dee][18-1] =  3; //  3;
      fT2d_DeeSCCons[dee-1][19-1] = 151; fT2d_DeeSCCons[dee][19-1] =  2; //  2;
      fT2d_DeeSCCons[dee-1][20-1] = 150; fT2d_DeeSCCons[dee][20-1] =  1; //  1;
      
      fT2d_DeeSCCons[dee-1][33-1] = 165; fT2d_DeeSCCons[dee][33-1] = 16; // 16;
      fT2d_DeeSCCons[dee-1][34-1] = 164; fT2d_DeeSCCons[dee][34-1] = 15; // 15;
      fT2d_DeeSCCons[dee-1][35-1] = 163; fT2d_DeeSCCons[dee][35-1] = 14; // 14;
      fT2d_DeeSCCons[dee-1][36-1] = 162; fT2d_DeeSCCons[dee][36-1] = 13; // 13;
      fT2d_DeeSCCons[dee-1][37-1] = 157; fT2d_DeeSCCons[dee][37-1] =  8; //  8;
      fT2d_DeeSCCons[dee-1][38-1] = 156; fT2d_DeeSCCons[dee][38-1] =  7; //  7;
      fT2d_DeeSCCons[dee-1][39-1] = 155; fT2d_DeeSCCons[dee][39-1] =  6; //  6;
      fT2d_DeeSCCons[dee-1][40-1] = 154; fT2d_DeeSCCons[dee][40-1] =  5; //  5;
      
      fT2d_DeeSCCons[dee-1][54-1] = 193; fT2d_DeeSCCons[dee][54-1] = 44; // 33; 
      fT2d_DeeSCCons[dee-1][55-1] = 186; fT2d_DeeSCCons[dee][55-1] = 37; // 31;
      fT2d_DeeSCCons[dee-1][56-1] = 179; fT2d_DeeSCCons[dee][56-1] = 30; // 27;
      fT2d_DeeSCCons[dee-1][57-1] = 173; fT2d_DeeSCCons[dee][57-1] = 24; // 24;
      fT2d_DeeSCCons[dee-1][58-1] = 169; fT2d_DeeSCCons[dee][58-1] = 20; // 20;
      fT2d_DeeSCCons[dee-1][59-1] = 166; fT2d_DeeSCCons[dee][59-1] = 17; // 17;
      fT2d_DeeSCCons[dee-1][60-1] = 182; fT2d_DeeSCCons[dee][60-1] = 33; // 30;    // 182a ;  33a;

      fT2d_DeeSCCons[dee-1][75-1] = 187; fT2d_DeeSCCons[dee][75-1] = 38; // 32;
      fT2d_DeeSCCons[dee-1][76-1] = 180; fT2d_DeeSCCons[dee][76-1] = 31; // 28;
      fT2d_DeeSCCons[dee-1][77-1] = 174; fT2d_DeeSCCons[dee][77-1] = 25; // 25;
      fT2d_DeeSCCons[dee-1][78-1] = 170; fT2d_DeeSCCons[dee][78-1] = 21; // 21;
      fT2d_DeeSCCons[dee-1][79-1] = 167; fT2d_DeeSCCons[dee][79-1] = 18; // 18;
      
      fT2d_DeeSCCons[dee-1][96-1] = 181; fT2d_DeeSCCons[dee][96-1] = 32; // 29;
      fT2d_DeeSCCons[dee-1][97-1] = 175; fT2d_DeeSCCons[dee][97-1] = 26; // 26;
      fT2d_DeeSCCons[dee-1][98-1] = 171; fT2d_DeeSCCons[dee][98-1] = 22; // 22;
      fT2d_DeeSCCons[dee-1][99-1] = 168; fT2d_DeeSCCons[dee][99-1] = 19; // 19;
      
      fT2d_DeeSCCons[dee-1][118-1] = 172; fT2d_DeeSCCons[dee][118-1] = 23; // 23;
      fT2d_DeeSCCons[dee-1][119-1] = 182; fT2d_DeeSCCons[dee][119-1] = 33; // 30;    // 182b ;  33b;
     
      //.......................................... (D1,S2 or D3,S8) AND (D2,S8 or D4,S2)
      fT2d_DeeSCCons[dee-1][32-1] = 178; fT2d_DeeSCCons[dee][32-1] = 29; // 25-3;   // 178c-207c ; 29c-58c;

      fT2d_DeeSCCons[dee-1][51-1] = 216; fT2d_DeeSCCons[dee][51-1] = 67; // 32;
      fT2d_DeeSCCons[dee-1][52-1] = 208; fT2d_DeeSCCons[dee][52-1] = 59; // 26;
      fT2d_DeeSCCons[dee-1][53-1] = 200; fT2d_DeeSCCons[dee][53-1] = 51; // 18;
      
      fT2d_DeeSCCons[dee-1][72-1] = 209; fT2d_DeeSCCons[dee][72-1] = 60; // 27;
      fT2d_DeeSCCons[dee-1][73-1] = 201; fT2d_DeeSCCons[dee][73-1] = 52; // 19;
      fT2d_DeeSCCons[dee-1][74-1] = 194; fT2d_DeeSCCons[dee][74-1] = 45; // 12;
      
      fT2d_DeeSCCons[dee-1][92-1] = 210; fT2d_DeeSCCons[dee][92-1] = 61; // 28;
      fT2d_DeeSCCons[dee-1][93-1] = 202; fT2d_DeeSCCons[dee][93-1] = 53; // 20;
      fT2d_DeeSCCons[dee-1][94-1] = 195; fT2d_DeeSCCons[dee][94-1] = 46; // 13;
      fT2d_DeeSCCons[dee-1][95-1] = 188; fT2d_DeeSCCons[dee][95-1] = 39; //  7;
      
      fT2d_DeeSCCons[dee-1][112-1] = 211; fT2d_DeeSCCons[dee][112-1] = 62; // 29;
      fT2d_DeeSCCons[dee-1][113-1] = 203; fT2d_DeeSCCons[dee][113-1] = 54; // 21;
      fT2d_DeeSCCons[dee-1][114-1] = 196; fT2d_DeeSCCons[dee][114-1] = 47; // 14;
      fT2d_DeeSCCons[dee-1][115-1] = 189; fT2d_DeeSCCons[dee][115-1] = 40; //  8;
      fT2d_DeeSCCons[dee-1][116-1] = 183; fT2d_DeeSCCons[dee][116-1] = 34; //  4;
      fT2d_DeeSCCons[dee-1][117-1] = 176; fT2d_DeeSCCons[dee][117-1] = 27; //  1;
      
      fT2d_DeeSCCons[dee-1][132-1] = 212; fT2d_DeeSCCons[dee][132-1] = 63; // 30;
      fT2d_DeeSCCons[dee-1][133-1] = 204; fT2d_DeeSCCons[dee][133-1] = 55; // 22;
      fT2d_DeeSCCons[dee-1][134-1] = 197; fT2d_DeeSCCons[dee][134-1] = 48; // 15;
      fT2d_DeeSCCons[dee-1][135-1] = 190; fT2d_DeeSCCons[dee][135-1] = 41; //  9;
      fT2d_DeeSCCons[dee-1][136-1] = 184; fT2d_DeeSCCons[dee][136-1] = 35; //  5;
      fT2d_DeeSCCons[dee-1][137-1] = 177; fT2d_DeeSCCons[dee][137-1] = 28; //  2;
      fT2d_DeeSCCons[dee-1][138-1] = 178; fT2d_DeeSCCons[dee][138-1] = 29; //  3; //  178a ;  29a;
      
      fT2d_DeeSCCons[dee-1][152-1] = 213; fT2d_DeeSCCons[dee][152-1] = 64; // 31;
      fT2d_DeeSCCons[dee-1][153-1] = 205; fT2d_DeeSCCons[dee][153-1] = 56; // 23;
      fT2d_DeeSCCons[dee-1][154-1] = 198; fT2d_DeeSCCons[dee][154-1] = 49; // 16;
      fT2d_DeeSCCons[dee-1][155-1] = 191; fT2d_DeeSCCons[dee][155-1] = 42; // 10;
      fT2d_DeeSCCons[dee-1][156-1] = 185; fT2d_DeeSCCons[dee][156-1] = 36; //  6;
      fT2d_DeeSCCons[dee-1][157-1] = 178; fT2d_DeeSCCons[dee][157-1] = 29; //  3; //  178b ;  29b;
      
      fT2d_DeeSCCons[dee-1][173-1] = 206; fT2d_DeeSCCons[dee][173-1] = 57; // 24;
      fT2d_DeeSCCons[dee-1][174-1] = 199; fT2d_DeeSCCons[dee][174-1] = 50; // 17;
      fT2d_DeeSCCons[dee-1][175-1] = 192; fT2d_DeeSCCons[dee][175-1] = 43; // 11;
      fT2d_DeeSCCons[dee-1][176-1] = 207; fT2d_DeeSCCons[dee][176-1] = 58; // 25; //  58a ;  207a;
 
      fT2d_DeeSCCons[dee-1][193-1] = 207; fT2d_DeeSCCons[dee][193-1] = 58; // 25; //  58b ;  207b;
    
      //.......................................... (D1,S3 or D3,S7) AND  (D2,S7 or D4,S3) 
      fT2d_DeeSCCons[dee-1][50-1] = 224; fT2d_DeeSCCons[dee][50-1] = 75; // 10;
      
      fT2d_DeeSCCons[dee-1][69-1] = 233; fT2d_DeeSCCons[dee][69-1] = 84; // 18;
      fT2d_DeeSCCons[dee-1][70-1] = 225; fT2d_DeeSCCons[dee][70-1] = 76; // 11;
      fT2d_DeeSCCons[dee-1][71-1] = 217; fT2d_DeeSCCons[dee][71-1] = 68; //  3;
      
      fT2d_DeeSCCons[dee-1][88-1] = 242; fT2d_DeeSCCons[dee][88-1] = 93; // 25;
      fT2d_DeeSCCons[dee-1][89-1] = 234; fT2d_DeeSCCons[dee][89-1] = 85; // 19;
      fT2d_DeeSCCons[dee-1][90-1] = 226; fT2d_DeeSCCons[dee][90-1] = 77; // 12;
      fT2d_DeeSCCons[dee-1][91-1] = 218; fT2d_DeeSCCons[dee][91-1] = 69; //  4;
      
      fT2d_DeeSCCons[dee-1][108-1] = 243; fT2d_DeeSCCons[dee][108-1] = 94; // 26;
      fT2d_DeeSCCons[dee-1][109-1] = 235; fT2d_DeeSCCons[dee][109-1] = 86; // 20;
      fT2d_DeeSCCons[dee-1][110-1] = 227; fT2d_DeeSCCons[dee][110-1] = 78; // 13;
      fT2d_DeeSCCons[dee-1][111-1] = 219; fT2d_DeeSCCons[dee][111-1] = 70; //  5;
      
      fT2d_DeeSCCons[dee-1][127-1] = 252; fT2d_DeeSCCons[dee][127-1] = 103; // 31;
      fT2d_DeeSCCons[dee-1][128-1] = 244; fT2d_DeeSCCons[dee][128-1] =  95; // 27;
      fT2d_DeeSCCons[dee-1][129-1] = 236; fT2d_DeeSCCons[dee][129-1] =  87; // 21;
      fT2d_DeeSCCons[dee-1][130-1] = 228; fT2d_DeeSCCons[dee][130-1] =  79; // 14;
      fT2d_DeeSCCons[dee-1][131-1] = 220; fT2d_DeeSCCons[dee][131-1] =  71; //  6;
      
      fT2d_DeeSCCons[dee-1][147-1] = 253; fT2d_DeeSCCons[dee][147-1] = 104; // 32;
      fT2d_DeeSCCons[dee-1][148-1] = 245; fT2d_DeeSCCons[dee][148-1] =  96; // 28;
      fT2d_DeeSCCons[dee-1][149-1] = 237; fT2d_DeeSCCons[dee][149-1] =  88; // 22;
      fT2d_DeeSCCons[dee-1][150-1] = 229; fT2d_DeeSCCons[dee][150-1] =  80; // 15;
      fT2d_DeeSCCons[dee-1][151-1] = 221; fT2d_DeeSCCons[dee][151-1] =  72; //  7;
      
      fT2d_DeeSCCons[dee-1][166-1] = 254; fT2d_DeeSCCons[dee][166-1] = 105; // 33;
      fT2d_DeeSCCons[dee-1][167-1] = 247; fT2d_DeeSCCons[dee][167-1] =  98; // 30;
      fT2d_DeeSCCons[dee-1][168-1] = 246; fT2d_DeeSCCons[dee][168-1] =  97; // 29;
      fT2d_DeeSCCons[dee-1][169-1] = 238; fT2d_DeeSCCons[dee][169-1] =  89; // 23;
      fT2d_DeeSCCons[dee-1][170-1] = 230; fT2d_DeeSCCons[dee][170-1] =  81; // 16;
      fT2d_DeeSCCons[dee-1][171-1] = 222; fT2d_DeeSCCons[dee][171-1] =  73; //  8;
      fT2d_DeeSCCons[dee-1][172-1] = 214; fT2d_DeeSCCons[dee][172-1] =  65; //  1;
      
      fT2d_DeeSCCons[dee-1][188-1] = 298; fT2d_DeeSCCons[dee][188-1] = 149; // 24; //  298a ;  149a;
      fT2d_DeeSCCons[dee-1][189-1] = 239; fT2d_DeeSCCons[dee][189-1] =  90; // 24;
      fT2d_DeeSCCons[dee-1][190-1] = 231; fT2d_DeeSCCons[dee][190-1] =  82; // 17;
      fT2d_DeeSCCons[dee-1][191-1] = 223; fT2d_DeeSCCons[dee][191-1] =  74; //  9;
      fT2d_DeeSCCons[dee-1][192-1] = 215; fT2d_DeeSCCons[dee][192-1] =  66; //  2;
  
      //.......................................... (D1,S4 or D3,S6) AND  (D2,S6 or D4,S4)
      fT2d_DeeSCCons[dee-1][29-1] = 261; fT2d_DeeSCCons[dee][29-1] = 112; // 14-21;   // 261a-268a ; 112a-119a;
      fT2d_DeeSCCons[dee-1][27-1] = 283; fT2d_DeeSCCons[dee][27-1] = 134; // 33;
      fT2d_DeeSCCons[dee-1][28-1] = 282; fT2d_DeeSCCons[dee][28-1] = 133; // 32;
      
      fT2d_DeeSCCons[dee-1][44-1] = 269; fT2d_DeeSCCons[dee][44-1] = 120; // 22;
      fT2d_DeeSCCons[dee-1][45-1] = 262; fT2d_DeeSCCons[dee][45-1] = 113; // 15;
      fT2d_DeeSCCons[dee-1][46-1] = 255; fT2d_DeeSCCons[dee][46-1] = 106; //  8;
      fT2d_DeeSCCons[dee-1][47-1] = 248; fT2d_DeeSCCons[dee][47-1] =  99; //  4;
      fT2d_DeeSCCons[dee-1][48-1] = 240; fT2d_DeeSCCons[dee][48-1] =  91; //  2;
      fT2d_DeeSCCons[dee-1][49-1] = 232; fT2d_DeeSCCons[dee][49-1] =  83; //  1;
      
      fT2d_DeeSCCons[dee-1][62-1] = 276; fT2d_DeeSCCons[dee][62-1] = 127; // 29;
      fT2d_DeeSCCons[dee-1][63-1] = 275; fT2d_DeeSCCons[dee][63-1] = 126; // 28;
      fT2d_DeeSCCons[dee-1][64-1] = 270; fT2d_DeeSCCons[dee][64-1] = 121; // 23;
      fT2d_DeeSCCons[dee-1][65-1] = 263; fT2d_DeeSCCons[dee][65-1] = 114; // 16;
      fT2d_DeeSCCons[dee-1][66-1] = 256; fT2d_DeeSCCons[dee][66-1] = 107; //  9;
      fT2d_DeeSCCons[dee-1][67-1] = 249; fT2d_DeeSCCons[dee][67-1] = 100; //  5;
      fT2d_DeeSCCons[dee-1][68-1] = 241; fT2d_DeeSCCons[dee][68-1] =  92; //  3;
      
      fT2d_DeeSCCons[dee-1][82-1] = 278; fT2d_DeeSCCons[dee][82-1] = 129; // 31;
      fT2d_DeeSCCons[dee-1][83-1] = 277; fT2d_DeeSCCons[dee][83-1] = 128; // 30;
      fT2d_DeeSCCons[dee-1][84-1] = 271; fT2d_DeeSCCons[dee][84-1] = 122; // 24;
      fT2d_DeeSCCons[dee-1][85-1] = 264; fT2d_DeeSCCons[dee][85-1] = 115; // 17;
      fT2d_DeeSCCons[dee-1][86-1] = 257; fT2d_DeeSCCons[dee][86-1] = 108; // 10;
      fT2d_DeeSCCons[dee-1][87-1] = 250; fT2d_DeeSCCons[dee][87-1] = 101; //  6;
      
      fT2d_DeeSCCons[dee-1][102-1] = 268; fT2d_DeeSCCons[dee][102-1] = 119; // 21; //  268c ;  119c;
      fT2d_DeeSCCons[dee-1][103-1] = 274; fT2d_DeeSCCons[dee][103-1] = 125; // 27;
      fT2d_DeeSCCons[dee-1][104-1] = 272; fT2d_DeeSCCons[dee][104-1] = 123; // 25;
      fT2d_DeeSCCons[dee-1][105-1] = 265; fT2d_DeeSCCons[dee][105-1] = 116; // 18;
      fT2d_DeeSCCons[dee-1][106-1] = 258; fT2d_DeeSCCons[dee][106-1] = 109; // 11;
      fT2d_DeeSCCons[dee-1][107-1] = 251; fT2d_DeeSCCons[dee][107-1] = 102; //  7;
      
      fT2d_DeeSCCons[dee-1][123-1] = 268; fT2d_DeeSCCons[dee][123-1] = 119; // 27; //  268b ;  119b;
      fT2d_DeeSCCons[dee-1][124-1] = 273; fT2d_DeeSCCons[dee][124-1] = 124; // 26;
      fT2d_DeeSCCons[dee-1][125-1] = 266; fT2d_DeeSCCons[dee][125-1] = 117; // 19;
      fT2d_DeeSCCons[dee-1][126-1] = 259; fT2d_DeeSCCons[dee][126-1] = 110; // 12;

      fT2d_DeeSCCons[dee-1][144-1] = 261; fT2d_DeeSCCons[dee][144-1] = 112; // 27; //  261c ;  112c;      
      fT2d_DeeSCCons[dee-1][145-1] = 267; fT2d_DeeSCCons[dee][145-1] = 118; // 20;
      fT2d_DeeSCCons[dee-1][146-1] = 260; fT2d_DeeSCCons[dee][146-1] = 111; // 13;

      fT2d_DeeSCCons[dee-1][165-1] = 261; fT2d_DeeSCCons[dee][165-1] = 112; // 27; //  261b ;  112b;
      
      //.......................................... D1 or D3, right half of S5 
      fT2d_DeeSCCons[dee-1][1-1] = 297; // 34;
      fT2d_DeeSCCons[dee-1][2-1] = 296; // 33;
      fT2d_DeeSCCons[dee-1][3-1] = 295; // 32;
      fT2d_DeeSCCons[dee-1][4-1] = 294; // 31;
      fT2d_DeeSCCons[dee-1][5-1] = 289; // 26;
      fT2d_DeeSCCons[dee-1][6-1] = 288; // 25;
      fT2d_DeeSCCons[dee-1][7-1] = 287; // 24;
      fT2d_DeeSCCons[dee-1][8-1] = 286; // 23;
      
      fT2d_DeeSCCons[dee-1][21-1] = 293; // 30;
      fT2d_DeeSCCons[dee-1][22-1] = 292; // 29;
      fT2d_DeeSCCons[dee-1][23-1] = 291; // 28;
      fT2d_DeeSCCons[dee-1][24-1] = 290; // 27;
      fT2d_DeeSCCons[dee-1][25-1] = 285; // 22;
      fT2d_DeeSCCons[dee-1][26-1] = 284; // 21;
      
      fT2d_DeeSCCons[dee-1][41-1] = 281; // 20; //  281a
      fT2d_DeeSCCons[dee-1][42-1] = 280; // 19;
      fT2d_DeeSCCons[dee-1][43-1] = 279; // 18;
      
      //.......................................... D2 or D4, left half of S5
      fT2d_DeeSCCons[dee][1-1] = 148; // 17;
      fT2d_DeeSCCons[dee][2-1] = 147; // 16;
      fT2d_DeeSCCons[dee][3-1] = 146; // 15;
      fT2d_DeeSCCons[dee][4-1] = 145; // 14;
      fT2d_DeeSCCons[dee][5-1] = 140; //  9;
      fT2d_DeeSCCons[dee][6-1] = 139; //  8;
      fT2d_DeeSCCons[dee][7-1] = 138; //  7;
      fT2d_DeeSCCons[dee][8-1] = 137; //  6;
      
      fT2d_DeeSCCons[dee][21-1] = 144; // 13;
      fT2d_DeeSCCons[dee][22-1] = 143; // 12;
      fT2d_DeeSCCons[dee][23-1] = 142; // 11;
      fT2d_DeeSCCons[dee][24-1] = 141; // 10;
      fT2d_DeeSCCons[dee][25-1] = 136; //  5;
      fT2d_DeeSCCons[dee][26-1] = 135; //  4;
     
      fT2d_DeeSCCons[dee][41-1] = 132; // 3; //  132a
      fT2d_DeeSCCons[dee][42-1] = 131; // 2;
      fT2d_DeeSCCons[dee][43-1] = 130; // 1;
    }

  //........................ ECNA numbers from numbers for constructions: fT2d_RecovDeeSC[][]

  for(Int_t i0EEDee=0; i0EEDee<MaxEEDee; i0EEDee++)
    {
      for(Int_t i_ecna=0; i_ecna<MaxDeeSC; i_ecna++)
	{
	  //....... test to avoid the -1 init value in 2nd index of array fT2d_RecovDeeSC[][]
	  //        (part of the matrix without real SC counterpart)
	  if( fT2d_DeeSCCons[i0EEDee][i_ecna] >= 0 && fT2d_DeeSCCons[i0EEDee][i_ecna] <= MaxEESCForCons )
	    {
	      fT2d_RecovDeeSC[i0EEDee][fT2d_DeeSCCons[i0EEDee][i_ecna]-1] = i_ecna+1;
	    }
	}
    }
}
//------------ (end of BuildEndcapSCTable) -------------------------

//===============================================================================
//
//        Get1DeeCrysFrom1DeeSCEcnaAnd0SCEcha
//        GetDeeCrysFromDeeEcha
//
//===============================================================================
Int_t TEcnaNumbering::Get1DeeCrysFrom1DeeSCEcnaAnd0SCEcha(const Int_t&  n1DeeSCEcna,
							  const Int_t&  i0SCEcha,
							  const TString& sDeeDir)
{
//get crystal number in Dee from SC number in Dee
// and from Electronic Channel number in super-crystal

  Int_t n1DeeCrys = 0;
  Int_t i0DeeDir = GetDeeDirIndex(sDeeDir);

  if( fT3dDeeCrys == 0 ){BuildEndcapCrysTable();}

  if( (n1DeeSCEcna >= 1) && (n1DeeSCEcna <= fEcal->MaxSCEcnaInDee()) )
    {
      if (i0SCEcha >=0 && i0SCEcha < fEcal->MaxCrysInSC())
	{
	  n1DeeCrys = fT3dDeeCrys[n1DeeSCEcna-1][i0SCEcha][i0DeeDir];
	}
      else
	{
	  n1DeeCrys = -2;   // Electronic Cnannel in Super-Crystal out of range
	  cout << "!TEcnaNumbering::Get1DeeCrysFrom1DeeSCEcnaAnd0SCEcha(...)> Electronic Channel in SuperCrystal = "
	       << i0SCEcha+1 << ". Out of range (range = [1," << fEcal->MaxCrysInSC() << "])" << fTTBELL << endl;
	}
    }
  else
    {
      n1DeeCrys = -3;   // Super-Crystal number in Dee out of range
      cout << "!TEcnaNumbering::Get1DeeCrysFrom1DeeSCEcnaAnd0SCEcha(...)> Super-Crystal number in Dee out of range."
	   << " n1DeeSCEcna = " << n1DeeSCEcna << fTTBELL << endl;
    }

  return n1DeeCrys;  // Range = [1,5000]
}

//===============================================================================
//
//                Get1SCEchaFrom1DeeCrys, Get1DeeSCEcnaFrom1DeeCrys
//
//===============================================================================

Int_t TEcnaNumbering::Get1SCEchaFrom1DeeCrys(const Int_t& n1DeeCrys, const TString& sDeeDir)
{
// get Electronic Channel number in Super-Crystal from Crystal ECNA number in Dee

  Int_t n1SCEcha = -1;
  Int_t iDeeDir  = GetDeeDirIndex(sDeeDir);

  if( n1DeeCrys >= 1 && n1DeeCrys <= fEcal->MaxCrysEcnaInDee() )
    {
      n1SCEcha = fT2dSCEcha[n1DeeCrys-1][iDeeDir];
    }
  else
    {
      n1SCEcha = -2;
      cout << "!TEcnaNumbering::Get1SCEchaFrom1DeeCrys(...)> Crystal number in Dee out of range."
	   << " n1DeeCrys = " << n1DeeCrys << "(max = " << fEcal->MaxCrysEcnaInDee() << ")" << fTTBELL << endl;
    }
  return n1SCEcha;   // range = [1,25]
}

Int_t TEcnaNumbering::Get1DeeSCEcnaFrom1DeeCrys(const Int_t& n1DeeCrys, const TString& sDeeDir)
{
// get Super-Crystal number in Dee from Crystal number in Dee

  Int_t n1DeeSCEcna = 0;
  Int_t iDeeDir    = GetDeeDirIndex(sDeeDir);
  
  if( n1DeeCrys >= 1 && n1DeeCrys <= fEcal->MaxCrysEcnaInDee() )
    {
      n1DeeSCEcna = fT2dDeeSC[n1DeeCrys-1][iDeeDir];
    }
  else
    {
      n1DeeSCEcna = -1;
      cout << "!TEcnaNumbering::Get1DeeSCEcnaFrom1DeeCrys(...)> Crystal number in Dee out of range."
	   << " n1DeeCrys = " << n1DeeCrys << "(max = " << fEcal->MaxCrysEcnaInDee() << ")" << fTTBELL << endl;
    }
  return n1DeeSCEcna;  // range = [1,200]
}

//===============================================================================
//
//          GetSCEchaFromDeeEcha
//          Get1DeeSCEcnaFromDeeEcha
//
//===============================================================================

Int_t TEcnaNumbering::Get1SCEchaFrom0DeeEcha(const Int_t& i0DeeEcha)
{
//get electronic channel number in super-crystal from electronic channel number in Dee

  Int_t i0DeeSC = i0DeeEcha/fEcal->MaxCrysInSC();
  Int_t n1SCEcha = i0DeeEcha - fEcal->MaxCrysInSC()*i0DeeSC + 1;

  return n1SCEcha;  //  range = [1,25]
}

Int_t TEcnaNumbering::Get1DeeSCEcnaFrom0DeeEcha(const Int_t& i0DeeEcha)
{
//get super-crystal number from electronic channel number in Dee

  Int_t n1DeeSC = i0DeeEcha/fEcal->MaxCrysInSC()+1;

  return n1DeeSC;  //  range = [1,200]
}

//--------------------------------------------------------------------------------
//
//       Correspondance (n1DeeNumber, DeeSC)  <->  (DS, DSSC, DeeSCCons)
//
//       GetDSFrom1DeeSCEcna,        GetDSSCFrom1DeeSCEcna,
//       GetDeeSCConsFrom1DeeSCEcna, Get1DeeSCEcnaFromDeeSCCons
//      
//       Get the values from the relevant arrays
//       with cross-check of the index values in argument
//--------------------------------------------------------------------------------

Int_t TEcnaNumbering::GetDSFrom1DeeSCEcna(const Int_t& n1DeeNumber, const Int_t& n1DeeSCEcna)
{
// Get Data Sector number from SC ECNA number in Dee 

  Int_t data_sector = -1;

  if( n1DeeNumber > 0 && n1DeeNumber <= fEcal->MaxDeeInEE() )
    {
      if( n1DeeSCEcna > 0  &&  n1DeeSCEcna <= fEcal->MaxSCEcnaInDee() )
	{
	  data_sector = fT2d_DS[n1DeeNumber-1][n1DeeSCEcna-1];
	}
      else
	{
	  cout << "!TEcnaNumbering::GetDSFrom1DeeSCEcna(...)> n1DeeSCEcna = " << n1DeeSCEcna
	       << ". Out of range ( range = [1," << fEcal->MaxSCEcnaInDee() << "] )"
	       << fTTBELL << endl;
	}
    }
  else
    {
      if( n1DeeNumber != 0 )
	{
	  cout << "!TEcnaNumbering::GetDSFrom1DeeSCEcna(...)> n1DeeNumber = " << n1DeeNumber 
	       << ". Out of range ( range = [1," << fEcal->MaxDeeInEE() << "] )"
	       << fTTBELL << endl;
	}
      else
	{
	  cout << "TEcnaNumbering::GetDSFrom1DeeSCEcna(...)> Dee = " << n1DeeNumber
	       << ". Out of range (range = [1," << fEcal->MaxDeeInEE() << "])" 
	       << fTTBELL << endl; 
	}
    }
  return data_sector;
}
//..........................................................................................
Int_t TEcnaNumbering::GetDSSCFrom1DeeSCEcna(const Int_t& n1DeeNumber, const Int_t& n1DeeSCEcna,
					    const Int_t& n1SCEcha)
{
  //.......... Get the correct SC number for the unconnected SC's (inner border)
  Int_t ds_sc = GetDSSCFrom1DeeSCEcna(n1DeeNumber, n1DeeSCEcna);

  if( n1DeeSCEcna == 29 || n1DeeSCEcna == 32 )
    {
      if( n1SCEcha == 11 )
	{
	  if( ds_sc == 14 ){ds_sc= 21;}  // 14 <=> 261/BR OR 112/BL
	}
      if( n1SCEcha == 1 || n1SCEcha == 2 || n1SCEcha == 3 ||
	  n1SCEcha == 6 || n1SCEcha == 7 )
	{
	  if( ds_sc ==  3 ){ds_sc = 25;}  // 3 <=> 178/TR OR 29/TL
	}
    }
  return ds_sc;
}
//..........................................................................................
Int_t TEcnaNumbering::GetDSSCFrom1DeeSCEcna(const Int_t& n1DeeNumber, const Int_t& n1DeeSCEcna)
{
// Get SC number in Data Sector from SC Ecna number in Dee 

  Int_t ds_sc = -1;

  if( n1DeeNumber > 0 && n1DeeNumber <= fEcal->MaxDeeInEE() )
    {
      if( n1DeeSCEcna > 0  &&  n1DeeSCEcna <= fEcal->MaxSCEcnaInDee() )
	{
	  ds_sc = fT2d_DSSC[n1DeeNumber-1][n1DeeSCEcna-1]; // 25 (not 3) or 14 (not 21) if n1DeeSCEcna = 32 or 29
	                                                   // 25 and 14 => 5 Xtals,  3 and 21 => 1 Xtal
	}
      else
	{
	  cout << "!TEcnaNumbering::GetDSSCFrom1DeeSCEcna(...)> n1DeeSCEcna = " << n1DeeSCEcna
	       << ". Out of range ( range = [1," << fEcal->MaxSCEcnaInDee() << "] )"
	       << fTTBELL << endl;
	}
    }
  else
    {
      if( n1DeeNumber != 0 )
	{
	  cout << "!TEcnaNumbering::GetDSSCFrom1DeeSCEcna(...)> n1DeeNumber = " << n1DeeNumber 
	       << ". Out of range ( range = [1," << fEcal->MaxDeeInEE() << "] )"
	       << fTTBELL << endl;
	}
      else
	{
	  cout << "TEcnaNumbering::GetDSSCFrom1DeeSCEcna(...)> Dee = " << n1DeeNumber
	       << ". Out of range (range = [1," << fEcal->MaxDeeInEE() << "])" 
	       << fTTBELL << endl;
	}
    }
  return ds_sc;
}
//..........................................................................................
Int_t TEcnaNumbering::GetDeeSCConsFrom1DeeSCEcna(const Int_t& n1DeeNumber, const Int_t& n1DeeSCEcna)
{
// Get SC number for Construction in Dee from SC ECNA number in Dee

  Int_t dee_sc_cons = -1;

  if( n1DeeNumber > 0 && n1DeeNumber <= fEcal->MaxDeeInEE() )
    {
      if( n1DeeSCEcna > 0  &&  n1DeeSCEcna <= fEcal->MaxSCEcnaInDee() )
	{
	  dee_sc_cons = fT2d_DeeSCCons[n1DeeNumber-1][n1DeeSCEcna-1];
	}
      else
	{
	  cout << "!TEcnaNumbering::GetDeeSCConsFrom1DeeSCEcna(...)> *** WARNING *** n1DeeSCEcna = " << n1DeeSCEcna
	       << ". Out of range ( range = [1," << fEcal->MaxSCEcnaInDee()
	       << "] ). Nb for const. forced to " << fT2d_DeeSCCons[n1DeeNumber-1][19] << "." << endl;
	  dee_sc_cons = fT2d_DeeSCCons[n1DeeNumber-1][19];
	}
    }
  else
    {
      if( n1DeeNumber != 0 )
	{
	  cout << "!TEcnaNumbering::GetDeeSCConsFrom1DeeSCEcna(...)> n1DeeNumber = " << n1DeeNumber 
	       << ". Out of range ( range = [1," << fEcal->MaxDeeInEE() << "] )"
	       << fTTBELL << endl;
	}
      else
	{
	  cout << "TEcnaNumbering::GetDeeSCConsFrom1DeeSCEcna(...)> Dee = " << n1DeeNumber
	       << ". Out of range (range = [1," << fEcal->MaxDeeInEE() << "])" 
	       << fTTBELL << endl;
	}
    }
  return dee_sc_cons;
}
//..........................................................................................
Int_t TEcnaNumbering::GetDeeSCConsFrom1DeeSCEcna(const Int_t& n1DeeNumber, const Int_t& n1DeeSCEcna,
						 const Int_t& n1SCEcha)
{
  //.......... Get the correct SC number (for cons) for the unconnected SC's (inner border)
  Int_t dee_sc_cons = GetDeeSCConsFrom1DeeSCEcna(n1DeeNumber, n1DeeSCEcna);

  if( n1DeeSCEcna == 29 || n1DeeSCEcna == 32 )
    {
      if( n1SCEcha == 11 )
	{
	  if( dee_sc_cons == 261 ){dee_sc_cons = 268;}  // 261<=>14/BR
	  if( dee_sc_cons == 112 ){dee_sc_cons = 119;}  // 112<=>14/BL
	}
      if( n1SCEcha == 1 ||  n1SCEcha == 2 || n1SCEcha == 3 ||
	  n1SCEcha == 6 ||  n1SCEcha == 7 )
	{
	  if( dee_sc_cons == 178 ){dee_sc_cons = 207;}  // 178<=>3/TR
	  if( dee_sc_cons ==  29 ){dee_sc_cons =  58;}  //  29<=>3/TL
	}
    }
  return dee_sc_cons;
}
//..........................................................................................
Int_t TEcnaNumbering::Get1DeeSCEcnaFromDeeSCCons(const Int_t& n1DeeNumber, const Int_t& DeeSCCons)
{
// Get SC Ecna number in Dee from SC number for Construction in Dee
 
  Int_t dee_sc_ecna = -1;

  if( n1DeeNumber > 0 && n1DeeNumber <= fEcal->MaxDeeInEE() )
    {
      Int_t off_set_cons = 0;
      if( n1DeeNumber == 1 || n1DeeNumber == 3 ){off_set_cons = fEcal->MaxSCForConsInDee();}
      
      if( DeeSCCons > off_set_cons  &&  DeeSCCons <= fEcal->MaxSCForConsInDee()+off_set_cons )
	{
	  dee_sc_ecna = fT2d_RecovDeeSC[n1DeeNumber-1][DeeSCCons-1];
	}
      else
	{
	  cout << "!TEcnaNumbering::Get1DeeSCEcnaFromDeeSCCons(...)> DeeSCCons = " << DeeSCCons
	       << ". Out of range ( range = [ " << off_set_cons+1
	       << "," << fEcal->MaxSCForConsInDee()+off_set_cons << "] )"
	       << fTTBELL << endl;
	}
    }
  else
    {
      if( n1DeeNumber != 0 )
	{
	  cout << "!TEcnaNumbering::Get1DeeSCEcnaFromDeeSCCons(...)> n1DeeNumber = " << n1DeeNumber 
	       << ". Out of range ( range = [1," << fEcal->MaxDeeInEE() << "] )"
	       << fTTBELL << endl;
	}
      else
	{
	  cout << "TEcnaNumbering::Get1DeeSCEcnaFromDeeSCCons(...)> Dee = " << n1DeeNumber
	       << ". Out of range (range = [1," << fEcal->MaxDeeInEE() << "])" 
	       << fTTBELL << endl;
	}
    }
  return dee_sc_ecna;
}

TString TEcnaNumbering::GetSCType(const Int_t& nb_for_cons)
{
// gives the special not connected SC's

  TString SCType = "Connected";   // => default type

  if( nb_for_cons == 182 || nb_for_cons ==  33 ){SCType = "NotConnected";}  // (D1,S1) (D3,S9) || (D2,S9) (D4,S1)

  if( nb_for_cons == 178 || nb_for_cons ==  29 ){SCType = "NotConnected";}  // (D1,S2) (D3,S8) || (D2,S8) (D4,S2)
  if( nb_for_cons == 207 || nb_for_cons ==  58 ){SCType = "NotConnected";}

  if( nb_for_cons == 298 || nb_for_cons == 149 ){SCType = "NotConnected";}  // (D1,S3) (D3,S7) || (D2,S7) (D4,S3)

  if( nb_for_cons == 261 || nb_for_cons == 112 ){SCType = "NotConnected";}  // (D1,S4) (D3,S6) || (D2,S6) (D4,S4)
  if( nb_for_cons == 268 || nb_for_cons == 119 ){SCType = "NotConnected";}

  if( nb_for_cons == 281 || nb_for_cons == 132 ){SCType = "NotConnected";}  // (D1,S5) (D3,S5) || (D2,S5) (D4,S5)

  if( nb_for_cons == 161 || nb_for_cons ==  12 ){SCType = "NotComplete";}   // (D1,S1) (D3,S9) || (D2,S9) (D4,S1)
  if( nb_for_cons == 216 || nb_for_cons ==  67 ){SCType = "NotComplete";}   // (D1,S2) (D3,S8) || (D2,S8) (D4,S2)
  if( nb_for_cons == 224 || nb_for_cons ==  75 ){SCType = "NotComplete";}   // (D1,S3) (D3,S7) || (D2,S7) (D4,S3)
  if( nb_for_cons == 286 || nb_for_cons == 137 ){SCType = "NotComplete";}   // (D1,S5) (D3,S5) || (D2,S5) (D4,S5)

  return SCType;
}

Int_t TEcnaNumbering::StexEchaForCons(const Int_t& n1DeeNumber, const Int_t& i0StexEcha)
{
  Int_t n1StexStin = Get1StexStinFrom0StexEcha(i0StexEcha);
  return fT2d_DeeSCCons[n1DeeNumber-1][n1StexStin-1];
}
  // return -1 if the SC does not correspond to a real SC; return the number for construction otherwise

//===========================================================================
//
//                        GetSCQuadFrom1DeeSCEcna
//
//===========================================================================  

TString  TEcnaNumbering::GetSCQuadFrom1DeeSCEcna(const Int_t& n1DeeSCEcna)
{
//gives the quadrant type ("top" or "bottom") from the SC number in Dee

  TString SCQuad = "top";   // => default value

  if (n1DeeSCEcna >=   1 && n1DeeSCEcna <=  10){SCQuad = "bottom";}
  if (n1DeeSCEcna >=  21 && n1DeeSCEcna <=  30){SCQuad = "bottom";}
  if (n1DeeSCEcna >=  41 && n1DeeSCEcna <=  50){SCQuad = "bottom";}
  if (n1DeeSCEcna >=  61 && n1DeeSCEcna <=  70){SCQuad = "bottom";}
  if (n1DeeSCEcna >=  81 && n1DeeSCEcna <=  90){SCQuad = "bottom";}
  if (n1DeeSCEcna >= 101 && n1DeeSCEcna <= 110){SCQuad = "bottom";}
  if (n1DeeSCEcna >= 121 && n1DeeSCEcna <= 130){SCQuad = "bottom";}
  if (n1DeeSCEcna >= 141 && n1DeeSCEcna <= 150){SCQuad = "bottom";}
  if (n1DeeSCEcna >= 161 && n1DeeSCEcna <= 170){SCQuad = "bottom";}
  if (n1DeeSCEcna >= 181 && n1DeeSCEcna <= 190){SCQuad = "bottom";}

  return SCQuad;
}
Int_t TEcnaNumbering::GetSCQuadTypeIndex(const TString& SCQuadType, const TString& sDeeDir)
{
//gives the index of the SC quadrant type (top/right, top/left, bottom/left, bottom/right)
// = quadrant number - 1

  Int_t itype = 0;   // => default
  if ( SCQuadType == "top"    && sDeeDir == "right" ){itype = 0;}
  if ( SCQuadType == "top"    && sDeeDir == "left"  ){itype = 1;}
  if ( SCQuadType == "bottom" && sDeeDir == "left"  ){itype = 2;}
  if ( SCQuadType == "bottom" && sDeeDir == "right" ){itype = 3;}
  return itype;
}
//===========================================================================
//
//                    GetEEDeeType, GetDeeDirViewedFromIP
//
//===========================================================================  
TString TEcnaNumbering::GetEEDeeEndcap(const Int_t& n1DeeNumber)
{
//gives the Endcap (EE+ or EE-) of the Dee (H. Heath, CMS NOTE 2006/027)

  TString eetype = "EE+";   // => default
  if ( n1DeeNumber == 1 || n1DeeNumber == 2 ){eetype = "EE+";}
  if ( n1DeeNumber == 3 || n1DeeNumber == 4 ){eetype = "EE-";}
  return eetype;
}
TString TEcnaNumbering::GetEEDeeType(const Int_t& n1DeeNumber)
{
//gives the EE +/- and Forward/Near of the Dee (H. Heath, CMS NOTE 2006/027)

  TString type = "EE+F";   // => default
  if ( n1DeeNumber == 1 ){type = "EE+F";}
  if ( n1DeeNumber == 2 ){type = "EE+N";}
  if ( n1DeeNumber == 3 ){type = "EE-N";}
  if ( n1DeeNumber == 4 ){type = "EE-F";}
  return type;
}

TString TEcnaNumbering::GetDeeDirViewedFromIP(const Int_t& n1DeeNumber)
{
//gives the direction (left/right) of the IX axis of the Dee
// looking from the interaction point

  TString sDeeDir = "right";   // => default
  if ( (n1DeeNumber == 1) || (n1DeeNumber == 3) ){sDeeDir = "right";}
  if ( (n1DeeNumber == 2) || (n1DeeNumber == 4) ){sDeeDir = "left" ;}
  return sDeeDir;
}
Int_t TEcnaNumbering::GetDeeDirIndex(const TString& sDeeDir)
{
//gives the index of the direction (left,right) of the IX axis of the Dee
// looking from the interaction point (right = 0, left = 1)

  Int_t iDeeDir = 0;   // => default
  if ( sDeeDir == "right" ){iDeeDir = 0;}
  if ( sDeeDir == "left"  ){iDeeDir = 1;}
  return iDeeDir;
}

//==============================================================================
//
//    GetIXCrysInSC,  GetJYCrysInSC
//    GetIXSCInDee,   GetJYSCInDee
//    GetIXCrysInDee, GetJYCrysInDConsFrom1DeeSCEcna(fStexNumber, StexStinEcna);
//
//==============================================================================
Int_t TEcnaNumbering::GetIXCrysInSC(const Int_t& n1DeeNumber, const Int_t& DeeSC,
				    const Int_t& i0SCEcha)
{
//Gives Crys IX in SC for a given (n1DeeNumber, DeeSC, i0SCEcha)
  
  TString SCQuadType = GetSCQuadFrom1DeeSCEcna(DeeSC);
  TString sDeeDir    = GetDeeDirViewedFromIP(n1DeeNumber);
  Int_t type_index   = GetSCQuadTypeIndex(SCQuadType, sDeeDir);
  Int_t IXCrysInSC   = fT2d_ich_IX[type_index][i0SCEcha+1] + 1;
  return IXCrysInSC;   // possible values: 1,2,3,4,5
}

Int_t TEcnaNumbering::GetIXSCInDee(const Int_t& DeeSC)
{
//Gives SC IX in Dee for a given (DeeSC)
  
  Int_t IXSCInDee = (DeeSC-1)/fEcal->MaxSCIYInDee() + 1;
  return IXSCInDee;  //  possible values: 1,2,...,9,10
}

Int_t TEcnaNumbering::GetIXCrysInDee(const Int_t& n1DeeNumber, const Int_t& DeeSC,
				     const Int_t& i0SCEcha)
{
//Gives Crys IX in Dee for a given (n1DeeNumber, DeeSC, i0SCEcha)

  Int_t IXCrysInDee =
    (GetIXSCInDee(DeeSC)-1)*fEcal->MaxCrysIXInSC() +
    GetIXCrysInSC(n1DeeNumber, DeeSC, i0SCEcha);
  return IXCrysInDee;  // possible values: 1,2,...,49,50
}
//---------------------------------------------------------------------------------
Int_t TEcnaNumbering::GetJYCrysInSC(const Int_t& n1DeeNumber, const Int_t& DeeSC,
				    const Int_t& i0SCEcha)
{
//Gives Crys JY in SC  for a given (n1DeeNumber, DeeSC, i0SCEcha)

  TString SCQuadType = GetSCQuadFrom1DeeSCEcna(DeeSC);
  TString sDeeDir    = GetDeeDirViewedFromIP(n1DeeNumber);
  Int_t type_index   = GetSCQuadTypeIndex(SCQuadType, sDeeDir);
  Int_t JYCrysInSC   = fT2d_jch_JY[type_index][i0SCEcha+1] + 1;
  return JYCrysInSC;   // possible values: 1,2,3,4,5
}

Int_t TEcnaNumbering::GetJYSCInDee(const Int_t& DeeSC)
{
//Gives SC JY in Dee for a given (n1DeeNumber, DeeSC, i0SCEcha)
  
  Int_t JYSCInDee = (DeeSC-1)%fEcal->MaxSCIYInDee() + 1;
  return JYSCInDee;  //  possible values: 1,2,...,19,20
}

Int_t TEcnaNumbering::GetJYCrysInDee(const Int_t& n1DeeNumber, const Int_t& DeeSC,
				     const Int_t& i0SCEcha)
{
//Gives Crys JY in Dee for a given (n1DeeNumber, DeeSC, i0SCEcha)

  Int_t JYCrysInDee =
    (GetJYSCInDee(DeeSC)-1)*fEcal->MaxCrysIYInSC() +
    GetJYCrysInSC(n1DeeNumber, DeeSC, i0SCEcha);
  return JYCrysInDee;  // possible values: 1,2,...,99,100 
}
//---------------------------------------------------------------------------------
Int_t TEcnaNumbering::GetMaxSCInDS(const Int_t& DeeDS)
{
// Gives the number of SC's in Data Sector DeeDS

  Int_t nb_of_sc = -1;
  if( DeeDS == 1 || DeeDS == 9 ){nb_of_sc = 33;}
  if( DeeDS == 2 || DeeDS == 8 ){nb_of_sc = 32;}
  if( DeeDS == 3 || DeeDS == 7 ){nb_of_sc = 34;}
  if( DeeDS == 4 || DeeDS == 6 ){nb_of_sc = 33;}
  if( DeeDS == 5){nb_of_sc = 34;}
  return nb_of_sc;
}


//==============================================================================
//
//       GetIXMin, GetIXMax,  GetIIXMin, GetIIXMax
//
//==============================================================================
Double_t TEcnaNumbering::GetIIXMin(const Int_t& DeeSC)
{
//Gives IIXMin for a given DeeSC , unit = crystal

  Double_t IX_min   = (Double_t)((DeeSC-1)/fEcal->MaxSCIYInDee())*fEcal->MaxCrysIXInSC() + 1.;
  return IX_min;
}

Double_t TEcnaNumbering::GetIIXMax(const Int_t& DeeSC)
{
//Gives IIXMax for a given DeeSC , unit = crystal

  Double_t IX_max  = ((Double_t)((DeeSC-1)/fEcal->MaxSCIYInDee())+1.)*fEcal->MaxCrysIXInSC();
  return IX_max;
}

Double_t TEcnaNumbering::GetIIXMin()
{
//Gives IIXMin for Dee , unit = Super-crystal

  Double_t i_IX_min = (Int_t)1.;
  return i_IX_min;
}

Double_t TEcnaNumbering::GetIIXMax()
{
//Gives IIXMax for Dee , unit = Super-crystal

  Double_t i_IX_max = (Int_t)fEcal->MaxSCIXInDee(); 
  return i_IX_max;
}

//==============================================================================
//
//     GetIIYMin, GetIIYMax
//
//==============================================================================
Double_t TEcnaNumbering::GetJIYMin(const Int_t& n1DeeNumber, const Int_t& DeeSC)
{
//Gives JIYMin for a given Super-Crystal

  Double_t IY_DeeSC = DeeSC%fEcal->MaxSCIYInDee();
  if( IY_DeeSC == 0. ){IY_DeeSC = fEcal->MaxSCIYInDee();}

  Double_t j_IY_min = (IY_DeeSC-1)*fEcal->MaxCrysIYInSC() + 1.;
 
  return j_IY_min;
}
//-----------------------------------------------------------------------------------------
Double_t TEcnaNumbering::GetJIYMax(const Int_t& n1DeeNumber, const Int_t& DeeSC)
{
//Gives JIYMax for a given Super-Crystal

  Double_t IY_DeeSC = DeeSC%fEcal->MaxSCIYInDee();
  if( IY_DeeSC == 0  ){IY_DeeSC = fEcal->MaxSCIYInDee();}

  Double_t j_IY_max = IY_DeeSC*fEcal->MaxCrysIYInSC();

  return j_IY_max;
}

//-----------------------------------------------------------------------------------------
Double_t TEcnaNumbering::GetJIYMin(const Int_t& n1DeeNumber)
{
//Gives JIYMin for a given Dee

  Double_t j_IY_min = (Double_t)1.;

  return j_IY_min;
}
//-----------------------------------------------------------------------------------------
Double_t TEcnaNumbering::GetJIYMax(const Int_t& n1DeeNumber)
{
//Gives JIYMax for a given Dee

  Double_t j_IY_max = (Double_t)fEcal->MaxSCIYInDee();

  return j_IY_max;
}
//=====================================================================
TString  TEcnaNumbering::GetDeeHalfEndcap(const Int_t& n1DeeNumber)
{
//gives the half-endcap of the Dee ("EE+" or "EE-")

  TString type = "EE-";   // => default value

  if ( n1DeeNumber == 1 || n1DeeNumber == 2 ){type = "EE+";}
  if ( n1DeeNumber == 3 || n1DeeNumber == 4 ){type = "EE-";}

  return type;
}
//==============================================================================
//
//   GetXDirectionEE, GetYDirectionEE, GetJYDirectionEE
//
//==============================================================================
TString TEcnaNumbering::GetXDirectionEE(const Int_t& n1DeeNumber)
{
  TString xdirection = "x";      // DEFAULT

  if( GetEEDeeType(n1DeeNumber) == "EE+F" ){xdirection = "-x";}   //  Dee 1
  if( GetEEDeeType(n1DeeNumber) == "EE+N" ){xdirection = "-x";}   //  Dee 2
  if( GetEEDeeType(n1DeeNumber) == "EE-N" ){xdirection = "x";}    //  Dee 3
  if( GetEEDeeType(n1DeeNumber) == "EE-F" ){xdirection = "x";}    //  Dee 4

  return xdirection;
}
//---------------------------------------------------------
TString TEcnaNumbering::GetYDirectionEE(const Int_t& n1DeeNumber)
{
  TString ydirection = "-x";      // DEFAULT

  if( GetEEDeeType(n1DeeNumber) == "endcap+" ){ydirection = "-x";}
  if( GetEEDeeType(n1DeeNumber) == "endcap-" ){ydirection = "-x";}

  return ydirection;
}

//---------------------------------------------------------
TString TEcnaNumbering::GetJYDirectionEE(const Int_t& n1DeeNumber)
{
  TString jydirection = "x";      // ALWAYS IN THIS CASE
  return jydirection;
}

//==========================================================================================================
//
//
//                               B A R R E L    A N D    E N D C A P 
//
//
//
//==========================================================================================================
//===============================================================================
//
//        Get1StexStinFrom0StexEcha
//     // GetStinEchaFromStexEcha
//        Get0StexEchaFrom1StexStinAnd0StinEcha
//        Get1StexCrysFrom1StexStinAnd0StinEcha
//
//===============================================================================
Int_t TEcnaNumbering::Get1StexStinFrom0StexEcha(const Int_t& i0StexEcha)
{
  Int_t n1StexStin = 0;

  if( fFlagSubDet == "EB" ){n1StexStin = Get1SMTowFrom0SMEcha(i0StexEcha);}
  if( fFlagSubDet == "EE" ){n1StexStin = Get1DeeSCEcnaFrom0DeeEcha(i0StexEcha);}

  return n1StexStin;
}

Int_t TEcnaNumbering::Get0StexEchaFrom1StexStinAnd0StinEcha(const Int_t& n1StexStin,
							    const Int_t& i0StinEcha)
{
// Electronic Channel number in Stex from Stin number in Stex
// and from Electronic Channel number in Stin

  Int_t StexEcha = (Int_t)(-1.);
                    
  if ( n1StexStin >  0 && n1StexStin <= fEcal->MaxStinEcnaInStex() &&
       i0StinEcha >= 0 && i0StinEcha < fEcal->MaxCrysInStin()  )
    {StexEcha = (n1StexStin-1)*fEcal->MaxCrysInStin() + i0StinEcha;}
  else
    {
      cout << "!TEcnaNumbering::Get0StexEchaFrom1StexStinAnd0StinEcha *** ERROR ***> VALUE"
	   << " OUT OF RANGE. Forced to -1. Argument values: n1StexStin = " << n1StexStin
	   << ", channel = " << i0StinEcha
	   << fTTBELL << endl;
    }
  return StexEcha;
}
Int_t TEcnaNumbering::Get1StexCrysFrom1StexStinAnd0StinEcha(const Int_t& n1StexStin,
							    const Int_t& i0StinEcha, const Int_t& StexNumber)
{
// Crystal number in Stex from Stin number in Stex
// and from Electronic Channel number in Stin (for the StexNumber_th Stex)
// argument StexNumber used only in "EE" case

  Int_t n1StexCrys = (Int_t)0;
  if( fFlagSubDet ==  "EB" ){n1StexCrys = Get1SMCrysFrom1SMTowAnd0TowEcha(n1StexStin, i0StinEcha);}
  if( fFlagSubDet ==  "EE" ){TString sDeeDir = GetDeeDirViewedFromIP(StexNumber);
  n1StexCrys = Get1DeeCrysFrom1DeeSCEcnaAnd0SCEcha(n1StexStin, i0StinEcha, sDeeDir);}
  
  return n1StexCrys;
}
//===============================================================================
//
//        GetIHocoMin, GetIHocoMax, GetVecoMin, GetVecoMax
//        GetJVecoMin, GetJVecoMax
//
//===============================================================================
Double_t TEcnaNumbering::GetIHocoMin(const Int_t& Stex, const Int_t& StexStin)
{
  Double_t IHocoMin = (Double_t)0.;
  if(fFlagSubDet == "EB" ){IHocoMin = GetIEtaMin(Stex, StexStin);}
  if(fFlagSubDet == "EE" ){IHocoMin = GetIIXMin(StexStin);}
  return IHocoMin;
}

Double_t TEcnaNumbering::GetIHocoMax(const Int_t& Stex, const Int_t& StexStin)
{
  Double_t IHocoMax = (Double_t)0.;
  if(fFlagSubDet == "EB" ){IHocoMax = GetIEtaMax(Stex, StexStin);}
  if(fFlagSubDet == "EE" ){IHocoMax = GetIIXMax(StexStin);}
  return IHocoMax;
}

Double_t TEcnaNumbering::GetVecoMin(const Int_t& Stex, const Int_t& StexStin)
{
  Double_t IVecoMin = (Double_t)0.;
  if(fFlagSubDet == "EB" ){IVecoMin = GetPhiMin(Stex, StexStin);}
  if(fFlagSubDet == "EE" ){IVecoMin = GetJIYMin(Stex, StexStin);}
  return IVecoMin;
}

Double_t TEcnaNumbering::GetVecoMax(const Int_t& Stex, const Int_t& StexStin)
{
  Double_t IVecoMax = (Double_t)0.;
  if(fFlagSubDet == "EB" ){IVecoMax = GetPhiMax(Stex, StexStin);}
  if(fFlagSubDet == "EE" ){IVecoMax = GetJIYMax(Stex, StexStin);}
  return IVecoMax;
}

Double_t TEcnaNumbering::GetJVecoMin(const Int_t& Stex, const Int_t& StexStin)
{
  Double_t JVecoMin = (Double_t)0.;
  if(fFlagSubDet == "EB" ){JVecoMin = GetJPhiMin(Stex, StexStin);}
  if(fFlagSubDet == "EE" ){JVecoMin = GetJIYMin(Stex, StexStin);}   // not used
  return JVecoMin;
}
Double_t TEcnaNumbering::GetJVecoMax(const Int_t& Stex, const Int_t& StexStin)
{
  Double_t JVecoMax = (Double_t)0.;
  if(fFlagSubDet == "EB" ){JVecoMax = GetJPhiMax(Stex, StexStin);}
  if(fFlagSubDet == "EE" ){JVecoMax = GetJIYMax(Stex, StexStin);}   // not used
  return JVecoMax;
}
//===========================================================================
//
//                        GetStexHalfBarrel   
//
//===========================================================================  
TString  TEcnaNumbering::GetStexHalfStas(const Int_t& SMNumber)
{
  TString half_stas = "EB? EE?";

  if( fFlagSubDet == "EB" ){half_stas = GetSMHalfBarrel(SMNumber);}
  if( fFlagSubDet == "EE" ){half_stas = GetDeeHalfEndcap(SMNumber);}

  return half_stas;
}
//===========================================================================
//
//                  GetSMFromFED, GetDSFromFED
//
//=========================================================================== 
Int_t TEcnaNumbering::GetSMFromFED(const Int_t& FEDNumber)
{
  Int_t EBSMNumber = 0;    // SM = Super Module
  if( FEDNumber >= 610 && FEDNumber <= 645 ){EBSMNumber = FEDNumber - 609;}
  return EBSMNumber;
}

Int_t TEcnaNumbering::GetDSFromFED(const Int_t& FEDNumber)
{
  Int_t EEDSNumber = 0;    // DS = Data Sector

  if( FEDNumber >= 600 && FEDNumber <= 609 ){EEDSNumber = FEDNumber - 599;} 
  if( FEDNumber >= 646 && FEDNumber <= 655 ){EEDSNumber = FEDNumber - 645;} 

  return EEDSNumber;
}

//--------------------------------------------
Int_t TEcnaNumbering::MaxCrysInStinEcna(const Int_t& n1DeeNumber, const Int_t& n1DeeSCEcna, const TString& s_option)
{
// Number of Xtals in "Ecna SC" for not complete and not connected SC's.
// Also valid for all connected and complete SC's and for towers of EB

  Int_t max_crys = fEcal->MaxCrysInStin();   // valid for EB and for connected and complete SC's of EE

  // Number of Xtals in SC for not complete and not connected SC's

  if(fFlagSubDet == "EE")
    {
      Int_t n_for_cons = GetDeeSCConsFrom1DeeSCEcna(n1DeeNumber, n1DeeSCEcna);

      //............ not complete SC's (inner border)
      if( n_for_cons ==  12 || n_for_cons ==  67 || n_for_cons ==  75 || n_for_cons == 137 ||
	  n_for_cons == 161 || n_for_cons == 216 || n_for_cons == 224 || n_for_cons == 286 ){max_crys = 20;}

      //............ not connected SC's
      if( (n_for_cons == 182 || n_for_cons ==  33) && (n1DeeSCEcna ==  60 || n1DeeSCEcna == 119) ){max_crys = 10;}

      if( (n_for_cons == 178 || n_for_cons ==  29) && (n1DeeSCEcna == 138 || n1DeeSCEcna == 157) ){max_crys = 10;}
      if( (n_for_cons == 207 || n_for_cons ==  58) && (n1DeeSCEcna == 176 || n1DeeSCEcna == 193) ){max_crys = 10;}

      if( (n_for_cons == 298 || n_for_cons == 149) && (n1DeeSCEcna == 188) ){max_crys = 10;}

      if( (n_for_cons == 261 || n_for_cons == 112) && (n1DeeSCEcna == 144 || n1DeeSCEcna == 165) ){max_crys = 10;}
      if( (n_for_cons == 268 || n_for_cons == 119) && (n1DeeSCEcna == 102 || n1DeeSCEcna == 123) ){max_crys = 10;}

      if( (n_for_cons == 281 || n_for_cons == 132) && (n1DeeSCEcna ==  41) ){max_crys = 10;}

      //............. not connected and mixed Ecna SC's
      if( s_option == "TEcnaRun" || s_option == "TEcnaRead" )
	{
	  if( s_option == "TEcnaRun" )
	    {
	      // special translation of Xtal 11 of SCEcna 29 and 32 to respectively Xtal 11 of SCEcna 10 and 11
	      if( n1DeeSCEcna == 29 || n1DeeSCEcna == 32 ){max_crys = 5;} 
	      if( n1DeeSCEcna == 10 || n1DeeSCEcna == 11 ){max_crys = 1;}
	    }
	  if( s_option == "TEcnaRead" )
	    {
	      //if( n1DeeSCEcna == 29 || n1DeeSCEcna == 32 ){max_crys = 6;}
	      if( n1DeeSCEcna == 29 || n1DeeSCEcna == 32 ){max_crys = 5;}
	      if( n1DeeSCEcna == 10 || n1DeeSCEcna == 11 ){max_crys = 1;}
	    }
	}
      else
	{
	  cout << "!TEcnaNumbering::MaxCrysInStinEcna(...)> " << s_option
	       << ": unknown option." << fTTBELL << endl;
	}
    }
  return max_crys;
}
