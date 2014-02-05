//----------Author's Name:FX Gentit and B.Fabbro  DSM/IRFU/SPP CEA-Saclay
//----------Copyright:Those valid for CEA software
//----------Modified:24/03/2011

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaHeader.h"

//--------------------------------------
//  TEcnaHeader.cc
//  Class creation: 03 Dec 2002
//  Documentation: see TEcnaHeader.h
//--------------------------------------

ClassImp(TEcnaHeader)
//____________________________________________________________________________
  
TEcnaHeader::TEcnaHeader(){Init();} // constructor without arguments

TEcnaHeader::TEcnaHeader(TEcnaObject* pObjectManager, const Text_t* name, const Text_t* title):TNamed(name,title)
{
  // Constructor with arguments for reading ROOT file.
  // Called in FileParameters(...) of TEcnaRead
  // Please give a name and a title containing info about what
  // you are doing and saving in the ROOT file

  // cout << "[Info Management] CLASS: TEcnaHeader.        CREATE OBJECT: this = " << this << endl;
  
  Init();
  Long_t i_this = (Long_t)this;
  pObjectManager->RegisterPointer("TEcnaHeader", i_this);
}

TEcnaHeader::~TEcnaHeader() {
  //destructor

 // cout << "[Info Management] CLASS: TEcnaHeader.        DESTROY OBJECT: this = " << this << endl;
}

void TEcnaHeader::Init()
{
 //Set default values in all variables and init the counters fxxxCalc

  //--------------------------------- INIT parameters

  fgMaxCar = (Int_t)512;

  //........................ RUN parameters

  fTypAna            = "!Analysis name> no info";
  fNbOfSamples       = (Int_t)0;
  fRunNumber         = (Int_t)0;
  fFirstReqEvtNumber = (Int_t)0;
  fLastReqEvtNumber  = (Int_t)0;
  fReqNbOfEvts       = (Int_t)0;
  fStex              = (Int_t)0;

  fStartTime     = (time_t)0;
  fStopTime      = (time_t)0;

  Int_t MaxCar = fgMaxCar;
  fStartDate.Resize(MaxCar);
  fStartDate     = "!Start date> no info";
  MaxCar = fgMaxCar;
  fStopDate.Resize(MaxCar);
  fStopDate      = "!Stop  date> no info";

  fRunType = 9999999;

  //--------------------------------- INIT counters
  fStinNumbersCalc  = 0;
  fNbOfEvtsCalc     = 0;
  fAdcEvtCalc       = 0;
  fMSpCalc          = 0;
  fSSpCalc          = 0;

  fCovCssCalc       = 0;
  fCorCssCalc       = 0;
  fHfCovCalc        = 0;
  fHfCorCalc        = 0;
  fLfCovCalc        = 0;
  fLfCorCalc        = 0;
  fLFccMoStinsCalc  = 0;
  fHFccMoStinsCalc  = 0;
  fPedCalc          = 0;
  fTnoCalc          = 0;
  fMeanCorssCalc    = 0;
  fLfnCalc          = 0;
  fHfnCalc          = 0;
  fSigCorssCalc     = 0;

  fAvPedCalc        = 0;
  fAvTnoCalc        = 0;
  fAvLfnCalc        = 0;
  fAvHfnCalc        = 0;
  fAvMeanCorssCalc  = 0;
  fAvSigCorssCalc   = 0;
}
// ------------ end of Init() ------------

void TEcnaHeader::HeaderParameters(const TString&      typ_ana,           const Int_t& nb_of_samples,
				   const Int_t& run_number,        const Int_t& aFirstReqEvtNumber,
				   const Int_t& aLastReqEvtNumber, const Int_t& aReqNbOfEvts,
				   const Int_t& Stex,              const Int_t& run_type
				   )
{
  // Constructor with arguments for reading DATA.
  // Called in GetReadyToReadData(...) of TEcnaRun
  // Please give a name and a title containing info about what
  // you are doing and saving in the ROOT file
 
 // cout << "[Info Management] CLASS: TEcnaHeader.        CREATE OBJECT: this = " << this << endl;

  Init();

  fTypAna            = typ_ana;
  fRunNumber         = run_number;
  fNbOfSamples       = nb_of_samples;
  fFirstReqEvtNumber = aFirstReqEvtNumber;
  fLastReqEvtNumber  = aLastReqEvtNumber;
  fReqNbOfEvts       = aReqNbOfEvts;
  fStex              = Stex;

  fRunType           = run_type;
}
//------------------------------------------------------------------------------

void TEcnaHeader::HeaderParameters(const TString&      typ_ana,           const Int_t& nb_of_samples,
				   const Int_t& run_number,        const Int_t& aFirstReqEvtNumber,
				   const Int_t& aLastReqEvtNumber, const Int_t& aReqNbOfEvts,
				   const Int_t& Stex)
 {
 // cout << "[Info Management] CLASS: TEcnaHeader.        CREATE OBJECT: this = " << this << endl;

  Init();

  fTypAna            = typ_ana;
  fRunNumber         = run_number;
  fNbOfSamples       = nb_of_samples;
  fFirstReqEvtNumber = aFirstReqEvtNumber;
  fLastReqEvtNumber  = aLastReqEvtNumber;
  fReqNbOfEvts       = aReqNbOfEvts;
  fStex              = Stex;
}

void TEcnaHeader::Print() {
  // Print the header
  cout << endl;
  cout << "     Header parameters " << endl;
  cout << endl;
  cout << "Run number                   : " << fRunNumber         << endl;
  cout << "First requested event number : " << fFirstReqEvtNumber << endl;
  cout << "Last requested event number  : " << fLastReqEvtNumber  << endl;
  cout << "Requested number of events   : " << fReqNbOfEvts       << endl;
  cout << "SM or Dee number             : " << fStex              << endl;
  cout << "Time first event             : " << fStartTime         << endl;
  cout << "Time last event              : " << fStopTime          << endl;
  cout << "Date first event             : " << fStartDate.Data()  << endl;
  cout << "Date last event              : " << fStopDate.Data()   << endl;
  cout << "Run type                     : " << fRunType           << endl;
  cout << endl;
  cout << "     Header counters " << endl;
  cout << endl;
  cout << "Stin Numbers                                 : "
       << fStinNumbersCalc << endl;
  cout << "Numbers of found evts                        : "
       << fNbOfEvtsCalc << endl;
  cout << "Samples as a function of time histograms     : "
       << fAdcEvtCalc << endl;
  cout << "Expectation values histogram                 : "
       << fMSpCalc    << endl;
  cout << "Variances histogram                          : "
       << fSSpCalc   << endl;
  cout << "Average total noise                         : "
       << fAvTnoCalc  << endl;
  cout << "Average low frequency noise                 : "
       << fAvLfnCalc  << endl;
  cout << "Average high frequency noise                : "
       << fAvHfnCalc  << endl;

  cout << "Nb of (sample,sample) covariance  matrices   : "
       << fCovCssCalc << endl;
  cout << "Nb of (sample,sample) correlation matrices   : "
       << fCorCssCalc << endl;
  cout << "Nb of (channel,channel) covariance  matrices : "
       << fHfCovCalc << endl;
  cout << "Nb of (channel,channel) correlation matrices : "
       << fHfCorCalc << endl;
  cout << "Nb of (channel,channel) cov mat mean on samp : "
       << fLfCovCalc << endl;
  cout << "Nb of (channel,channel) cor mat mean on samp : "
       << fLfCorCalc << endl;
  cout << "Nb of mean cov(c,c) mean on samp, all Stins  : "
       << fLfCovCalc << endl;
  cout << "Nb of mean cor(c,c) mean on samp, all Stins  : "
       << fLfCorCalc << endl;

  cout << "Exp. val. of the exp. val. of the samples    : "
       << fPedCalc     << endl;
  cout << "Expect. val. of the sigmas of the samples    : "
       << fTnoCalc    << endl;
  cout << "Expect. val. of the (samp,samp) correlations : "
       << fMeanCorssCalc << endl;

  cout << "Sigmas of the exp. val. of the samples       : "
       << fLfnCalc     << endl;
  cout << "Sigmas of the sigmas of the samples          : "
       << fHfnCalc     << endl;
  cout << "Sigmas of the (samp,samp) correlations       : "
       << fSigCorssCalc     << endl;

  cout << "Average pedestals                           : "
       << fAvPedCalc  << endl;
  cout << "Average mean cor(s,s)                    : "
       << fAvMeanCorssCalc << endl;
  cout << "Average sigma of Cor(s,s)                   : "
       << fAvSigCorssCalc << endl;
  cout << endl;
}
