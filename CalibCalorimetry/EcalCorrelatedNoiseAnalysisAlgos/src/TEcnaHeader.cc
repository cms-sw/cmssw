//----------Author's Name:FX Gentit and B.Fabbro  DSM/IRFU/SPP CEA-Saclay
//----------Copyright:Those valid for CEA software
//----------Modified:24/03/2011

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaHeader.h"

//--------------------------------------
//  TEcnaHeader.cc
//  Class creation: 03 Dec 2002
//  Documentation: see TEcnaHeader.h
//--------------------------------------

ClassImp(TEcnaHeader);
//____________________________________________________________________________

TEcnaHeader::TEcnaHeader() { Init(); }  // constructor without arguments

TEcnaHeader::TEcnaHeader(TEcnaObject* pObjectManager, const Text_t* name, const Text_t* title) : TNamed(name, title) {
  // Constructor with arguments for reading ROOT file.
  // Called in FileParameters(...) of TEcnaRead
  // Please give a name and a title containing info about what
  // you are doing and saving in the ROOT file

  // std::cout << "[Info Management] CLASS: TEcnaHeader.        CREATE OBJECT: this = " << this << std::endl;

  Init();
  Long_t i_this = (Long_t)this;
  pObjectManager->RegisterPointer("TEcnaHeader", i_this);
}

TEcnaHeader::~TEcnaHeader() {
  //destructor

  // std::cout << "[Info Management] CLASS: TEcnaHeader.        DESTROY OBJECT: this = " << this << std::endl;
}

void TEcnaHeader::Init() {
  //Set default values in all variables and init the counters fxxxCalc

  //--------------------------------- INIT parameters

  fgMaxCar = (Int_t)512;

  //........................ RUN parameters

  fTypAna = "!Analysis name> no info";
  fNbOfSamples = (Int_t)0;
  fRunNumber = (Int_t)0;
  fFirstReqEvtNumber = (Int_t)0;
  fLastReqEvtNumber = (Int_t)0;
  fReqNbOfEvts = (Int_t)0;
  fStex = (Int_t)0;

  fStartTime = (time_t)0;
  fStopTime = (time_t)0;

  Int_t MaxCar = fgMaxCar;
  fStartDate.Resize(MaxCar);
  fStartDate = "!Start date> no info";
  MaxCar = fgMaxCar;
  fStopDate.Resize(MaxCar);
  fStopDate = "!Stop  date> no info";

  fRunType = 9999999;

  //--------------------------------- INIT counters
  fStinNumbersCalc = 0;
  fNbOfEvtsCalc = 0;
  fAdcEvtCalc = 0;
  fMSpCalc = 0;
  fSSpCalc = 0;

  fCovCssCalc = 0;
  fCorCssCalc = 0;
  fHfCovCalc = 0;
  fHfCorCalc = 0;
  fLfCovCalc = 0;
  fLfCorCalc = 0;
  fLFccMoStinsCalc = 0;
  fHFccMoStinsCalc = 0;
  fPedCalc = 0;
  fTnoCalc = 0;
  fMeanCorssCalc = 0;
  fLfnCalc = 0;
  fHfnCalc = 0;
  fSigCorssCalc = 0;

  fAvPedCalc = 0;
  fAvTnoCalc = 0;
  fAvLfnCalc = 0;
  fAvHfnCalc = 0;
  fAvMeanCorssCalc = 0;
  fAvSigCorssCalc = 0;
}
// ------------ end of Init() ------------

void TEcnaHeader::HeaderParameters(const TString& typ_ana,
                                   const Int_t& nb_of_samples,
                                   const Int_t& run_number,
                                   const Int_t& aFirstReqEvtNumber,
                                   const Int_t& aLastReqEvtNumber,
                                   const Int_t& aReqNbOfEvts,
                                   const Int_t& Stex,
                                   const Int_t& run_type) {
  // Constructor with arguments for reading DATA.
  // Called in GetReadyToReadData(...) of TEcnaRun
  // Please give a name and a title containing info about what
  // you are doing and saving in the ROOT file

  // std::cout << "[Info Management] CLASS: TEcnaHeader.        CREATE OBJECT: this = " << this << std::endl;

  Init();

  fTypAna = typ_ana;
  fRunNumber = run_number;
  fNbOfSamples = nb_of_samples;
  fFirstReqEvtNumber = aFirstReqEvtNumber;
  fLastReqEvtNumber = aLastReqEvtNumber;
  fReqNbOfEvts = aReqNbOfEvts;
  fStex = Stex;

  fRunType = run_type;
}
//------------------------------------------------------------------------------

void TEcnaHeader::HeaderParameters(const TString& typ_ana,
                                   const Int_t& nb_of_samples,
                                   const Int_t& run_number,
                                   const Int_t& aFirstReqEvtNumber,
                                   const Int_t& aLastReqEvtNumber,
                                   const Int_t& aReqNbOfEvts,
                                   const Int_t& Stex) {
  // std::cout << "[Info Management] CLASS: TEcnaHeader.        CREATE OBJECT: this = " << this << std::endl;

  Init();

  fTypAna = typ_ana;
  fRunNumber = run_number;
  fNbOfSamples = nb_of_samples;
  fFirstReqEvtNumber = aFirstReqEvtNumber;
  fLastReqEvtNumber = aLastReqEvtNumber;
  fReqNbOfEvts = aReqNbOfEvts;
  fStex = Stex;
}

void TEcnaHeader::Print() {
  // Print the header
  std::cout << std::endl;
  std::cout << "     Header parameters " << std::endl;
  std::cout << std::endl;
  std::cout << "Run number                   : " << fRunNumber << std::endl;
  std::cout << "First requested event number : " << fFirstReqEvtNumber << std::endl;
  std::cout << "Last requested event number  : " << fLastReqEvtNumber << std::endl;
  std::cout << "Requested number of events   : " << fReqNbOfEvts << std::endl;
  std::cout << "SM or Dee number             : " << fStex << std::endl;
  std::cout << "Time first event             : " << fStartTime << std::endl;
  std::cout << "Time last event              : " << fStopTime << std::endl;
  std::cout << "Date first event             : " << fStartDate.Data() << std::endl;
  std::cout << "Date last event              : " << fStopDate.Data() << std::endl;
  std::cout << "Run type                     : " << fRunType << std::endl;
  std::cout << std::endl;
  std::cout << "     Header counters " << std::endl;
  std::cout << std::endl;
  std::cout << "Stin Numbers                                 : " << fStinNumbersCalc << std::endl;
  std::cout << "Numbers of found evts                        : " << fNbOfEvtsCalc << std::endl;
  std::cout << "Samples as a function of time histograms     : " << fAdcEvtCalc << std::endl;
  std::cout << "Expectation values histogram                 : " << fMSpCalc << std::endl;
  std::cout << "Variances histogram                          : " << fSSpCalc << std::endl;
  std::cout << "Average total noise                         : " << fAvTnoCalc << std::endl;
  std::cout << "Average low frequency noise                 : " << fAvLfnCalc << std::endl;
  std::cout << "Average high frequency noise                : " << fAvHfnCalc << std::endl;

  std::cout << "Nb of (sample,sample) covariance  matrices   : " << fCovCssCalc << std::endl;
  std::cout << "Nb of (sample,sample) correlation matrices   : " << fCorCssCalc << std::endl;
  std::cout << "Nb of (channel,channel) covariance  matrices : " << fHfCovCalc << std::endl;
  std::cout << "Nb of (channel,channel) correlation matrices : " << fHfCorCalc << std::endl;
  std::cout << "Nb of (channel,channel) cov mat mean on samp : " << fLfCovCalc << std::endl;
  std::cout << "Nb of (channel,channel) cor mat mean on samp : " << fLfCorCalc << std::endl;
  std::cout << "Nb of mean cov(c,c) mean on samp, all Stins  : " << fLfCovCalc << std::endl;
  std::cout << "Nb of mean cor(c,c) mean on samp, all Stins  : " << fLfCorCalc << std::endl;

  std::cout << "Exp. val. of the exp. val. of the samples    : " << fPedCalc << std::endl;
  std::cout << "Expect. val. of the sigmas of the samples    : " << fTnoCalc << std::endl;
  std::cout << "Expect. val. of the (samp,samp) correlations : " << fMeanCorssCalc << std::endl;

  std::cout << "Sigmas of the exp. val. of the samples       : " << fLfnCalc << std::endl;
  std::cout << "Sigmas of the sigmas of the samples          : " << fHfnCalc << std::endl;
  std::cout << "Sigmas of the (samp,samp) correlations       : " << fSigCorssCalc << std::endl;

  std::cout << "Average pedestals                           : " << fAvPedCalc << std::endl;
  std::cout << "Average mean cor(s,s)                    : " << fAvMeanCorssCalc << std::endl;
  std::cout << "Average sigma of Cor(s,s)                   : " << fAvSigCorssCalc << std::endl;
  std::cout << std::endl;
}
