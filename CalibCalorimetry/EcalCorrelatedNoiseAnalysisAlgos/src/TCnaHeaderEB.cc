//----------Author's Name:FX Gentit and B.Fabbro  DAPNIA/SPP CEN Saclay
//----------Copyright:Those valid for CEA software
//----------Modified:07/06/2007

#include "Riostream.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEBParameters.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaHeaderEB.h"

ClassImp(TCnaHeaderEB)
  //____________________________________________________________________________

  TCnaHeaderEB::TCnaHeaderEB(){
  // Constructor without argument. Call to Init()

  Init();
}

TCnaHeaderEB::TCnaHeaderEB(Text_t        *name,          Text_t        *title,
			   TString       typ_ana,        const Int_t&  run_number, 
			   const Int_t&  first_evt,      const Int_t&  nb_of_taken_evts,
			   const Int_t&  super_module,   const Int_t&  nentries
			   ):TNamed(name,title) {

  // Constructor with arguments for reading DATA.
  // Called in GetReadyToReadData(...) of TCnaRunEB
  // Please give a name and a title containing info about what
  // you are doing and saving in the ROOT file

  Init();

  fTypAna        = typ_ana;
  fRunNumber     = run_number;
  fFirstEvt      = first_evt;
  fNbOfTakenEvts = nb_of_taken_evts;
  fSuperModule   = super_module;

  fNentries      = nentries;

  fNbBinsSampTime = fNbOfTakenEvts;
}
//------------------------------------------------------------------------------

TCnaHeaderEB::TCnaHeaderEB(Text_t        *name,        Text_t        *title,
			   TString       typ_ana,      const Int_t&  run_number, 
			   const Int_t&  first_evt,    const Int_t&  nb_of_taken_evts,
			   const Int_t&  super_module):TNamed(name,title) {
  
  // Constructor with arguments for reading ROOT file.
  // Called in GetReadyToReadRootfile(...) of TCnaReadEB
  // Please give a name and a title containing info about what
  // you are doing and saving in the ROOT file

  Init();

  fTypAna        = typ_ana;
  fRunNumber     = run_number;
  fFirstEvt      = first_evt;
  fNbOfTakenEvts = nb_of_taken_evts;
  fSuperModule   = super_module;

  fNbBinsSampTime = fNbOfTakenEvts;
}

TCnaHeaderEB::~TCnaHeaderEB() {
  //destructor

}

void TCnaHeaderEB::Init() {
  //Set default values in all variables and init the counters fxxxCalc

  //--------------------------------- INIT parameters

  //........................ RUN parameters

  fTypAna        = "!Analysis name> no info";
  fRunNumber     = (Int_t)0;
  fFirstEvt      = (Int_t)0;
  fNbOfTakenEvts = (Int_t)0;
  fSuperModule   = (Int_t)0;
  fStartTime     = (time_t)0;
  fStopTime      = (time_t)0;
  fStartDate     = "!Start date> no info";
  fStopDate      = "!Stop  date> no info";
  fNentries      = (Int_t)0;

  //....................... CMS/ECAL parameters

  TEBParameters* MyEcal = new TEBParameters();   fCnew++;

  fMaxTowEtaInSM = MyEcal->fMaxTowEtaInSM;
  fMaxTowPhiInSM = MyEcal->fMaxTowPhiInSM;
  fMaxTowInSM    = MyEcal->fMaxTowInSM;
  fMaxCrysInTow  = MyEcal->fMaxCrysInTow;
  fMaxSampADC    = MyEcal->fMaxSampADC;
  fMaxCrysInSM   = MyEcal->fMaxCrysInSM;

  //....................... CNA internal parameters
  fNbBinsADC      = (Int_t)100;      // 100 bins for histos of the event distributions
  fNbBinsSampTime = MyEcal->fMaxEvtsInBurstPedRun; // nb of bins for histos of the sample as a function of the event
                                                   // = nb of evts per burst (i.e. per gain) in a pedestal run  

  delete MyEcal;                                     fCdelete++;

  //--------------------------------- INIT counters
  fTowerNumbersCalc  = 0;
  fLastEvtNumberCalc = 0;
  fEvtNbInLoopCalc   = 0;
  fSampTimeCalc      = 0;
  fEvCalc            = 0;
  fVarCalc           = 0;
  fEvtsCalc          = 0;
  fCovCssCalc        = 0;
  fCorCssCalc        = 0;
  fCovSccCalc        = 0;
  fCorSccCalc        = 0;
  fCovSccMosCalc     = 0;
  fCorSccMosCalc     = 0;
  fCovMosccMotCalc   = 0;
  fCorMosccMotCalc   = 0;
  fEvEvCalc          = 0;
  fEvSigCalc         = 0;
  fEvCorCssCalc      = 0;
  fSigEvCalc         = 0;
  fSigSigCalc        = 0;
  fSigCorCssCalc     = 0;

  fSvCorrecCovCssCalc  = 0;
  fCovCorrecCovCssCalc = 0;
  fCorCorrecCovCssCalc = 0;
}

void TCnaHeaderEB::Print() {
  // Print the header
  cout << endl;
  cout << "     Header parameters " << endl;
  cout << endl;
  cout << "Run number                   : " << fRunNumber        << endl;
  cout << "First taken event            : " << fFirstEvt         << endl;
  cout << "Nb of taken events           : " << fNbOfTakenEvts    << endl;
  cout << "Super-module number          : " << fSuperModule      << endl;
  cout << "Time first taken event       : " << fStartTime        << endl;
  cout << "Time last  taken event       : " << fStopTime         << endl;
  cout << "Date first taken event       : " << fStartDate ;
  cout << "Date last  taken event       : " << fStopDate  ;
  cout << "Number of entries            : " << fNentries         << endl;
  cout << endl;
  cout << "Max nb of towers in SM       : " << fMaxTowInSM       << endl;
  cout << "Max nN of crystals in tower  : " << fMaxCrysInTow     << endl;
  cout << " ( => Max nb of crystals in SM =  " << fMaxCrysInSM << " ) " << endl;
  cout << endl;
  cout << "Max nb of samples ADC        : " << fMaxSampADC       << endl;
  cout << "Nb of bins for ADC distribs  : " << fNbBinsADC        << endl;
  //cout << "Nb of points for evol graphs : " << fNbBinsEvol       << endl;
  cout << "Nb of bins for sample/evt    : " << fNbBinsSampTime   << endl;
  cout << endl;

  cout << "     Header counters " << endl;
  cout << endl;
  cout << "Tower Numbers                                : "
       << fTowerNumbersCalc << endl;
  cout << "Numbers of found evts                        : "
       << fLastEvtNumberCalc << endl;
  cout << "Event numbers in loop of the data reading    : "
       << fEvtNbInLoopCalc << endl;
  cout << "Samples as a function of time histograms     : "
       << fSampTimeCalc << endl;
  cout << "Expectation values histogram                 : "
       << fEvCalc    << endl;
  cout << "Variances histogram                          : "
       << fVarCalc   << endl;
  cout << "Nb of events histograms (ADC's values)       : "
       << fEvtsCalc  << endl;
  cout << "Nb of (sample,sample) covariance  matrices   : "
       << fCovCssCalc << endl;
  cout << "Nb of (sample,sample) correlation matrices   : "
       << fCorCssCalc << endl;
  cout << "Nb of (channel,channel) covariance  matrices : "
       << fCovSccCalc << endl;
  cout << "Nb of (channel,channel) correlation matrices : "
       << fCorSccCalc << endl;
  cout << "Nb of (channel,channel) cov mat mean on samp : "
       << fCovSccMosCalc << endl;
  cout << "Nb of (channel,channel) cor mat mean on samp : "
       << fCorSccMosCalc << endl;
  cout << "Nb of mean cov(c,c) mean on samp, all towers : "
       << fCovSccMosCalc << endl;
  cout << "Nb of mean cor(c,c) mean on samp, all towers : "
       << fCorSccMosCalc << endl;

  cout << "Exp. val. of the exp. val. of the samples    : "
       << fEvEvCalc     << endl;
  cout << "Expect. val. of the sigmas of the samples    : "
       << fEvSigCalc    << endl;
  cout << "Expect. val. of the (samp,samp) correlations : "
       << fEvCorCssCalc << endl;

  cout << "Sigmas of the exp. val. of the samples       : "
       << fSigEvCalc     << endl;
  cout << "Sigmas of the sigmas of the samples          : "
       << fSigSigCalc     << endl;
  cout << "Sigmas of the (samp,samp) correlations       : "
       << fSigCorCssCalc     << endl;

  cout << "Corrections to the sample values             : "
       << fSvCorrecCovCssCalc  << endl;
  cout << "Corrections to the (samp,samp) covariances   : "
       << fCovCorrecCovCssCalc << endl;
  cout << "Corrections to the (samp,samp) correlations  : "
       << fCorCorrecCovCssCalc << endl;
  cout << endl;
}
