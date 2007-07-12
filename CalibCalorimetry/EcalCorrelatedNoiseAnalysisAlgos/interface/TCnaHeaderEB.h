#ifndef ROOT_TCnaHeaderEB
#define ROOT_TCnaHeaderEB

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCnaHeaderEB   Header of CNA ROOT file                                 //
//                                                                      //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include "TString.h"
#include "TNamed.h"
#include <time.h>

class TCnaHeaderEB : public TNamed {

protected:

  void      Init();

public:

  Int_t   fCnew;
  Int_t   fCdelete;

  //....... Header parameters
  TString fTypAna;              // name of the analysis (default = "std")
  Int_t   fRunNumber;           // Run number
  Int_t   fFirstEvt;            // First taken event number
  Int_t   fNbOfTakenEvts;       // Number of taken evts
  Int_t   fSuperModule;         // SuperModule number
  TString fStartDate;           // Start date
  TString fStopDate;            // Stop date
  time_t  fStartTime;           // Start time
  time_t  fStopTime;            // Stop time
  Int_t   fNentries;            // Number of events in the run

  Int_t   fMaxTowEtaInSM;       // Nb of towers in eta in a SuperModule
  Int_t   fMaxTowPhiInSM;       // Nb of towers in phi in a SuperModule
  Int_t   fMaxTowInSM;          // Maximum number of towers in a SuperModule ( = fSizeSMEta*fSizeSMPhi)
  Int_t   fMaxCrysInTow;        // Maximum number of crystals in tower ( tower = square, so sizex = sizey = sqrt(fMaxCrysInTow) )
  Int_t   fMaxSampADC;          // Maximum number of samples
  Int_t   fMaxCrysInSM;         // Maximum number of channels ( = fMaxTowInSM x fMaxCrysInTow )

  Int_t   fNbBinsADC;           // Number of bins for the ADC event distributions
  Int_t   fNbBinsSampTime;      // Number of bins for the histos of samples as a function of time
  Int_t   fNbBinsEvol;          // Number of points for the evolution graphs

  //....... Header counters
  Int_t   fTowerNumbersCalc;    // Nb of entries of type TowerNumbers
  Int_t   fLastEvtNumberCalc;   // Nb of entries of type LastEvtNumber
  Int_t   fEvtNbInLoopCalc;     // Nb of entries of type EvtNbInLoop

  Int_t   fSampTimeCalc;        // *Nb of entries of type SampTime
  Int_t   fEvCalc;              // *Nb of entries of type Ev
  Int_t   fVarCalc;             // *Nb of entries of type Var
  Int_t   fEvtsCalc;            // *Nb of entries of type Evts
  Int_t   fCovCssCalc;          // *Nb of entries of type CovCss
  Int_t   fCorCssCalc;          // *Nb of entries of type CorCss
  Int_t   fCovSccCalc;          // *Nb of entries of type CovScc
  Int_t   fCorSccCalc;          // *Nb of entries of type CorScc
  Int_t   fCovSccMosCalc;       // *Nb of entries of type CovSccMos
  Int_t   fCorSccMosCalc;       // *Nb of entries of type CorSccMos
  Int_t   fCovMosccMotCalc;     // *Nb of entries of type CovMosccMot
  Int_t   fCorMosccMotCalc;     // *Nb of entries of type CorMosccMot
  Int_t   fEvEvCalc;            // *Nb of entries of type EvEv
  Int_t   fEvSigCalc;           // *Nb of entries of type EvSig
  Int_t   fEvCorCssCalc;        // *Nb of entries of type EvCorCss
  Int_t   fSigEvCalc;           // *Nb of entries of type SigEv
  Int_t   fSigSigCalc;          // *Nb of entries of type SigSig
  Int_t   fSigCorCssCalc;       // *Nb of entries of type SigCorCss
  Int_t   fSvCorrecCovCssCalc;  // *Nb of entries of type SvCorrecCovCss
  Int_t   fCovCorrecCovCssCalc; // *Nb of entries of type CovCorrecCovCss
  Int_t   fCorCorrecCovCssCalc; // *Nb of entries of type CorCorrecCovCss

  //------------------------------- methods

  TCnaHeaderEB(); 
  TCnaHeaderEB(Text_t*,  Text_t*, TString,
	     const Int_t&,    const Int_t&,  const Int_t&,  const Int_t&,
	     const  Int_t&);
  TCnaHeaderEB(Text_t*,  Text_t*, TString,
	     const Int_t&,    const Int_t&,  const Int_t&,  const Int_t&);
  ~TCnaHeaderEB();

  void Print();
  ClassDef(TCnaHeaderEB,1)  //Header of CNA ROOT file
};
#endif
