#ifndef ROOT_TEcnaHeader
#define ROOT_TEcnaHeader

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEcnaHeader   Header of CNA ROOT file                                 //
//                                                                      //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include "TROOT.h"
#include "TObject.h"
#include "TString.h"
#include "TNamed.h"
#include "Riostream.h"
#include <time.h>
#include <TMath.h>

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaObject.h"

///-----------------------------------------------------------
///   TEcnaHeader.h
///   Update: 16/02/2011
///   Authors:   FX Gentit, B.Fabbro (bernard.fabbro@cea.fr)
///              DSM/IRFU/SPP CEA-Saclay
///   Copyright: Those valid for CEA sofware
///
///   ECNA web page:
///     http://cms-fabbro.web.cern.ch/cms-fabbro/
///     cna_new/Correlated_Noise_Analysis/ECNA_cna_1.htm
///-----------------------------------------------------------
///
///   Header of ECNA result file (.root file)
///
  
class TEcnaHeader : public TNamed {

protected:

  void      Init();

public:

  Int_t   fCnew;
  Int_t   fCdelete;

  Int_t   fgMaxCar;   // Max nb of caracters for char*

  //....... Header parameters
  TString fTypAna;              // name of the analysis (default = "std")
  Int_t   fNbOfSamples;         // Number of samples for calculations
  Int_t   fRunNumber;           // Run number
  Int_t   fFirstReqEvtNumber;   // First requested event number
  Int_t   fLastReqEvtNumber;    // Number of taken evts
  Int_t   fReqNbOfEvts;         // Requested number of events
  Int_t   fStex;                // Stex number

  time_t  fStartTime;           // Start time
  time_t  fStopTime;            // Stop time
  TString fStartDate;           // Start date
  TString fStopDate;            // Stop date

  Int_t   fRunType;             // run type

  //....... Header counters
  Int_t   fStinNumbersCalc;     // Nb of entries of type StinNumbers
  Int_t   fNbOfEvtsCalc;        // Nb of entries of type NbOfEvts

  Int_t   fAdcEvtCalc;          // *Nb of entries of type SampTime
  Int_t   fMSpCalc;             // *Nb of entries of type Ev
  Int_t   fSSpCalc;             // *Nb of entries of type Var
  Int_t   fAvTnoCalc;           // *Nb of entries of type AvTotn
  Int_t   fAvLfnCalc;           // *Nb of entries of type AvLfn
  Int_t   fAvHfnCalc;           // *Nb of entries of type AvHfn

  Int_t   fCovCssCalc;          // *Nb of entries of type CovCss
  Int_t   fCorCssCalc;          // *Nb of entries of type CorCss
  Int_t   fHfCovCalc;           // *Nb of entries of type HfCov
  Int_t   fHfCorCalc;           // *Nb of entries of type HfCor
  Int_t   fLfCovCalc;           // *Nb of entries of type LfCov
  Int_t   fLfCorCalc;           // *Nb of entries of type LfCor
  Int_t   fLFccMoStinsCalc;     // *Nb of entries of type LFccMoStins
  Int_t   fHFccMoStinsCalc;     // *Nb of entries of type HFccMoStins
  Int_t   fPedCalc;             // *Nb of entries of type Ped
  Int_t   fTnoCalc;             // *Nb of entries of type Tno
  Int_t   fMeanCorssCalc;       // *Nb of entries of type EvCorCss
  Int_t   fLfnCalc;             // *Nb of entries of type Lfn
  Int_t   fHfnCalc;             // *Nb of entries of type Hfn
  Int_t   fSigCorssCalc;        // *Nb of entries of type SigCorCss

  Int_t   fAvPedCalc;           // *Nb of entries of type AvPed
  Int_t   fAvMeanCorssCalc;     // *Nb of entries of type AvEvCorss
  Int_t   fAvSigCorssCalc;      // *Nb of entries of type AvSigCorss

  //------------------------------- methods
  TEcnaHeader();
  TEcnaHeader(TEcnaObject*, const Text_t*, const Text_t*);
  //TEcnaHeader(const Text_t*, const Text_t*);
  ~TEcnaHeader();

//  void HeaderParameters(Text_t*, Text_t*, const TString&,      const Int_t&, 
// 			  const Int_t&,     const Int_t&, const Int_t&, const Int_t&,
// 		       	const  Int_t&);
//  void HeaderParameters(Text_t*,  Text_t*, const TString&,    const Int_t&, 
// 			const Int_t&,    const Int_t&, const Int_t&, const Int_t&);

  void HeaderParameters( const TString&,      const Int_t&, const Int_t&,
			 const Int_t&, const Int_t&, const Int_t&, const Int_t&);
  void HeaderParameters( const TString&,      const Int_t&, const Int_t&,
			 const Int_t&, const Int_t&, const Int_t&, const Int_t&, const Int_t&);

  void Print();
  ClassDef(TEcnaHeader,1)  //Header of CNA ROOT file
};
#endif
