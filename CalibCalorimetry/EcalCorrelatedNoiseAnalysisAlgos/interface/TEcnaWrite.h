#ifndef ZTR_TEcnaWrite
#define ZTR_TEcnaWrite

#include <Riostream.h>

#include "TObject.h"
#include "TSystem.h"
#include "Riostream.h"
#include <cmath>
#include <ctime>

#include "TVectorD.h"
#include "TMatrixD.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaObject.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaNumbering.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParEcal.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParPaths.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParCout.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaHeader.h"

///-----------------------------------------------------------
///   TEcnaWrite.h
///   Update: 05/10/2012
///   Authors:   B.Fabbro (bernard.fabbro@cea.fr), FX Gentit
///              DSM/IRFU/SPP CEA-Saclay
///   Copyright: Those valid for CEA sofware
///
///   ECNA web page:
///     http://cms-fabbro.web.cern.ch/cms-fabbro/
///     cna_new/Correlated_Noise_Analysis/ECNA_main_page.htm
///-----------------------------------------------------------

class TEcnaWrite : public TObject {
private:
  //..... Attributes

  Int_t fCnew, fCdelete;
  Int_t fCnewRoot, fCdeleteRoot;

  Int_t fgMaxCar;  // Max nb of caracters for char*

  TString fTTBELL;

  Int_t fCnaCommand, fCnaError;

  //...............................................................

  Int_t fFlagPrint;
  Int_t fCodePrintComments, fCodePrintWarnings, fCodePrintAllComments, fCodePrintNoComment;

  TEcnaParEcal* fEcal;
  TString fFlagSubDet;

  TEcnaNumbering* fEcalNumbering;
  TEcnaParPaths* fCnaParPaths;
  TEcnaParCout* fCnaParCout;

  std::ofstream fFcout_f;

  //...................................... Codes for file names
  Int_t fCodeHeaderAscii;
  Int_t fCodeRoot;

  Int_t fCodeNbOfEvts;
  Int_t fCodePed;
  Int_t fCodeTno;
  Int_t fCodeLfn;
  Int_t fCodeHfn;
  Int_t fCodeMeanCorss;
  Int_t fCodeSigCorss;

  Int_t fCodeCovCss;
  Int_t fCodeCorCss;

  Int_t fCodeAdcEvt;
  Int_t fCodeMSp;
  Int_t fCodeSSp;

  Int_t fCodeAvPed;
  Int_t fCodeAvTno;
  Int_t fCodeAvMeanCorss;
  Int_t fCodeAvSigCorss;

  Int_t fCodeLfCov;
  Int_t fCodeLfCor;
  Int_t fCodeHfCov;
  Int_t fCodeHfCor;

  Int_t fCodeLFccMoStins;
  Int_t fCodeHFccMoStins;

  //..........................................................................
  Int_t fSectChanSizeX, fSectChanSizeY;
  Int_t fSectSampSizeX, fSectSampSizeY;

  Int_t fNbChanByLine;  // Nb channels by line (for ASCII results file)
  Int_t fNbSampByLine;  // Nb samples by line  (for ASCII results file)
  Int_t fUserSamp;      // Current sample  number (for ASCII results file)
  Int_t fStexStinUser;  // Current Stin number in Stex
  Int_t fStinEchaUser;  // Current electronic channel number in Stin
                        // for ASCII results file ([0,24] for EB, [1,25] for EE)

  Double_t** fjustap_2d_ev;
  Double_t* fjustap_1d_ev;

  Double_t** fjustap_2d_var;
  Double_t* fjustap_1d_var;

  Double_t** fjustap_2d_cc;
  Double_t* fjustap_1d_cc;

  Double_t** fjustap_2d_ss;
  Double_t* fjustap_1d_ss;

  //.................... Private methods

  void fAsciiFileWriteHeader(const Int_t&);
  void fT2dWriteAscii(const Int_t&, const Int_t&, const Int_t&, const Int_t&, const TMatrixD&);

public:
  //..... Public attributes

  TString fAnaType;
  Int_t fNbOfSamples;
  Int_t fRunNumber;
  Int_t fFirstReqEvtNumber;
  Int_t fLastReqEvtNumber;
  Int_t fReqNbOfEvts;
  Int_t fStexNumber;
  TString fStexName;
  TString fStinName;

  TString fPathForAsciiFiles;
  TString fStartDate, fStopDate;
  time_t fStartTime, fStopTime;

  TString fRootFileNameShort;  // name of  the results ROOT file
  TString fRootFileName;       // name of  the results ROOT file with its path = fPathRoot/fRootFileNameShort

  TString fAsciiFileName;       // name of  the results ASCII file
  TString fAsciiFileNameShort;  // name of  the results ASCII file = fPathAscii/fAsciiFileNameShort

  //..... Methods

  TEcnaWrite();
  TEcnaWrite(TEcnaObject*, const TString&);
  TEcnaWrite(const TString&, TEcnaParPaths*, TEcnaParCout*, TEcnaParEcal*, TEcnaNumbering*);

  ~TEcnaWrite() override;

  void Init();
  void SetEcalSubDetector(const TString&);
  void SetEcalSubDetector(const TString&, TEcnaParEcal*, TEcnaNumbering*);

  //...................................................... making file name method
  void fMakeResultsFileName();  // => default: arg = fCodeRoot
  void fMakeResultsFileName(const Int_t&);

  //.................................... ASCII writing file methods
  // void    WriteAsciiSampleMeans();    // methode a remettre ?
  // void    WriteAsciiSampleSigmas();   // methode a remettre ?

  void WriteAsciiCovariancesBetweenSamples(const Int_t&, const Int_t&, const Int_t&, const TMatrixD&);
  void WriteAsciiCorrelationsBetweenSamples(const Int_t&, const Int_t&, const Int_t&, const TMatrixD&);

  void WriteAsciiHisto(const TString&, const Int_t&, const TVectorD&);

  //...........................................................................
  const TString& GetAsciiFileName() const;
  const TString& GetRootFileName() const;
  const TString& GetRootFileNameShort() const;
  const TString& GetAnalysisName() const;

  Int_t GetNbOfSamples();
  Int_t GetRunNumber();
  Int_t GetFirstReqEvtNumber();
  Int_t GetReqNbOfEvts();
  Int_t GetStexNumber();

  Int_t NumberOfEventsAnalysis(Int_t**, const Int_t&, const Int_t&, const Int_t&);  // Called by TEcnaRun
  Int_t NumberOfEventsAnalysis(Int_t*, const Int_t&, const Int_t&, const Int_t&);   // Called by TEcnaRead

  void RegisterFileParameters(const TString&,
                              const Int_t&,
                              const Int_t&,
                              const Int_t&,
                              const Int_t&,
                              const Int_t&,
                              const Int_t&,
                              const TString&,
                              const TString&,
                              const time_t,
                              const time_t);

  void RegisterFileParameters(
      const TString&, const Int_t&, const Int_t&, const Int_t&, const Int_t&, const Int_t&, const Int_t&);

  ClassDefOverride(TEcnaWrite, 1)  // Writing in file (.ascii, .root) methods for CNA (Correlated Noises Analysis)
};

#endif  //    ZTR_TEcnaParameter
