#ifndef CL_EcnaAnalyzer_H
#define CL_EcnaAnalyzer_H

// -*- C++ -*-
//
// Package:    EcalCorrelatedNoiseAnalysisModules
// Class:      EcnaAnalyzer
//
/**\class EcnaAnalyzer EcnaAnalyzer.cc
 CalibCalorimetry/EcalCorrelatedNoiseAnalysisModules/src/EcnaAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Bernard Fabbro
//         Created:  Fri Jun  2 10:27:01 CEST 2006
// $Id: EcnaAnalyzer.h,v 1.3 2013/04/05 20:17:20 wmtan Exp $
//
//

// system include files
#include "Riostream.h"
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <csignal>
#include <sys/time.h>

// ROOT include files
#include "TObject.h"
#include "TString.h"
#include "TSystem.h"
#include "TTreeIndex.h"
#include "TVectorD.h"

// CMSSW include files
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Provenance/interface/EventID.h"

#include "DataFormats/Provenance/interface/Timestamp.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

// user include files
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaNumbering.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaObject.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParEcal.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParPaths.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRun.h"

///-----------------------------------------------------------
///   EcnaAnalyzer.h
///   Update: 11/0'/2011
///   Authors:   B.Fabbro (bernard.fabbro@cea.fr)
///              DSM/IRFU/SPP CEA-Saclay
///   Copyright: Those valid for CEA sofware
///
///   ECNA web page:
///     http://cms-fabbro.web.cern.ch/cms-fabbro/
///     cna_new/Correlated_Noise_Analysis/ECNA_cna_1.htm
///-----------------------------------------------------------
///
///----------------------------------- Analysis name codes
///------------------------------------------
///
///      TString  AnalysisName: code for the analysis. According to this code,
///                             the analyzer selects the event type
///                             (PEDESTAL_STD, PEDESTAL_GAP, LASER_STD, etc...)
///                             and some other event characteristics
///                             (example: the gain in pedestal runs:
///                              AnalysisName = "Ped1" or "Ped6" or "Ped12")
///                             The string AnalysisName is automatically
///                             included in the name of the results files
///
///                  AnalysisName  RunType         Gain    DBLS (Dynamic
///                  BaseLine Substraction)
///                  ..........................................
///
///                  AdcAny        any run type       0    No
///
///                  AdcPed1       fPEDESTAL_STD      3    No
///                  AdcPed6       fPEDESTAL_STD      2    No
///                  AdcPed12      fPEDESTAL_STD      1    No
///
///                  AdcPeg12      fPEDESTAL_GAP      1    No
///
///                  AdcLaser      fLASER_STD         0    No
///                  AdcPes12      fPEDSIM            0    No
///
///                  AdcPhys       fPHYSICS_GLOBAL    0    No
///
///
///                  AdcSPed1      fPEDESTAL_STD      3    Yes
///                  AdcSPed6      fPEDESTAL_STD      2    Yes
///                  AdcSPed12     fPEDESTAL_STD      1    Yes
///
///                  AdcSPeg12     fPEDESTAL_GAP      1    Yes
///
///                  AdcSLaser     fLASER_STD         0    Yes
///                  AdcSPes12     fPEDSIM            0    Yes
///
///--------------------------------------------------------------------------------------------------

//
// class declaration
//

class EcnaAnalyzer : public edm::one::EDAnalyzer<> {
public:
  enum { kChannels = 1700, kGains = 3, kFirstGainId = 1 };

  explicit EcnaAnalyzer(const edm::ParameterSet &);
  ~EcnaAnalyzer() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  TString runtype(const Int_t &);
  Int_t gainvalue(const Int_t &);
  void CheckMsg(const Int_t &, const Int_t &);
  void CheckMsg(const Int_t &);
  Bool_t AnalysisOutcome(const TString &);

private:
  constexpr static Int_t fgMaxCar = 512;  // Max nb of caracters for char*
  TString fTTBELL;

  // ----------member data ---------------------------
  unsigned int verbosity_;
  Int_t nChannels_;
  Int_t iEvent_;  // should be removed when we can access class EventID
  std::string eventHeaderProducer_;
  std::string digiProducer_;
  std::string eventHeaderCollection_;
  std::string EBdigiCollection_;
  std::string EEdigiCollection_;
  edm::EDGetTokenT<EcalRawDataCollection> eventHeaderToken_;
  edm::EDGetTokenT<EBDigiCollection> EBdigiToken_;
  edm::EDGetTokenT<EEDigiCollection> EEdigiToken_;

  TString sAnalysisName_;
  TString sNbOfSamples_;
  TString sFirstReqEvent_;
  TString sLastReqEvent_;
  TString sReqNbOfEvts_;
  TString sStexName_;
  TString sStexNumber_;

  Bool_t fOutcomeError;

  Int_t fEvtNumber;
  Int_t fEvtNumberMemo;
  Int_t fRecNumber;
  Int_t fCurrentEventNumber;
  Int_t fNbOfSelectedEvents;

  std::vector<Int_t> fBuildEventDistribBad;
  std::vector<Int_t> fBuildEventDistribGood;

  TString fCfgAnalyzerParametersFilePath;  // absolute path for the analyzer
                                           // parameters files (/afs/etc...)
  TString fCfgAnalyzerParametersFileName;  // name of the analyzer parameters file
  std::ifstream fFcin_f;

  TString fAnalysisName;
  Int_t fChozenGainNumber;     // determined from fAnalysisName
  Int_t fChozenRunTypeNumber;  // determined from fAnalysisName
  TString fDynBaseLineSub;     // determined from fAnalysisName

  Int_t fNbOfSamples;
  Int_t fRunNumber;
  Int_t fRunTypeNumber;
  Int_t fFirstReqEvent;
  Int_t fLastReqEvent;
  TString fStexName;
  Int_t fStexNumber;

  Int_t fReqNbOfEvts;
  Int_t fMgpaGainNumber;

  Int_t fSMIndexBegin;
  Int_t fSMIndexStop;
  Int_t fDeeIndexBegin;
  Int_t fDeeIndexStop;
  Int_t fStexIndexBegin;
  Int_t fStexIndexStop;

  Int_t fFedTcc;
  std::vector<Int_t> fSMFromFedTcc;
  std::vector<Int_t> fESFromFedTcc;
  Int_t fTreatedFedOrder;
  std::vector<Int_t> fFedStatusOrder;
  Int_t fFedId;
  std::vector<std::string> fDeeNumberString;

  Int_t fMaxTreatedStexCounter = 0;
  Int_t fDeeDS5Memo1;
  Int_t fDeeDS5Memo2;
  std::vector<Int_t> fStexDigiOK;
  std::vector<Int_t> fStexNbOfTreatedEvents;
  std::vector<Int_t> fStexStatus;

  Int_t fMaxFedUnitCounter = 0;
  std::vector<Int_t> fFedStatus;
  std::vector<Int_t> fFedDigiOK;
  std::vector<Int_t> fFedNbOfTreatedEvents;

  Int_t fMemoCutOK;
  Int_t fNbOfTreatedStexs;
  std::vector<Int_t> fNbOfTreatedFedsInDee;
  std::vector<Int_t> fNbOfTreatedFedsInStex;

  Int_t fANY_RUN;
  Int_t fPEDESTAL_STD;
  Int_t fPEDESTAL_GAP;
  Int_t fLASER_STD;
  Int_t fPHYSICS_GLOBAL;
  Int_t fPEDSIM;

  std::vector<time_t> fTimeFirst;
  std::vector<time_t> fTimeLast;
  std::vector<TString> fDateFirst;
  std::vector<TString> fDateLast;

  std::vector<Int_t> fMemoDateFirstEvent;

  TEcnaObject fMyEcnaEBObjectManager;
  TEcnaObject fMyEcnaEEObjectManager;

  std::vector<std::unique_ptr<TEcnaRun>> fMyCnaEBSM;
  std::vector<std::unique_ptr<TEcnaRun>> fMyCnaEEDee;

  TEcnaNumbering fMyEBNumbering;
  TEcnaParEcal fMyEBEcal;

  TEcnaNumbering fMyEENumbering;
  TEcnaParEcal fMyEEEcal;

  //  Int_t** fT2d_LastEvt; // 2D array[channel][sample] max nb of evts read for
  //  a given (channel,sample) Int_t*  fT1d_LastEvt;

  constexpr static Int_t fMaxRunTypeCounter = 26;
  std::array<Int_t, fMaxRunTypeCounter> fRunTypeCounter;

  constexpr static Int_t fMaxMgpaGainCounter = 4;  // Because chozen gain = 0,1,2,3
  std::array<Int_t, fMaxMgpaGainCounter> fMgpaGainCounter;

  constexpr static Int_t fMaxFedIdCounter = 54;
  std::array<Int_t, fMaxFedIdCounter> fFedIdCounter;
};

#endif
