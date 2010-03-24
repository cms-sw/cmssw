#ifndef ERRORSANALYZER_HH
#define ERRORSANALYZER_HH

// -*- C++ -*-
//
// Package:    ErrorsAnalyzer
// Class:      ErrorsAnalyzer
// 
/**\class ErrorsAnalyzer ErrorsAnalyzer.cc MuonAnalysis/MomentumScaleCalibration/plugins/ErrorsAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Marco De Mattia
//         Created:  Thu Sep 11 12:16:00 CEST 2008
// $Id: ErrorsAnalyzer.h,v 1.1 2010/03/24 15:15:19 demattia Exp $
//
//

// system include files
#include <memory>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <TH1D.h>
#include <TProfile.h>
#include <TString.h>
#include <TCanvas.h>
#include <TGraphAsymmErrors.h>
#include <TROOT.h>

#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/RootTreeHandler.h"

//
// class declaration
//

using namespace std;

class ErrorsAnalyzer : public edm::EDAnalyzer
{
public:
  explicit ErrorsAnalyzer(const edm::ParameterSet&);
  ~ErrorsAnalyzer();

private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void fillHistograms();
  void drawHistograms(const TProfile * histo, const TProfile * histoPlusErr, const TProfile * histoMinusErr, const TString & type);
  void fillValueError();
  virtual void endJob() {};

  TString treeFileName_;
  int resolFitType_;
  uint32_t maxEvents_;
  TString outputFileName_;
  int ptBins_;
  double ptMin_;
  double ptMax_;
  int etaBins_;
  double etaMin_;
  double etaMax_;
  bool debug_;

  vector<double> parameters_;
  vector<double> errors_;
  vector<int> errorFactors_;

  vector<double> valuePlusError_;
  vector<double> valueMinusError_;

  TProfile * sigmaPtVsEta_;
  TProfile * sigmaPtVsEtaPlusErr_;
  TProfile * sigmaPtVsEtaMinusErr_;

  TProfile * sigmaPtVsPt_;
  TProfile * sigmaPtVsPtPlusErr_;
  TProfile * sigmaPtVsPtMinusErr_;
};

#endif // RESOLUTIONANALYZER_HH
